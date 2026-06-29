"""
36_topX_threshold_rsa.py
Continuous cross-day RSA at the strongest pairs only — does the top-X% of
the within-day signal correlation distribution preserve its cross-day
Pearson r as the threshold becomes stricter?

For each cross-day pair (A–B, B–C, A–C) in each mouse, treated separately
(NOT averaged within mouse):

  (1) Compute the signal correlation matrices for the two days on the
      matched-cell set (the same set used by the rest of the thesis;
      Figure 3, etc.).
  (2) Flatten the upper-triangle off-diagonal entries into a 1-D vector
      of pair-correlation values per day.
  (3) Rank pairs by mean |r| across the two days (symmetric — doesn't
      privilege either day).
  (4) For each threshold % in [25, 20, 15, 10, 5, 1]:
        Select the top-X% pairs by mean |r|.
        Compute Pearson r between those selected pairs' day-A values
        and their day-B values.
        This is the continuous cross-day RSA at that threshold.

Across mice:
  For each (cross_day_pair, threshold), compute cohort mean and 95%
  bootstrap CI by resampling MICE with replacement (B = 10 000).
  Each mouse contributes ONE observation per (cross_day_pair × threshold);
  A–B and B–C are kept as distinct cross-day pairs (not collapsed into a
  single 1-day estimate), so each mouse contributes 3 observations per
  threshold (one A–B, one B–C, one A–C).

The figure shows three lines (A–B purple, B–C magenta, A–C yellow) with
shaded 95% bootstrap CI bands. Visual interpretation: if A–C's CI band
sits below A–B and B–C across the threshold range, the strongest pairs
preserve their cross-day similarity more at 1-day than at 2-day —
time-graded drift visible at the strongest pairs.

Inputs (per numeric-name container):
  {root}/{cid}/{A,B,C}/events_matrix.npy
  {root}/{cid}/{A,B,C}/movie_repeat_frames.npy

Outputs:
  outputs/movie1/topX_threshold_rsa.csv         per-pair table
  outputs/movie1/topX_threshold_rsa_stats.csv   cohort means + bootstrap CIs
  outputs/movie1/topX_threshold_rsa.png         figure

Usage:
  python3 scripts/36_topX_threshold_rsa.py
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-10
SESSIONS = ["A", "B", "C"]
PAIRS = [("A", "B", 1, "#a05a8a"),   # purple — 1 day
         ("B", "C", 1, "#d63384"),   # magenta — 1 day
         ("A", "C", 2, "#ffd60a")]   # yellow — 2 day
THRESHOLDS = [25, 20, 15, 10, 5, 1]   # % of pairs kept (top X%)

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--boot", type=int, default=10000,
                    help="Bootstrap resamples for CI (default 10 000).")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

ROOT = Path(args.root)
rng = np.random.default_rng(args.seed)


# Per-session signal correlation matrix (matched cells, fulldata 10-rep,
# frame rate — same recipe as scripts 28, 31, 36)

def present_mask(events):
    return ~np.all(np.isnan(events), axis=1)


def session_signal_corr(events, frames, cell_idx):
    """Signal correlation matrix on matched cells, fulldata 10-rep at frame
    rate. Returns n×n matrix; NaN rows/cols for silent cells."""
    ev_p = events[cell_idx, :].T
    L_common = int(min(int(ef) - int(sf) + 1 for sf, ef in frames))
    reps = []
    for sf, _ in frames:
        sf = int(sf)
        reps.append(ev_p[sf: sf + L_common, :])
    T = np.stack(reps, axis=0)
    psth = T.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S = np.corrcoef(psth, rowvar=False)
    dead = psth.var(axis=0) <= EPS
    S[dead, :] = np.nan
    S[:, dead] = np.nan
    return S


def top_pct_cross_day_rsa(C_a, C_b, top_pct):
    """Rank pairs by mean |r| across the two days, keep top-X% by that mean,
    compute Pearson r between the two days' values at those selected pairs."""
    n = C_a.shape[0]
    iu = np.triu_indices(n, k=1)
    a = C_a[iu]
    b = C_b[iu]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return np.nan
    a = a[mask]
    b = b[mask]
    # Symmetric ranking score
    score = (np.abs(a) + np.abs(b)) / 2.0
    # Pick top X%
    n_keep = max(3, int(round(len(score) * top_pct / 100.0)))
    keep_idx = np.argsort(score)[::-1][:n_keep]
    a_sel = a[keep_idx]
    b_sel = b[keep_idx]
    if np.std(a_sel) < EPS or np.std(b_sel) < EPS:
        return np.nan
    return float(np.corrcoef(a_sel, b_sel)[0, 1])


# Walk containers and compute per-pair table

print("=" * 70)
print(" Computing top-X% threshold cross-day RSA per mouse")
print("=" * 70)

cid_dirs = [p for p in sorted(ROOT.iterdir())
            if p.is_dir() and p.name.isdigit()
            and all((p / s / "events_matrix.npy").exists() for s in SESSIONS)]
print(f"  Found {len(cid_dirs)} numeric containers with all 3 events_matrix.npy")

rows = []
for cid_dir in cid_dirs:
    cid = cid_dir.name
    ev = {s: np.load(cid_dir / s / "events_matrix.npy") for s in SESSIONS}
    fr = {s: np.load(cid_dir / s / "movie_repeat_frames.npy") for s in SESSIONS}
    matched = np.where(present_mask(ev["A"])
                       & present_mask(ev["B"])
                       & present_mask(ev["C"]))[0]
    n_cells = int(len(matched))
    if n_cells < 10:
        print(f"  skip {cid}: only {n_cells} matched cells (need ≥ 10)")
        continue

    # Per-session signal correlation matrices on matched cells
    mats = {s: session_signal_corr(ev[s], fr[s], matched) for s in SESSIONS}

    for a, b, gap, _color in PAIRS:
        for thresh in THRESHOLDS:
            r = top_pct_cross_day_rsa(mats[a], mats[b], thresh)
            rows.append(dict(
                container_id=cid, n_cells=n_cells,
                pair=f"{a}-{b}", days_apart=gap,
                threshold_pct=thresh,
                cross_day_rsa=r,
            ))
    print(f"  {cid}: done ({n_cells} cells)")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "topX_threshold_rsa.csv", index=False)
print(f"\nSaved per-pair table: {ROOT / 'topX_threshold_rsa.csv'} "
      f"({len(df)} rows, {df['container_id'].nunique()} mice)")


# Cohort aggregation with 95% bootstrap CIs (resample mice with replacement)

def boot_mean_ci(values, B=10000, alpha=0.05, rng=None):
    """Mean + percentile bootstrap CI on a 1D array. Resampling at the
    observation level (which here = per-mouse, since each mouse contributes
    one value per (pair, threshold))."""
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    idx = rng.integers(0, n, size=(B, n))
    boot_means = x[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


print(f"\nComputing cohort means + 95% bootstrap CIs ({args.boot} resamples)")
stat_rows = []
for a, b, gap, _color in PAIRS:
    for thresh in THRESHOLDS:
        sub = df[(df["pair"] == f"{a}-{b}") & (df["threshold_pct"] == thresh)]
        vals = sub["cross_day_rsa"].dropna().values
        mean, lo, hi = boot_mean_ci(vals, B=args.boot, rng=rng)
        stat_rows.append(dict(
            pair=f"{a}-{b}", days_apart=gap,
            threshold_pct=thresh,
            n_mice=len(vals),
            mean=mean, ci_lo=lo, ci_hi=hi,
        ))
stats = pd.DataFrame(stat_rows)
stats.to_csv(ROOT / "topX_threshold_rsa_stats.csv", index=False)

print("\nCohort summary (mean ± 95% bootstrap CI):")
for thresh in THRESHOLDS:
    print(f"\n  Threshold = top {thresh}% strongest pairs:")
    for a, b, gap, _ in PAIRS:
        row = stats[(stats["pair"] == f"{a}-{b}")
                    & (stats["threshold_pct"] == thresh)].iloc[0]
        sig = "*" if (row["ci_lo"] > 0 or row["ci_hi"] < 0) else " "
        print(f"    {a}-{b} ({gap}d, n={int(row['n_mice'])}): "
              f"mean = {row['mean']:+.3f}  "
              f"95% CI [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}] {sig}")


# Figure: cross-day RSA vs threshold % strict (right) ← → permissive (left)

fig, ax = plt.subplots(figsize=(9, 6))

# Per-mouse traces (thin, semi-transparent)
for a, b, gap, color in PAIRS:
    per = df[df["pair"] == f"{a}-{b}"]
    for cid, g in per.groupby("container_id"):
        g_sorted = g.sort_values("threshold_pct", ascending=False)
        ax.plot(g_sorted["threshold_pct"], g_sorted["cross_day_rsa"],
                color=color, lw=0.5, alpha=0.18, zorder=1)

# Cohort mean + 95% bootstrap CI per pair
for a, b, gap, color in PAIRS:
    sub = stats[stats["pair"] == f"{a}-{b}"].sort_values("threshold_pct",
                                                         ascending=False)
    x = sub["threshold_pct"].values
    ax.plot(x, sub["mean"].values, color=color, lw=2.8, marker="o", ms=7,
            mew=0, label=f"{a}-{b} ({gap} day{'s' if gap > 1 else ''})",
            zorder=4)
    ax.fill_between(x, sub["ci_lo"].values, sub["ci_hi"].values,
                    color=color, alpha=0.22, zorder=2)

ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(THRESHOLDS)
ax.set_xticklabels([f"{t}%" for t in THRESHOLDS])
ax.invert_xaxis()  # strict on right, permissive on left
ax.set_xlabel("Top X % of pairs kept (strict → permissive)", fontsize=12)
ax.set_ylabel("Cross-day Pearson r at top-X% pairs", fontsize=12)
ax.set_title(f"Cross-day RSA on the strongest pairs — "
             f"{df['container_id'].nunique()} mice\n"
             "Per-mouse traces (light) + cohort mean ± 95% bootstrap CI (bold)",
             fontsize=12)
ax.grid(alpha=0.3)
ax.legend(loc="best", fontsize=10, frameon=True)

plt.tight_layout()
out_png = ROOT / "topX_threshold_rsa.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out_png}")
print(f"Saved {ROOT / 'topX_threshold_rsa_stats.csv'}")
