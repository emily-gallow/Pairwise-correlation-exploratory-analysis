"""
38_top_overlap_enrichment.py

Top-X% pair overlap across days — does a pair that is among the strongest
on one day remain among the strongest on the next day?

For each cross-day pair (A-B, B-C, A-C) and each percentile X in
{25, 20, 15, 10, 5, 1}:

  (1) Rank pairs by |r| on the EARLIER day (asymmetric — no Berkson
      conditioning).
  (2) Select the top X% of those pairs.
  (3) Independently, select the top X% of pairs on the LATER day (by
      |r| on the later day).
  (4) Compute the overlap:
         overlap = |top_X_earlier ∩ top_X_later| / |top_X_earlier|
      Because both sets have the same size by construction, this is the
      asymmetric overlap that the user prescribed; it equals the symmetric
      Jaccard's numerator divided by the set size and reduces to
      |intersection| / k where k = top-X count.
  (5) Compute fold enrichment over chance:
         fold = observed_overlap / (X / 100)
      The chance baseline X/100 is the expected overlap for two
      independent equal-sized random subsets of size k drawn from N total
      pairs (E[|intersection|] = k * (k/N), divided by k gives k/N = X/100).

What this asks: do the pairs that are strongest on one day remain among
the strongest pairs on the next day?

Honest framing:
  - This is the cleanest version of the strong-pair persistence question.
    Selection is asymmetric (rank by earlier day only) so no Berkson
    artifact; overlap is a binary set-membership question so range
    restriction is irrelevant; the fold-enrichment baseline is exact and
    well-defined.
  - Still affected by rate bias to the same degree as the earlier-day
    ranking (top-X% selection is biased toward high-rate, high-reliability
    cell pairs), but the chance baseline accounts for the size of the
    selected set only — it doesn't model what the selection biases toward,
    so a fold > 1 IS biological persistence of the same biased set across
    days, not an artifact.

Statistical inference:
  - Per-mouse fold enrichment, cohort 95% bootstrap CI (resample mice).
  - Sign-flip permutation test: is observed_overlap > chance_overlap at
    the cohort level?  H0: per-mouse (overlap - chance) = 0; two-sided.
  - Paired 1-day-vs-2-day test at each threshold:
        d_i = mean(fold_AB, fold_BC) - fold_AC
      sign-flip permutation, two-sided.

Inputs (per numeric-name container):
  {root}/{cid}/{A,B,C}/events_matrix.npy
  {root}/{cid}/{A,B,C}/movie_repeat_frames.npy

Outputs:
  outputs/movie1/top_overlap_enrichment.csv        per-mouse table
  outputs/movie1/top_overlap_enrichment_stats.csv  cohort means + CIs + tests
  outputs/movie1/top_overlap_enrichment.png        figure (2 panels)

Usage:
  python3 scripts/38_top_overlap_enrichment.py
  python3 scripts/38_top_overlap_enrichment.py --boot 20000 --perm 20000
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

# Cross-day pairs (earlier, later, days_apart, colour matching the rest of the thesis)
PAIRS = [("A", "B", 1, "#a05a8a"),   # purple - 1 day
         ("B", "C", 1, "#d63384"),   # magenta - 1 day
         ("A", "C", 2, "#ffd60a")]   # yellow  - 2 days

PERCENTILES = [25, 20, 15, 10, 5, 1]   # strict → permissive on the x-axis (we invert)

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--perm", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

ROOT = Path(args.root)
rng = np.random.default_rng(args.seed)


# Helpers (signal-correlation recipe identical to script 28, 31, 36, 37)


def present_mask(events):
    return ~np.all(np.isnan(events), axis=1)


def session_signal_corr(events, frames, cell_idx):
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


def boot_mean_ci(values, B, alpha, rng):
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


def sign_flip_p(d, B, rng):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    n = len(d)
    if n < 2: return np.nan
    obs = float(np.mean(d))
    signs = rng.choice([-1, 1], size=(B, n)).astype(float)
    null = (signs * d).mean(axis=1)
    return float((np.sum(np.abs(null) >= np.abs(obs)) + 1) / (B + 1))


# Walk containers — per-mouse top-X overlap and fold enrichment

print("=" * 70)
print(" Top-X pair overlap across days (asymmetric selection, fold over chance)")
print("=" * 70)

cid_dirs = [p for p in sorted(ROOT.iterdir())
            if p.is_dir() and p.name.isdigit()
            and all((p / s / "events_matrix.npy").exists() for s in SESSIONS)]
print(f"  Found {len(cid_dirs)} numeric containers")

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
        print(f"  skip {cid}: only {n_cells} matched cells")
        continue

    mats = {s: session_signal_corr(ev[s], fr[s], matched) for s in SESSIONS}
    iu = np.triu_indices(n_cells, k=1)

    for earlier, later, gap, _color in PAIRS:
        r_e = mats[earlier][iu]
        r_l = mats[later][iu]
        mask = np.isfinite(r_e) & np.isfinite(r_l)
        if mask.sum() < 50:
            continue
        abs_e = np.abs(r_e[mask])
        abs_l = np.abs(r_l[mask])
        n_pairs = len(abs_e)
        order_e = np.argsort(abs_e)
        order_l = np.argsort(abs_l)

        for X in PERCENTILES:
            n_keep = max(3, int(round(n_pairs * X / 100.0)))
            top_e = set(order_e[-n_keep:].tolist())
            top_l = set(order_l[-n_keep:].tolist())
            overlap = len(top_e & top_l) / n_keep
            chance = n_keep / n_pairs                  # exact baseline
            fold   = overlap / chance if chance > 0 else np.nan
            rows.append(dict(
                container_id=cid, n_cells=n_cells, n_pairs=n_pairs,
                pair=f"{earlier}-{later}", days_apart=gap,
                percentile=X, n_keep=n_keep,
                chance_overlap=chance, overlap=overlap,
                fold_enrichment=fold,
            ))
    print(f"  {cid}: done ({n_cells} cells, {n_pairs} pairs)")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "top_overlap_enrichment.csv", index=False)
print(f"\nSaved per-mouse table: {ROOT / 'top_overlap_enrichment.csv'} "
      f"({len(df)} rows, {df['container_id'].nunique()} mice)")


# Cohort aggregation + above-chance & paired 1d-vs-2d tests

print(f"\nBootstrap CIs ({args.boot} resamples) + permutation tests ({args.perm} perms)")
stat_rows = []
for earlier, later, gap, _color in PAIRS:
    for X in PERCENTILES:
        sub = df[(df["pair"] == f"{earlier}-{later}") & (df["percentile"] == X)]
        if sub.empty: continue
        m_ov,  lo_ov,  hi_ov  = boot_mean_ci(sub["overlap"].values, args.boot, 0.05, rng)
        m_fo,  lo_fo,  hi_fo  = boot_mean_ci(sub["fold_enrichment"].values, args.boot, 0.05, rng)
        # Above-chance test: per-mouse d = overlap - chance, H0 mean(d)=0
        d_above = (sub["overlap"].values - sub["chance_overlap"].values)
        p_above = sign_flip_p(d_above, args.perm, rng)
        stat_rows.append(dict(
            pair=f"{earlier}-{later}", days_apart=gap, percentile=X,
            n_mice=len(sub),
            chance_overlap=float(sub["chance_overlap"].mean()),
            overlap_mean=m_ov, overlap_lo=lo_ov, overlap_hi=hi_ov,
            fold_mean=m_fo, fold_lo=lo_fo, fold_hi=hi_fo,
            above_chance_diff=float(d_above.mean()),
            above_chance_p=p_above,
        ))

stats = pd.DataFrame(stat_rows)

# Paired 1d-vs-2d test on fold enrichment at each threshold
paired_rows = []
for X in PERCENTILES:
    piv = (df[df["percentile"] == X]
             .pivot_table(index="container_id", columns="pair",
                          values="fold_enrichment", aggfunc="first")
             .dropna())
    if {"A-B", "B-C", "A-C"}.issubset(piv.columns):
        d = 0.5 * (piv["A-B"].values + piv["B-C"].values) - piv["A-C"].values
        n = len(d)
        mean_d = float(np.mean(d))
        if n > 1:
            p = sign_flip_p(d, args.perm, rng)
            lo, hi = boot_mean_ci(d, args.boot, 0.05, rng)[1:]
        else:
            p = lo = hi = np.nan
    else:
        n = 0; mean_d = p = lo = hi = np.nan
    paired_rows.append(dict(percentile=X, n_mice=n,
                            mean_fold_d=mean_d, ci_lo=lo, ci_hi=hi, perm_p=p))

paired = pd.DataFrame(paired_rows)

# Save both as one CSV stack with a row-type column
stats["row_type"] = "per_pair"
paired["row_type"] = "paired_1d_vs_2d"
stack = pd.concat([stats, paired], ignore_index=True, sort=False)
stack.to_csv(ROOT / "top_overlap_enrichment_stats.csv", index=False)

# Console summary
print("\nCohort summary (per cross-day pair):")
for earlier, later, gap, _ in PAIRS:
    pname = f"{earlier}-{later}"
    sub = stats[stats["pair"] == pname]
    print(f"\n  {pname}  ({gap} day{'s' if gap > 1 else ''}):")
    print(f"    {'X%':>3}  {'overlap':>20}  {'chance':>7}  {'fold':>16}  {'above-chance p':>14}")
    for _, r in sub.iterrows():
        sig = "*" if r["above_chance_p"] < 0.05 else " "
        print(f"    {int(r['percentile']):>3}  "
              f"{r['overlap_mean']:.3f} [{r['overlap_lo']:.3f},{r['overlap_hi']:.3f}]  "
              f"{r['chance_overlap']:.3f}  "
              f"{r['fold_mean']:5.2f}x [{r['fold_lo']:5.2f},{r['fold_hi']:5.2f}]  "
              f"  p={r['above_chance_p']:.4f}{sig}")

print("\nPaired 1d-vs-2d fold-enrichment per threshold "
      "(d = mean(fold_AB, fold_BC) - fold_AC):")
for _, r in paired.iterrows():
    sig = "*" if np.isfinite(r["perm_p"]) and r["perm_p"] < 0.05 else " "
    print(f"  {int(r['percentile']):>3}%  n={int(r['n_mice'])}  "
          f"d = {r['mean_fold_d']:+.2f}  "
          f"95% CI [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  "
          f"perm p = {r['perm_p']:.4f}{sig}")


# Figure: 2 panels
#   Left:  observed overlap fraction vs threshold (chance baseline as dashed)
#   Right: fold enrichment vs threshold (chance = 1.0 horizontal)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

# Per-mouse traces (light, semi-transparent)
for earlier, later, gap, color in PAIRS:
    per = df[df["pair"] == f"{earlier}-{later}"]
    for cid, g in per.groupby("container_id"):
        g_sorted = g.sort_values("percentile", ascending=False)
        axes[0].plot(g_sorted["percentile"], g_sorted["overlap"],
                     color=color, lw=0.5, alpha=0.18, zorder=1)
        axes[1].plot(g_sorted["percentile"], g_sorted["fold_enrichment"],
                     color=color, lw=0.5, alpha=0.18, zorder=1)

# Cohort means + 95% bootstrap CI bands
for earlier, later, gap, color in PAIRS:
    sub = stats[stats["pair"] == f"{earlier}-{later}"].sort_values("percentile",
                                                                   ascending=False)
    xs = sub["percentile"].values
    axes[0].plot(xs, sub["overlap_mean"].values, color=color, lw=2.6,
                 marker="o", ms=7, mew=0,
                 label=f"{earlier}-{later} ({gap} day{'s' if gap > 1 else ''})",
                 zorder=4)
    axes[0].fill_between(xs, sub["overlap_lo"].values, sub["overlap_hi"].values,
                         color=color, alpha=0.22, zorder=2)
    axes[1].plot(xs, sub["fold_mean"].values, color=color, lw=2.6,
                 marker="o", ms=7, mew=0,
                 label=f"{earlier}-{later} ({gap} day{'s' if gap > 1 else ''})",
                 zorder=4)
    axes[1].fill_between(xs, sub["fold_lo"].values, sub["fold_hi"].values,
                         color=color, alpha=0.22, zorder=2)

# Chance baselines
xs_all = np.array(PERCENTILES, dtype=float)
axes[0].plot(xs_all, xs_all / 100.0, color="black", ls="--", lw=1.2,
             label="Chance (= X/100)", zorder=3)
axes[1].axhline(1.0, color="black", ls="--", lw=1.2,
                label="Chance (= 1.0×)", zorder=3)

for ax in axes:
    ax.set_xticks(PERCENTILES)
    ax.set_xticklabels([f"{t}%" for t in PERCENTILES])
    ax.invert_xaxis()      # permissive (25%) on the left, strict (1%) on the right
    ax.set_xlabel("Threshold X% (rank pairs by |r| on the earlier day)",
                  fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, frameon=True)

axes[0].set_ylabel("Top-X overlap fraction "
                   r"$|top_X^{earlier} \cap top_X^{later}| / |top_X^{earlier}|$",
                   fontsize=11)
axes[0].set_title("Observed top-X overlap vs chance", fontsize=12)
axes[1].set_ylabel("Fold enrichment (observed / chance)", fontsize=11)
axes[1].set_title("Fold enrichment over chance", fontsize=12)

fig.suptitle(
    f"Do the pairs strongest on one day remain among the strongest on the next?  "
    f"({df['container_id'].nunique()} mice)\n"
    "Asymmetric selection: rank by |r| on the earlier day, overlap with top X% on the later day. "
    "Per-mouse traces (light) + cohort mean ± 95% bootstrap CI (bold).",
    fontsize=11, y=1.02)

plt.tight_layout()
out_png = ROOT / "top_overlap_enrichment.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out_png}")
print(f"Saved {ROOT / 'top_overlap_enrichment.csv'}")
print(f"Saved {ROOT / 'top_overlap_enrichment_stats.csv'}")
