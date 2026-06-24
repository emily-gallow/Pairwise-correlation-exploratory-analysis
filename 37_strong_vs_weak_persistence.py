"""
37_strong_vs_weak_persistence.py
================================
Asymmetric strong-vs-weak pair persistence across days.

Does selecting the strongest pairs on an earlier day predict elevated pair
correlation on a later day?  And how does that compare to selecting the
weakest pairs on the earlier day?

For each cross-day pair (A-B, B-C, A-C) we use ASYMMETRIC selection:
  Rank pairs by |r| on the EARLIER day only.
  This avoids the Berkson / regression-to-the-mean artifact that conditioning
  on a symmetric (sum-of-|r|) score introduces, and is biologically the
  forward-in-time question: "if a pair is strong NOW, is it still strong
  LATER?"

For each percentile X in {1, 5, 10, 15, 20, 25}:

  Top X%    = pairs with the highest |r_earlier|
  Bottom X% = pairs with the lowest  |r_earlier|

We measure two persistence signatures on day Y:
  (1) Mean |r_Y| of the selected pairs (strength-on-day-Y).
      Compare to the population mean |r_Y| across all pairs.
      Interpretation: do these pairs collectively remain elevated
      (or depressed) on day Y?
  (2) Bin-overlap fraction: of the X% selected on day X, what fraction
      are still in the top (or bottom) X% on day Y?
      Compare to X/100 (chance baseline for two independent X%-size sets).

Across mice (n = 17 in the standard cohort):
  Mouse-level bootstrap 95% CI on each metric.
  Sign-flip permutation test for the TOP - BOTTOM difference in mean |r_Y|
  at each (pair, percentile), one observation per mouse, B = 10 000.

Honest framing:
  - "Reoccur" is unpacked in two complementary senses (set membership and
    elevated mean strength).
  - Numbers are not symmetric: strong-end persistence is substantial; weak-
    end persistence is at or below chance. This asymmetry is the headline
    finding.
  - This analysis is NOT subject to the Berkson conditioning artifact
    visible in script 36 (topX_threshold_rsa), because each percentile bin
    is selected on day X alone — measurement on day Y is unconditional.
  - Rate bias still applies: high-rate cell pairs tend to have higher |r|,
    so the "top X%" set is rate-biased. Both directions of the comparison
    (top and bottom) inherit this bias but the TOP-vs-BOTTOM contrast is
    what we test.

Inputs (per numeric-name container):
  {root}/{cid}/{A,B,C}/events_matrix.npy
  {root}/{cid}/{A,B,C}/movie_repeat_frames.npy

Outputs:
  outputs/movie1/strong_vs_weak_persistence.csv         per-mouse table
  outputs/movie1/strong_vs_weak_persistence_stats.csv   cohort means + CIs + tests
  outputs/movie1/strong_vs_weak_persistence.png         figure (2 rows x 3 cols)

Usage:
  python3 scripts/37_strong_vs_weak_persistence.py
  python3 scripts/37_strong_vs_weak_persistence.py --boot 20000 --perm 20000
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

# Cross-day pairs: (earlier, later, days_apart, colour for "this pair")
PAIRS = [("A", "B", 1, "#a05a8a"),   # purple - 1 day
         ("B", "C", 1, "#d63384"),   # magenta - 1 day
         ("A", "C", 2, "#ffd60a")]   # yellow  - 2 days

PERCENTILES = [1, 5, 10, 15, 20, 25]

TOP_COLOR    = "#c1272d"   # red - strong pairs
BOT_COLOR    = "#2166ac"   # blue - weak pairs
BASE_COLOR   = "#555555"   # grey - population baseline

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--perm", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

ROOT = Path(args.root)
rng = np.random.default_rng(args.seed)


# ---------------------------------------------------------------------------
# Helpers (same recipe as scripts 28, 31, 36)
# ---------------------------------------------------------------------------

def present_mask(events):
    return ~np.all(np.isnan(events), axis=1)


def session_signal_corr(events, frames, cell_idx):
    """Signal correlation matrix on matched cells, fulldata 10-rep at frame
    rate. Returns n x n matrix; NaN rows/cols for silent cells."""
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
    """Two-sided sign-flip permutation p-value for paired mean(d) = 0."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    n = len(d)
    if n < 2: return np.nan
    obs = float(np.mean(d))
    signs = rng.choice([-1, 1], size=(B, n)).astype(float)
    null = (signs * d).mean(axis=1)
    return float((np.sum(np.abs(null) >= np.abs(obs)) + 1) / (B + 1))


# ---------------------------------------------------------------------------
# Walk containers, compute per-mouse persistence metrics
# ---------------------------------------------------------------------------

print("=" * 70)
print(" Strong-vs-weak pair persistence across days (asymmetric selection)")
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
    n = n_cells
    iu = np.triu_indices(n, k=1)

    for earlier, later, gap, _color in PAIRS:
        r_e = mats[earlier][iu]
        r_l = mats[later][iu]
        mask = np.isfinite(r_e) & np.isfinite(r_l)
        if mask.sum() < 50:
            continue
        r_e, r_l = r_e[mask], r_l[mask]
        abs_e, abs_l = np.abs(r_e), np.abs(r_l)
        n_pairs = len(r_e)
        pop_mean_l = float(abs_l.mean())
        order_e = np.argsort(abs_e)     # ascending: [0] weakest, [-1] strongest
        order_l = np.argsort(abs_l)

        for X in PERCENTILES:
            n_keep = max(3, int(round(n_pairs * X / 100.0)))
            top_e_idx  = order_e[-n_keep:]
            bot_e_idx  = order_e[:n_keep]
            top_l_set  = set(order_l[-n_keep:].tolist())
            bot_l_set  = set(order_l[:n_keep].tolist())

            rows.append(dict(
                container_id=cid, n_cells=n_cells, n_pairs=n_pairs,
                pair=f"{earlier}-{later}", days_apart=gap, percentile=X,
                pop_mean_abs_later   = pop_mean_l,
                # Top
                top_mean_abs_earlier = float(abs_e[top_e_idx].mean()),
                top_mean_abs_later   = float(abs_l[top_e_idx].mean()),
                top_overlap          = len(set(top_e_idx.tolist()) & top_l_set) / n_keep,
                # Bottom
                bot_mean_abs_earlier = float(abs_e[bot_e_idx].mean()),
                bot_mean_abs_later   = float(abs_l[bot_e_idx].mean()),
                bot_overlap          = len(set(bot_e_idx.tolist()) & bot_l_set) / n_keep,
            ))
    print(f"  {cid}: done ({n_cells} cells, {n_pairs} pairs)")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "strong_vs_weak_persistence.csv", index=False)
print(f"\nSaved per-mouse table: {ROOT / 'strong_vs_weak_persistence.csv'} "
      f"({len(df)} rows, {df['container_id'].nunique()} mice)")


# ---------------------------------------------------------------------------
# Cohort aggregation + sign-flip TOP-vs-BOTTOM tests
# ---------------------------------------------------------------------------

print(f"\nBootstrap CIs ({args.boot} resamples) + sign-flip tests ({args.perm} perms)")

stat_rows = []
for earlier, later, gap, _color in PAIRS:
    for X in PERCENTILES:
        sub = df[(df["pair"] == f"{earlier}-{later}") & (df["percentile"] == X)]
        if sub.empty:
            continue
        # Cohort means + CIs
        out = dict(pair=f"{earlier}-{later}", days_apart=gap, percentile=X,
                   n_mice=len(sub),
                   pop_mean_abs_later=float(sub["pop_mean_abs_later"].mean()))
        for col_in, key in [
            ("top_mean_abs_later", "top_mean_abs_later"),
            ("bot_mean_abs_later", "bot_mean_abs_later"),
            ("top_overlap",        "top_overlap"),
            ("bot_overlap",        "bot_overlap"),
        ]:
            m, lo, hi = boot_mean_ci(sub[col_in].values, args.boot, 0.05, rng)
            out[f"{key}_mean"] = m
            out[f"{key}_lo"]   = lo
            out[f"{key}_hi"]   = hi
        # TOP - BOTTOM mean |r_later| with sign-flip
        d = (sub["top_mean_abs_later"].values
             - sub["bot_mean_abs_later"].values)
        m_d, lo_d, hi_d = boot_mean_ci(d, args.boot, 0.05, rng)
        p = sign_flip_p(d, args.perm, rng)
        out["top_minus_bot_mean"] = m_d
        out["top_minus_bot_lo"]   = lo_d
        out["top_minus_bot_hi"]   = hi_d
        out["top_minus_bot_p"]    = p
        # TOP - BOTTOM overlap diff (also sign-flip)
        d2 = sub["top_overlap"].values - sub["bot_overlap"].values
        m_d2, lo_d2, hi_d2 = boot_mean_ci(d2, args.boot, 0.05, rng)
        p2 = sign_flip_p(d2, args.perm, rng)
        out["top_minus_bot_overlap_mean"] = m_d2
        out["top_minus_bot_overlap_lo"]   = lo_d2
        out["top_minus_bot_overlap_hi"]   = hi_d2
        out["top_minus_bot_overlap_p"]    = p2
        stat_rows.append(out)

stats = pd.DataFrame(stat_rows)
stats.to_csv(ROOT / "strong_vs_weak_persistence_stats.csv", index=False)

# Console summary, one block per pair
for earlier, later, gap, _color in PAIRS:
    pname = f"{earlier}-{later}"
    print(f"\n  ----- {pname} ({gap} day) -----")
    sub = stats[stats["pair"] == pname]
    print(f"  Population mean |r_{later}| (all pairs): "
          f"{sub['pop_mean_abs_later'].iloc[0]:.4f}")
    print(f"\n  {'X%':>3}  {'top mean|rY|':>20}  {'bot mean|rY|':>20}  "
          f"{'top - bot':>22}  {'top overlap':>18}  {'bot overlap':>18}")
    for _, r in sub.iterrows():
        sig = "*" if r["top_minus_bot_p"] < 0.05 else " "
        print(f"  {int(r['percentile']):>3}  "
              f"{r['top_mean_abs_later_mean']:.3f}[{r['top_mean_abs_later_lo']:.3f},{r['top_mean_abs_later_hi']:.3f}]  "
              f"{r['bot_mean_abs_later_mean']:.3f}[{r['bot_mean_abs_later_lo']:.3f},{r['bot_mean_abs_later_hi']:.3f}]  "
              f"{r['top_minus_bot_mean']:+.3f}[{r['top_minus_bot_lo']:+.3f},{r['top_minus_bot_hi']:+.3f}] p={r['top_minus_bot_p']:.3f}{sig}  "
              f"{r['top_overlap_mean']:.3f}[{r['top_overlap_lo']:.3f},{r['top_overlap_hi']:.3f}]  "
              f"{r['bot_overlap_mean']:.3f}[{r['bot_overlap_lo']:.3f},{r['bot_overlap_hi']:.3f}]")


# ---------------------------------------------------------------------------
# Figure: 2 rows x 3 cols
#   Row 1: mean |r_later| at selected pairs vs percentile, with population baseline
#   Row 2: bin-overlap fraction vs percentile, with chance baseline
#   Cols : A-B (1d), B-C (1d), A-C (2d)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), sharex=True)

for col_i, (earlier, later, gap, pair_color) in enumerate(PAIRS):
    pname = f"{earlier}-{later}"
    sub = stats[stats["pair"] == pname].sort_values("percentile")
    xs = sub["percentile"].values
    pop_base = float(sub["pop_mean_abs_later"].iloc[0])

    # --- Row 1: mean |r_later| -------------------------------------------
    ax = axes[0, col_i]
    # Top
    ax.plot(xs, sub["top_mean_abs_later_mean"].values, color=TOP_COLOR,
            lw=2.6, marker="o", ms=7, label="Top X%")
    ax.fill_between(xs, sub["top_mean_abs_later_lo"].values,
                    sub["top_mean_abs_later_hi"].values,
                    color=TOP_COLOR, alpha=0.20)
    # Bottom
    ax.plot(xs, sub["bot_mean_abs_later_mean"].values, color=BOT_COLOR,
            lw=2.6, marker="s", ms=7, label="Bottom X%")
    ax.fill_between(xs, sub["bot_mean_abs_later_lo"].values,
                    sub["bot_mean_abs_later_hi"].values,
                    color=BOT_COLOR, alpha=0.20)
    # Population baseline (mean |r_later| across all pairs)
    ax.axhline(pop_base, color=BASE_COLOR, lw=1.4, ls="--",
               label=f"Population mean |r_{later}|")
    # Significance stars
    for _, r in sub.iterrows():
        if r["top_minus_bot_p"] < 0.001: tag = "***"
        elif r["top_minus_bot_p"] < 0.01: tag = "**"
        elif r["top_minus_bot_p"] < 0.05: tag = "*"
        else: tag = ""
        if tag:
            yp = max(r["top_mean_abs_later_hi"], r["bot_mean_abs_later_hi"])
            ax.text(r["percentile"], yp * 1.04, tag, ha="center",
                    fontsize=8, color="black")
    ax.set_title(f"{pname}  ({gap} day{'s' if gap > 1 else ''})",
                 fontsize=11, color=pair_color, fontweight="bold")
    ax.set_ylabel(f"Mean |r_{later}| of selected pairs" if col_i == 0 else "",
                  fontsize=11)
    ax.grid(alpha=0.3)
    if col_i == 2:
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    # --- Row 2: bin-overlap fraction -------------------------------------
    ax = axes[1, col_i]
    ax.plot(xs, sub["top_overlap_mean"].values, color=TOP_COLOR,
            lw=2.6, marker="o", ms=7, label="Top X%")
    ax.fill_between(xs, sub["top_overlap_lo"].values,
                    sub["top_overlap_hi"].values,
                    color=TOP_COLOR, alpha=0.20)
    ax.plot(xs, sub["bot_overlap_mean"].values, color=BOT_COLOR,
            lw=2.6, marker="s", ms=7, label="Bottom X%")
    ax.fill_between(xs, sub["bot_overlap_lo"].values,
                    sub["bot_overlap_hi"].values,
                    color=BOT_COLOR, alpha=0.20)
    # Chance baseline: X/100 (i.e. percentile/100)
    ax.plot(xs, xs / 100.0, color=BASE_COLOR, lw=1.4, ls="--",
            label="Chance (= X/100)")
    ax.set_xlabel("Percentile X (%)", fontsize=11)
    ax.set_ylabel("Bin-overlap fraction (kept in bin on day Y)"
                  if col_i == 0 else "", fontsize=11)
    ax.set_xticks(PERCENTILES)
    ax.grid(alpha=0.3)
    if col_i == 2:
        ax.legend(loc="upper left", fontsize=8, frameon=True)

fig.suptitle(
    f"Strong-vs-weak pair persistence across days  —  {df['container_id'].nunique()} mice\n"
    "Asymmetric selection: rank by |r| on the earlier day, measure on the later day. "
    "TOP - BOTTOM sign-flip p: * < 0.05, ** < 0.01, *** < 0.001",
    fontsize=12, y=1.00)

plt.tight_layout()
out_png = ROOT / "strong_vs_weak_persistence.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out_png}")
print(f"Saved {ROOT / 'strong_vs_weak_persistence.csv'}")
print(f"Saved {ROOT / 'strong_vs_weak_persistence_stats.csv'}")
