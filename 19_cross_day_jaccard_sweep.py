"""
19_cross_day_jaccard_sweep.py
Cross-day functional stability vs a within-day reliability ceiling, swept
across binarisation thresholds.

This fuses three earlier pieces:
  * the density-threshold SWEEP (script 17),
  * the CROSS-DAY adjacency comparison on matched cells (script 18),
  * the SPLIT-HALF reliability CEILING idea (script 16's RSA noise ceiling).

For the 52 cells tracked across all three days we build each session's
signal-correlation matrix (trial-averaged response correlation on natural
movie 1 — the same substrate as the cross-day adjacency figure). Then, at each
density threshold (top X% of |weights|), we binarise and measure:

  DATA-MATCHING (important): the ceiling is a 5-vs-5 split, so to compare
  fairly EVERY network here is built from 5 repeats. A 10-repeat cross-day
  network would be less noisy than a 5-repeat split-half and would make the
  ceiling look artificially close — an unfair, stability-flattering comparison.
  Matching the repeat budget removes that bias.

  CEILING curve  J_split :
      Within ONE day, split the 10 movie repeats into halves (1-5 vs 6-10),
      build an adjacency from each half, and take their Jaccard. This is the
      best overlap achievable given trial-to-trial noise alone (same day, same
      cells, same movie), so it is the maximum a cross-day curve could reach.
      Computed for A, B and C; plotted as the mean (shaded = min..max range).

  CROSS-DAY curves  J_AB, J_BC, J_AC :
      Jaccard overlap of two different days' edge sets — functional stability.
      Each day's network is a 5-repeat half; we average the Jaccard over all
      four half-pairings (a_h1×b_h1, a_h1×b_h2, a_h2×b_h1, a_h2×b_h2) so the
      estimation noise matches the split-half ceiling exactly.

  CHANCE floor  d/(2-d) :
      Expected Jaccard for two independent top-d% edge sets.

Reading the plot: a cross-day curve sitting near the ceiling => functional
structure is as stable across days as it is within a day (drift undetectable
above trial noise). A cross-day curve well below the ceiling but above chance
=> partial, real drift.

NOTE: all three sessions view natural_movie_one (the only common stimulus), so
every curve is same-movie; differences are day-to-day, NOT stimulus changes.

Substrate gate: requires true L0 events (aborts on dF/F). Numpy only — no SDK.

Inputs (per session S): outputs/movie1/{S}/events_matrix.npy,
                        outputs/movie1/{S}/movie_repeat_frames.npy
Outputs: outputs/movie1/cross_day_jaccard_sweep.png
         outputs/movie1/cross_day_jaccard_sweep.csv

Usage:
  python3 scripts/19_cross_day_jaccard_sweep.py
  python3 scripts/19_cross_day_jaccard_sweep.py --min-pct 1 --max-pct 50 --step 1
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Config / CLI

SESSIONS = ["A", "B", "C"]
DAY      = {"A": "day 95", "B": "day 96", "C": "day 97"}
FPS      = 30.0
EPS      = 1e-10
CROSS    = [("A", "B"), ("B", "C"), ("A", "C")]
CROSS_COL = {("A", "B"): "#1f77b4", ("B", "C"): "#2ca02c", ("A", "C"): "#9467bd"}

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--min-pct", type=float, default=1.0)
parser.add_argument("--max-pct", type=float, default=50.0)
parser.add_argument("--step", type=float, default=1.0)
parser.add_argument("--bin-ms", type=float, default=33.0,
                    help="Bin width (ms); default 33 ~ frame rate.")
args = parser.parse_args()

ROOT       = Path(args.root)
BIN_FRAMES = max(1, int(round(args.bin_ms / 1000.0 * FPS)))
densities  = np.arange(args.min_pct, args.max_pct + 1e-9, args.step) / 100.0

print("=" * 70)
print("  Cross-day Jaccard sweep vs within-day split-half ceiling")
print(f"  root={ROOT}  density {args.min_pct:g}-{args.max_pct:g}%  "
      f"bin={args.bin_ms:.0f} ms")
print("=" * 70)


# Helpers

def present_mask(ev):
    return ~np.all(np.isnan(ev), axis=1)


def bin_epoch(epoch, bf):
    L, n = epoch.shape
    nb = L // bf
    return epoch[: nb * bf].reshape(nb, bf, n).sum(axis=1)


def build_tensor(ev_cols, frames, bf):
    Lc = int(min(int(ef) - int(sf) + 1 for sf, ef in frames))
    reps = [bin_epoch(ev_cols[int(sf): int(sf) + Lc, :], bf) for sf, _ in frames]
    return np.stack(reps, axis=0)                       # (R, n_bins, n_cells)


def signal_corr(T):
    """Trial-averaged response correlation across time-bins (zero-var -> NaN)."""
    mat = T.mean(axis=0)
    dead = mat.var(axis=0) <= EPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        C = np.corrcoef(mat, rowvar=False)
    C[dead, :] = np.nan
    C[:, dead] = np.nan
    return C


def binarise_abs(C, density):
    """Top-`density` of |edges|; diagonal 0; symmetric; returns upper-tri bool."""
    C2 = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C2, 0.0)
    n = C2.shape[0]
    iu = np.triu_indices(n, k=1)
    a = np.abs(C2[iu])
    k = max(1, int(round(density * a.size)))
    keep = np.argsort(-a)[:k]
    mask = np.zeros(a.size, dtype=bool)
    mask[keep] = True
    return mask                                         # bool over upper-tri pairs


def jaccard(mask1, mask2):
    inter = int(np.logical_and(mask1, mask2).sum())
    union = int(np.logical_or(mask1, mask2).sum())
    return inter / union if union else np.nan


# Build matched-cell correlation matrices (full + the two repeat-halves)

present, events, frames = {}, {}, {}
for s in SESSIONS:
    ev = np.load(ROOT / s / "events_matrix.npy")
    flat = ev[~np.isnan(ev)]
    if np.mean(flat < 0) > 0.01:
        raise ValueError(f"Session {s}: events_matrix is dF/F, not L0 events. "
                         f"Regenerate with the fixed script 02.")
    events[s] = ev
    frames[s] = np.load(ROOT / s / "movie_repeat_frames.npy")
    present[s] = present_mask(ev)

shared = np.where(present["A"] & present["B"] & present["C"])[0]
print(f"\nMatched cells (tracked across all 3 days): {len(shared)}")

corr_h1, corr_h2 = {}, {}     # two 5-repeat half-networks per session
for s in SESSIONS:
    T = build_tensor(events[s][shared, :].T, frames[s], BIN_FRAMES)
    h = T.shape[0] // 2
    corr_h1[s] = signal_corr(T[:h])     # repeats 1..h
    corr_h2[s] = signal_corr(T[h:])     # repeats h+1..R
    print(f"  Session {s} ({DAY[s]}): {T.shape[0]} repeats -> two {h}-repeat "
          f"halves, {T.shape[1]} bins")

# Sweep

cross_curves = {p: [] for p in CROSS}
ceiling_per  = {s: [] for s in SESSIONS}
chance_curve = []

for d in densities:
    # binarise both 5-repeat halves of every session at this density
    bh1 = {s: binarise_abs(corr_h1[s], d) for s in SESSIONS}
    bh2 = {s: binarise_abs(corr_h2[s], d) for s in SESSIONS}
    # within-day split-half ceiling (h1 vs h2), per session
    for s in SESSIONS:
        ceiling_per[s].append(jaccard(bh1[s], bh2[s]))
    # cross-day: average Jaccard over all four half-pairings (matched noise)
    for a, b in CROSS:
        combos = [jaccard(bh1[a], bh1[b]), jaccard(bh1[a], bh2[b]),
                  jaccard(bh2[a], bh1[b]), jaccard(bh2[a], bh2[b])]
        cross_curves[(a, b)].append(np.nanmean(combos))
    chance_curve.append(d / (2 - d))

cross_curves = {p: np.array(v) for p, v in cross_curves.items()}
ceiling_arr  = np.vstack([ceiling_per[s] for s in SESSIONS])     # (3, n_dens)
ceiling_mean = ceiling_arr.mean(axis=0)
ceiling_lo, ceiling_hi = ceiling_arr.min(axis=0), ceiling_arr.max(axis=0)
chance_curve = np.array(chance_curve)

# CSV

out = pd.DataFrame({"density_pct": densities * 100,
                    "J_AB": cross_curves[("A", "B")],
                    "J_BC": cross_curves[("B", "C")],
                    "J_AC": cross_curves[("A", "C")],
                    "ceiling_mean": ceiling_mean,
                    "ceiling_min": ceiling_lo,
                    "ceiling_max": ceiling_hi,
                    "chance": chance_curve})
out.to_csv(ROOT / "cross_day_jaccard_sweep.csv", index=False)
print(f"\nSaved {ROOT / 'cross_day_jaccard_sweep.csv'}")

# quick read at threshold 90% (equivalently: top 10% strongest |r| kept)
i10 = int(np.argmin(np.abs(densities * 100 - 10)))
print(f"\n  At threshold 90%  (top 10% strongest |r| kept):")
for p in CROSS:
    print(f"    J_{p[0]}{p[1]} = {cross_curves[p][i10]:.3f}")
print(f"    ceiling (within-day) = {ceiling_mean[i10]:.3f}  "
      f"[{ceiling_lo[i10]:.3f}-{ceiling_hi[i10]:.3f}]")
print(f"    chance = {chance_curve[i10]:.3f}")

# Figure

fig, ax = plt.subplots(figsize=(9, 6.2))
# Convention: Threshold % = % of weakest |r| pairs zeroed (higher = stricter).
# Internal sweep is "top X% kept"; relabel + sort here.
threshold_pct = (1.0 - densities) * 100.0
order = np.argsort(threshold_pct)
x = threshold_pct[order]

# ceiling (bold black) + range band
ax.fill_between(x, ceiling_lo[order], ceiling_hi[order], color="black", alpha=0.10, lw=0)
ax.plot(x, ceiling_mean[order], color="black", lw=2.6,
        label="within-day ceiling — split-half (mean of A,B,C)")

# cross-day curves
labels = {("A", "B"): "A–B  (days 95↔96, adjacent)",
          ("B", "C"): "B–C  (days 96↔97)",
          ("A", "C"): "A–C  (days 95↔97, longest gap)"}
for p in CROSS:
    ax.plot(x, cross_curves[p][order], color=CROSS_COL[p], lw=2, marker="o", ms=3,
            label=labels[p])

# chance floor
ax.plot(x, chance_curve[order], color="grey", ls=":", lw=1.4, label="chance  (d/(2−d))")

ax.set_xlabel("Threshold %  (% of weakest |r| pairs zeroed)", fontsize=11)
ax.set_ylabel("Jaccard index (binary edge overlap)", fontsize=11)
ax.set_title("Cross-day functional stability vs within-day reliability ceiling\n"
             f"{len(shared)} matched cells · natural_movie_one · |weight| "
             f"thresholding · signal correlation · all networks = 5-repeat "
             f"(data-matched)", fontsize=10.5)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, None)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
out_png = ROOT / "cross_day_jaccard_sweep.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")
