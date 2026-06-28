"""
22_noise_drift_zoom.py

Companion to script 21. Zooms into the noise-correlation panel of the
multi-animal day-gap synthesis so the bulk of the data (≈ 0 to ≈ 0.1 r) is
visible, and quantifies the magnitude of the drop from 1-day apart to 2-days
apart.

For each animal:
  * 1-day value  = mean of A–B and B–C cross-day RSA (paired, same day-gap)
  * 2-day value  = A–C cross-day RSA
  * paired drop  = 1-day − 2-day
  * % drop       = 100 × paired drop / 1-day value

Group-level statistics (across animals):
  * mean ± SEM at 1-day apart
  * mean ± SEM at 2-day apart
  * mean paired drop ± SEM
  * mean % drop (of the 1-day baseline)

Plotted as a single zoomed panel: per-animal grey traces + dots (same colour
palette as Figure 3) and a bold black mean ± SEM line connecting the two
day-gap means. The y-axis is capped at --ymax (default 0.10) so the bulk of
the data is legible; any animal whose values exceed the cap is reported in the
annotation rather than silently hidden.

Inputs:
  outputs/movie1/day_gap_scatter_multi_<metric>.csv  (from script 21)

Outputs:
  outputs/movie1/noise_drift_zoom_<metric>.png
  outputs/movie1/noise_drift_zoom_<metric>.csv  — per-animal table

Usage:
  python3 scripts/22_noise_drift_zoom.py
  python3 scripts/22_noise_drift_zoom.py --metric fulldata --ymax 0.15
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Config / CLI

PAIR_COLOR = {"A-B": "#a05a8a",   # purple (matches scripts 20/21)
              "B-C": "#d63384",   # magenta
              "A-C": "#ffd60a"}   # yellow
JITTER = {"A-B": -0.05, "B-C": +0.05, "A-C": 0.0}

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1",
                    help="Folder containing day_gap_scatter_multi_<metric>.csv.")
parser.add_argument("--metric", choices=["matched", "fulldata"], default="matched",
                    help="Which RSA estimate to read.")
parser.add_argument("--ymax", type=float, default=0.10,
                    help="Upper y-axis limit for the zoom (default 0.10). "
                         "Animals above this are reported but not plotted.")
args = parser.parse_args()

ROOT = Path(args.root)
in_csv = ROOT / f"day_gap_scatter_multi_{args.metric}.csv"
if not in_csv.exists():
    raise SystemExit(f"Missing {in_csv}. Run scripts/21_day_gap_multi.py first.")

df = pd.read_csv(in_csv)
noise = df[df["type"] == "noise"].copy()
if noise.empty:
    raise SystemExit("No noise rows in input CSV.")

# Per-animal pairing: 1-day value (mean of A-B and B-C) vs 2-day value (A-C)

per_animal = []
for cid, g in noise.groupby("container_id"):
    one_day_vals = g[g["days_apart"] == 1]["r"].astype(float).tolist()
    two_day_vals = g[g["days_apart"] == 2]["r"].astype(float).tolist()
    if not one_day_vals or not two_day_vals:
        continue
    one_day_mean = float(np.mean(one_day_vals))     # mean of A-B and B-C
    two_day      = float(two_day_vals[0])           # A-C is the only 2-day point
    per_animal.append(dict(
        container_id   = cid,
        one_day_mean   = one_day_mean,
        two_day        = two_day,
        paired_drop    = one_day_mean - two_day,
        pct_drop       = (100 * (one_day_mean - two_day) / one_day_mean
                          if one_day_mean != 0 else np.nan),
    ))
pa = pd.DataFrame(per_animal).sort_values("container_id").reset_index(drop=True)
n_animals = len(pa)

pa.to_csv(ROOT / f"noise_drift_zoom_{args.metric}.csv", index=False)

# Group-level statistics

def mean_sem(x):
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, s

mean_1, sem_1 = mean_sem(pa["one_day_mean"])
mean_2, sem_2 = mean_sem(pa["two_day"])
mean_d, sem_d = mean_sem(pa["paired_drop"])
mean_pct, sem_pct = mean_sem(pa["pct_drop"].dropna())

print(f"\nn = {n_animals} animals")
print(f"  1-day mean ± SEM = {mean_1:.4f} ± {sem_1:.4f}")
print(f"  2-day mean ± SEM = {mean_2:.4f} ± {sem_2:.4f}")
print(f"  paired drop      = {mean_d:+.4f} ± {sem_d:.4f}")
print(f"  % drop           = {mean_pct:+.1f}% ± {sem_pct:.1f}%")

# How many animals are above the zoom range and will be clipped?
above_1d = (pa["one_day_mean"] > args.ymax).sum()
above_2d = (pa["two_day"]      > args.ymax).sum()
clipped_animals = int(((pa["one_day_mean"] > args.ymax) |
                       (pa["two_day"]      > args.ymax)).sum())

# Figure

fig, ax = plt.subplots(figsize=(9, 6.2))

# Per-animal grey traces and pair-coloured dots, using the full noise table so
# the original 3-dot structure (A-B, B-C, A-C) is preserved.
for cid, g in noise.groupby("container_id"):
    g_sorted = g.sort_values(["days_apart", "pair"])
    xs = [float(row["days_apart"]) + JITTER[row["pair"]]
          for _, row in g_sorted.iterrows()]
    ys = g_sorted["r"].astype(float).tolist()
    ax.plot(xs, ys, color="grey", lw=1.0, alpha=0.4, zorder=1)
    for x, y, pair in zip(xs, ys, g_sorted["pair"]):
        ax.scatter(x, y, s=60, color=PAIR_COLOR.get(pair, "#888888"),
                   edgecolors="black", linewidth=0.6, alpha=0.7, zorder=2)

# Bold group-level mean ± SEM line connecting the two day-gap means.
ax.errorbar([1, 2], [mean_1, mean_2], yerr=[sem_1, sem_2],
            color="black", lw=3.0, marker="D", ms=11, mew=0,
            capsize=7, capthick=2.5, zorder=5,
            label=f"group mean ± SEM (n = {n_animals})")

# Y-axis cap so the bulk is legible.
ax.set_ylim(0, args.ymax)
ax.set_xlim(0.6, 2.4)
ax.set_xticks([1, 2])
ax.set_xlabel("Days apart", fontsize=12)
ax.set_ylabel("Noise correlation cross-day RSA (Pearson r)", fontsize=12)
ax.set_title("Noise correlation cross-day drift — zoomed view "
             f"({n_animals} mice · RSA: "
             f"{'matched' if args.metric == 'matched' else 'full data'})",
             fontsize=12)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=10, frameon=True)

# Quantification box: drop magnitudes + clipped-animal note.
annot_lines = [
    f"1-day mean: {mean_1:.3f} ± {sem_1:.3f}",
    f"2-day mean: {mean_2:.3f} ± {sem_2:.3f}",
    f"paired drop: {mean_d:+.3f} ± {sem_d:.3f}",
    f"% drop:      {mean_pct:+.1f}% ± {sem_pct:.1f}%",
]
if clipped_animals > 0:
    annot_lines.append(f"({clipped_animals} animal(s) clipped above y = {args.ymax:g})")

ax.text(0.025, 0.97, "\n".join(annot_lines),
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="grey",
                  alpha=0.92, lw=0.8))

# Pair-colour legend
pair_handles = [mpatches.Patch(color=PAIR_COLOR["A-B"], label="A–B (1 day)"),
                mpatches.Patch(color=PAIR_COLOR["B-C"], label="B–C (1 day)"),
                mpatches.Patch(color=PAIR_COLOR["A-C"], label="A–C (2 days)")]
fig.legend(handles=pair_handles, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.01), frameon=True)

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_png = ROOT / f"noise_drift_zoom_{args.metric}.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out_png}")
print(f"Saved {ROOT / f'noise_drift_zoom_{args.metric}.csv'}")
