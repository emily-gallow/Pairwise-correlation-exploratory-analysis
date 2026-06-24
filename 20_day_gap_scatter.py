"""
20_day_gap_scatter.py
=====================
Synthesis figure for ONE animal: cross-day RSA similarity plotted against the
number of days between sessions.

Two panels side by side:
  left  — SIGNAL correlation structure stability
  right — NOISE  correlation structure stability

Each panel has THREE points (one per cross-day pair):
  A–B  (1 day apart)
  B–C  (1 day apart)
  A–C  (2 days apart)
connected by a line so the trend with day gap is visible.

Designed to scale: when more animals/containers are processed, every animal
contributes its own 3-point trace and we can overlay multiple traces or
aggregate them.

Inputs:
  outputs/movie1/cross_session_rsa.csv   produced by script 16.
      Columns: type ('signal'/'noise'), AB, BC, AC  (matched 5-rep mean of
      4 half-pairings, comparable to the within-day split-half ceiling).

Outputs:
  outputs/movie1/day_gap_scatter.png
  outputs/movie1/day_gap_scatter.csv     long-format table (pair, gap, type, r)

Usage:
  python3 scripts/20_day_gap_scatter.py
  python3 scripts/20_day_gap_scatter.py --metric fulldata   # use 10-rep r instead
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------

PAIRS     = [("A", "B"), ("B", "C"), ("A", "C")]
PAIR_GAP  = {("A", "B"): 1, ("B", "C"): 1, ("A", "C"): 2}
# Colours match the cross-day adjacency figure (script 18).
PAIR_COLOR = {("A", "B"): "#a05a8a",   # purple
              ("B", "C"): "#d63384",   # magenta
              ("A", "C"): "#ffd60a"}   # yellow

# Small horizontal jitter so A–B and B–C (both at 1 day) don't overlap on the
# plot. Doesn't change the underlying day-gap value, only the marker position.
JITTER = {("A", "B"): -0.05, ("B", "C"): +0.05, ("A", "C"): 0.0}

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1",
                    help="Folder containing cross_session_rsa.csv (default: "
                         "outputs/movie1).")
parser.add_argument("--metric", choices=["matched", "fulldata"], default="matched",
                    help="Which RSA estimate to use: 'matched' (5-rep mean of "
                         "4 half-pairings; consistent with the within-day "
                         "ceiling) or 'fulldata' (10-rep cross-day similarity).")
args = parser.parse_args()

ROOT = Path(args.root)

# ---------------------------------------------------------------------------
# Load values
# ---------------------------------------------------------------------------

df = pd.read_csv(ROOT / "cross_session_rsa.csv")

# Which CSV columns hold the per-pair values?
if args.metric == "matched":
    col_map = {("A", "B"): "AB", ("B", "C"): "BC", ("A", "C"): "AC"}
    metric_label = "matched (5-rep, mean of 4 half-pairings)"
else:
    col_map = {("A", "B"): "AB_fulldata",
               ("B", "C"): "BC_fulldata",
               ("A", "C"): "AC_fulldata"}
    metric_label = "full data (10-rep cross-day)"

rows = {}
for tp in ("signal", "noise"):
    sub = df[df["type"] == tp]
    if sub.empty:
        raise ValueError(f"No '{tp}' row in {ROOT / 'cross_session_rsa.csv'}")
    rows[tp] = sub.iloc[0]

# Long-format table for downstream / multi-animal aggregation later.
long_rows = []
for tp in ("signal", "noise"):
    for pair in PAIRS:
        long_rows.append(dict(
            pair=f"{pair[0]}-{pair[1]}",
            days_apart=PAIR_GAP[pair],
            type=tp,
            metric=args.metric,
            r=float(rows[tp][col_map[pair]]),
        ))
long_df = pd.DataFrame(long_rows)
long_df.to_csv(ROOT / "day_gap_scatter.csv", index=False)
print(long_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Figure: signal | noise, 3 dots + line per panel
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

for ax, tp, panel_title in [(axes[0], "signal", "Signal correlation"),
                             (axes[1], "noise",  "Noise correlation")]:
    # Collect (x_jittered, y, pair) for plotting, sorted by pair order so the
    # connecting line goes A-B -> B-C -> A-C.
    pts = []
    for pair in PAIRS:
        gap = PAIR_GAP[pair]
        x   = gap + JITTER[pair]
        y   = float(rows[tp][col_map[pair]])
        pts.append((x, y, pair))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    # Connecting line (drawn first so dots sit on top).
    ax.plot(xs, ys, color="grey", lw=1.5, alpha=0.7, zorder=1)

    # Individual dots, coloured by pair.
    for x, y, pair in pts:
        ax.scatter(x, y, s=180, color=PAIR_COLOR[pair],
                   edgecolors="black", linewidth=1.2, zorder=2,
                   label=f"{pair[0]}–{pair[1]}")

    ax.set_xlabel("Days apart", fontsize=12)
    ax.set_ylabel("Cross-day similarity (RSA Pearson r)", fontsize=12)
    ax.set_title(panel_title, fontsize=13)
    ax.set_xticks([1, 2])
    ax.set_xlim(0.6, 2.4)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="best", frameon=True)

# Derive the container label and matched-cell count from the data rather than
# hardcoding (so per-container batch runs are labelled correctly, not as the
# original container 661437138).
cid_label = ROOT.name if ROOT.name.isdigit() else "661437138"
try:
    n_cells = int(np.load(ROOT / "A" / "signal_corr.npy").shape[0])
    cells_str = f"{n_cells} matched cells"
except Exception:
    cells_str = "matched cells"
fig.suptitle(
    f"Cross-day correlation structure stability vs day gap — container "
    f"{cid_label} (VISp)\n{cells_str} · natural_movie_one · RSA: "
    f"{metric_label}",
    fontsize=12, y=1.02)
plt.tight_layout()
# Filename: matched -> default name; fulldata -> tag the file so both can
# coexist in the outputs folder for direct comparison.
suffix = "" if args.metric == "matched" else f"_{args.metric}"
out_png = ROOT / f"day_gap_scatter{suffix}.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out_png}")
print(f"Saved {ROOT / 'day_gap_scatter.csv'}")
