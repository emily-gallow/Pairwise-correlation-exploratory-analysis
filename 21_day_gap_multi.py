"""
21_day_gap_multi.py
===================
Multi-animal version of the day-gap synthesis figure (script 20).

Walks outputs/movie1/*/day_gap_scatter.csv (one per container produced by
script 20), concatenates them with a container_id column, and produces:

  outputs/movie1/day_gap_scatter_multi_<metric>.png   — same layout as script 20
      (signal | noise), but with one 3-dot trace per animal overlaid.
  outputs/movie1/day_gap_scatter_multi_<metric>.csv   — long-format table for
      reproducibility (container_id, pair, days_apart, type, metric, r).

Two panels side by side: SIGNAL correlation (left), NOISE correlation (right).
On each panel, every animal contributes three coloured dots (A–B, B–C, A–C)
connected by a grey line — identical styling to script 20, repeated for all
containers.

Designed to be run after the batch pipeline (scripts/run_containers.py) so
every container under outputs/movie1/<container_id>/ has its day_gap_scatter.csv
populated.

Usage:
  python3 scripts/21_day_gap_multi.py
  python3 scripts/21_day_gap_multi.py --metric fulldata
  python3 scripts/21_day_gap_multi.py --root outputs/movie1 --min-cells 50
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------

PAIR_COLOR = {"A-B": "#a05a8a",   # purple (matches script 20)
              "B-C": "#d63384",   # magenta
              "A-C": "#ffd60a"}   # yellow
PAIR_ORDER = ["A-B", "B-C", "A-C"]
# Same jitter as script 20 so A–B and B–C (both 1 day) don't overlap.
JITTER = {"A-B": -0.05, "B-C": +0.05, "A-C": 0.0}

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1",
                    help="Folder containing per-container subfolders.")
parser.add_argument("--metric", choices=["matched", "fulldata"], default="matched",
                    help="Which RSA estimate to aggregate: 'matched' (5-rep "
                         "mean of 4 half-pairings, compares to ceiling) or "
                         "'fulldata' (10-rep cross-day).")
parser.add_argument("--min-cells", type=int, default=0,
                    help="Optional: filter out containers with fewer than this "
                         "many matched cells (read from cross_session_rsa.csv "
                         "via the matched ratio if present, otherwise ignored).")
args = parser.parse_args()

ROOT = Path(args.root)

# ---------------------------------------------------------------------------
# Walk container subfolders and collect day_gap_scatter.csv files.
# ---------------------------------------------------------------------------

# CRITICAL: only numeric-named container folders. Earlier exploratory runs
# left behind 'not using?'-style folders that can leak stale data into the
# cohort aggregation. Match the fix in scripts/21b and 36.
found = sorted([p for p in ROOT.glob("*/day_gap_scatter.csv")
                if p.parent.name.isdigit()])
if not found:
    raise SystemExit(
        f"No day_gap_scatter.csv found under {ROOT}/*/.\n"
        f"Run the batch pipeline first: python3 scripts/run_containers.py ..."
    )
print(f"Found {len(found)} numeric-name container CSV files.")

frames = []
for csv_path in found:
    container_id = csv_path.parent.name
    df_c = pd.read_csv(csv_path)
    df_c["container_id"] = container_id
    frames.append(df_c)
all_df = pd.concat(frames, ignore_index=True)
all_df = all_df[all_df["metric"] == args.metric].copy()
if all_df.empty:
    raise SystemExit(f"No rows for metric={args.metric}. Try --metric matched.")

# Save the concatenated long-format table for reproducibility.
out_csv = ROOT / f"day_gap_scatter_multi_{args.metric}.csv"
all_df.to_csv(out_csv, index=False)
print(f"Saved {out_csv}  ({len(all_df)} rows from "
      f"{all_df['container_id'].nunique()} containers)")

# ---------------------------------------------------------------------------
# Figure: signal | noise — one 3-dot trace per animal (script 20 × N)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)

for ax, tp, panel_title in [(axes[0], "signal", "Signal correlation"),
                             (axes[1], "noise",  "Noise correlation")]:
    sub = all_df[all_df["type"] == tp].copy()
    if sub.empty:
        ax.text(0.5, 0.5, f"no {tp} rows", ha="center", va="center",
                transform=ax.transAxes)
        continue

    for _, g in sub.groupby("container_id"):
        g_sorted = g.set_index("pair").loc[PAIR_ORDER].reset_index()
        xs = [float(row["days_apart"]) + JITTER[row["pair"]]
              for _, row in g_sorted.iterrows()]
        ys = g_sorted["r"].astype(float).tolist()

        ax.plot(xs, ys, color="grey", lw=1.2, alpha=0.55, zorder=1)
        for x, y, pair in zip(xs, ys, g_sorted["pair"]):
            ax.scatter(x, y, s=70, color=PAIR_COLOR.get(pair, "#888888"),
                       edgecolors="black", linewidth=0.8, alpha=0.85, zorder=2)

    ax.set_xlabel("Days apart", fontsize=12)
    ax.set_ylabel("Cross-day similarity (RSA Pearson r)", fontsize=12)
    ax.set_title(panel_title, fontsize=13)
    ax.set_xticks([1, 2])
    ax.set_xlim(0.6, 2.4)
    ax.grid(alpha=0.3)

pair_handles = [mpatches.Patch(color=PAIR_COLOR["A-B"], label="A–B (1 day)"),
                mpatches.Patch(color=PAIR_COLOR["B-C"], label="B–C (1 day)"),
                mpatches.Patch(color=PAIR_COLOR["A-C"], label="A–C (2 days)")]
fig.legend(handles=pair_handles, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02), frameon=True)

n_animals = all_df["container_id"].nunique()
metric_label = ("matched (5-rep, mean of 4 half-pairings)"
                if args.metric == "matched"
                else "full data (10-rep cross-day)")
fig.suptitle(
    f"Cross-day correlation structure stability vs day gap — "
    f"{n_animals} mice\n"
    f"Each grey trace = one animal · RSA: {metric_label}",
    fontsize=12, y=1.03)

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_png = ROOT / f"day_gap_scatter_multi_{args.metric}.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")
