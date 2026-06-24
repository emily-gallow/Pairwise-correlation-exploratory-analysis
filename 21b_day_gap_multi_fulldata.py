"""
21b_day_gap_multi_fulldata.py
=============================
Build the multi-mouse fulldata 10-rep cross-day RSA figure directly from each
container's cross_session_rsa.csv (which already has AB/BC/AC fulldata columns).

This is needed because the batch runner ran script 20 with the matched default,
so per-container day_gap_scatter.csv files only contain matched rows. Going
through cross_session_rsa.csv bypasses that.

Outputs:
  outputs/movie1/day_gap_scatter_multi_fulldata.{csv,png}

Usage:
  python3 scripts/21b_day_gap_multi_fulldata.py
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path("outputs/movie1")
PAIR_COLOR = {"A-B": "#a05a8a", "B-C": "#d63384", "A-C": "#ffd60a"}
PAIR_ORDER = ["A-B", "B-C", "A-C"]
JITTER = {"A-B": -0.05, "B-C": +0.05, "A-C": 0.0}

# ---------------------------------------------------------------------------
# Aggregate cohort table
# ---------------------------------------------------------------------------
# Only include numeric-named container folders — earlier exploratory runs left
# behind a 'not using?' folder containing stale cross_session_rsa.csv from a
# different methodology, which would otherwise inflate the cohort by 1 mouse.
rows = []
for f in sorted(ROOT.glob("*/cross_session_rsa.csv")):
    cid = f.parent.name
    if not cid.isdigit():
        print(f"  skipping non-numeric container folder: {cid!r}")
        continue
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        tp = r["type"]
        for pair, col in (("A-B", "AB_fulldata"),
                          ("B-C", "BC_fulldata"),
                          ("A-C", "AC_fulldata")):
            rows.append(dict(container_id=cid, type=tp, pair=pair,
                             days_apart=1 if pair in ("A-B","B-C") else 2,
                             metric="fulldata", r=float(r[col])))
all_df = pd.DataFrame(rows)
out_csv = ROOT / "day_gap_scatter_multi_fulldata.csv"
all_df.to_csv(out_csv, index=False)
n_animals = all_df["container_id"].nunique()
print(f"Cohort fulldata table: {n_animals} containers, {len(all_df)} rows -> {out_csv}")

# ---------------------------------------------------------------------------
# Figure: same layout as script 21, signal | noise, shared y-axis
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)

for ax, tp, title in [(axes[0], "signal", "Signal correlation"),
                      (axes[1], "noise",  "Noise correlation")]:
    sub = all_df[all_df["type"] == tp].copy()
    for _, g in sub.groupby("container_id"):
        g_sorted = g.set_index("pair").loc[PAIR_ORDER].reset_index()
        xs = [float(row["days_apart"]) + JITTER[row["pair"]]
              for _, row in g_sorted.iterrows()]
        ys = g_sorted["r"].astype(float).tolist()
        ax.plot(xs, ys, color="grey", lw=1.2, alpha=0.55, zorder=1)
        for x, y, pair in zip(xs, ys, g_sorted["pair"]):
            ax.scatter(x, y, s=70, color=PAIR_COLOR[pair],
                       edgecolors="black", linewidth=0.8, alpha=0.85, zorder=2)
    ax.set_xlabel("Days apart", fontsize=12)
    ax.set_ylabel("Cross-day similarity (RSA Pearson r)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([1, 2])
    ax.set_xlim(0.6, 2.4)
    ax.grid(alpha=0.3)

pair_handles = [mpatches.Patch(color=PAIR_COLOR["A-B"], label="A–B (1 day)"),
                mpatches.Patch(color=PAIR_COLOR["B-C"], label="B–C (1 day)"),
                mpatches.Patch(color=PAIR_COLOR["A-C"], label="A–C (2 days)")]
fig.legend(handles=pair_handles, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02), frameon=True)

fig.suptitle(
    f"Cross-day correlation structure stability vs day gap — {n_animals} mice\n"
    "Each grey trace = one animal · RSA: full data (10-rep cross-day)",
    fontsize=12, y=1.03)

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_png = ROOT / "day_gap_scatter_multi_fulldata.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")
