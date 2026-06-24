"""
41b_per_mouse_centred_check.py
==============================
Robustness check for Figure 5: per-mouse mean-centred cross-day RSA vs
trial-level running mismatch.

Each mouse contributes 3 cross-day RSA values (A-B, B-C, A-C). Different
mice have very different baseline RSA (some sit around 0.10, others around
0.40+), which is a large source of between-mouse variability in the pooled
Figure 5 analysis. This script removes that between-mouse variance by
subtracting each mouse's mean RSA across its 3 cross-day pairs, then asks
the same question on the centred data:

  Within each mouse, do higher-running-mismatch pairs have lower (mouse-
  centred) cross-day RSA?

Mean-centring (not z-scoring) is used because with only 3 observations
per mouse the per-mouse SD is unstable; mean-centring is the conservative
choice that removes only the fixed per-mouse baseline.

Outcome interpretation:
  - If the centred Spearman rho is more negative than the raw pooled rho
    (and tighter): the Figure 5 visible negative trend was hidden by
    between-mouse variance, and the within-mouse running effect is real.
  - If the centred rho is similar to the raw rho: the trend is the same,
    cohort size is the binding constraint.
  - If the centred rho reverses or vanishes: the visible negative trend
    in Figure 5 was driven by between-mouse variance, not within-mouse
    behaviour.

Inputs:
  outputs/movie1/session_rsa_trial_mismatch.csv  (from script 41)

Outputs:
  outputs/movie1/session_rsa_trial_mismatch_centred.csv         per-row centred table
  outputs/movie1/session_rsa_trial_mismatch_centred_stats.csv   cohort stats
  outputs/movie1/session_rsa_trial_mismatch_centred.png         single comparison plot

Usage:
  python3 scripts/41b_per_mouse_centred_check.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-10
ROOT = Path("outputs/movie1")

PAIR_COLOR = {"A-B": "#a05a8a", "B-C": "#d63384", "A-C": "#ffd60a"}
PAIR_LABEL = {"A-B": "A-B (1 day)", "B-C": "B-C (1 day)", "A-C": "A-C (2 days)"}

parser = argparse.ArgumentParser()
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--perm", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rankdata_avg(x):
    x = np.asarray(x, float); n = len(x)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(n, float); r[order] = np.arange(1, n + 1, dtype=float)
    sx = x[order]; i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]: j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def spearman_rho(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4: return np.nan
    ra, rb = rankdata_avg(a[m]), rankdata_avg(b[m])
    if np.std(ra) < EPS or np.std(rb) < EPS: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def mouse_level_bootstrap_rho(sub, x_col, y_col, B, rng):
    obs = spearman_rho(sub[x_col].values, sub[y_col].values)
    mice = sub["container_id"].unique(); n = len(mice)
    rhos = np.empty(B, float)
    for b in range(B):
        sel = rng.choice(mice, size=n, replace=True)
        chunks = [sub[sub["container_id"] == m] for m in sel]
        s = pd.concat(chunks, ignore_index=True)
        rhos[b] = spearman_rho(s[x_col].values, s[y_col].values)
    rhos = rhos[np.isfinite(rhos)]
    if len(rhos) == 0: return obs, np.nan, np.nan
    return obs, float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def within_mouse_y_shuffle_p(sub, x_col, y_col, B, rng):
    obs = spearman_rho(sub[x_col].values, sub[y_col].values)
    if not np.isfinite(obs): return np.nan
    groups = [g[y_col].values for _, g in sub.groupby("container_id", sort=False)]
    x = sub[x_col].values
    nulls = np.empty(B, float)
    for b in range(B):
        permuted = [rng.permutation(g) for g in groups]
        y = np.concatenate(permuted)
        nulls[b] = spearman_rho(x, y)
    nulls = nulls[np.isfinite(nulls)]
    return float((np.sum(np.abs(nulls) >= np.abs(obs)) + 1) / (len(nulls) + 1))


# ---------------------------------------------------------------------------
# Load + per-mouse mean-centre
# ---------------------------------------------------------------------------

src = ROOT / "session_rsa_trial_mismatch.csv"
df = pd.read_csv(src)
ana = df.dropna(subset=["session_rsa", "mean_trial_pair_running_mismatch"]).copy()

ana["rsa_mouse_mean"] = ana.groupby("container_id")["session_rsa"].transform("mean")
ana["rsa_centred"]    = ana["session_rsa"] - ana["rsa_mouse_mean"]

ana.to_csv(ROOT / "session_rsa_trial_mismatch_centred.csv", index=False)
print(f"Loaded {src.name}  ({len(ana)} rows, {ana['container_id'].nunique()} mice)")
print(f"Per-mouse RSA mean range: [{ana['rsa_mouse_mean'].min():.3f}, "
      f"{ana['rsa_mouse_mean'].max():.3f}]")
print(f"Centred RSA range: [{ana['rsa_centred'].min():+.3f}, "
      f"{ana['rsa_centred'].max():+.3f}]")


# ---------------------------------------------------------------------------
# Formal stats on centred RSA (same design as Figure 5)
# ---------------------------------------------------------------------------

print(f"\nFormal stats on per-mouse-centred RSA "
      f"(mouse-level bootstrap B = {args.boot:,}; perm B = {args.perm:,}):")
stat_rows = []
for label, sub in [("all (1d + 2d)", ana),
                   ("1-day only",    ana[ana["days_apart"] == 1]),
                   ("2-day only",    ana[ana["days_apart"] == 2])]:
    rho, lo, hi = mouse_level_bootstrap_rho(
        sub, "mean_trial_pair_running_mismatch", "rsa_centred", args.boot, rng)
    p = within_mouse_y_shuffle_p(
        sub, "mean_trial_pair_running_mismatch", "rsa_centred", args.perm, rng)
    print(f"  {label:>16}: n = {len(sub):>2}  "
          f"rho = {rho:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  perm p = {p:.4f}")
    stat_rows.append(dict(subset=label, n=len(sub),
                          rho_centred=rho, ci_lo=lo, ci_hi=hi, perm_p=p))

# Compare to raw stats (from existing stats csv) for direct readout
raw_stats = pd.read_csv(ROOT / "session_rsa_trial_mismatch_stats.csv")
print("\nComparison vs RAW Figure 5 stats:\n")
print(f"  {'subset':>16}  {'rho_raw':>9}  {'rho_centred':>12}  {'p_raw':>7}  {'p_centred':>10}")
for r_raw, r_cen in zip(raw_stats.to_dict("records"), stat_rows):
    print(f"  {r_raw['subset']:>16}  {r_raw['rho']:+.3f}     {r_cen['rho_centred']:+.3f}        "
          f"{r_raw['perm_p']:.3f}    {r_cen['perm_p']:.3f}")

pd.DataFrame(stat_rows).to_csv(ROOT / "session_rsa_trial_mismatch_centred_stats.csv",
                                index=False)


# ---------------------------------------------------------------------------
# Plot (single panel) — centred RSA vs running mismatch
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10.0, 6.2))

for pair in ["A-B", "B-C", "A-C"]:
    sub = ana[ana["pair"] == pair]
    color = PAIR_COLOR[pair]
    ax.scatter(sub["mean_trial_pair_running_mismatch"], sub["rsa_centred"],
               s=85, color=color, alpha=0.85,
               edgecolors="black", linewidth=0.7,
               label=PAIR_LABEL[pair], zorder=3)
    x = sub["mean_trial_pair_running_mismatch"].values
    y = sub["rsa_centred"].values
    if len(x) >= 3 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, slope * xx + intercept, color=color, lw=1.6,
                alpha=0.65, ls="--", zorder=2)

ax.axhline(0, color="black", lw=0.5, zorder=1)
ax.set_xlabel("Mean trial-pair running mismatch  (cm/s scale, 0-1)", fontsize=11)
ax.set_ylabel("Per-mouse-centred cross-day RSA  (RSA − mouse mean RSA)",
              fontsize=11)
ax.set_title(f"Robustness check — per-mouse mean-centred RSA vs trial-level running mismatch  "
             f"({ana['container_id'].nunique()} mice, {len(ana)} session-pair points)\n"
             "Mean-centring removes between-mouse baseline RSA differences; "
             "asks within-mouse: does higher mismatch predict lower RSA?",
             fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=10, frameon=True)

def fmt_p(p):
    if not np.isfinite(p): return "n.s."
    if p < 0.001: return "p < 0.001"
    return f"p = {p:.3f}"

s_all, s_1d, s_2d = stat_rows[0], stat_rows[1], stat_rows[2]
raw_all = raw_stats.iloc[0]; raw_1d = raw_stats.iloc[1]; raw_2d = raw_stats.iloc[2]
ann = (
    f"Centred (this plot)  |  Raw (Fig 5):\n"
    f"  All pairs:  ρ = {s_all['rho_centred']:+.3f}  CI [{s_all['ci_lo']:+.3f}, {s_all['ci_hi']:+.3f}]  "
    f"{fmt_p(s_all['perm_p'])}    |  ρ = {raw_all['rho']:+.3f}  {fmt_p(raw_all['perm_p'])}\n"
    f"  1-day:      ρ = {s_1d['rho_centred']:+.3f}  CI [{s_1d['ci_lo']:+.3f}, {s_1d['ci_hi']:+.3f}]  "
    f"{fmt_p(s_1d['perm_p'])}    |  ρ = {raw_1d['rho']:+.3f}  {fmt_p(raw_1d['perm_p'])}\n"
    f"  2-day:      ρ = {s_2d['rho_centred']:+.3f}  CI [{s_2d['ci_lo']:+.3f}, {s_2d['ci_hi']:+.3f}]  "
    f"{fmt_p(s_2d['perm_p'])}    |  ρ = {raw_2d['rho']:+.3f}  {fmt_p(raw_2d['perm_p'])}"
)
ax.text(0.98, 0.02, ann, transform=ax.transAxes,
        fontsize=8.5, va="bottom", ha="right", family="monospace",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.92,
                  boxstyle="round,pad=0.4"))

plt.tight_layout()
out_png = ROOT / "session_rsa_trial_mismatch_centred.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out_png}")
print(f"Saved {ROOT / 'session_rsa_trial_mismatch_centred.csv'}")
print(f"Saved {ROOT / 'session_rsa_trial_mismatch_centred_stats.csv'}")
