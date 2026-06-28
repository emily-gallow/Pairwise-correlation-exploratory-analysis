"""
27_formal_stats.py
Formal across-animal statistics for every headline cross-day comparison.

scipy is not available in the analysis environment, so all inference is done
with numpy only via:

  * paired sign-flip permutation test (exact, distribution-free):
      H0: per-animal difference d_i has zero mean.
      For B random sign-flip patterns we compute mean(d * signs); the
      two-sided p-value is mean(|null| >= |observed|).
  * bootstrap 95 % CI for the paired mean: resample animals with replacement
      B times, take 2.5/97.5 percentiles of the resampled mean.
  * Cohen's d_z for paired data = mean(d) / std(d, ddof=1).

This script tests:

  HEADLINE — cross-day RSA, 1 day vs 2 days, across mice:
    For each metric in {matched (5-rep mean of 4 half-pairings),
                        fulldata (10-rep cross-day)}:
      For each type in {signal, noise}:
        per-animal 1-day value = mean(A-B, B-C);  2-day = A-C
        paired test on d = 1-day - 2-day.

  STATE SPLIT — same paired test, restricted to mice whose state-subset
  values exist for both day-gaps; for state in {all, stationary, running}
  and type in {signal, noise}.

  BEHAVIOURAL REGRESSION (script 23) — sanity that running R² is ≈ 0:
    one-sample sign-flip test on per-session median R² (H0: median = 0,
    H1: > 0) — this isn't the headline but it's nice to nail down.

Outputs:
  outputs/movie1/stats/formal_stats.csv      one row per test
  outputs/movie1/stats/formal_stats.md       human-readable table

Usage:
  python3 scripts/27_formal_stats.py
  python3 scripts/27_formal_stats.py --boot 20000 --perm 20000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1",
                    help="Root with multi-animal CSVs.")
parser.add_argument("--boot", type=int, default=10000,
                    help="Bootstrap resamples (default 10 000).")
parser.add_argument("--perm", type=int, default=10000,
                    help="Permutation iterations (default 10 000).")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

ROOT = Path(args.root)
OUT  = ROOT / "stats"
OUT.mkdir(exist_ok=True)
rng  = np.random.default_rng(args.seed)

# Inference primitives

def paired_perm_test(d, B=10000, rng=None):
    """Two-sided sign-flip permutation p-value for paired mean(d) = 0.
    Returns (observed_mean, p, B_used)."""
    if rng is None: rng = np.random.default_rng()
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return float(np.mean(d)) if n else np.nan, np.nan, 0
    obs = float(np.mean(d))
    # Each iteration: random sign-flip per element
    signs = rng.choice([-1, 1], size=(B, n)).astype(float)
    null_means = (signs * d).mean(axis=1)
    p = float(np.mean(np.abs(null_means) >= np.abs(obs)))
    # Make sure p>0 even at the extreme; use (count+1)/(B+1) convention
    p_adj = (np.sum(np.abs(null_means) >= np.abs(obs)) + 1) / (B + 1)
    return obs, p_adj, B


def bootstrap_ci(d, B=10000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for the mean."""
    if rng is None: rng = np.random.default_rng()
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    idx = rng.integers(0, n, size=(B, n))
    boot_means = d[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100*alpha/2))
    hi = float(np.percentile(boot_means, 100*(1 - alpha/2)))
    return lo, hi


def cohens_dz(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2: return np.nan
    s = float(np.std(d, ddof=1))
    return float(np.mean(d) / s) if s > 0 else np.nan


def paired_test_block(d, name, extra=None):
    """Run perm + bootstrap, return a result dict."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    mean_d = float(np.mean(d)) if n else np.nan
    sem_d  = float(np.std(d, ddof=1)/np.sqrt(n)) if n > 1 else np.nan
    obs, p, B = paired_perm_test(d, B=args.perm, rng=rng)
    lo, hi    = bootstrap_ci(d, B=args.boot, rng=rng)
    dz        = cohens_dz(d)
    out = dict(
        test=name, n_animals=n,
        mean_diff=mean_d, sem_diff=sem_d,
        ci95_lo=lo, ci95_hi=hi,
        perm_p=p, perm_B=B,
        cohens_dz=dz,
    )
    if extra: out.update(extra)
    return out

# Build per-animal 1-day vs 2-day paired tables for matched & fulldata

def per_animal_pairs(df):
    """For each (container, type), return 1-day value (mean A-B & B-C) and
    2-day value (A-C)."""
    out = []
    for (cid, tp), g in df.groupby(["container_id", "type"]):
        v1 = g[g["days_apart"] == 1]["r"].astype(float).tolist()
        v2 = g[g["days_apart"] == 2]["r"].astype(float).tolist()
        if not v1 or not v2:
            continue
        out.append(dict(container_id=cid, type=tp,
                        one_day=float(np.mean(v1)),
                        two_day=float(v2[0])))
    return pd.DataFrame(out)


rows = []

for metric in ("matched", "fulldata"):
    df = pd.read_csv(ROOT / f"day_gap_scatter_multi_{metric}.csv")
    pa = per_animal_pairs(df)
    for tp in ("signal", "noise"):
        sub = pa[pa["type"] == tp].copy()
        if sub.empty: continue
        sub["d"] = sub["one_day"] - sub["two_day"]      # 1-day minus 2-day
        sub["one_day_minus_two_day_pct"] = (
            100 * sub["d"] / sub["one_day"].replace(0, np.nan))
        res = paired_test_block(
            sub["d"].values,
            name=f"headline {metric}/{tp} (1day vs 2day)",
            extra=dict(metric=metric, type=tp,
                       mean_1day=float(sub["one_day"].mean()),
                       mean_2day=float(sub["two_day"].mean())),
        )
        rows.append(res)

# Note: previous versions of this script also computed state-split paired
# tests (from script 25's state_split_rsa.csv) and a behaviour-regression
# sanity check (from script 23's behaviour_regress_summary_ALL.csv). Those
# analyses have been removed from the project — only the headline matched/
# fulldata × signal/noise paired tests remain. State-context tests for
# running- and pupil-based session-pair classifications are produced by
# scripts 29, 33, 34 (state-match family) and reported separately, not via
# this script.

res_df = pd.DataFrame(rows)
res_df.to_csv(OUT / "formal_stats.csv", index=False)

# Markdown report
def fmt_p(p):
    if not np.isfinite(p): return "—"
    if p < 0.001: return f"p < 0.001"
    return f"p = {p:.3f}"

def fmt_ci(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)): return "—"
    return f"[{lo:+.4f}, {hi:+.4f}]"

lines = [
    "# Formal statistics — paired tests across mice",
    "",
    "All p-values are two-sided paired sign-flip permutation tests "
    f"(B = {args.perm:,} iterations, distribution-free). 95% CI is "
    f"percentile bootstrap on the per-animal paired difference "
    f"(B = {args.boot:,}). Effect size is Cohen's d_z for paired data.",
    "",
    "*Sign convention:* `mean_diff = mean(1-day) − mean(2-day)`. A positive "
    "value means the 1-day correlation is higher than the 2-day one, i.e. "
    "time-graded drift in the expected direction.",
    "",
]

# Group: headline only (state-split and regression families removed)
def write_group(title, mask):
    sub = res_df[mask].copy()
    if sub.empty: return
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| test | n | mean(1d−2d) | 95% CI | dz | perm-p |")
    lines.append("|------|---|-------------|--------|------|--------|")
    for _, r in sub.iterrows():
        lines.append(
            f"| {r['test']} | {int(r['n_animals'])} | "
            f"{r['mean_diff']:+.4f} | {fmt_ci(r['ci95_lo'], r['ci95_hi'])} | "
            f"{r['cohens_dz']:+.2f} | {fmt_p(r['perm_p'])} |")
    lines.append("")

write_group("Headline cross-day RSA — matched vs fulldata",
            res_df["test"].str.startswith("headline"))

(OUT / "formal_stats.md").write_text("\n".join(lines))
print((OUT / "formal_stats.md").read_text())

print(f"\nSaved {OUT / 'formal_stats.csv'}")
print(f"Saved {OUT / 'formal_stats.md'}")
