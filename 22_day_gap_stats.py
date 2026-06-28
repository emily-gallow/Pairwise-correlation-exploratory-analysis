"""
22_day_gap_stats.py
Quantify the cross-day RSA drop from 1-day to 2-day gaps across animals.

Reads per-container day_gap_scatter.csv files (same inputs as script 21) and
writes several CSV summaries:

  day_gap_stats_per_animal_<metric>.csv
      One row per animal × type (signal / noise): per-pair r values,
      collapsed 1-day reference, deltas, slope across 3 points, drop flags.

  day_gap_stats_group_<metric>.csv
      Population-level descriptive stats and hypothesis tests for each
      contrast (mean 1-day vs 2-day; A–B vs A–C; B–C vs A–C).

  day_gap_stats_by_pair_<metric>.csv
      Marginal mean ± SEM of r at each pair (A–B, B–C, A–C) across animals.

  day_gap_stats_mixed_lm_<metric>.csv
      Linear mixed model: r ~ days_apart + (1 | container_id), one row per
      type (signal / noise). Uses all 3 points per animal.

Usage:
  python3 scripts/22_day_gap_stats.py
  python3 scripts/22_day_gap_stats.py --metric matched
  python3 scripts/22_day_gap_stats.py --metric all
  python3 scripts/22_day_gap_stats.py --root outputs/movie1 --bootstrap-n 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

PAIR_ORDER = ["A-B", "B-C", "A-C"]
CONTRASTS = (
    ("mean_1day_vs_2day", "r_1day_mean", "r_AC"),
    ("AB_vs_AC", "r_AB", "r_AC"),
    ("BC_vs_AC", "r_BC", "r_AC"),
)

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1",
                    help="Folder containing per-container subfolders.")
parser.add_argument("--metric", default="all",
                    help="'matched', 'fulldata', or 'all' (default: every "
                         "metric present with >=3 animals).")
parser.add_argument("--bootstrap-n", type=int, default=10_000,
                    help="Bootstrap replicates for 95%% CI on mean delta.")
parser.add_argument("--seed", type=int, default=0,
                    help="RNG seed for bootstrap.")
args = parser.parse_args()

ROOT = Path(args.root)
RNG = np.random.default_rng(args.seed)


def load_long(metric: str) -> pd.DataFrame:
    frames = []
    # CRITICAL: only numeric-named container folders (skip 'not using?' etc.)
    for csv_path in sorted(ROOT.glob("*/day_gap_scatter.csv")):
        if not csv_path.parent.name.isdigit():
            continue
        df_c = pd.read_csv(csv_path)
        if metric not in df_c["metric"].values:
            continue
        df_c = df_c[df_c["metric"] == metric].copy()
        df_c["container_id"] = csv_path.parent.name
        frames.append(df_c)
    if not frames:
        raise SystemExit(f"No rows for metric={metric!r} under {ROOT}/*/")
    return pd.concat(frames, ignore_index=True)


def pair_lookup(g: pd.DataFrame) -> dict[str, float]:
    by_pair = g.set_index("pair")["r"]
    return {p: float(by_pair[p]) for p in PAIR_ORDER}


def cohens_d_paired(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return np.nan
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else np.nan


def bootstrap_ci(values: np.ndarray, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_tests(x2: np.ndarray, x1: np.ndarray) -> dict:
    """Tests for x2 - x1 (e.g. 2-day minus 1-day reference)."""
    diff = x2 - x1
    n = len(diff)
    out = dict(n_animals=n)
    if n < 2:
        out.update(t_stat=np.nan, t_pvalue=np.nan,
                   wilcoxon_stat=np.nan, wilcoxon_pvalue=np.nan,
                   sign_test_p=np.nan, cohens_d_paired=np.nan,
                   bootstrap_ci_delta_lo=np.nan, bootstrap_ci_delta_hi=np.nan)
        return out

    t_res = stats.ttest_rel(x2, x1, nan_policy="omit")
    try:
        w_res = stats.wilcoxon(x2, x1)
        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    n_drop = int(np.sum(diff < 0))
    # H0: P(drop) = 0.5
    sign_p = float(stats.binomtest(n_drop, n, 0.5).pvalue)

    lo, hi = bootstrap_ci(diff, args.bootstrap_n)
    out.update(
        t_stat=float(t_res.statistic),
        t_pvalue=float(t_res.pvalue),
        wilcoxon_stat=w_stat,
        wilcoxon_pvalue=w_p,
        sign_test_p=sign_p,
        cohens_d_paired=cohens_d_paired(diff),
        bootstrap_ci_delta_lo=lo,
        bootstrap_ci_delta_hi=hi,
    )
    return out


def per_animal_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cid, tp), g in long_df.groupby(["container_id", "type"]):
        pr = pair_lookup(g)
        r_ab, r_bc, r_ac = pr["A-B"], pr["B-C"], pr["A-C"]
        r_1day_mean = 0.5 * (r_ab + r_bc)
        lr = stats.linregress(g["days_apart"], g["r"])
        rows.append(dict(
            container_id=cid,
            type=tp,
            metric=g["metric"].iloc[0],
            r_AB=r_ab,
            r_BC=r_bc,
            r_AC=r_ac,
            r_1day_mean=r_1day_mean,
            delta_mean_1day=r_ac - r_1day_mean,
            delta_AB=r_ac - r_ab,
            delta_BC=r_ac - r_bc,
            slope_3pt=float(lr.slope),
            slope_3pt_stderr=float(lr.stderr),
            slope_3pt_pvalue=float(lr.pvalue),
            dropped_vs_mean_1day=bool(r_ac < r_1day_mean),
            dropped_vs_AB=bool(r_ac < r_ab),
            dropped_vs_BC=bool(r_ac < r_bc),
        ))
    return pd.DataFrame(rows)


def group_table(per_animal: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for tp in ("signal", "noise"):
        sub = per_animal[per_animal["type"] == tp]
        if sub.empty:
            continue
        for contrast_name, col_1day, col_2day in CONTRASTS:
            x1 = sub[col_1day].to_numpy(dtype=float)
            x2 = sub[col_2day].to_numpy(dtype=float)
            diff = x2 - x1
            tests = paired_tests(x2, x1)

            slopes = sub["slope_3pt"].to_numpy(dtype=float)
            slope_t = (stats.ttest_1samp(slopes, 0.0, nan_policy="omit")
                       if len(slopes) >= 2
                       else stats.TtestResult(np.nan, np.nan))

            rows.append(dict(
                metric=metric,
                type=tp,
                contrast=contrast_name,
                mean_r_1day_ref=float(np.mean(x1)),
                sem_r_1day_ref=float(stats.sem(x1)),
                mean_r_2day=float(np.mean(x2)),
                sem_r_2day=float(stats.sem(x2)),
                mean_delta=float(np.mean(diff)),
                sem_delta=float(stats.sem(diff)),
                median_delta=float(np.median(diff)),
                std_delta=float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan,
                fraction_dropped=float(np.mean(diff < 0)),
                n_dropped=int(np.sum(diff < 0)),
                mean_slope_3pt=float(np.mean(slopes)),
                sem_slope_3pt=float(stats.sem(slopes)),
                slope_one_sample_t=float(slope_t.statistic),
                slope_one_sample_p=float(slope_t.pvalue),
                **tests,
            ))
    return pd.DataFrame(rows)


def by_pair_table(long_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for tp in ("signal", "noise"):
        sub = long_df[long_df["type"] == tp]
        for pair in PAIR_ORDER:
            vals = sub.loc[sub["pair"] == pair, "r"].astype(float)
            if vals.empty:
                continue
            rows.append(dict(
                metric=metric,
                type=tp,
                pair=pair,
                days_apart=int(sub.loc[sub["pair"] == pair, "days_apart"].iloc[0]),
                n_animals=int(vals.shape[0]),
                mean_r=float(vals.mean()),
                sem_r=float(stats.sem(vals)),
                median_r=float(vals.median()),
                std_r=float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                min_r=float(vals.min()),
                max_r=float(vals.max()),
            ))
    return pd.DataFrame(rows)


def mixed_lm_table(long_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    if not HAS_STATSMODELS:
        return pd.DataFrame([dict(
            metric=metric, type="all", note="statsmodels not installed",
        )])
    for tp in ("signal", "noise"):
        sub = long_df[long_df["type"] == tp].copy()
        if sub["container_id"].nunique() < 3:
            rows.append(dict(
                metric=metric, type=tp, n_animals=sub["container_id"].nunique(),
                n_obs=len(sub), beta_days_apart=np.nan, se_days_apart=np.nan,
                p_days_apart=np.nan, intercept=np.nan,
                note="too few animals for mixed LM",
            ))
            continue
        try:
            fit = smf.mixedlm("r ~ days_apart", sub, groups=sub["container_id"]).fit(
                reml=True, method="lbfgs"
            )
            rows.append(dict(
                metric=metric,
                type=tp,
                n_animals=sub["container_id"].nunique(),
                n_obs=len(sub),
                beta_days_apart=float(fit.params["days_apart"]),
                se_days_apart=float(fit.bse["days_apart"]),
                p_days_apart=float(fit.pvalues["days_apart"]),
                intercept=float(fit.params["Intercept"]),
                note="",
            ))
        except Exception as exc:  # noqa: BLE001 — surface model failure in CSV
            rows.append(dict(
                metric=metric, type=tp,
                n_animals=sub["container_id"].nunique(),
                n_obs=len(sub),
                beta_days_apart=np.nan, se_days_apart=np.nan,
                p_days_apart=np.nan, intercept=np.nan,
                note=f"mixed LM failed: {exc}",
            ))
    return pd.DataFrame(rows)


def metrics_to_run() -> list[str]:
    if args.metric != "all":
        return [args.metric]
    found = set()
    # CRITICAL: only numeric-named container folders.
    for csv_path in ROOT.glob("*/day_gap_scatter.csv"):
        if not csv_path.parent.name.isdigit():
            continue
        found.update(pd.read_csv(csv_path)["metric"].unique())
    # Prefer matched first; only include fulldata if enough animals.
    out = []
    for m in ("matched", "fulldata"):
        if m not in found:
            continue
        long_df = load_long(m)
        if long_df["container_id"].nunique() >= 3:
            out.append(m)
        else:
            print(f"Skipping metric={m!r}: only "
                  f"{long_df['container_id'].nunique()} animal(s) with data.")
    if not out:
        raise SystemExit("No metric with >=3 animals found.")
    return out


def main():
    for metric in metrics_to_run():
        print(f"\n=== metric: {metric} ===")
        long_df = load_long(metric)
        n = long_df["container_id"].nunique()
        print(f"  {n} animals, {len(long_df)} rows")

        per_animal = per_animal_table(long_df)
        group = group_table(per_animal, metric)
        by_pair = by_pair_table(long_df, metric)
        mixed = mixed_lm_table(long_df, metric)

        paths = {
            "per_animal": ROOT / f"day_gap_stats_per_animal_{metric}.csv",
            "group": ROOT / f"day_gap_stats_group_{metric}.csv",
            "by_pair": ROOT / f"day_gap_stats_by_pair_{metric}.csv",
            "mixed_lm": ROOT / f"day_gap_stats_mixed_lm_{metric}.csv",
        }
        per_animal.to_csv(paths["per_animal"], index=False)
        group.to_csv(paths["group"], index=False)
        by_pair.to_csv(paths["by_pair"], index=False)
        mixed.to_csv(paths["mixed_lm"], index=False)

        for name, path in paths.items():
            print(f"  Saved {path}  ({name})")

        # Console highlight for the primary contrast.
        primary = group[
            (group["type"] == "signal") &
            (group["contrast"] == "mean_1day_vs_2day")
        ]
        if len(primary):
            r = primary.iloc[0]
            print(f"  Signal / mean_1day_vs_2day: "
                  f"Δ={r['mean_delta']:.4f} (SEM {r['sem_delta']:.4f}), "
                  f"{r['n_dropped']}/{r['n_animals']:.0f} dropped, "
                  f"p(t)={r['t_pvalue']:.4g}, p(Wilcoxon)={r['wilcoxon_pvalue']:.4g}")


if __name__ == "__main__":
    main()
