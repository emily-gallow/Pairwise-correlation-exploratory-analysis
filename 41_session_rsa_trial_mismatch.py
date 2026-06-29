"""
41_session_rsa_trial_mismatch.py
Session-level 10-trial cross-day RSA vs trial-level mean running mismatch.

Same biological question as script 40 (does running mismatch predict lower
cross-day RSA?), but with the LESS NOISY matrix estimator. Instead of
computing trial-pair RSAs from single-trial pairwise correlation matrices
(noisy because each matrix uses only ~900 frames from one trial), this
script uses the full 10-trial PSTH-based session-level signal correlation
matrices — the same estimator used by the headline Figure 3 RSA — and
pairs that with a trial-level summary of running mismatch.

Per (mouse, session pair):
  y = session-level cross-day RSA
      = Pearson r between off-diagonal upper-triangle vectors of the two
        10-trial PSTH-based signal correlation matrices on the matched-
        cell set (the recipe from scripts 28/31, Figure 3).
  x = mean trial-pair running mismatch
      = mean over (i, j) in 10 x 10 trial-pairs of
        | running_frac_X_trial_i  −  running_frac_Y_trial_j |
      where running_frac per trial = fraction of trial frames with
      |speed| > 1 cm/s (matches the threshold from Figure 5).

One point per (mouse, session pair). 48 points total (16 mice x 3 pairs).
Three colours by cross-day pair (A-B, B-C, A-C).

This is the "10-trial-PSTH" companion to script 40's "single-trial-PSTH"
version. The trade-off:
  - script 40 (single-trial):   noisy matrices, 4800 points,
                                no useful trend visible
  - this script (10-trial):     stable matrices (matches headline RSA),
                                48 points, x-axis is summarised rather
                                than per-trial-pair

Inputs (per numeric-name container):
  {root}/{cid}/{A,B,C}/events_matrix.npy
  {root}/{cid}/{A,B,C}/movie_repeat_frames.npy
  {root}/{cid}/{A,B,C}/running_speed.npy

Outputs:
  outputs/movie1/session_rsa_trial_mismatch.csv
  outputs/movie1/session_rsa_trial_mismatch.png

Usage:
  python3 scripts/41_session_rsa_trial_mismatch.py
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
ROOT = Path("outputs/movie1")
SESSIONS = ["A", "B", "C"]
PAIRS = [("A", "B", 1, "#a05a8a"),
         ("B", "C", 1, "#d63384"),
         ("A", "C", 2, "#ffd60a")]
RUN_THRESH = 1.0   # cm/s; matches Figure 5 binary running threshold

parser = argparse.ArgumentParser()
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--perm", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)


# Helpers

def present_mask(events):
    return ~np.all(np.isnan(events), axis=1)


def session_signal_corr_10trial(events, frames, cell_idx):
    """10-trial PSTH-based signal correlation matrix on matched cells.
    Same recipe as scripts 28/31 (headline Figure 3/4 estimator)."""
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


def per_trial_pupil_mean(pupil, sf, ef):
    """Mean pupil area during a single trial.
    pupil shape: (2, T) where row 0 = timestamps, row 1 = pupil size."""
    p = pupil[1, sf: ef + 1]
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return np.nan
    return float(np.mean(p))


def z_score(values):
    """Z-score a 1D array, leaving NaNs as NaN. If SD = 0 returns zeros."""
    a = np.asarray(values, float)
    mask = np.isfinite(a)
    if mask.sum() < 2: return np.full_like(a, np.nan)
    m = a[mask].mean(); s = a[mask].std(ddof=1)
    if s < EPS: return np.where(mask, 0.0, np.nan)
    z = np.full_like(a, np.nan)
    z[mask] = (a[mask] - m) / s
    return z


def per_trial_running_frac(running, sf, ef, threshold=RUN_THRESH):
    s = running[0, sf: ef + 1]
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.nan
    return float(np.mean(np.abs(s) > threshold))


def matrix_rsa(C1, C2):
    n = C1.shape[0]
    iu = np.triu_indices(n, k=1)
    a, b = C1[iu], C2[iu]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return np.nan
    a, b = a[m], b[m]
    if np.std(a) < EPS or np.std(b) < EPS: return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def mean_trial_pair_mismatch(run_X, run_Y):
    """Mean over the 10x10 = 100 trial-pairs of |run_X_i - run_Y_j|."""
    diffs = []
    for i in range(len(run_X)):
        for j in range(len(run_Y)):
            ri, rj = run_X[i], run_Y[j]
            if np.isfinite(ri) and np.isfinite(rj):
                diffs.append(abs(ri - rj))
    return float(np.mean(diffs)) if diffs else np.nan


def rankdata_avg(x):
    x = np.asarray(x, float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(n, float)
    r[order] = np.arange(1, n + 1, dtype=float)
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


def mouse_level_bootstrap_rho(sub_df, x_col, y_col, mouse_col, B, rng):
    """Bootstrap Spearman rho by resampling MICE with replacement, then
    taking all of each chosen mouse's rows. Returns (obs_rho, lo, hi)."""
    obs = spearman_rho(sub_df[x_col].values, sub_df[y_col].values)
    mice = sub_df[mouse_col].unique()
    n = len(mice)
    rhos = np.empty(B, float)
    for b in range(B):
        sel = rng.choice(mice, size=n, replace=True)
        chunks = [sub_df[sub_df[mouse_col] == m] for m in sel]
        sub = pd.concat(chunks, ignore_index=True)
        rhos[b] = spearman_rho(sub[x_col].values, sub[y_col].values)
    rhos = rhos[np.isfinite(rhos)]
    if len(rhos) == 0:
        return obs, np.nan, np.nan
    return obs, float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def within_mouse_y_shuffle_p(sub_df, x_col, y_col, mouse_col, B, rng):
    """Two-sided permutation p: shuffle y WITHIN each mouse (preserves
    mouse-level structure, breaks x-y association). H0: rho = 0."""
    obs = spearman_rho(sub_df[x_col].values, sub_df[y_col].values)
    if not np.isfinite(obs): return np.nan
    groups = [g[y_col].values for _, g in sub_df.groupby(mouse_col, sort=False)]
    x = sub_df[x_col].values
    nulls = np.empty(B, float)
    for b in range(B):
        permuted = [rng.permutation(g) for g in groups]
        y = np.concatenate(permuted)
        nulls[b] = spearman_rho(x, y)
    nulls = nulls[np.isfinite(nulls)]
    return float((np.sum(np.abs(nulls) >= np.abs(obs)) + 1) / (len(nulls) + 1))


# Main loop

print("=" * 70)
print(" Session-level 10-trial RSA vs trial-level running mismatch (mean)")
print("=" * 70)

rows = []
for cid_dir in sorted(ROOT.iterdir()):
    if not (cid_dir.is_dir() and cid_dir.name.isdigit()):
        continue
    needed = ["events_matrix.npy", "movie_repeat_frames.npy",
              "running_speed.npy", "pupil_size.npy"]
    if not all((cid_dir / s / f).exists() for s in SESSIONS for f in needed):
        print(f"  skip {cid_dir.name}: missing running or pupil data")
        continue

    ev = {s: np.load(cid_dir / s / "events_matrix.npy") for s in SESSIONS}
    fr = {s: np.load(cid_dir / s / "movie_repeat_frames.npy") for s in SESSIONS}
    rn = {s: np.load(cid_dir / s / "running_speed.npy") for s in SESSIONS}
    pu = {s: np.load(cid_dir / s / "pupil_size.npy") for s in SESSIONS}

    matched = np.where(present_mask(ev["A"])
                       & present_mask(ev["B"])
                       & present_mask(ev["C"]))[0]
    if len(matched) < 10:
        print(f"  skip {cid_dir.name}: only {len(matched)} matched cells")
        continue

    # 10-trial PSTH-based session-level signal correlation matrices
    sig_mats = {s: session_signal_corr_10trial(ev[s], fr[s], matched)
                for s in SESSIONS}

    # Per-trial running fractions and per-trial mean pupil
    trial_runs, trial_pupil_raw = {s: [] for s in SESSIONS}, {s: [] for s in SESSIONS}
    bad = False
    for s in SESSIONS:
        if len(fr[s]) != 10:
            bad = True; break
        for sf, ef in fr[s]:
            sf, ef = int(sf), int(ef)
            trial_runs[s].append(per_trial_running_frac(rn[s], sf, ef))
            trial_pupil_raw[s].append(per_trial_pupil_mean(pu[s], sf, ef))
    if bad: continue

    # Per-mouse z-score for pupil across all 30 trials (10 per session × 3 sessions)
    all_pupil = np.array(trial_pupil_raw["A"] + trial_pupil_raw["B"] + trial_pupil_raw["C"])
    z_all = z_score(all_pupil)
    trial_pupil_z = {"A": list(z_all[:10]), "B": list(z_all[10:20]), "C": list(z_all[20:30])}

    for s1, s2, gap, _color in PAIRS:
        signal_rsa = matrix_rsa(sig_mats[s1], sig_mats[s2])
        run_mismatch = mean_trial_pair_mismatch(trial_runs[s1], trial_runs[s2])
        pup_mismatch = mean_trial_pair_mismatch(trial_pupil_z[s1], trial_pupil_z[s2])
        rows.append(dict(
            container_id=cid_dir.name, pair=f"{s1}-{s2}", days_apart=gap,
            signal_rsa=signal_rsa,
            mean_trial_pair_running_mismatch=run_mismatch,
            mean_trial_pair_pupil_mismatch=pup_mismatch,
        ))
    print(f"  {cid_dir.name}: done")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "session_rsa_trial_mismatch.csv", index=False)
print(f"\n  Saved {ROOT / 'session_rsa_trial_mismatch.csv'}  "
      f"({len(df)} rows from {df['container_id'].nunique()} mice)")


# Formal stats: Spearman rho with mouse-level bootstrap CI + within-mouse
# y-shuffle permutation. Run for SIGNAL and NOISE RSA separately.
# All pairs / 1-day only / 2-day only.

ana = df.dropna(subset=["signal_rsa",
                        "mean_trial_pair_running_mismatch",
                        "mean_trial_pair_pupil_mismatch"]).copy()

print(f"\nFormal stats (mouse-level bootstrap B = {args.boot:,}; "
      f"within-mouse y-shuffle perm B = {args.perm:,}):")

stat_rows = []
for proxy_label, proxy_col in [("running", "mean_trial_pair_running_mismatch"),
                                ("pupil",   "mean_trial_pair_pupil_mismatch")]:
    print(f"\n  --- {proxy_label.upper()} mismatch vs signal RSA ---")
    for label, sub in [("all (1d + 2d)", ana),
                       ("1-day only",    ana[ana["days_apart"] == 1]),
                       ("2-day only",    ana[ana["days_apart"] == 2])]:
        obs, lo, hi = mouse_level_bootstrap_rho(
            sub, proxy_col, "signal_rsa",
            "container_id", args.boot, rng)
        p = within_mouse_y_shuffle_p(
            sub, proxy_col, "signal_rsa",
            "container_id", args.perm, rng)
        print(f"  {label:>16}: n = {len(sub):>2}  "
              f"rho = {obs:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  perm p = {p:.4f}")
        stat_rows.append(dict(proxy=proxy_label, subset=label, n=len(sub),
                              rho=obs, ci_lo=lo, ci_hi=hi, perm_p=p))

pd.DataFrame(stat_rows).to_csv(ROOT / "session_rsa_trial_mismatch_stats.csv",
                                index=False)
print(f"\n  Saved {ROOT / 'session_rsa_trial_mismatch_stats.csv'}")

# Cache for plot annotation
run_all = stat_rows[0]; run_1d = stat_rows[1]; run_2d = stat_rows[2]
pup_all = stat_rows[3]; pup_1d = stat_rows[4]; pup_2d = stat_rows[5]


# Plot: 2 panels (signal left, noise right)

def fmt_p(p):
    if not np.isfinite(p): return "n.s."
    if p < 0.001: return "p < 0.001"
    return f"p = {p:.3f}"


fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))

panels = [
    ("mean_trial_pair_running_mismatch",
     "Running mismatch",
     "Mean trial-pair running mismatch\n(|fraction time running > 1 cm/s, day X − day Y|, mean over 100 trial-pairs)",
     axes[0], (run_all, run_1d, run_2d)),
    ("mean_trial_pair_pupil_mismatch",
     "Pupil mismatch",
     "Mean trial-pair pupil mismatch\n(|z-scored mean pupil area, day X − day Y|, mean over 100 trial-pairs)",
     axes[1], (pup_all, pup_1d, pup_2d)),
]

for x_col, panel_title, x_label, ax, stats_triplet in panels:
    s_all, s_1d, s_2d = stats_triplet

    for s1, s2, gap, color in PAIRS:
        sub = df[df["pair"] == f"{s1}-{s2}"].dropna(
            subset=["signal_rsa", x_col])
        if sub.empty: continue
        ax.scatter(sub[x_col], sub["signal_rsa"],
                   s=80, color=color, alpha=0.85,
                   edgecolors="black", linewidth=0.6,
                   label=f"{s1}-{s2}  ({gap} day{'s' if gap > 1 else ''})",
                   zorder=3)
        # Per-colour linear fit
        x = sub[x_col].values
        y = sub["signal_rsa"].values
        if len(x) >= 3 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, slope * xx + intercept, color=color, lw=1.5,
                    alpha=0.65, ls="--", zorder=2)

    ax.axhline(0, color="black", lw=0.5, zorder=1)
    ax.set_xlabel(x_label, fontsize=10.5)
    ax.set_ylabel("Session-level cross-day signal RSA\n(10-trial PSTH-based)",
                  fontsize=10.5)
    ax.set_title(panel_title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    ann = (
        f"Per-mouse-resampled Spearman ρ:\n"
        f"  All pairs:  ρ = {s_all['rho']:+.3f}  "
        f"CI [{s_all['ci_lo']:+.3f}, {s_all['ci_hi']:+.3f}]  "
        f"{fmt_p(s_all['perm_p'])}\n"
        f"  1-day only: ρ = {s_1d['rho']:+.3f}  "
        f"CI [{s_1d['ci_lo']:+.3f}, {s_1d['ci_hi']:+.3f}]  "
        f"{fmt_p(s_1d['perm_p'])}\n"
        f"  2-day only: ρ = {s_2d['rho']:+.3f}  "
        f"CI [{s_2d['ci_lo']:+.3f}, {s_2d['ci_hi']:+.3f}]  "
        f"{fmt_p(s_2d['perm_p'])}"
    )
    ax.text(0.98, 0.02, ann, transform=ax.transAxes,
            fontsize=8.5, va="bottom", ha="right", family="monospace",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.92,
                      boxstyle="round,pad=0.4"))

n_mice = ana["container_id"].nunique()
n_pts = len(ana)
fig.suptitle(f"Cross-day signal RSA vs trial-level behavioural mismatch — running (left) and pupil (right)  "
             f"({n_mice} mice, {n_pts} session-pair points per panel)\n"
             "10-trial PSTH-based signal RSA (same as Figs 3, 4); "
             "x summarises the 10×10 trial-pair behavioural-mismatch matrix per session pair.",
             fontsize=11, y=1.02)

plt.tight_layout()
out_png = ROOT / "session_rsa_trial_mismatch.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")
