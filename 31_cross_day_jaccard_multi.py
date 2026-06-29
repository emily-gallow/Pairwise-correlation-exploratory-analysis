"""
31_cross_day_jaccard_multi.py

Cohort-level cross-day Jaccard sweep — multi-animal replacement for the
single-animal Figure 2 (cross_day_adjacency.png) in the thesis figure set.

For every container with all three sessions' events_matrix.npy and a usable
matched-cell set, recompute:
  per session, the signal and noise correlation matrices on the matched cell
    set at frame rate (33 ms bins, 10 trials per session — the fulldata
    estimator that matches Figure 3's headline);
  for each density d ∈ [1%, 50%] (i.e. keep the top d-fraction of |r| pairs)
    binarise each session's matrix into an edge set, then compute the
    Jaccard overlap |E_a ∩ E_b| / |E_a ∪ E_b| for every cross-day pair
    (A–B, B–C, A–C) for both signal and noise.

The figure shows per-animal Jaccard curves as light coloured traces (one per
cross-day pair) and the cohort mean ± SEM as bold lines, with the analytic
chance baseline d/(2−d) as a dotted reference. Two panels side by side:
signal (left) and noise (right). Same colour convention as Figure 3 — A–B
purple, B–C magenta, A–C yellow.

This replaces the single-animal cross-day adjacency figure (which was
susceptible to per-animal cherry-picking) with a cohort-level view: every
mouse contributes three curves per panel, and the cohort mean is the visual
headline. Direction of the cross-day pair (1-day vs 2-day) is preserved as a
multi-animal trend rather than a one-mouse anecdote.

Inputs (per container under --root):
  {root}/{cid}/{A,B,C}/events_matrix.npy
  {root}/{cid}/{A,B,C}/movie_repeat_frames.npy

Outputs:
  outputs/movie1/cross_day_jaccard_multi.csv   long-format Jaccard table
  outputs/movie1/cross_day_jaccard_multi.png   signal | noise side by side

Usage:
  python3 scripts/31_cross_day_jaccard_multi.py
  python3 scripts/31_cross_day_jaccard_multi.py --n-densities 30
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = 30.0
EPS = 1e-10
SESSIONS = ["A", "B", "C"]
# (a, b, day_gap, colour)
PAIRS = [("A", "B", 1, "#a05a8a"),    # purple — adjacent days
         ("B", "C", 1, "#d63384"),    # magenta — adjacent days
         ("A", "C", 2, "#ffd60a")]    # yellow — 2-day gap

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--n-densities", type=int, default=50,
                    help="Number of density steps across the swept range. "
                         "Default 50.")
parser.add_argument("--d-min", type=float, default=0.01,
                    help="Lower density (strict end, top X%% kept). Default 0.01 "
                         "= top 1%% kept = threshold 99%%.")
parser.add_argument("--d-max", type=float, default=0.25,
                    help="Upper density (permissive end). Default 0.25 = top "
                         "25%% kept = threshold 75%%. This focuses the sweep on "
                         "the strict region where cross-day signal is most "
                         "distinguishable from chance.")
args = parser.parse_args()

ROOT = Path(args.root)
densities = np.linspace(args.d_min, args.d_max, args.n_densities)


# Per-session signal & noise matrices (fulldata 10-rep, frame rate)

def present_mask(events): return ~np.all(np.isnan(events), axis=1)


def session_corrs(events, frames, cell_idx):
    """Return signal + noise correlation matrices on the matched cell set,
    computed at frame rate (33 ms) using all 10 trials of the session."""
    ev_p = events[cell_idx, :].T                              # (T, n_cells)
    L_common = int(min(int(ef) - int(sf) + 1 for sf, ef in frames))
    reps = []
    for sf, _ in frames:
        sf = int(sf)
        reps.append(ev_p[sf: sf + L_common, :])
    T_tensor = np.stack(reps, axis=0)                         # (R, nb, n_p)
    R, nb, n_p = T_tensor.shape

    # Signal: PSTH across trials → corr across bins
    psth = T_tensor.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S = np.corrcoef(psth, rowvar=False)
    dead_s = psth.var(axis=0) <= EPS
    S[dead_s, :] = np.nan; S[:, dead_s] = np.nan

    # Noise: z-scored residuals pooled over (rep × bin)
    resid = T_tensor - psth[None, :, :]
    sd_rb = T_tensor.std(axis=0)
    nz = sd_rb > EPS
    z = resid / np.where(nz, sd_rb, 1.0)[None, :, :]
    z = z * nz[None, :, :]
    Zmat = z.reshape(R*nb, n_p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        N = np.corrcoef(Zmat, rowvar=False)
    dead_n = Zmat.var(axis=0) <= EPS
    N[dead_n, :] = np.nan; N[:, dead_n] = np.nan

    return dict(signal=S, noise=N)


def binary_at_density(C, d):
    """Boolean symmetric (n, n) matrix where 1 = pair is in the top-d
    fraction of |r| pairs (excluding NaN values). Diagonal is zero."""
    n = C.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = np.abs(C[iu])
    valid = np.isfinite(vals)
    if not valid.any():
        return np.zeros((n, n), dtype=bool)
    thr = np.quantile(vals[valid], 1.0 - d)
    abs_C = np.where(np.isfinite(C), np.abs(C), -1.0)
    B = abs_C >= thr
    np.fill_diagonal(B, False)
    return B


def jaccard(A_bool, B_bool):
    """Off-diagonal Jaccard between two boolean adjacency matrices."""
    n = A_bool.shape[0]
    iu = np.triu_indices(n, k=1)
    a = A_bool[iu]; b = B_bool[iu]
    union = int((a | b).sum())
    if union == 0: return np.nan
    return float((a & b).sum()) / union


# Walk containers and compute Jaccard curves

cid_dirs = [p for p in sorted(ROOT.iterdir())
            if p.is_dir() and p.name.isdigit()
            and all((p / s / "events_matrix.npy").exists() for s in SESSIONS)]
print(f"Found {len(cid_dirs)} containers with all 3 sessions.")

rows = []
for cid_dir in cid_dirs:
    cid = cid_dir.name
    ev = {s: np.load(cid_dir / s / "events_matrix.npy") for s in SESSIONS}
    fr = {s: np.load(cid_dir / s / "movie_repeat_frames.npy") for s in SESSIONS}
    matched = np.where(present_mask(ev["A"])
                       & present_mask(ev["B"])
                       & present_mask(ev["C"]))[0]
    if len(matched) < 5:
        continue

    # Per-session correlation matrices (computed once per session)
    mats = {s: session_corrs(ev[s], fr[s], matched) for s in SESSIONS}

    for d in densities:
        for tp in ("signal", "noise"):
            binA = binary_at_density(mats["A"][tp], d)
            binB = binary_at_density(mats["B"][tp], d)
            binC = binary_at_density(mats["C"][tp], d)
            for (a, b, gap, _), pair_label in zip(
                    PAIRS, ["A-B", "B-C", "A-C"]):
                bx = {"A": binA, "B": binB, "C": binC}
                j = jaccard(bx[a], bx[b])
                rows.append(dict(container_id=cid, density=float(d), type=tp,
                                 pair=pair_label, days_apart=gap, jaccard=j,
                                 n_matched=int(len(matched))))
    print(f"  {cid}: done ({len(matched)} matched cells)")

df = pd.DataFrame(rows)
out_csv = ROOT / "cross_day_jaccard_multi.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}  ({len(df)} rows from "
      f"{df['container_id'].nunique()} containers)")

# Figure: signal | noise, per-animal grey traces + cohort mean ± SEM

n_animals = df["container_id"].nunique()
chance = densities / (2 - densities)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=False)

# Convert density d (top-d kept) to threshold % zeroed = 100 × (1 − d) so the
# axis matches the single-animal Figure S5 convention: x = 50 (permissive,
# top 50% kept) on the left to x = 99 (strict, top 1% kept) on the right.
def density_to_threshold(d): return 100.0 * (1.0 - d)

for ax, tp, title in [(axes[0], "signal", "Signal correlation"),
                       (axes[1], "noise",  "Noise correlation")]:
    sub = df[df["type"] == tp].copy()
    # Per-animal traces — thin, semi-transparent, coloured by pair
    for (a, b, gap, color), pair_label in zip(PAIRS, ["A-B", "B-C", "A-C"]):
        per = sub[sub["pair"] == pair_label]
        for cid, g in per.groupby("container_id"):
            g = g.sort_values("density")
            ax.plot(density_to_threshold(g["density"]), g["jaccard"],
                    color=color, lw=0.6, alpha=0.18, zorder=1)
    # Cohort mean ± SEM per pair (bold)
    for (a, b, gap, color), pair_label in zip(PAIRS, ["A-B", "B-C", "A-C"]):
        per = sub[sub["pair"] == pair_label]
        agg = per.groupby("density")["jaccard"].agg(["mean",
            lambda x: np.std(x, ddof=1)/np.sqrt(len(x)) if len(x) > 1 else 0])
        agg.columns = ["mean", "sem"]
        x_vals = density_to_threshold(agg.index.values)
        ax.plot(x_vals, agg["mean"], color=color, lw=2.6,
                label=f"{pair_label} ({gap} day{'s' if gap > 1 else ''})",
                zorder=4)
        ax.fill_between(x_vals, agg["mean"] - agg["sem"],
                        agg["mean"] + agg["sem"], color=color,
                        alpha=0.22, zorder=3)
    # Chance reference, also on the threshold-zeroed x-axis
    ax.plot(density_to_threshold(densities), chance,
            color="grey", ls="--", lw=1.2,
            label="chance d / (2 − d)", zorder=2)
    ax.set_xlabel("Threshold %  (% of weakest |r| pairs zeroed)", fontsize=12)
    ax.set_ylabel("Cross-day Jaccard", fontsize=12)
    ax.set_title(title, fontsize=13)
    # Display strict end of the sweep (75–99 %) where the cross-day signal is
    # most clearly distinguishable from the chance baseline.
    ax.set_xlim(density_to_threshold(args.d_max), density_to_threshold(args.d_min))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, frameon=True)

fig.suptitle(
    f"Cross-day correlation-network overlap vs density — {n_animals} mice "
    f"(fulldata 10-rep)\n"
    "Per-animal Jaccards (light traces) and cohort mean ± SEM (bold), with "
    "analytic chance baseline.",
    fontsize=12, y=1.02)
plt.tight_layout(rect=[0, 0.02, 1, 1])

out_png = ROOT / "cross_day_jaccard_multi.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")


print("\n--- Cohort means at key densities (fulldata 10-rep) ---")
unique_densities = sorted(df["density"].unique())
for d_target in (0.05, 0.10, 0.20):
    # Snap to the closest unique density value actually present in the sweep,
    # so we report exactly one row per (type, pair) per target — no double-
    # counting from windowing tolerance.
    d_actual = min(unique_densities, key=lambda x: abs(x - d_target))
    if abs(d_actual - d_target) > 0.01:
        continue  # target outside swept range; skip
    print(f"\nTarget density = {d_target*100:.0f}% "
          f"(actual = {d_actual*100:.2f}%, chance = {d_actual/(2-d_actual):.3f})")
    for tp in ("signal", "noise"):
        for pair_label in ("A-B", "B-C", "A-C"):
            sub = df[(df["type"] == tp)
                     & (df["pair"] == pair_label)
                     & (df["density"] == d_actual)]
            if not len(sub): continue
            m = sub["jaccard"].mean()
            s = sub["jaccard"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0
            ch = d_actual / (2 - d_actual)
            print(f"  {tp:6}  {pair_label:>4}  mean = {m:.3f} ± {s:.3f}  "
                  f"(n = {len(sub)} mice; chance = {ch:.3f}; "
                  f"ratio over chance = {m/ch:.2f}×)")
