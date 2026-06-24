"""
16_cross_session_rsa.py
=======================
Second-order RSA — is the pairwise-correlation STRUCTURE preserved across days?

Single-neuron tuning is known to drift across days (Deitch/Rubin/Ziv 2021 on
this very dataset). The project question is one level up: even as individual
cells change, is the *relational* structure — which pairs are co-tuned (signal)
and which co-fluctuate (noise) — conserved? That is a second-order question, so
we use Representational Similarity Analysis:

  1. For each session restrict to the cells tracked across ALL three days
     (present-in-all, aligned by the canonical container ordering).
  2. Compute that session's signal- and noise-correlation matrices (frame-rate,
     same conventions as script 15).
  3. Take the upper-triangle vector of pairwise weights (one number per cell
     pair). The "second-order similarity" between two days is the correlation
     of these weight vectors. Stacking all day-pairs gives a 3x3 RSA matrix.

  SCATTER: day-i weight vs day-j weight, one point per cell pair — the raw
  picture behind each RSA number.

  NOISE CEILING (RSA diagonal): a low cross-day similarity can mean either the
  structure truly drifts OR that 10 repeats simply cannot estimate it. To tell
  these apart we put each session's SPLIT-HALF reliability (repeats 1-5 vs 6-10)
  on the diagonal. Cross-day off-diagonals should be read against that ceiling:
  cross-day ~ within-day reliability => structure is as stable as the data can
  resolve; cross-day << reliability => genuine drift.

  DATA-MATCHING (consistency with script 19): the split-half ceiling uses
  5-repeat halves, so the cross-day off-diagonals are ALSO computed from 5-rep
  halves — averaged over the four half-pairings {h1_i,h2_i}x{h1_j,h2_j} — rather
  than from the full 10-repeat matrices. Comparing a 10-rep cross-day estimate
  to a 5-rep ceiling would unfairly flatter stability (more data = less noise =
  higher similarity). The full-data (10-rep) cross-day values are still reported
  and annotated on the scatters for reference, clearly labelled as inflated.

Substrate: requires true L0 events (script 02 fixed loader). A gate aborts on
dF/F. Runs on the cached per-session matrices (numpy only) — no AllenSDK needed.

Inputs (per session S):
  outputs/movie1/{S}/events_matrix.npy        (n_canonical x n_timepoints)
  outputs/movie1/{S}/movie_repeat_frames.npy  (n_repeats x 2)

Outputs:
  outputs/movie1/cross_session_rsa.png   scatters (2x3) + RSA heatmaps (2)
  outputs/movie1/cross_session_rsa.csv   RSA values + split-half reliabilities

Usage:
  python3 scripts/16_cross_session_rsa.py
  python3 scripts/16_cross_session_rsa.py --bin-ms 100 --metric spearman
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------

SESSIONS = ["A", "B", "C"]
DAY      = {"A": "day 95", "B": "day 96", "C": "day 97"}
FPS      = 30.0
EPS      = 1e-10
PAIRS    = [("A", "B"), ("B", "C"), ("A", "C")]

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1",
                    help="Root holding {A,B,C}/ session folders.")
parser.add_argument("--bin-ms", type=float, default=33.0,
                    help="Bin width (ms). Default 33 ms ~ 1 frame (frame rate).")
parser.add_argument("--metric", choices=["pearson", "spearman"], default="pearson",
                    help="Second-order similarity metric for the RSA matrix and "
                         "the scatter annotations.")
args = parser.parse_args()

ROOT       = Path(args.root)
BIN_FRAMES = max(1, int(round(args.bin_ms / 1000.0 * FPS)))

print("=" * 70)
print("  Cross-session second-order RSA — matched cells, natural movie 1")
print(f"  root={ROOT}  bin={args.bin_ms:.0f} ms (~{BIN_FRAMES} frame"
      f"{'s' if BIN_FRAMES != 1 else ''})  metric={args.metric}")
print("=" * 70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def present_mask(events):
    return ~np.all(np.isnan(events), axis=1)


def bin_epoch(epoch, bin_frames):
    L, n = epoch.shape
    nb = L // bin_frames
    return epoch[: nb * bin_frames].reshape(nb, bin_frames, n).sum(axis=1)


def build_tensor(ev_cols, frames, bin_frames):
    """ev_cols: (T, n_cells) for the matched cells. Returns r[rep, bin, cell]."""
    L_common = int(min(int(ef) - int(sf) + 1 for sf, ef in frames))
    reps = [bin_epoch(ev_cols[int(sf): int(sf) + L_common, :], bin_frames)
            for sf, _ in frames]
    return np.stack(reps, axis=0)                       # (R, n_bins, n_cells)


def corr_dead(mat_samples):
    """Pearson corr across columns; zero-variance columns -> NaN."""
    dead = mat_samples.var(axis=0) <= EPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        C = np.corrcoef(mat_samples, rowvar=False)
    C[dead, :] = np.nan
    C[:, dead] = np.nan
    return C


def signal_corr(T):
    """Corr of trial-averaged responses (PSTH) across time-bins."""
    return corr_dead(T.mean(axis=0))


def noise_corr(T):
    """Corr of z-scored (per-bin) trial-to-trial residuals, pooled over (rep,bin)."""
    psth = T.mean(axis=0)
    resid = T - psth[None, :, :]
    sd = T.std(axis=0)
    nz = sd > EPS
    z = resid / np.where(nz, sd, 1.0)[None, :, :]
    z = z * nz[None, :, :]
    return corr_dead(z.reshape(T.shape[0] * T.shape[1], T.shape[2]))


def offdiag(C):
    n = C.shape[0]
    return C[np.triu_indices(n, k=1)]


def _rank(x):
    """Average-free ordinal ranks (ties negligible for continuous r values)."""
    order = x.argsort()
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(len(x), dtype=float)
    return r


def similarity(vi, vj, metric):
    """Correlation of two pairwise-weight vectors over their common valid pairs."""
    good = ~np.isnan(vi) & ~np.isnan(vj)
    if good.sum() < 3:
        return np.nan, good
    a, b = vi[good], vj[good]
    if metric == "spearman":
        a, b = _rank(a), _rank(b)
    return float(np.corrcoef(a, b)[0, 1]), good


# ---------------------------------------------------------------------------
# Load matched cells + per-session correlation matrices
# ---------------------------------------------------------------------------

present = {}
events  = {}
frames  = {}
for s in SESSIONS:
    ev = np.load(ROOT / s / "events_matrix.npy")
    flat = ev[~np.isnan(ev)]
    if np.mean(flat < 0) > 0.01:
        raise ValueError(f"Session {s}: events_matrix is {100*np.mean(flat<0):.1f}% "
                         f"negative — that's dF/F, not L0 events. Regenerate via "
                         f"the fixed script 02 before running RSA.")
    events[s]  = ev
    frames[s]  = np.load(ROOT / s / "movie_repeat_frames.npy")
    present[s] = present_mask(ev)

# Cells tracked across all three days (canonical ordering already aligns them).
shared = np.where(present["A"] & present["B"] & present["C"])[0]
print(f"\nCells tracked across all 3 days (matched set): {len(shared)}")

# Per-session matched correlation matrices (full data + the two repeat-halves).
sig = {}   # session -> signal corr (n_shared x n_shared)
noi = {}   # session -> noise  corr
sig_half = {}   # session -> (half1, half2) signal corr
noi_half = {}
for s in SESSIONS:
    ev_cols = events[s][shared, :].T                    # (T, n_shared)
    T = build_tensor(ev_cols, frames[s], BIN_FRAMES)    # (R, n_bins, n_shared)
    R = T.shape[0]
    sig[s] = signal_corr(T)
    noi[s] = noise_corr(T)
    h = R // 2
    sig_half[s] = (signal_corr(T[:h]), signal_corr(T[h:]))
    noi_half[s] = (noise_corr(T[:h]), noise_corr(T[h:]))
    print(f"  Session {s} ({DAY[s]}): {R} repeats x {T.shape[1]} bins "
          f"on {len(shared)} matched cells")

# ---------------------------------------------------------------------------
# RSA matrices (3x3) + split-half reliability on the diagonal
# ---------------------------------------------------------------------------

def rsa_matched(half_by_session):
    """Data-matched RSA: EVERY entry is estimated from 5-repeat half-networks,
    so cross-day similarity carries the same estimation noise as the split-half
    ceiling (no 10-vs-5 advantage that would flatter stability).

      diagonal[i]      = split-half reliability  sim(h1_i, h2_i)   (the ceiling)
      off-diagonal[i,j]= mean cross-day similarity over the 4 half-pairings
                         {h1_i,h2_i} x {h1_j,h2_j}
    """
    M = np.full((3, 3), np.nan)
    for i, si in enumerate(SESSIONS):
        h1i, h2i = half_by_session[si]
        M[i, i] = similarity(offdiag(h1i), offdiag(h2i), args.metric)[0]
        for j, sj in enumerate(SESSIONS):
            if j > i:
                h1j, h2j = half_by_session[sj]
                combos = [similarity(offdiag(a), offdiag(b), args.metric)[0]
                          for a in (h1i, h2i) for b in (h1j, h2j)]
                M[i, j] = M[j, i] = float(np.nanmean(combos))
    return M


def rsa_fulldata(corr_by_session):
    """Cross-day similarity using the FULL 10-repeat matrices (best weight
    estimate; shown on the scatters). Off-diagonal only; diagonal trivially 1."""
    M = np.eye(3)
    for i, si in enumerate(SESSIONS):
        for j, sj in enumerate(SESSIONS):
            if j > i:
                M[i, j] = M[j, i] = similarity(offdiag(corr_by_session[si]),
                                               offdiag(corr_by_session[sj]),
                                               args.metric)[0]
    return M

# matched (headline, comparable to the split-half ceiling)
rsa_sig = rsa_matched(sig_half)
rsa_noi = rsa_matched(noi_half)
# full-data cross-day (for the scatter annotations / transparency)
full_sig = rsa_fulldata(sig)
full_noi = rsa_fulldata(noi)
IDX = {s: k for k, s in enumerate(SESSIONS)}

# ---------------------------------------------------------------------------
# Report + CSV
# ---------------------------------------------------------------------------

rows = []
for label, M, F in [("signal", rsa_sig, full_sig), ("noise", rsa_noi, full_noi)]:
    print(f"\n  {label.upper()} RSA — DATA-MATCHED (off-diag = cross-day [5-rep, "
          f"mean of 4 pairings], diag = split-half reliability):")
    for i, si in enumerate(SESSIONS):
        print("    " + "  ".join(f"{M[i, j]:+.3f}" for j in range(3)) + f"   <- {si}")
    cross = [M[0, 1], M[1, 2], M[0, 2]]
    rel   = [M[0, 0], M[1, 1], M[2, 2]]
    full_cross = [F[0, 1], F[1, 2], F[0, 2]]
    ratio = np.nanmean(cross) / np.nanmean(rel) if np.nanmean(rel) else np.nan
    print(f"    mean cross-day (matched) = {np.nanmean(cross):+.3f}   "
          f"mean ceiling = {np.nanmean(rel):+.3f}   "
          f"cross/ceiling = {ratio:.2f}")
    print(f"    [reference] mean cross-day FULL-DATA (10-rep) = "
          f"{np.nanmean(full_cross):+.3f}  (inflated vs the 5-rep ceiling)")
    rows.append(dict(type=label,
                     AB=M[0, 1], BC=M[1, 2], AC=M[0, 2],
                     rel_A=M[0, 0], rel_B=M[1, 1], rel_C=M[2, 2],
                     mean_cross_day_matched=np.nanmean(cross),
                     mean_ceiling=np.nanmean(rel),
                     cross_over_ceiling=ratio,
                     AB_fulldata=F[0, 1], BC_fulldata=F[1, 2], AC_fulldata=F[0, 2],
                     mean_cross_day_fulldata=np.nanmean(full_cross)))
pd.DataFrame(rows).to_csv(ROOT / "cross_session_rsa.csv", index=False)
print(f"\nSaved {ROOT / 'cross_session_rsa.csv'}")

# ---------------------------------------------------------------------------
# Figure: rows = signal / noise; cols 0-2 = day-pair scatters; col 3 = RSA heatmap
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
# scatter substrate = 5-rep half (h1) so the figure is on the same data budget
# as the matched RSA heatmap (off-diagonals = mean over 4 half-pairings).
row_defs = [("signal", sig_half,  rsa_sig, "#1f77b4"),
            ("noise",  noi_half,  rsa_noi, "#d62728")]

for ri, (label, half_by_session, M, color) in enumerate(row_defs):
    # --- scatters of 5-rep h1 weights for each day-pair ---
    for ci, (a, b) in enumerate(PAIRS):
        ax = axes[ri, ci]
        ia, ib = IDX[a], IDX[b]
        h1_a, _ = half_by_session[a]
        h1_b, _ = half_by_session[b]
        vi, vj = offdiag(h1_a), offdiag(h1_b)
        good = ~np.isnan(vi) & ~np.isnan(vj)
        ax.scatter(vi[good], vj[good], s=6, alpha=0.25, color=color,
                   edgecolors="none")
        lim = [np.nanmin([vi[good].min(), vj[good].min()]),
               np.nanmax([vi[good].max(), vj[good].max()])]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6)      # identity line
        ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
        # title r = matched mean over 4 half-pairings (= the heatmap off-diagonal)
        ax.set_title(f"{label}: {a} vs {b}   "
                     f"{args.metric} r = {M[ia, ib]:+.2f}", fontsize=9)
        ax.set_xlabel(f"{label} weight — {a} h1 ({DAY[a]})", fontsize=8)
        ax.set_ylabel(f"{label} weight — {b} h1 ({DAY[b]})", fontsize=8)
        ax.grid(alpha=0.2)

    # --- RSA heatmap (data-matched; diag = ceiling) ---
    ax = axes[ri, 3]
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(SESSIONS); ax.set_yticklabels(SESSIONS)
    cross = np.nanmean([M[0, 1], M[1, 2], M[0, 2]])
    rel   = np.nanmean([M[0, 0], M[1, 1], M[2, 2]])
    ax.set_title(f"{label} RSA — data-matched (5-rep)\n"
                 f"diag = ceiling · cross/ceiling = {cross/rel:.2f}", fontsize=9)
    for i in range(3):
        for j in range(3):
            if not np.isnan(M[i, j]):
                edge = (i == j)
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        fontsize=9,
                        color="black" if abs(M[i, j]) < 0.6 else "white",
                        fontweight="bold" if edge else "normal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

fig.suptitle(
    f"Cross-day stability of correlation structure (second-order RSA) — "
    f"{len(shared)} matched cells, natural movie 1, bin {args.bin_ms:.0f} ms\n"
    f"all panels DATA-MATCHED (5-rep): scatters show h1 weights per session "
    f"(one point per cell pair); RSA heatmap diagonal = within-day "
    f"split-half reliability ceiling",
    fontsize=12, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_png = ROOT / "cross_session_rsa.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")
