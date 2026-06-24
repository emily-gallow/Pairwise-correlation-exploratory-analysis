"""
17_signal_noise_jaccard.py
==========================
Compare the TOPOLOGY of the signal- and noise-correlation networks, within
each session, by sweeping a density threshold and measuring how much their
binarised edge sets overlap (Jaccard similarity).

Motivation:
  Signal correlation (tuning similarity) and noise correlation (shared
  trial-to-trial variability) are two distinct pairwise interaction types.
  Do the cell pairs that are most strongly co-tuned (signal) coincide with
  the pairs that are most strongly functionally coupled (noise)? Rather than
  commit to one cut-off, we sweep network DENSITY: at each density p we keep
  the top-p% strongest edges of EACH matrix SEPARATELY (so both graphs have
  the same number of edges) and compute the Jaccard overlap of the two edge
  sets.

  Jaccard(A,B) = |A ∩ B| / |A ∪ B|.  With |A| = |B| = k this reduces to
  inter / (2k - inter).

  CHANCE BASELINE: if the two top-k edge sets were drawn independently from
  the P candidate pairs, the expected Jaccard is
      E[J] = d / (2 - d),     d = k / P  (the density.)
  This depends only on density, not on P, so it is one dashed curve shared by
  all sessions. Points ABOVE it mean signal and noise networks share structure
  beyond what equal edge counts force; points ON it mean the two networks pick
  their strong edges essentially independently.

Edge ranking:
  --rank value (default): rank by raw correlation, descending -> "top p% =
     strongest positive connections" (the usual functional-connectivity
     convention; FC graphs are normally built on positive coupling).
  --rank abs: rank by |r| -> also pulls in strong anti-correlations.

Inputs (per session S, from script 15):
  outputs/movie1/{S}/signal_corr.npy   (n_present x n_present)
  outputs/movie1/{S}/noise_corr.npy    (n_present x n_present)
  (signal and noise share the same cell set / ordering within a session)

Outputs:
  outputs/movie1/signal_noise_jaccard.csv      density-vs-Jaccard per session
  outputs/movie1/signal_noise_jaccard.png      Jaccard-vs-density curves (A/B/C)
  outputs/movie1/signal_noise_adjacency.png    binarised adjacency at one density

DATA SUBSTRATE (honesty):
  The figures auto-detect whether the underlying matrices were built from true
  L0 events or (provisionally) from dF/F, by inspecting events_matrix.npy. dF/F
  at frame rate carries the GCaMP6f decay-kernel autocorrelation, so any
  dF/F-based result here is PROVISIONAL and should be re-run once the
  events_matrix is regenerated from Allen L0 events (loader fixed in script 02).

Usage:
  python3 scripts/17_signal_noise_jaccard.py
  python3 scripts/17_signal_noise_jaccard.py --rank abs --adj-density 10
  python3 scripts/17_signal_noise_jaccard.py --min-pct 1 --max-pct 50 --step 1
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------

SESSIONS = ["A", "B", "C"]
DAY      = {"A": "day 95", "B": "day 96", "C": "day 97"}
COL      = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1",
                    help="Root holding {A,B,C}/ session folders.")
parser.add_argument("--min-pct", type=float, default=1.0,
                    help="Smallest network density to test (top %% of edges).")
parser.add_argument("--max-pct", type=float, default=50.0,
                    help="Largest network density to test (top %% of edges).")
parser.add_argument("--step", type=float, default=1.0,
                    help="Density step in percent.")
parser.add_argument("--rank", choices=["value", "abs"], default="value",
                    help="Rank edges by raw correlation (value) or magnitude (abs).")
parser.add_argument("--adj-density", type=float, nargs="+", default=[10.0, 50.0],
                    help="Densities (top %% kept) at which to draw binarised "
                         "adjacency comparison figures. Accepts multiple values: "
                         "e.g. --adj-density 10 50 renders one figure per density.")
args = parser.parse_args()

ROOT = Path(args.root)
densities = np.arange(args.min_pct, args.max_pct + 1e-9, args.step) / 100.0

print("=" * 70)
print("  Signal vs noise network overlap (Jaccard) across density thresholds")
print(f"  root={ROOT}  density {args.min_pct:g}–{args.max_pct:g}% step {args.step:g}%"
      f"  rank={args.rank}")
print("=" * 70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def upper_vec(C):
    """Upper-triangular off-diagonal vector + its index tuple + n."""
    n = C.shape[0]
    iu = np.triu_indices(n, k=1)
    return C[iu], iu, n


def topk_mask(vec, k, rank):
    """Boolean mask of the top-k entries of vec (descending), by value or |value|."""
    key = np.abs(vec) if rank == "abs" else vec
    order = np.argsort(-key)
    m = np.zeros(vec.shape[0], dtype=bool)
    m[order[:k]] = True
    return m


def eig_order(C):
    """Order cells by leading eigenvector of |C| (NaN->0); groups co-active cells."""
    M = np.nan_to_num(C, nan=0.0).copy()
    np.fill_diagonal(M, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, V = np.linalg.eigh(M)
    return np.argsort(V[:, np.argmax(w)])


def vec_to_mat(mask_full, iu, n):
    """Symmetric binary adjacency from an upper-tri boolean vector."""
    M = np.zeros((n, n), dtype=int)
    M[iu] = mask_full.astype(int)
    return M + M.T


def detect_substrate(root):
    """Inspect events_matrix.npy to label the data substrate honestly."""
    try:
        ev = np.load(Path(root) / "A" / "events_matrix.npy")
        flat = ev[~np.isnan(ev)]
        neg = float(np.mean(flat < 0))
        if neg <= 0.01:
            return "L0 events", neg
        return "dF/F  [PROVISIONAL — regenerate from L0 events]", neg
    except Exception:
        return "unknown substrate", float("nan")


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

substrate, neg_frac = detect_substrate(ROOT)
print(f"\n  Data substrate: {substrate}"
      + (f"  ({100*neg_frac:.1f}% negative values)" if neg_frac == neg_frac else ""))

rows = []
per_session = {}

for s in SESSIONS:
    sig = np.load(ROOT / s / "signal_corr.npy")
    noi = np.load(ROOT / s / "noise_corr.npy")

    sv, iu, n = upper_vec(sig)
    nv, _, _  = upper_vec(noi)
    valid = ~np.isnan(sv) & ~np.isnan(nv)
    svv, nvv = sv[valid], nv[valid]
    P = int(valid.sum())
    valid_pos = np.where(valid)[0]

    jac = np.full(len(densities), np.nan)
    jch = np.full(len(densities), np.nan)
    for i, d in enumerate(densities):
        k = max(1, int(round(d * P)))
        ma = topk_mask(svv, k, args.rank)
        mb = topk_mask(nvv, k, args.rank)
        inter = int((ma & mb).sum())
        union = int((ma | mb).sum())
        jac[i] = inter / union if union else np.nan
        dd = k / P
        jch[i] = dd / (2 - dd)
        rows.append(dict(session=s, density_pct=round(d * 100, 4), n_edges=k,
                         n_pairs=P, jaccard=jac[i], jaccard_chance=jch[i],
                         ratio_obs_over_chance=(jac[i] / jch[i] if jch[i] > 0 else np.nan)))

    per_session[s] = dict(jac=jac, jch=jch, P=P, n=n, sig=sig, noi=noi,
                          sv=sv, nv=nv, valid=valid, valid_pos=valid_pos, iu=iu)

    # report a few representative densities
    def at(pct):
        i = int(np.argmin(np.abs(densities * 100 - pct)))
        return jac[i], jac[i] / jch[i] if jch[i] > 0 else np.nan
    j5, r5   = at(5);  j10, r10 = at(10);  j20, r20 = at(20)
    print(f"\n  Session {s} ({DAY[s]}): {P} valid pairs ({n} cells)")
    print(f"    Jaccard @ top  5%: {j5:.3f}  ({r5:.2f}× chance)")
    print(f"    Jaccard @ top 10%: {j10:.3f}  ({r10:.2f}× chance)")
    print(f"    Jaccard @ top 20%: {j20:.3f}  ({r20:.2f}× chance)")

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

df = pd.DataFrame(rows)
out_csv = ROOT / "signal_noise_jaccard.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")

# ---------------------------------------------------------------------------
# Figure 1 — Jaccard vs density curves
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.5, 6.0))
# Convention: Threshold % = % of weakest |r| pairs zeroed (higher = stricter).
# Internally we still compute "top X% kept"; this is just a relabel + sort.
threshold_pct = (1.0 - densities) * 100.0
order = np.argsort(threshold_pct)
thr = threshold_pct[order]
chance_sorted = (densities / (2 - densities))[order]
for s in SESSIONS:
    ps = per_session[s]
    ax.plot(thr, ps["jac"][order], color=COL[s], lw=2.5, marker="o", ms=3.5,
            label=f"Session {s} ({DAY[s]})")
ax.plot(thr, chance_sorted, color="grey", ls=":", lw=1.2, label="chance")
ax.set_xlabel("Threshold %  (% of weakest |r| pairs zeroed)", fontsize=13)
ax.set_ylabel("Similarity (Jaccard)", fontsize=13)
_rank_lbl = "|r|" if args.rank == "abs" else "signed r"
ax.set_title(f"Signal vs noise network overlap — within session   "
             f"(rank by {_rank_lbl}, {substrate})", fontsize=10.5)
ax.tick_params(labelsize=10)
ax.grid(alpha=0.25)
ax.legend(fontsize=10.5, loc="upper right", frameon=True)
ax.set_xlim(thr.min(), thr.max())
ax.set_ylim(0, None)
plt.tight_layout()
out_png = ROOT / "signal_noise_jaccard.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")

# ---------------------------------------------------------------------------
# Figure 1b — RATIO over chance (curves rise bottom-left to top-right)
# ---------------------------------------------------------------------------
# Same x-axis (Threshold %), but y = Jaccard / chance. At permissive thresholds
# both are similar so ratio ~ 1; as the threshold tightens, chance falls faster
# than the observed Jaccard, so the ratio grows — showing how *meaningful* the
# overlap is, not just how many edges survive.
fig, ax = plt.subplots(figsize=(8.5, 6.0))
for s in SESSIONS:
    ps = per_session[s]
    ratio = ps["jac"][order] / chance_sorted
    ax.plot(thr, ratio, color=COL[s], lw=2.5, marker="o", ms=3.5,
            label=f"Session {s} ({DAY[s]})")
ax.axhline(1.0, color="grey", ls=":", lw=1.2, label="chance (ratio = 1)")
ax.set_xlabel("Threshold %  (% of weakest |r| pairs zeroed)", fontsize=13)
ax.set_ylabel("Similarity ratio  (Jaccard / chance)", fontsize=13)
ax.set_title(f"Signal vs noise network overlap — ratio over chance\n"
             f"(within session, rank by {_rank_lbl}, {substrate})",
             fontsize=10.5)
ax.tick_params(labelsize=10)
ax.grid(alpha=0.25)
ax.legend(fontsize=10.5, loc="upper left", frameon=True)
ax.set_xlim(thr.min(), thr.max())
ax.set_ylim(0.5, None)
plt.tight_layout()
out_png_ratio = ROOT / "signal_noise_jaccard_ratio.png"
fig.savefig(out_png_ratio, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png_ratio}")

# ---------------------------------------------------------------------------
# Figure 2 — binarised adjacency comparison at one or more densities
# ---------------------------------------------------------------------------
# Loops over args.adj_density (now a list) so we can render the strict view
# (top 10% kept = threshold 90%) AND a permissive view (top 50% = threshold 50%)
# side by side as separate files.

ov_cmap = ListedColormap(["#f5f5f5", "#1f77b4", "#ff7f0e", "#7b3fa0"])
ov_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)

for adj_density_pct in args.adj_density:
    adj_d = adj_density_pct / 100.0
    threshold_pct = (1.0 - adj_d) * 100.0    # "% of weakest |r| zeroed"

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13))
    for i, s in enumerate(SESSIONS):
        ps = per_session[s]
        n, iu, P = ps["n"], ps["iu"], ps["P"]
        k = max(1, int(round(adj_d * P)))

        ma_v = topk_mask(ps["sv"][ps["valid"]], k, args.rank)
        mb_v = topk_mask(ps["nv"][ps["valid"]], k, args.rank)
        ma_full = np.zeros(ps["sv"].shape[0], dtype=bool); ma_full[ps["valid_pos"]] = ma_v
        mb_full = np.zeros(ps["nv"].shape[0], dtype=bool); mb_full[ps["valid_pos"]] = mb_v

        A = vec_to_mat(ma_full, iu, n)      # signal adjacency
        B = vec_to_mat(mb_full, iu, n)      # noise adjacency
        order = eig_order(ps["sig"])
        A, B = A[np.ix_(order, order)], B[np.ix_(order, order)]

        # overlap map: 0 none / 1 signal-only / 2 noise-only / 3 both
        OV = np.zeros_like(A)
        OV[(A == 1) & (B == 0)] = 1
        OV[(A == 0) & (B == 1)] = 2
        OV[(A == 1) & (B == 1)] = 3

        shared = int(((A == 1) & (B == 1)).sum() // 2)
        jac_here = shared / (2 * k - shared) if (2 * k - shared) else np.nan

        axes[i, 0].imshow(A, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 0].set_title(f"Session {s} — SIGNAL adjacency\n"
                             f"(threshold {threshold_pct:g}%, top {adj_density_pct:g}%, "
                             f"{k} edges)", fontsize=9)
        axes[i, 1].imshow(B, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 1].set_title(f"Session {s} — NOISE adjacency\n"
                             f"(threshold {threshold_pct:g}%, top {adj_density_pct:g}%, "
                             f"{k} edges)", fontsize=9)
        axes[i, 2].imshow(OV, cmap=ov_cmap, norm=ov_norm, interpolation="nearest")
        axes[i, 2].set_title(f"Session {s} — overlap\nJaccard = {jac_here:.3f}", fontsize=9)
        for j in range(3):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    legend = [Patch(facecolor="#1f77b4", label="signal only"),
              Patch(facecolor="#ff7f0e", label="noise only"),
              Patch(facecolor="#7b3fa0", label="both (shared edge)"),
              Patch(facecolor="#f5f5f5", edgecolor="grey", label="neither")]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Binarised signal vs noise adjacency — threshold "
                 f"{threshold_pct:g}% (top {adj_density_pct:g}% strongest |r| kept)\n"
                 f"cells eig-sorted · substrate: {substrate}",
                 fontsize=11, y=0.998)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    # Filename: keep signal_noise_adjacency.png for density 10% (back-compat);
    # use a density-tagged suffix for any others.
    if abs(adj_density_pct - 10.0) < 1e-6:
        out_png2 = ROOT / "signal_noise_adjacency.png"
    else:
        out_png2 = ROOT / f"signal_noise_adjacency_d{int(round(adj_density_pct))}.png"
    fig.savefig(out_png2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png2}")

print("\nInterpretation: read each session's curve against the dashed chance "
      "line. A curve sitting near chance means the strongest signal edges and "
      "the strongest noise edges are largely different cell pairs; a curve "
      "well above chance means co-tuned pairs and co-fluctuating pairs are the "
      "same pairs (a shared 'like-to-like' backbone).")
