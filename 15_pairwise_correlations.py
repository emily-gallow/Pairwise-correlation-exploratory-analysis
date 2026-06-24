"""
15_pairwise_correlations.py
===========================
Per-session SIGNAL and NOISE correlation matrices for natural movie 1,
the first analysis of the pairwise-correlation / functional-connectivity
direction.

TEMPORAL RESOLUTION (why frame-rate, not wide bins):
  Natural movie 1 is a continuously changing stimulus, so the principled
  unit of analysis is a single movie frame: each ~33 ms frame (the 2P
  acquisition rate is ~30 Hz) is effectively its own stimulus condition.
  SIGNAL is then the reproducible frame-by-frame response (the PSTH) and
  NOISE is the trial-to-trial deviation AT THAT FRAME.

  Binning the movie into wide windows (e.g. 500 ms ~ 15 frames) merges many
  distinct stimulus moments into one "condition." That (a) blurs the fine
  temporal tuning that distinguishes cells in the SIGNAL term and (b) lets
  stimulus-driven, co-tuned structure leak into the residual that we call
  NOISE, and it inflates noise-correlation magnitude (rNC grows with the
  counting window — Bair/Zohary/Newsome 2001; Cohen & Kohn 2011). So the
  DEFAULT here is frame resolution (--bin-ms 33 ~ 1 frame).

  Caveat that frame resolution does NOT remove: L0-deconvolved events are
  sparse and we have only ~10 repeats, so the per-frame across-trial variance
  used to z-score residuals is often zero. We report that sparsity explicitly
  (frac_zero_var_frames) and provide a --sweep robustness check across
  timescales so the reported structure can be confirmed to be stable, not an
  artifact of one bin width.

Definitions:
  Build a response tensor r[rep, bin, cell] over the R (~10) repeats. With
  the default bin width each "bin" is one acquisition frame.

  SIGNAL correlation (rsc_signal):
    Correlation, across time-bins, of the two cells' TRIAL-AVERAGED
    responses (PSTHs). Captures tuning similarity — do the cells respond
    to the same moments of the movie?

  NOISE correlation (rsc_noise):
    Correlation, across all (rep, bin) samples, of the two cells'
    TRIAL-TO-TRIAL RESIDUALS (response minus that bin's PSTH), z-scored
    within each bin across repeats (Cohen-Kohn convention, so each bin
    contributes equally and stimulus drive is removed). Captures shared
    trial-to-trial variability — functional coupling not explained by the
    stimulus. This is the substrate whose cross-day stability the project
    asks about.

Cell-set policy (MATCHED across all three days — for figure-wide consistency):
  This script computes WITHIN-SESSION structure on the SAME cell set used by
  the cross-day plots (scripts 16, 18, 19) — the cells tracked across all
  three days (~52 for the original container). Keeping every plot on the
  same population means a "cell in row 17" of the within-session matrix is
  the same neuron as "cell in row 17" of the cross-day figures, and the
  methods section can say "all analyses are on the matched 52 cells."

  An earlier dual-set policy used each session's full per-session present
  set (86/85/102); the switch to the matched set was made for consistency,
  with no meaningful loss of statistical power (1326 pairs per session is
  plenty for within-session structure).

A cell is "present" in a session if its row in events_matrix is not
entirely NaN; "matched" means present in A, B, and C.

Inputs (per session S, from scripts 02/03):
  outputs/movie1/{S}/events_matrix.npy        (n_canonical x n_timepoints)
  outputs/movie1/{S}/movie_repeat_frames.npy  (n_repeats x 2)

Outputs (per session S):
  outputs/movie1/{S}/signal_corr.npy        (n_present x n_present)
  outputs/movie1/{S}/noise_corr.npy         (n_present x n_present)
  outputs/movie1/{S}/corr_cell_index.npy    indices into canonical ordering,
                                            row/col order of the matrices
Shared:
  outputs/movie1/pairwise_corr_summary.csv  one row per session
  outputs/movie1/pairwise_corr_matrices.png 3 sessions x (signal, noise, scatter)

Usage:
  python3 scripts/15_pairwise_correlations.py                # frame-rate (default)
  python3 scripts/15_pairwise_correlations.py --sweep        # + timescale robustness check
  python3 scripts/15_pairwise_correlations.py --bin-ms 100   # light smoothing
  python3 scripts/15_pairwise_correlations.py --root outputs/movie1/661437138
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
# Configuration / CLI
# ---------------------------------------------------------------------------

SESSIONS = ["A", "B", "C"]
FPS      = 30.0
DAY      = {"A": "day 95", "B": "day 96", "C": "day 97"}
EPS      = 1e-10

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1",
                    help="Root holding {A,B,C}/ session folders. For the "
                         "multi-container batch, point at a single "
                         "container's folder, e.g. outputs/movie1/<cid>.")
parser.add_argument("--bin-ms", type=float, default=33.0,
                    help="Bin width in milliseconds for the within-repeat "
                         "time axis (default 33 ms ~ 1 frame at 30 Hz = frame "
                         "resolution). Sets the temporal scale of the noise "
                         "correlations; wider bins merge distinct movie frames "
                         "into one stimulus condition (discouraged).")
parser.add_argument("--sweep", action="store_true",
                    help="Run a timescale robustness check across several bin "
                         "widths (33/100/250/500 ms) and print how mean noise "
                         "correlation and noise STRUCTURE change with timescale. "
                         "Does not overwrite the main frame-rate outputs.")
parser.add_argument("--clim", type=float, default=None,
                    help="Colour-scale limit for the SIGNAL correlation heatmaps. "
                         "Default: auto-set to the 99th percentile of |r| across "
                         "off-diagonal pairs pooled across all three sessions.")
parser.add_argument("--noise-clim", type=float, default=None,
                    help="Colour-scale limit for the NOISE correlation heatmaps. "
                         "Default: auto-set to the 99th percentile of |r| across "
                         "off-diagonal pairs pooled across all three sessions.")
parser.add_argument("--clim-pct", type=float, default=99.0,
                    help="Percentile used by the auto colour limit (default 99). "
                         "Tighter values (e.g. 95) saturate more pairs; looser "
                         "values (e.g. 99.9) show only the very strongest as red.")
args = parser.parse_args()

ROOT       = Path(args.root)
BIN_FRAMES = max(1, int(round(args.bin_ms / 1000.0 * FPS)))
print("=" * 68)
print("  Per-session signal & noise correlations — natural movie 1")
print(f"  root={ROOT}  bin={args.bin_ms:.0f} ms (~{BIN_FRAMES} frame"
      f"{'s' if BIN_FRAMES != 1 else ''})"
      f"{'  [FRAME RESOLUTION]' if BIN_FRAMES == 1 else ''}")
print("=" * 68)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def present_mask(events):
    """A cell is present this session if its row isn't entirely NaN."""
    return ~np.all(np.isnan(events), axis=1)


def bin_epoch(epoch, bin_frames):
    """(L, n_cells) -> (n_bins, n_cells) by summing events within each bin.
    bin_frames == 1 is a pass-through (native frame resolution)."""
    L, n_cells = epoch.shape
    n_bins = L // bin_frames
    trimmed = epoch[: n_bins * bin_frames]
    return trimmed.reshape(n_bins, bin_frames, n_cells).sum(axis=1)


def build_tensor(events_present, frames, bin_frames):
    """Return r[rep, bin, cell] over repeats, common-length & bin-trimmed."""
    L_common = int(min(int(ef) - int(sf) + 1 for sf, ef in frames))
    n_bins = L_common // bin_frames
    if n_bins < 2:
        raise ValueError(f"Only {n_bins} bins per repeat at this bin width; "
                         f"reduce --bin-ms.")
    reps = []
    for sf, _ in frames:
        sf = int(sf)
        epoch = events_present[sf: sf + L_common, :]      # (L_common, n_cells)
        reps.append(bin_epoch(epoch, bin_frames))         # (n_bins, n_cells)
    return np.stack(reps, axis=0), n_bins, L_common       # (R, n_bins, n_cells)


def corr_with_dead_mask(mat_samples):
    """Pearson corr across columns of (n_samples, n_cells); columns with
    ~zero variance -> NaN row/col. Returns (n_cells, n_cells), dead_mask."""
    var = mat_samples.var(axis=0)
    dead = var <= EPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        C = np.corrcoef(mat_samples, rowvar=False)
    C[dead, :] = np.nan
    C[:, dead] = np.nan
    return C, dead


def offdiag(C):
    """Upper-triangular off-diagonal values, NaNs dropped."""
    n = C.shape[0]
    iu = np.triu_indices(n, k=1)
    v = C[iu]
    return v[~np.isnan(v)]


def eig_order(C):
    """1-D ordering of cells by the leading eigenvector of |C| (NaN->0).
    Groups co-correlated cells for display."""
    M = np.nan_to_num(C, nan=0.0).copy()
    np.fill_diagonal(M, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, V = np.linalg.eigh(M)
    lead = V[:, np.argmax(w)]
    return np.argsort(lead)


def session_corrs(events, frames, bin_frames, cell_idx=None):
    """Core computation for one session at a given bin width.

    cell_idx (optional): explicit canonical-row indices to use. If None,
    falls back to this session's present set. Pass a shared (matched) index
    to get the same cells across all sessions, which is the policy used now.

    Returns a dict with the signal/noise matrices, the canonical cell
    indices, and sparsity diagnostics."""
    if cell_idx is None:
        pmask = present_mask(events)
        idx   = np.where(pmask)[0]
    else:
        idx = np.asarray(cell_idx)
    ev_p  = events[idx, :].T                                # (T, n_cells)
    n_p   = ev_p.shape[1]

    T, n_bins, L_common = build_tensor(ev_p, frames, bin_frames)   # (R,n_bins,n_p)
    R = T.shape[0]

    # --- Signal correlation: corr of PSTHs across bins ---------------------
    psth = T.mean(axis=0)                                   # (n_bins, n_present)
    signal_corr, _ = corr_with_dead_mask(psth)              # corr across bins

    # --- Noise correlation: corr of z-scored residuals across (rep,bin) ----
    resid = T - psth[None, :, :]                            # (R, n_bins, n_p)
    sd_rb = T.std(axis=0)                                   # (n_bins, n_p) across reps
    nz    = sd_rb > EPS
    sd_safe = np.where(nz, sd_rb, 1.0)
    z = resid / sd_safe[None, :, :]
    # zero out (bin,cell) entries with no across-rep variance
    z = z * nz[None, :, :]
    Zmat = z.reshape(R * n_bins, n_p)                       # (samples, n_present)
    noise_corr, dead_noise = corr_with_dead_mask(Zmat)

    # --- Sparsity diagnostics (honesty about frame-rate viability) ---------
    # Fraction of (bin,cell) entries that carry NO across-trial variability:
    # these contribute nothing to the noise estimate. High -> data-limited.
    frac_zero_var = float(np.mean(~nz))
    frac_dead     = float(np.mean(dead_noise))              # fully-silent cells
    mean_event    = float(np.nanmean(T))                    # mean events / bin / cell / rep

    return dict(signal=signal_corr, noise=noise_corr, idx=idx,
                n_p=n_p, R=R, n_bins=n_bins, L_common=L_common,
                frac_zero_var=frac_zero_var, frac_dead=frac_dead,
                mean_event=mean_event)


# ---------------------------------------------------------------------------
# Per-session computation (headline = frame resolution)
# ---------------------------------------------------------------------------

results   = {}     # S -> dict of arrays/stats
summary   = []
loaded    = {}     # S -> (events, frames) reused by the sweep

# Precompute the matched cell set (present in ALL three sessions). Every
# within-session matrix below is then computed on this same population, so
# row 17 in any plot refers to the same neuron everywhere.
present_per = {}
for s in SESSIONS:
    ev = np.load(ROOT / s / "events_matrix.npy")
    present_per[s] = present_mask(ev)
shared_idx = np.where(
    present_per["A"] & present_per["B"] & present_per["C"]
)[0]
print(f"\nMatched cells tracked across all 3 days: {len(shared_idx)}")
print("Within-session signal & noise matrices computed on this matched set "
      "(consistent with scripts 16, 18, 19).\n")

for s in SESSIONS:
    sd = ROOT / s
    events = np.load(sd / "events_matrix.npy")             # (n_canon, T)
    frames = np.load(sd / "movie_repeat_frames.npy")       # (R, 2)
    loaded[s] = (events, frames)

    res = session_corrs(events, frames, BIN_FRAMES, cell_idx=shared_idx)
    signal_corr, noise_corr, idx = res["signal"], res["noise"], res["idx"]
    n_p, R, n_bins = res["n_p"], res["R"], res["n_bins"]

    # --- Save matrices -----------------------------------------------------
    np.save(sd / "signal_corr.npy", signal_corr)
    np.save(sd / "noise_corr.npy", noise_corr)
    np.save(sd / "corr_cell_index.npy", idx)

    # --- Summary stats -----------------------------------------------------
    sig_od   = offdiag(signal_corr)
    noi_od   = offdiag(noise_corr)
    # signal-noise relationship across pairs (limited-range structure)
    n = signal_corr.shape[0]
    iu = np.triu_indices(n, k=1)
    sflat, nflat = signal_corr[iu], noise_corr[iu]
    good = ~np.isnan(sflat) & ~np.isnan(nflat)
    if good.sum() > 2:
        sn_r = float(np.corrcoef(sflat[good], nflat[good])[0, 1])
    else:
        sn_r = np.nan

    row = dict(
        session            = s,
        day                = DAY[s],
        n_present_cells    = int(n_p),
        n_repeats          = int(R),
        n_bins             = int(n_bins),
        bin_ms             = args.bin_ms,
        bin_frames         = int(BIN_FRAMES),
        repeat_len_frames  = int(res["L_common"]),
        n_valid_pairs      = int(good.sum()),
        mean_signal_corr   = float(np.mean(sig_od)) if len(sig_od) else np.nan,
        mean_noise_corr    = float(np.mean(noi_od)) if len(noi_od) else np.nan,
        median_noise_corr  = float(np.median(noi_od)) if len(noi_od) else np.nan,
        std_noise_corr     = float(np.std(noi_od)) if len(noi_od) else np.nan,
        frac_noise_pos     = float(np.mean(noi_od > 0)) if len(noi_od) else np.nan,
        signal_noise_pair_r= sn_r,
        # sparsity diagnostics
        mean_event_per_bin = res["mean_event"],
        frac_zero_var_bins = res["frac_zero_var"],
        frac_dead_cells    = res["frac_dead"],
    )
    summary.append(row)
    results[s] = dict(signal=signal_corr, noise=noise_corr,
                      sflat=sflat[good], nflat=nflat[good])

    print(f"\nSession {s} ({DAY[s]}): {n_p} matched cells, "
          f"{R} repeats x {n_bins} bins")
    print(f"  mean signal r = {row['mean_signal_corr']:+.3f}   "
          f"mean noise r = {row['mean_noise_corr']:+.3f} "
          f"(median {row['median_noise_corr']:+.3f}, "
          f"{100*row['frac_noise_pos']:.0f}% positive)")
    print(f"  signal-vs-noise pair correlation (limited-range) r = {sn_r:+.3f}")
    print(f"  [sparsity] mean events/bin/cell = {res['mean_event']:.4f}   "
          f"zero-variance (bin,cell) = {100*res['frac_zero_var']:.1f}%   "
          f"fully-silent cells = {100*res['frac_dead']:.1f}%")

# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

df = pd.DataFrame(summary)
out_csv = ROOT / "pairwise_corr_summary.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Timescale robustness sweep (does NOT overwrite main outputs)
# ---------------------------------------------------------------------------

if args.sweep:
    sweep_ms     = [33.0, 100.0, 250.0, 500.0]
    sweep_frames = sorted({max(1, int(round(m / 1000.0 * FPS))) for m in sweep_ms})
    print("\n" + "=" * 68)
    print("  TIMESCALE ROBUSTNESS SWEEP")
    print("  Q: does the noise-correlation STRUCTURE survive a change of bin")
    print("     width, or is it an artifact of one timescale?")
    print("  'struct r vs frame' = Pearson r of the pairwise noise-corr vector")
    print("     at that bin width against the frame-resolution vector")
    print("     (same cells, same pairs). Near 1.0 => structure preserved.")
    print("=" * 68)
    header = (f"  {'session':<9}{'bin(ms)':>8}{'frames':>7}{'mean noise r':>14}"
              f"{'zero-var %':>12}{'struct r vs frame':>20}")
    for s in SESSIONS:
        print()
        print(header)
        events, frames = loaded[s]
        # reference = finest resolution (first in sweep_frames)
        ref_vec = None
        iu = None
        for bf in sweep_frames:
            res = session_corrs(events, frames, bf)
            C = res["noise"]
            if iu is None:
                iu = np.triu_indices(C.shape[0], k=1)
            vec = C[iu]
            if ref_vec is None:
                ref_vec = vec
            both = ~np.isnan(vec) & ~np.isnan(ref_vec)
            struct_r = (float(np.corrcoef(vec[both], ref_vec[both])[0, 1])
                        if both.sum() > 2 else np.nan)
            mean_noise = float(np.nanmean(vec))
            bin_ms = bf / FPS * 1000.0
            tag = "  <- frame res" if bf == sweep_frames[0] else ""
            print(f"  {s:<9}{bin_ms:>8.0f}{bf:>7d}{mean_noise:>+14.3f}"
                  f"{100*res['frac_zero_var']:>11.1f}%{struct_r:>20.3f}{tag}")
    print("\n  Interpretation: mean noise r should RISE with bin width (longer")
    print("  counting windows accumulate shared variability); if 'struct r vs")
    print("  frame' stays high, the *pattern* of which pairs are coupled is")
    print("  robust to timescale even though the magnitude scales.")

# ---------------------------------------------------------------------------
# Figure: rows = sessions, cols = signal heatmap | noise heatmap | scatter
# ---------------------------------------------------------------------------

def auto_clim(matrices, percentile=99.0):
    """Data-driven colour limit: the given percentile of |off-diagonal r| values
    pooled across the supplied matrices. Excludes the diagonal (which is 1.0)
    and any NaN entries. Returned rounded to 2 decimals for cleaner labels."""
    pooled = []
    for C in matrices:
        n = C.shape[0]
        iu = np.triu_indices(n, k=1)
        vals = np.abs(C[iu])
        pooled.append(vals[~np.isnan(vals)])
    pooled = np.concatenate(pooled)
    return float(np.round(np.percentile(pooled, percentile), 2))


# Compute data-driven colour limits unless the user passed explicit values.
sig_mats = [results[s]["signal"] for s in SESSIONS]
noi_mats = [results[s]["noise"]  for s in SESSIONS]
clim       = args.clim       if args.clim       is not None else auto_clim(sig_mats, args.clim_pct)
noise_clim = args.noise_clim if args.noise_clim is not None else auto_clim(noi_mats, args.clim_pct)
print(f"\nColour limits ({args.clim_pct:g}-th percentile of |r|, "
      f"pooled across sessions): signal ±{clim:g}, noise ±{noise_clim:g}")

# Shared symmetric axis limit for the signal-vs-noise scatter panels — so
# x and y are on the same scale and the magnitude asymmetry between signal
# and noise is visually honest. Uses the global max |value| across both
# variables and all sessions, with 5% padding so the strongest points sit
# slightly inside the panel.
all_sig = np.concatenate([results[s]["sflat"] for s in SESSIONS])
all_noi = np.concatenate([results[s]["nflat"] for s in SESSIONS])
scatter_lim = float(np.nanmax(np.abs(np.concatenate([all_sig, all_noi])))) * 1.05
print(f"Scatter axis limit (shared signal/noise, max |value| + 5%): "
      f"±{scatter_lim:.3f}")

fig, axes = plt.subplots(3, 3, figsize=(15, 14))

for i, s in enumerate(SESSIONS):
    Csig = results[s]["signal"]
    Cnoi = results[s]["noise"]
    order = eig_order(Csig)                 # shared ordering within a session

    for j, (C, name, cl) in enumerate([(Csig, "signal", clim),
                                        (Cnoi, "noise", noise_clim)]):
        ax = axes[i, j]
        Cs = C[np.ix_(order, order)]
        im = ax.imshow(Cs, vmin=-cl, vmax=cl, cmap="RdBu_r",
                       interpolation="nearest", aspect="equal")
        ax.set_title(f"Session {s} ({DAY[s]}) — {name} correlation "
                     f"(±{cl:g})\n{C.shape[0]} matched cells (eig-sorted)",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

    # scatter: signal vs noise per pair — SHARED x/y scale so the magnitude
    # asymmetry between signal and noise correlations is visually honest.
    ax = axes[i, 2]
    ax.scatter(results[s]["sflat"], results[s]["nflat"], s=4, alpha=0.25,
               color="steelblue", edgecolors="none")
    r = df[df["session"] == s]["signal_noise_pair_r"].iloc[0]
    ax.set_title(f"Session {s} — noise vs signal corr\n(pair r = {r:+.3f})",
                 fontsize=9)
    ax.set_xlabel("signal correlation", fontsize=8)
    ax.set_ylabel("noise correlation", fontsize=8)
    ax.set_xlim(-scatter_lim, scatter_lim)
    ax.set_ylim(-scatter_lim, scatter_lim)
    ax.set_aspect("equal", adjustable="box")
    # y = x identity line — points below it mean signal > noise in magnitude
    ax.plot([-scatter_lim, scatter_lim], [-scatter_lim, scatter_lim],
            color="black", ls="--", lw=0.7, alpha=0.4)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.grid(alpha=0.2)

bin_lbl = (f"frame resolution, ~{args.bin_ms:.0f} ms"
           if BIN_FRAMES == 1 else f"bin {args.bin_ms:.0f} ms")
fig.suptitle(
    f"Within-session signal & noise correlation structure — natural movie 1 "
    f"({bin_lbl})\nmatched {len(shared_idx)} cells (tracked across all 3 days)",
    fontsize=11, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
out_png = ROOT / "pairwise_corr_matrices.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out_png}")
print(f"\nNext: script 16 — cross-session stability of the correlation "
      f"structure (matched cells: 52 for 3-way, 57/65/61 pairwise).")
