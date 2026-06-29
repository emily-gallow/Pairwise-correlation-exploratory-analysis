"""
28_timescale_sweep.py

Timescale sensitivity sweep on the cross-day RSA — does the noise null
survive at intermediate bin widths where noise correlations are typically
most cleanly measurable (Bair, Zohary & Newsome 2001; Cohen & Kohn 2011)?

For each bin width in {33, 100, 250, 500} ms (= {1, 3, 8, 15} frames at 30 Hz):
  per container, per session, recompute the matched-cell signal & noise
  correlation matrices using the standard recipe (script 15 idiom);
  per container, compute cross-day RSA r for (A-B, B-C, A-C) × (signal, noise);
  per container, build the per-animal 1-day value (mean of A-B, B-C) and
  2-day value (A-C);
  run a paired sign-flip permutation test (1-day vs 2-day) across animals.

Headline output:
  bin width on the x-axis, per-animal paired drop (1d − 2d) ± SEM and
  permutation p-value annotated, for signal and noise side-by-side.

This is the right experiment to answer "is the noise null an SNR floor at
frame rate, or a real null?" — if the null persists at 250 ms (where noise
correlations are largest and most reliably estimated), the null is real;
if it inverts, the frame-rate result was SNR-limited.

The sweep uses the FULLDATA 10-rep cross-day RSA (no half-splitting),
mirroring the headline recommendation from results_summary.md.

Inputs (per container):
  {root}/{cid}/{A,B,C}/events_matrix.npy
  {root}/{cid}/{A,B,C}/movie_repeat_frames.npy

Outputs:
  {root}/timescale_sweep_rsa.csv      one row per (container, pair, type, bin_ms)
  {root}/timescale_sweep_stats.csv    paired-test result per (bin_ms, type)
  {root}/timescale_sweep.png          headline figure

Usage:
  python3 scripts/28_timescale_sweep.py
  python3 scripts/28_timescale_sweep.py --bin-ms 33 100 250 500 1000
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
PAIRS = [("A", "B", 1), ("B", "C", 1), ("A", "C", 2)]

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", default="outputs/movie1")
parser.add_argument("--bin-ms", type=int, nargs="+",
                    default=[33, 100, 250, 500],
                    help="Bin widths (ms) to sweep. Default 33/100/250/500.")
parser.add_argument("--perm", type=int, default=10000)
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

ROOT = Path(args.root)
rng = np.random.default_rng(args.seed)

# Core math (mirrors script 15 — kept inline so this script is self-contained)

def present_mask(events): return ~np.all(np.isnan(events), axis=1)


def bin_epoch(epoch, bf):
    L, n = epoch.shape
    nb = L // bf
    return epoch[:nb*bf].reshape(nb, bf, n).sum(axis=1)


def build_tensor(ev_p, frames, bf):
    L_common = int(min(int(ef) - int(sf) + 1 for sf, ef in frames))
    nb = L_common // bf
    reps = []
    for sf, _ in frames:
        sf = int(sf)
        reps.append(bin_epoch(ev_p[sf: sf + L_common, :], bf))
    return np.stack(reps, axis=0), nb


def session_corrs(events, frames, bf, cell_idx):
    """Returns dict with signal & noise corr matrices on the given cell set."""
    ev_p = events[cell_idx, :].T                              # (T, n_cells)
    T, nb = build_tensor(ev_p, frames, bf)                    # (R, nb, n_cells)
    R, _, n_p = T.shape
    psth = T.mean(axis=0)                                     # (nb, n_p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S = np.corrcoef(psth, rowvar=False)
    dead_s = psth.var(axis=0) <= EPS
    S[dead_s, :] = np.nan; S[:, dead_s] = np.nan
    # Noise
    resid = T - psth[None, :, :]
    sd_rb = T.std(axis=0)
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


def upper_tri_corr(A, B):
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    a, b = A[iu], B[iu]
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[good], b[good])[0, 1])


# Sweep

cid_dirs = [p for p in sorted(ROOT.iterdir())
            if p.is_dir() and p.name.isdigit()
            and (p / "A" / "events_matrix.npy").exists()
            and (p / "B" / "events_matrix.npy").exists()
            and (p / "C" / "events_matrix.npy").exists()]
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
    for bin_ms in args.bin_ms:
        bf = max(1, int(round(bin_ms / 1000.0 * FPS)))
        mats = {s: session_corrs(ev[s], fr[s], bf, matched) for s in SESSIONS}
        for a, b, gap in PAIRS:
            for tp in ("signal", "noise"):
                r = upper_tri_corr(mats[a][tp], mats[b][tp])
                rows.append(dict(container_id=cid, bin_ms=bin_ms, bin_frames=bf,
                                 pair=f"{a}-{b}", days_apart=gap, type=tp, r=r,
                                 n_matched_cells=int(len(matched))))
    print(f"  {cid}: done ({len(matched)} matched cells)")

raw = pd.DataFrame(rows)
raw.to_csv(ROOT / "timescale_sweep_rsa.csv", index=False)
print(f"\nSaved {ROOT / 'timescale_sweep_rsa.csv'}  ({len(raw)} rows)")

# Per-animal 1-day vs 2-day, paired tests per (bin_ms, type)

def paired_perm(d, B=10000, rng=None):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < 2: return float(np.mean(d)) if len(d) else np.nan, np.nan
    obs = float(np.mean(d))
    s = rng.choice([-1, 1], size=(B, len(d))).astype(float)
    null = (s * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (B + 1)
    return obs, p


def boot_ci(d, B=10000, rng=None):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < 2: return np.nan, np.nan
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


stat_rows = []
for bin_ms in args.bin_ms:
    for tp in ("signal", "noise"):
        sub = raw[(raw["bin_ms"] == bin_ms) & (raw["type"] == tp)]
        # Per-animal 1d (mean A-B and B-C) vs 2d (A-C)
        pa = []
        for cid, g in sub.groupby("container_id"):
            v1 = g[g["days_apart"] == 1]["r"].astype(float)
            v2 = g[g["days_apart"] == 2]["r"].astype(float)
            if v1.empty or v2.empty: continue
            pa.append(dict(cid=cid, one=float(v1.mean()),
                           two=float(v2.iloc[0])))
        pa_df = pd.DataFrame(pa)
        if pa_df.empty: continue
        d = (pa_df["one"] - pa_df["two"]).values
        mean_d, p = paired_perm(d, B=args.perm, rng=rng)
        lo, hi = boot_ci(d, B=args.boot, rng=rng)
        n = len(d)
        sem = float(np.std(d, ddof=1)/np.sqrt(n)) if n > 1 else np.nan
        stat_rows.append(dict(
            bin_ms=bin_ms, type=tp, n_animals=n,
            mean_1day=float(pa_df["one"].mean()),
            mean_2day=float(pa_df["two"].mean()),
            mean_diff=mean_d, sem_diff=sem,
            ci_lo=lo, ci_hi=hi,
            cohens_dz=float(np.mean(d)/np.std(d, ddof=1)) if np.std(d, ddof=1) > 0 else np.nan,
            perm_p=p,
        ))

stats = pd.DataFrame(stat_rows)
stats.to_csv(ROOT / "timescale_sweep_stats.csv", index=False)
print("\nPaired-test results per bin width:")
print(stats.to_string(index=False))

# Figure

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
xs = np.array(args.bin_ms, dtype=float)

for ax, tp, color, title in [(axes[0], "signal", "#1f77b4", "Signal correlation"),
                              (axes[1], "noise",  "#d62728", "Noise correlation")]:
    sub = stats[stats["type"] == tp].sort_values("bin_ms")
    ax.errorbar(sub["bin_ms"], sub["mean_diff"], yerr=sub["sem_diff"],
                color=color, lw=2.5, marker="D", ms=10, mew=0,
                capsize=6, capthick=2)
    # CI band
    ax.fill_between(sub["bin_ms"], sub["ci_lo"], sub["ci_hi"],
                    color=color, alpha=0.15, label="95% bootstrap CI")
    ax.axhline(0, color="black", lw=0.8)
    # p-value annotation
    for _, r in sub.iterrows():
        sig = ""
        if r["perm_p"] < 0.001: sig = "***"
        elif r["perm_p"] < 0.01: sig = "**"
        elif r["perm_p"] < 0.05: sig = "*"
        ax.text(r["bin_ms"], r["mean_diff"] + (0.005 if tp=="signal" else 0.01),
                f"p={r['perm_p']:.3f}{sig}\nn={int(r['n_animals'])}",
                ha="center", fontsize=8.5, color="black")
    ax.set_xscale("log")
    ax.set_xticks(args.bin_ms)
    ax.set_xticklabels([str(b) for b in args.bin_ms])
    ax.set_xlabel("Bin width (ms, log scale)", fontsize=12)
    ax.set_ylabel("Paired drop (1-day − 2-day, Pearson r)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

fig.suptitle("Timescale sensitivity — cross-day RSA paired drop vs bin width\n"
             "(positive = 1-day more similar than 2-day; "
             "frame rate = 33 ms)",
             fontsize=12, y=1.03)
plt.tight_layout(rect=[0, 0.02, 1, 1])
out_png = ROOT / "timescale_sweep.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out_png}")
print(f"Saved {ROOT / 'timescale_sweep_stats.csv'}")
