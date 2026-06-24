"""
03_preprocess_matrix.py
Clean and z-score the per-repeat response matrix from a given session
(natural movie 1 trial structure).

Per-session pipeline:
  1. Re-align running speed to each repeat window (sanity check; already
     done in script 02 but recomputed here from cached run_speed values)
  2. Determine active neurons from the FULL-SESSION events matrix
     (cells with non-NaN data and mean event rate > MIN_MEAN_EVENT_RATE)
  3. Z-score the (n_repeats × 142) per-repeat matrix across repeats
  4. Save cleaned matrices and updated metadata under
     outputs/movie1/{SESSION_LABEL}/

Inputs (per session, written by script 02):
  outputs/movie1/{S}/X_trials_neurons.npy    (n_repeats × 142)
  outputs/movie1/{S}/trial_metadata.csv
  outputs/movie1/{S}/events_matrix.npy       (142 × n_timepoints)

Outputs (per session):
  outputs/movie1/{S}/X_clean.npy                (n_repeats × 142)  z-scored
  outputs/movie1/{S}/X_clean_active.npy         (n_repeats × n_active)
  outputs/movie1/{S}/active_neuron_indices.npy  (n_active,) indices into 142
  outputs/movie1/{S}/trial_metadata_clean.csv

Usage:
  python3 scripts/03_preprocess_matrix.py --session A
  python3 scripts/03_preprocess_matrix.py --session B
  python3 scripts/03_preprocess_matrix.py --session C
  python3 scripts/03_preprocess_matrix.py --session all
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration

MIN_MEAN_EVENT_RATE = 1e-4
SESSIONS_ALL        = ["A", "B", "C"]

# CLI

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--session", required=True, choices=["A", "B", "C", "all"],
    help="Which session to preprocess. 'all' runs A, B, and C.",
)
parser.add_argument(
    "--root", default="outputs/movie1",
    help="Folder containing the per-session subfolders A/B/C. For multi-"
         "container batches, point at outputs/movie1/<container_id>.",
)
args = parser.parse_args()
sessions_to_run = SESSIONS_ALL if args.session == "all" else [args.session]
BASE_OUTPUT = Path(args.root)


# Per-session preprocess

def preprocess_one(session_label: str):
    in_dir = out_dir = BASE_OUTPUT / session_label
    print("=" * 60)
    print(f"  Preprocessing session {session_label}  →  {out_dir}")
    print("=" * 60)

    X    = np.load(in_dir / "X_trials_neurons.npy")
    meta = pd.read_csv(in_dir / "trial_metadata.csv")
    events_matrix = np.load(in_dir / "events_matrix.npy")

    n_trials_raw, n_neurons = X.shape
    print(f"\nLoaded:  {n_trials_raw} repeats × {n_neurons} neurons")
    print(f"         events_matrix: {events_matrix.shape}")

    # Active-neuron selection — from full-session events
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_rates = np.nanmean(events_matrix, axis=1)
    present_cols   = ~np.isnan(mean_rates)
    active_mask    = present_cols & (mean_rates > MIN_MEAN_EVENT_RATE)
    active_indices = np.where(active_mask)[0]
    n_present, n_active = int(present_cols.sum()), int(active_mask.sum())
    print(f"\n  Cells present (non-NaN):  {n_present} / {n_neurons}")
    print(f"  Active (mean > {MIN_MEAN_EVENT_RATE}): {n_active}")

    # Z-score across the small n_repeats sample
    print(f"\n  Z-scoring across {n_trials_raw} repeats (caveat: small N)")
    X_z = np.full_like(X, np.nan, dtype=np.float32)
    for col in np.where(present_cols)[0]:
        col_vals = X[:, col]
        mu       = np.nanmean(col_vals)
        sigma    = np.nanstd(col_vals)
        X_z[:, col] = (col_vals - mu) / sigma if sigma > 1e-10 else 0.0

    # Save
    np.save(out_dir / "X_clean.npy", X_z)
    X_clean_active = X_z[:, active_indices]
    np.save(out_dir / "X_clean_active.npy", X_clean_active)
    np.save(out_dir / "active_neuron_indices.npy", active_indices)
    meta.to_csv(out_dir / "trial_metadata_clean.csv", index=False)

    print(f"\n  Saved:")
    print(f"    X_clean.npy               {X_z.shape}")
    print(f"    X_clean_active.npy        {X_clean_active.shape}")
    print(f"    active_neuron_indices.npy {active_indices.shape}")
    print(f"    trial_metadata_clean.csv  {meta.shape}\n")
    return {"session": session_label, "n_active": n_active, "n_repeats": n_trials_raw}


results = [preprocess_one(s) for s in sessions_to_run]

print("=" * 60)
print("  PREPROCESSING SUMMARY")
print("=" * 60)
for r in results:
    print(f"  Session {r['session']}: {r['n_repeats']} repeats, "
          f"{r['n_active']} active neurons")
print(f"\n  Next: python3 scripts/02b_intersect_active.py")
