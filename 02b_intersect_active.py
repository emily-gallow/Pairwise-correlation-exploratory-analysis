"""
02b_intersect_active.py
=======================
After scripts 02 and 03 have run for all three sessions, compute the
intersection of active cells across A, B, and C. This is the canonical
subset of neurons used for cross-session analyses (RDM, assembly comparison,
Jaccard) — guaranteed to be present and active in every session.

Inputs:
  outputs/movie1/A/active_neuron_indices.npy
  outputs/movie1/B/active_neuron_indices.npy
  outputs/movie1/C/active_neuron_indices.npy

Outputs:
  outputs/movie1/active_neuron_indices_intersect.npy
      (n_intersect,) indices into the 142-cell canonical container ordering
  outputs/movie1/active_summary.csv
      Per-session active count + intersection size

Usage:
  python3 scripts/02b_intersect_active.py
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="outputs/movie1",
                    help="Folder containing the per-session subfolders A/B/C. "
                         "For multi-container batches point at "
                         "outputs/movie1/<container_id>.")
args = parser.parse_args()

BASE_OUTPUT = Path(args.root)
SESSIONS    = ["A", "B", "C"]

print("=" * 60)
print("  Cross-session active-cell intersection")
print("=" * 60)

active_sets = {}
for s in SESSIONS:
    path = BASE_OUTPUT / s / "active_neuron_indices.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts 02 + 03 for session {s} first."
        )
    idx = np.load(path)
    active_sets[s] = set(int(i) for i in idx)
    print(f"  Session {s}: {len(active_sets[s])} active cells")

intersect = sorted(set.intersection(*active_sets.values()))
union     = sorted(set.union       (*active_sets.values()))

print(f"\n  Intersection (active in all 3): {len(intersect)} cells")
print(f"  Union (active in any):           {len(union)} cells")

# Per-cell membership table for inspection
all_ids = sorted(set.union(*active_sets.values()))
table = pd.DataFrame({"cell_row": all_ids})
for s in SESSIONS:
    table[f"active_in_{s}"] = table["cell_row"].apply(lambda r: r in active_sets[s])
table["active_in_all"] = table[[f"active_in_{s}" for s in SESSIONS]].all(axis=1)

intersect_arr = np.array(intersect, dtype=np.int64)
np.save(BASE_OUTPUT / "active_neuron_indices_intersect.npy", intersect_arr)
table.to_csv(BASE_OUTPUT / "active_summary.csv", index=False)

print(f"\n  Saved:")
print(f"    {BASE_OUTPUT / 'active_neuron_indices_intersect.npy'}  shape ({len(intersect)},)")
print(f"    {BASE_OUTPUT / 'active_summary.csv'}")
print(f"\n  Use active_neuron_indices_intersect.npy as the canonical cell")
print(f"  subset for all cross-session analyses (raster comparison, RDM, etc.)")
