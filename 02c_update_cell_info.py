"""
02c_update_cell_info.py
Regenerate cell_info.csv for all three sessions with the natural-movie-1
per-cell stats (reliability_nm1, peak_dff_nm1) instead of the natural-scenes
stats (pref_image_ns, image_sel_ns, reliability_ns).

This is a fast, targeted updater — no events matrix, no trial computation,
no NWB download. Just queries Allen's cell specimens table and writes
new cell_info.csv files in place.

Usage:
  python3 scripts/02c_update_cell_info.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

CONTAINER_ID = 661437138
SESSIONS     = {"A": 661437140, "B": 662351346, "C": 662358233}
CACHE_DIR    = Path.home() / "allen_cache" / "visual_coding"
BASE_OUTPUT  = Path("outputs") / "movie1"

KEEP_COLS = ["cell_specimen_id", "area", "imaging_depth",
             "rf_center_on_x_lsn", "rf_center_on_y_lsn",
             "reliability_nm1", "peak_dff_nm1"]

print("=" * 60)
print("  Regenerating cell_info.csv with natural-movie-1 metadata")
print("=" * 60)

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

boc = BrainObservatoryCache(manifest_file=str(CACHE_DIR / "manifest.json"))

# Container-level cell specimens (same 142 cells for all three sessions)
cell_specs = boc.get_cell_specimens(experiment_container_ids=[CONTAINER_ID])
cell_table = pd.DataFrame(cell_specs)
container_ids = list(cell_table["cell_specimen_id"])
print(f"\nContainer cells: {len(container_ids)}")

missing = [c for c in KEEP_COLS if c not in cell_table.columns]
if missing:
    print(f"WARNING: Allen cell_specimens table is missing: {missing}")
    print(f"         These columns will be omitted from cell_info.csv")
keep = [c for c in KEEP_COLS if c in cell_table.columns]
print(f"Columns to keep: {keep}")

# Build the base cell_info dataframe (same for all sessions)
cell_info_base = cell_table[keep].copy()
cell_info_base = (cell_info_base
                  .set_index("cell_specimen_id")
                  .reindex(container_ids)
                  .reset_index())
cell_info_base = cell_info_base.rename(columns={
    "rf_center_on_x_lsn": "x",
    "rf_center_on_y_lsn": "y",
})

# Per-session: just add the present_in_session_X flag using each session's cells
for label, exp_id in SESSIONS.items():
    session_dir = BASE_OUTPUT / label
    if not session_dir.exists():
        print(f"\n  Session {label}: skipping (directory {session_dir} not found)")
        continue

    dataset = boc.get_ophys_experiment_data(exp_id)
    session_ids = list(dataset.get_cell_specimen_ids())
    present_mask = np.array([cid in session_ids for cid in container_ids])

    ci = cell_info_base.copy()
    ci[f"present_in_session_{label.lower()}"] = present_mask
    out_path = session_dir / "cell_info.csv"
    ci.to_csv(out_path, index=False)
    print(f"\n  Session {label}: wrote {out_path}")
    print(f"    columns: {list(ci.columns)}")
    print(f"    {present_mask.sum()} / {len(container_ids)} cells present")
    non_null = ci.notna().sum()
    print(f"    non-null per column:")
    for c in ci.columns:
        if c == "cell_specimen_id": continue
        print(f"      {c}: {non_null[c]} / {len(ci)}")

print("\nDone.")
