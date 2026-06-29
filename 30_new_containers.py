"""
30_new_containers.py
After rebuilding the catalog with broader scope (script 01b without
--require-pupil and/or with broader --cre/--areas), this script diffs the
new catalog against the existing per-container output folders and prints:

  * the list of newly-available complete container IDs you don't yet have
  * the exact run_containers.py command to add them
  * a brief sanity check (cre line / area / depth / matched-cell count)

Usage:
  python3 scripts/30_new_containers.py
  python3 scripts/30_new_containers.py --catalog outputs/movie1/container_catalog.csv
"""

import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--catalog", default="outputs/movie1/container_catalog.csv")
parser.add_argument("--out-root", default="outputs/movie1",
                    help="Folder with per-container subfolders (existing cohort).")
parser.add_argument("--min-matched-cells", type=int, default=50,
                    help="Skip containers with fewer than this many catalogued "
                         "matched cells (default 50 — same threshold the existing "
                         "cohort comfortably exceeds).")
args = parser.parse_args()

OUT = Path(args.out_root)
cat = pd.read_csv(args.catalog)
comp = cat[cat["complete"].astype(bool)].copy()
print(f"Catalog has {len(comp)} complete containers in scope.")

# Current cohort: every existing numeric subfolder under outputs/movie1/
existing = {int(p.name) for p in OUT.iterdir()
            if p.is_dir() and p.name.isdigit()}
print(f"Existing cohort (subfolders under {OUT}): {len(existing)} containers.")

# New = complete in catalog AND not yet a folder AND meets min cell count
new = comp[~comp["container_id"].astype(int).isin(existing)].copy()
new = new[new["n_matched_cells"] >= args.min_matched_cells].copy()
new = new.sort_values("n_matched_cells", ascending=False)

print(f"\nNew complete containers to add: {len(new)}")
if len(new) == 0:
    print("Nothing to add. Either you haven't rebuilt the catalog with broader "
          "scope yet, or no new viable containers exist in that scope.")
    raise SystemExit(0)

print("\n--- Newly available containers ---")
print(new[["container_id","cre_line","area","imaging_depth",
           "reporter_line","n_matched_cells","has_pupil_tracking"]]
      .to_string(index=False))

ids = " ".join(str(int(x)) for x in new["container_id"])
print("\n--- Command to add them ---")
print(f"python3 scripts/run_containers.py --containers {ids}")
print(f"\n(Each container takes ~30–45 min for downloads + pipeline.)")
