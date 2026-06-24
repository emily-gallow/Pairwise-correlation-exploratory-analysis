"""
01b_catalog_containers.py
=========================
Multi-container discovery for the cross-session natural-movie-1
drift / functional-connectivity study.

FIRST step of scaling the pipeline beyond the single original
container (661437138). It catalogs EVERY Visual Coding 2P experiment
container that has a complete three-session set (A, B, and C or C2), so the
cross-session analysis can be run across many animals.

  Natural Movie One is presented in all three session types by Allen design
  (three_session_A, three_session_B, three_session_C / three_session_C2),
  so it is the common stimulus across days for every complete container.
  Cells are pre-matched within a container via a stable cell_specimen_id,
  which gives the tracked-across-days population for free — exactly the
  setup used in the original container.

  Metadata only — no NWB trace downloads here. The full cell-specimen table
  is fetched once (one cached download) and grouped by container to get the
  authoritative matched-cell count per container. Use --no-cell-count to
  skip even that and produce the session catalog quickly.

Output:
  outputs/movie1/container_catalog.csv
    one row per COMPLETE container:
      container_id, cre_line, area, imaging_depth, reporter_line,
      session_A_id, session_B_id, session_C_id, session_C_type,
      n_matched_cells, is_original, complete

  The generalized loader (next step) consumes this CSV: pick the rows you
  want (by cre_line / area / depth / matched-cell count) and feed their
  container_ids to the batch runner.

Usage:
  python3 scripts/01b_catalog_containers.py
  python3 scripts/01b_catalog_containers.py --areas VISp
  python3 scripts/01b_catalog_containers.py --cre Slc17a7-IRES2-Cre
  python3 scripts/01b_catalog_containers.py --no-cell-count
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Configuration

CACHE_DIR   = Path.home() / "allen_cache" / "visual_coding"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_OUTPUT = Path("outputs") / "movie1"
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

ORIGINAL_CONTAINER = 661437138          # the container used so far (for flagging)

# Allen session-type strings. Natural movie 1 is present in all of these.
SESSION_A_TYPES = {"three_session_a"}
SESSION_B_TYPES = {"three_session_b"}
SESSION_C_TYPES = {"three_session_c", "three_session_c2"}

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--areas", nargs="*", default=None,
                    help="Restrict to these targeted structures (e.g. VISp VISl). "
                         "Default: all areas.")
parser.add_argument("--cre", nargs="*", default=None,
                    help="Restrict to these cre lines (e.g. Slc17a7-IRES2-Cre). "
                         "Default: all cre lines.")
parser.add_argument("--no-cell-count", action="store_true",
                    help="Skip the matched-cell count (faster; omits the big "
                         "cell-specimen table download). n_matched_cells will be -1.")
parser.add_argument("--require-pupil", action="store_true",
                    help="Only output containers where eye-tracking did NOT fail "
                         "on any of the three sessions (i.e. pupil/eye traces are "
                         "available). The column is always written; this flag "
                         "filters the final CSV.")
parser.add_argument("--min-matched-cells", type=int, default=0,
                    help="Only output containers with at least this many matched "
                         "cells in the cell-specimen table (default 0 = no filter).")
args = parser.parse_args()

print("=" * 68)
print("  Visual Coding 2P — complete-container catalog (natural movie 1)")
print("=" * 68)
print(f"\nCache: {CACHE_DIR}")
print("Connecting to BrainObservatoryCache...")

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

boc = BrainObservatoryCache(manifest_file=str(CACHE_DIR / "manifest.json"))
print("Connected.\n")

# 1. Containers (non-failed) + optional metadata filters

cont_kwargs = {}
if args.areas:
    cont_kwargs["targeted_structures"] = args.areas
if args.cre:
    cont_kwargs["cre_lines"] = args.cre

containers = boc.get_experiment_containers(**cont_kwargs)
cont_meta  = {c["id"]: c for c in containers}
valid_cids = set(cont_meta.keys())
print(f"Containers returned (non-failed): {len(valid_cids)}")
if args.areas:
    print(f"  area filter:  {args.areas}")
if args.cre:
    print(f"  cre filter:   {args.cre}")

# 2. Experiments → group by container, resolve A / B / C session ids

exps = boc.get_ophys_experiments()
by_cont      = defaultdict(dict)     # cid -> {session_type_lower: experiment_id}
eye_fail     = {}                    # experiment_id -> bool (True = eye tracking failed)
for e in exps:
    cid = e.get("experiment_container_id")
    if cid not in valid_cids:
        continue
    stype = str(e.get("session_type", "") or "").lower().strip()
    if stype:
        by_cont[cid][stype] = e["id"]
    # 'fail_eye_tracking' is the standard Allen metadata flag for whether eye
    # tracking (incl. pupil) was usable for this experiment. Some SDK builds
    # omit it; we treat missing as "unknown" (=> permissive).
    eye_fail[e["id"]] = bool(e.get("fail_eye_tracking", False))


def resolve_sessions(stype_map):
    """Return (A_id, B_id, C_id, C_type) from a {session_type: exp_id} map."""
    a = next((eid for st, eid in stype_map.items() if st in SESSION_A_TYPES), None)
    b = next((eid for st, eid in stype_map.items() if st in SESSION_B_TYPES), None)
    c, c_type = None, None
    for st, eid in stype_map.items():
        if st in SESSION_C_TYPES:
            c, c_type = eid, st
            break
    return a, b, c, c_type


def all_have_pupil(*eids):
    """Return True if eye tracking did NOT fail for every supplied experiment."""
    return all(eid is not None and not eye_fail.get(eid, False) for eid in eids)

# 3. Matched-cell count per container (one cached download, then group)

matched_per_cont = {}
if not args.no_cell_count:
    print("\nFetching cell-specimen table (one cached download)...")
    cells_df = pd.DataFrame(boc.get_cell_specimens())
    if "experiment_container_id" in cells_df.columns:
        matched_per_cont = (
            cells_df.groupby("experiment_container_id")["cell_specimen_id"]
            .nunique()
            .to_dict()
        )
        print(f"  Counted matched cells for {len(matched_per_cont)} containers.")
    else:
        print("  WARNING: 'experiment_container_id' not in cell-specimen table; "
              "matched counts unavailable.")
else:
    print("\nSkipping matched-cell count (--no-cell-count).")

# 4. Build the catalog

rows = []
for cid in sorted(valid_cids):
    a, b, c, c_type = resolve_sessions(by_cont.get(cid, {}))
    complete = all(x is not None for x in (a, b, c))
    meta = cont_meta[cid]
    rows.append(dict(
        container_id      = cid,
        cre_line          = meta.get("cre_line", "unknown"),
        area              = meta.get("targeted_structure", "unknown"),
        imaging_depth     = meta.get("imaging_depth", -1),
        reporter_line     = meta.get("reporter_line", ""),
        session_A_id      = a,
        session_B_id      = b,
        session_C_id      = c,
        session_C_type    = c_type,
        n_matched_cells   = int(matched_per_cont.get(cid, -1)),
        # Pupil tracking availability — True only if eye tracking did NOT fail
        # on all three sessions. Behavioural-state side analyses (running, pupil)
        # need this, so it's exposed as a column for downstream filtering.
        has_pupil_tracking = bool(complete and all_have_pupil(a, b, c)),
        is_original       = (cid == ORIGINAL_CONTAINER),
        complete          = complete,
    ))

df = pd.DataFrame(rows)
complete_df = df[df["complete"]].copy()

if args.require_pupil:
    complete_df = complete_df[complete_df["has_pupil_tracking"]].copy()
    print(f"\n  --require-pupil applied: {len(complete_df)} containers remaining.")
if args.min_matched_cells > 0:
    complete_df = complete_df[
        complete_df["n_matched_cells"] >= args.min_matched_cells
    ].copy()
    print(f"  --min-matched-cells {args.min_matched_cells} applied: "
          f"{len(complete_df)} containers remaining.")

complete_df = complete_df.sort_values(
    ["n_matched_cells", "container_id"], ascending=[False, True]
)

out_path = BASE_OUTPUT / "container_catalog.csv"
# Save the full table (filtered complete first), but keep incomplete rows for
# the record.
df_sorted = pd.concat([
    complete_df,
    df[~df["complete"]].sort_values("container_id"),
], ignore_index=True)
df_sorted.to_csv(out_path, index=False)

# 5. Console summary

print(f"\n{'=' * 68}")
print("  RESULTS")
print(f"{'=' * 68}")
print(f"  Containers examined:          {len(df)}")
print(f"  Complete (A + B + C/C2):      {len(complete_df)}")

if len(complete_df):
    print(f"\n  Complete containers by area × cre line:")
    breakdown = (complete_df
                 .groupby(["area", "cre_line"])
                 .size()
                 .reset_index(name="n")
                 .sort_values("n", ascending=False))
    for _, r in breakdown.iterrows():
        print(f"    {r['area']:<8} {r['cre_line']:<24} {int(r['n']):>4}")

    if not args.no_cell_count:
        mc = complete_df["n_matched_cells"]
        mc = mc[mc >= 0]
        if len(mc):
            print(f"\n  Matched-cell count across complete containers:")
            print(f"    min {mc.min()}  median {int(mc.median())}  "
                  f"max {mc.max()}  (n={len(mc)})")
            print(f"    containers with >=80 matched cells:  "
                  f"{int((mc >= 80).sum())}")
            print(f"    containers with >=120 matched cells: "
                  f"{int((mc >= 120).sum())}")

    # How many match the original container's profile
    orig = df[df["container_id"] == ORIGINAL_CONTAINER]
    if len(orig):
        o = orig.iloc[0]
        same_profile = complete_df[
            (complete_df["cre_line"] == o["cre_line"]) &
            (complete_df["area"] == o["area"])
        ]
        print(f"\n  Original container {ORIGINAL_CONTAINER}: "
              f"{o['cre_line']}, {o['area']}, {o['imaging_depth']}µm")
        print(f"  Complete containers matching its cre+area "
              f"({o['cre_line']}, {o['area']}): {len(same_profile)}")

print(f"\n  Saved catalog → {out_path}")
print(f"\n  Next: choose the inclusion set from this CSV (by cre_line / area /")
print(f"        depth / n_matched_cells), then run the generalized loader")
print(f"        over those container_ids.")
print()
