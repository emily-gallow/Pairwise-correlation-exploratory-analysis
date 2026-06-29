"""
run_containers.py
Batch runner: takes a list of experiment container IDs and runs the full
pipeline (02 -> 03 -> 02b -> 15 -> 16 -> 17 -> 18 -> 19 -> 20) for each one,
writing per-container outputs to outputs/movie1/<container_id>/.

This is the workhorse for scaling the analysis from a single animal to many.
Each container's downloads (events NWB files, ~hundreds of MB per session)
happen the first time it's seen; subsequent runs hit the AllenSDK cache.

Usage:
  # Pass container IDs directly:
  python3 scripts/run_containers.py --containers 661437138 712178916 ...

  # Or read them from the catalog CSV produced by 01b (top N rows):
  python3 scripts/run_containers.py --from-catalog outputs/movie1/container_catalog.csv --n 20

  # Only run a subset of the chain (e.g. skip 18-20 for now):
  python3 scripts/run_containers.py --containers 661437138 --steps 02 03 02b 15 16 17

  # Continue after a failure (skip containers whose final CSV exists):
  python3 scripts/run_containers.py --from-catalog ... --skip-existing

Notes:
  - The script intentionally invokes the per-step scripts as subprocesses so
    each one runs with its own Python state. Failures in one container don't
    crash the others — the runner prints a clear error and continues.
  - Each container writes everything to outputs/movie1/<container_id>/...
    The original single-container outputs at outputs/movie1/ are untouched
    (so the existing pipeline stays reproducible).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# All steps + the args each one needs. Steps with --root take it; others use
# the --container flag (02 only) and write to the container subfolder.
CHAIN = [
    # (step name,  script,                                 extra args)
    ("02",  "scripts/02_load_visual_coding_session.py",   ["--session", "all"]),
    ("03",  "scripts/03_preprocess_matrix.py",            ["--session", "all"]),
    ("02b", "scripts/02b_intersect_active.py",            []),
    ("15",  "scripts/15_pairwise_correlations.py",        []),
    ("16",  "scripts/16_cross_session_rsa.py",            []),
    ("17",  "scripts/17_signal_noise_jaccard.py",         ["--rank", "abs"]),
    ("18",  "scripts/18_cross_day_adjacency.py",          []),     # uses --container
    ("19",  "scripts/19_cross_day_jaccard_sweep.py",      []),
    ("20",  "scripts/20_day_gap_scatter.py",              []),
]

BASE_OUTPUT = Path("outputs") / "movie1"


def container_root(cid: int) -> Path:
    return BASE_OUTPUT / str(cid)


def run_step(cid: int, step_name: str, script: str, extra: list[str]) -> bool:
    """Run one step for one container. Returns True on success."""
    cmd = [sys.executable, script] + list(extra)
    # Steps after the loader take --root pointing at the container folder.
    if step_name == "02":
        cmd += ["--container", str(cid)]
    elif step_name == "18":
        cmd += ["--container", str(cid)]    # script 18 also uses container id
    else:
        cmd += ["--root", str(container_root(cid))]
    print(f"  [container {cid}] step {step_name}:  {' '.join(cmd[1:])}")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"    FAILED ({dt:.1f}s)")
        # Print last 20 lines of stderr so the failure is debuggable.
        for line in (res.stderr.splitlines() or ["(no stderr)"])[-20:]:
            print(f"      {line}")
        return False
    print(f"    ok ({dt:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--containers", type=int, nargs="+",
                     help="Explicit list of container IDs to run.")
    src.add_argument("--from-catalog", type=str,
                     help="Path to a container_catalog.csv (from script 01b). "
                          "Containers are taken from rows where complete=True.")
    parser.add_argument("--n", type=int, default=None,
                        help="When using --from-catalog, only take the top N "
                             "rows (catalog is sorted by matched-cell count "
                             "descending).")
    parser.add_argument("--steps", type=str, nargs="+", default=None,
                        help="Run only these named steps (e.g. 02 03 02b 15). "
                             "Default: run the whole chain.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip containers that already have a "
                             "day_gap_scatter.csv (assumed complete).")
    args = parser.parse_args()

    # Resolve container list.
    if args.containers:
        cids = list(args.containers)
    else:
        cat = pd.read_csv(args.from_catalog)
        cat = cat[cat["complete"].astype(bool)].copy()
        if args.n is not None:
            cat = cat.head(args.n)
        cids = [int(x) for x in cat["container_id"].tolist()]
    print(f"Running pipeline over {len(cids)} containers: {cids}")

    # Resolve step list.
    if args.steps is None:
        chain = CHAIN
    else:
        wanted = set(args.steps)
        chain = [c for c in CHAIN if c[0] in wanted]
        unknown = wanted - {c[0] for c in CHAIN}
        if unknown:
            print(f"WARNING: unknown step names ignored: {sorted(unknown)}")

    summary = []
    for cid in cids:
        print("=" * 70)
        print(f"  CONTAINER {cid}")
        print("=" * 70)
        if args.skip_existing and (container_root(cid) / "day_gap_scatter.csv").exists():
            print(f"  skipped (day_gap_scatter.csv exists)")
            summary.append((cid, "skipped"))
            continue
        all_ok = True
        for step_name, script, extra in chain:
            ok = run_step(cid, step_name, script, extra)
            if not ok:
                all_ok = False
                break
        summary.append((cid, "ok" if all_ok else "failed"))

    print("\n" + "=" * 70)
    print("  BATCH SUMMARY")
    print("=" * 70)
    for cid, status in summary:
        print(f"  {cid}: {status}")
    n_ok = sum(1 for _, s in summary if s == "ok")
    print(f"\n  {n_ok}/{len(summary)} containers completed successfully.")


if __name__ == "__main__":
    main()
