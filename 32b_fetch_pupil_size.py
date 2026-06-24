"""
32b_fetch_pupil_size.py
=======================

Fetch raw pupil-area timecourses from Allen eye-gaze files, resample them onto
the same event timestamps used by the existing session outputs, and save
pupil_size.npy under outputs/movie1/<container>/<session>/.

This is intended to recover pupil data for containers where script 02's
built-in NWB pupil getters failed, but eye-gaze mappings are available via
AllenSDK's get_ophys_pupil_data(..., suppress_pupil_data=False).

Usage:
  .venv/bin/python scripts/32b_fetch_pupil_size.py
  .venv/bin/python scripts/32b_fetch_pupil_size.py --container 680156909
  .venv/bin/python scripts/32b_fetch_pupil_size.py --overwrite
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SES = {
    "A": {"three_session_a"},
    "B": {"three_session_b"},
    "C": {"three_session_c", "three_session_c2"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / "allen_cache" / "visual_coding"),
        help="AllenSDK manifest directory.",
    )
    parser.add_argument(
        "--catalog",
        default="outputs/movie1/container_catalog.csv",
        help="Container catalog CSV from script 01b.",
    )
    parser.add_argument(
        "--out-root",
        default="outputs/movie1",
        help="Root output folder containing container session subfolders.",
    )
    parser.add_argument(
        "--container",
        type=int,
        nargs="*",
        help="Optional container id(s) to process. If omitted, all viable catalog containers are used.",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        choices=["A", "B", "C"],
        default=["A", "B", "C"],
        help="Session labels to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pupil_size.npy files.",
    )
    return parser.parse_args()


def _clean_series(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("expected 1D sequence")
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.full_like(x, np.nan)
    idx = np.arange(len(x))
    x = np.interp(idx, idx[good], x[good])
    return x


def _uniq_sorted(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) != len(y):
        raise ValueError("timestamps and values must be same length")
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    if len(t) == 0:
        return t, y
    mask = np.ones(len(t), dtype=bool)
    mask[1:] = t[1:] != t[:-1]
    if mask.all():
        return t, y
    # average duplicated timestamps
    uniq_t = []
    uniq_y = []
    i = 0
    while i < len(t):
        j = i + 1
        while j < len(t) and t[j] == t[i]:
            j += 1
        uniq_t.append(t[i])
        uniq_y.append(np.nanmean(y[i:j]))
        i = j
    return np.asarray(uniq_t), np.asarray(uniq_y)


def _resample_pupil(event_ts, pupil_ts, pupil_area):
    if len(pupil_ts) == 0 or len(pupil_area) == 0:
        return np.full(len(event_ts), np.nan, dtype=float)
    pupil_area = _clean_series(pupil_area)
    pupil_ts, pupil_area = _uniq_sorted(pupil_ts, pupil_area)
    if len(pupil_ts) == 0:
        return np.full(len(event_ts), np.nan, dtype=float)
    if not np.all(np.diff(pupil_ts) > 0):
        # fallback to monotonic unique timestamps
        pupil_ts, pupil_area = _uniq_sorted(pupil_ts, pupil_area)
    if len(pupil_ts) == 1:
        return np.full(len(event_ts), pupil_area[0], dtype=float)
    return np.interp(event_ts, pupil_ts, pupil_area)


def _resolve_session_ids(boc, container_id):
    exps = boc.get_ophys_experiments(experiment_container_ids=[container_id])
    session_ids = {}
    for exp in exps:
        st = str(exp.get("session_type", "") or "").lower().strip()
        for label, names in SES.items():
            if st in names:
                session_ids[label] = exp["id"]
                break
    return session_ids


def process_session(boc, container_id, session_label, out_root, overwrite):
    session_dir = Path(out_root) / str(container_id) / session_label
    if not session_dir.exists():
        return {
            "container_id": container_id,
            "session": session_label,
            "status": "missing_session_dir",
            "notes": "session folder not present",
        }
    events_path = session_dir / "events_matrix.npy"
    if not events_path.exists():
        return {
            "container_id": container_id,
            "session": session_label,
            "status": "missing_events_matrix",
            "notes": "events_matrix.npy not found",
        }
    pupil_path = session_dir / "pupil_size.npy"
    if pupil_path.exists() and not overwrite:
        return {
            "container_id": container_id,
            "session": session_label,
            "status": "skipped_exists",
            "notes": "pupil_size.npy already exists",
        }

    # Resolve experiment id and load pupil data.
    session_ids = _resolve_session_ids(boc, container_id)
    eid = session_ids.get(session_label)
    if eid is None:
        return {
            "container_id": container_id,
            "session": session_label,
            "status": "missing_experiment_id",
            "notes": "no experiment id for session label",
        }

    try:
        ds = boc.get_ophys_experiment_data(eid)
    except Exception as exc:
        return {
            "container_id": container_id,
            "session": session_label,
            "experiment_id": eid,
            "status": "failed_load_dataset",
            "notes": str(exc),
        }

    event_ts, _ = ds.get_dff_traces()
    events = np.load(events_path)
    T = events.shape[1]
    if len(event_ts) != T:
        old_len = len(event_ts)
        event_ts = event_ts[:T]
        if len(event_ts) != T:
            return {
                "container_id": container_id,
                "session": session_label,
                "experiment_id": eid,
                "status": "timestamp_length_mismatch",
                "notes": f"dff timestamps length {old_len} vs events {T}",
            }

    try:
        df = boc.get_ophys_pupil_data(ophys_experiment_id=eid, suppress_pupil_data=False)
    except Exception as exc:
        return {
            "container_id": container_id,
            "session": session_label,
            "experiment_id": eid,
            "status": "failed_get_ophys_pupil_data",
            "notes": str(exc),
        }

    if "raw_pupil_area" not in df.columns:
        return {
            "container_id": container_id,
            "session": session_label,
            "experiment_id": eid,
            "status": "missing_raw_pupil_area",
            "notes": f"columns={list(df.columns)}",
        }

    pupil_ts = np.asarray(df.index, dtype=float)
    pupil_area = np.asarray(df["raw_pupil_area"].values, dtype=float)
    if pupil_ts.ndim != 1 or pupil_area.ndim != 1:
        return {
            "container_id": container_id,
            "session": session_label,
            "experiment_id": eid,
            "status": "bad_pupil_shape",
            "notes": f"pupil_ts shape={pupil_ts.shape} pupil_area shape={pupil_area.shape}",
        }

    output_area = _resample_pupil(event_ts, pupil_ts, pupil_area)
    np.save(pupil_path, np.vstack([event_ts, output_area]))
    flag_path = session_dir / "pupil_unavailable.flag"
    if flag_path.exists():
        flag_path.unlink()

    return {
        "container_id": container_id,
        "session": session_label,
        "experiment_id": eid,
        "status": "saved",
        "notes": "pupil_size.npy written",
        "n_pupil_rows": len(pupil_area),
        "n_event_frames": T,
    }


def main():
    args = parse_args()
    out_root = Path(args.out_root)

    if not Path(args.catalog).exists():
        raise SystemExit(f"Catalog not found: {args.catalog}")
    catalog = pd.read_csv(args.catalog)
    if "complete" in catalog.columns:
        catalog = catalog[catalog["complete"].astype(bool)]
    if "has_pupil_tracking" in catalog.columns:
        catalog = catalog[catalog["has_pupil_tracking"].astype(bool)]
    container_ids = sorted(catalog["container_id"].astype(int).unique())

    if args.container:
        selected = [int(c) for c in args.container]
        missing = set(selected) - set(container_ids)
        if missing:
            print(f"Warning: requested containers not in viable catalog set: {sorted(missing)}")
        container_ids = [c for c in selected if c in container_ids]

    if len(container_ids) == 0:
        raise SystemExit("No containers selected for processing.")

    print(f"Using AllenSDK cache: {args.cache_dir}")
    print(f"Processing {len(container_ids)} container(s): {container_ids}")
    print(f"Writing resampled pupil_size.npy under {out_root}")

    try:
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    except ImportError as exc:
        raise SystemExit(f"AllenSDK import failed: {exc}")
    boc = BrainObservatoryCache(manifest_file=str(Path(args.cache_dir) / "manifest.json"))

    rows = []
    for container_id in container_ids:
        for session_label in args.sessions:
            row = process_session(boc, container_id, session_label, out_root, args.overwrite)
            rows.append(row)
            print(f"{container_id}/{session_label}: {row['status']}" + (f" ({row.get('notes')})" if row.get('notes') else ""))

    out_csv = out_root / "pupil_size_fetch_status.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved fetch status to {out_csv}")


if __name__ == "__main__":
    main()
