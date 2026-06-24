"""
02_load_visual_coding_session.py
================================
Load one Allen Brain Observatory Visual Coding session for container 661437138
and extract NATURAL MOVIE ONE responses, with per-session output organisation
so the cross-session pipeline (Sessions A, B, C) can consume them in parallel.

Container 661437138 sessions (all imaged on consecutive days):
  A  — 661437140  (three_session_A:  drifting gratings + natural_movie_one + natural_movie_three)
  B  — 662351346  (three_session_B:  static gratings + natural_scenes + natural_movie_one)
  C  — 662358233  (three_session_C2: locally sparse noise + natural_movie_one + natural_movie_two)

Natural movie 1 is the only stimulus present in all three sessions, which is
why the cross-session drift analysis pivots on it.

Trial structure used here:
  Each presentation of the ~30-second movie is one TRIAL ("repeat").
  Typically 10 repeats per session. Per-trial response is the mean event
  rate over the full ~30 s repeat window.

Per-session outputs (under outputs/movie1/{SESSION_LABEL}/):
  events_matrix.npy         (142 × n_timepoints) full session, canonical cell ordering
  trial_metadata.csv        per-repeat: trial_id, repeat_index, start_time,
                            end_time, duration_s, n_frames, mean_running_speed
  X_trials_neurons.npy      (n_repeats × 142)  per-repeat mean response
  movie_repeat_frames.npy   (n_repeats × 2)    [start_frame, end_frame] per repeat
  cell_info.csv             cell_specimen_id, present_in_session flag, x, y

Usage:
  python3 scripts/02_load_visual_coding_session.py --session A
  python3 scripts/02_load_visual_coding_session.py --session B
  python3 scripts/02_load_visual_coding_session.py --session C
  python3 scripts/02_load_visual_coding_session.py --session all

Note:
  First run for each session triggers NWB download (~0.5–2 GB).
  Subsequent runs use cache.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONTAINER = 661437138             # the original container; can override

CACHE_DIR     = Path.home() / "allen_cache" / "visual_coding"
BASE_OUTPUT   = Path("outputs") / "movie1"
STIMULUS_NAME = "natural_movie_one"

# Allen session-type strings. Natural movie 1 is present in all of these.
SESSION_A_TYPES = {"three_session_a"}
SESSION_B_TYPES = {"three_session_b"}
SESSION_C_TYPES = {"three_session_c", "three_session_c2"}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--session", required=True, choices=["A", "B", "C", "all"],
    help="Which session to load. 'all' runs A, B, and C sequentially.",
)
parser.add_argument(
    "--container", type=int, default=DEFAULT_CONTAINER,
    help="Experiment container ID. Output goes to "
         "outputs/movie1/<container_id>/{A,B,C}/ for multi-container runs.",
)
args = parser.parse_args()

CONTAINER_ID = args.container
sessions_to_run = ["A", "B", "C"] if args.session == "all" else [args.session]
# Container-specific output folder so each animal has its own namespace.
CONTAINER_OUTPUT = BASE_OUTPUT / str(CONTAINER_ID)
CONTAINER_OUTPUT.mkdir(parents=True, exist_ok=True)

# Resolve A/B/C session IDs from the container dynamically (replaces the
# previously-hardcoded SESSIONS dict). This is what makes the script work for
# any container, not just 661437138.
from allensdk.core.brain_observatory_cache import BrainObservatoryCache as _BOC
_resolver_boc = _BOC(manifest_file=str(CACHE_DIR / "manifest.json"))
_container_exps = _resolver_boc.get_ophys_experiments(
    experiment_container_ids=[CONTAINER_ID]
)
SESSIONS = {}
for _e in _container_exps:
    _stype = str(_e.get("session_type", "") or "").lower().strip()
    if _stype in SESSION_A_TYPES:
        SESSIONS["A"] = _e["id"]
    elif _stype in SESSION_B_TYPES:
        SESSIONS["B"] = _e["id"]
    elif _stype in SESSION_C_TYPES:
        SESSIONS["C"] = _e["id"]
missing = [k for k in ("A", "B", "C") if k not in SESSIONS]
if missing:
    raise RuntimeError(
        f"Container {CONTAINER_ID}: missing session(s) {missing}. "
        f"Found session types: "
        f"{[str(e.get('session_type')) for e in _container_exps]}"
    )
print(f"Container {CONTAINER_ID}: resolved sessions A={SESSIONS['A']}, "
      f"B={SESSIONS['B']}, C={SESSIONS['C']}")


# ---------------------------------------------------------------------------
# Per-session loader
# ---------------------------------------------------------------------------

def load_one_session(session_label: str):
    experiment_id = SESSIONS[session_label]
    out_dir       = CONTAINER_OUTPUT / session_label
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  Session {session_label}  (experiment {experiment_id})  →  {out_dir}")
    print("=" * 65)

    from allensdk.core.brain_observatory_cache import BrainObservatoryCache

    boc = BrainObservatoryCache(manifest_file=str(CACHE_DIR / "manifest.json"))
    print("\nFetching experiment data (downloads NWB if not cached)...")
    dataset = boc.get_ophys_experiment_data(experiment_id)
    print("  Done.\n")

    # -- 1. L0-deconvolved events matrix ------------------------------------
    # Allen publishes L0 events for the Visual Coding 2P dataset. Pull them
    # EXPLICITLY via the cache, rather than relying on dataset.get_l0_events()
    # which is absent in some SDK builds. The previous hasattr() fallback
    # silently substituted dF/F — a signed trace autocorrelated at ~0.95 per
    # frame by the GCaMP6f decay kernel, which is the wrong substrate for
    # frame-rate noise correlations (the kernel masquerades as fast coupling).
    print("--- Step 1: L0-deconvolved events matrix ---")
    # Timestamps and cell ordering come from the dF/F trace API; the events
    # array shares that exact time base and cell-specimen ordering.
    timestamps, _dff = dataset.get_dff_traces()
    events = boc.get_ophys_experiment_events(experiment_id)   # (n_cells, T)
    signal_name = "L0 events"
    if events.shape[1] != timestamps.shape[0]:
        n = min(events.shape[1], timestamps.shape[0])
        print(f"  Note: trimming events/timestamps to common length {n} "
              f"(events {events.shape[1]} vs timestamps {timestamps.shape[0]}).")
        events, timestamps = events[:, :n], timestamps[:n]
    print(f"  Signal ({signal_name}): {events.shape}  (neurons × timepoints)")
    print(f"  Frame rate: {1 / np.median(np.diff(timestamps)):.2f} Hz")
    print(f"  Duration:   {timestamps[-1] - timestamps[0]:.1f} s")

    # Sanity gate: genuine L0 events are NON-NEGATIVE and SPARSE. If the loaded
    # signal is mostly signed/dense it is dF/F, and the run must NOT proceed
    # silently (this is exactly the bug that produced the earlier dF/F matrices).
    neg_frac  = float(np.mean(events < 0))
    zero_frac = float(np.mean(events == 0))
    print(f"  Event sanity: {100 * neg_frac:.2f}% negative, "
          f"{100 * zero_frac:.1f}% exactly zero "
          f"(expect ~0% negative, high % zero for true L0 events)")
    if neg_frac > 0.01:
        raise ValueError(
            f"Loaded signal for experiment {experiment_id} has "
            f"{100 * neg_frac:.1f}% negative values — this is dF/F, NOT L0 "
            f"events. Check boc.get_ophys_experiment_events() / the events "
            f"download. Refusing to write a mislabelled events_matrix."
        )

    # -- 2. Natural movie 1 stimulus table → per-repeat ---------------------
    print(f"\n--- Step 2: {STIMULUS_NAME} stimulus table ---")
    stim_table_raw = dataset.get_stimulus_table(STIMULUS_NAME)
    print(f"  Raw stim table: {stim_table_raw.shape}")
    if "repeat" not in stim_table_raw.columns:
        raise ValueError(
            f"Expected 'repeat' column; got {list(stim_table_raw.columns)}"
        )
    per_repeat = (
        stim_table_raw
        .groupby("repeat")
        .agg(start_frame=("start", "min"), end_frame=("end", "max"))
        .reset_index()
        .sort_values("repeat")
        .reset_index(drop=True)
    )
    per_repeat["start_time"] = [
        float(timestamps[int(f)]) for f in per_repeat["start_frame"]
    ]
    per_repeat["end_time"] = [
        float(timestamps[min(int(f), len(timestamps) - 1)])
        for f in per_repeat["end_frame"]
    ]
    per_repeat["duration_s"] = per_repeat["end_time"] - per_repeat["start_time"]
    n_repeats = len(per_repeat)
    print(f"  Per-repeat: {n_repeats} repeats, mean duration "
          f"{per_repeat['duration_s'].mean():.2f}s")

    # -- 3. Cell alignment to canonical 142-cell ordering -------------------
    print("\n--- Step 3: Cell alignment ---")
    cell_specs_raw      = boc.get_cell_specimens(experiment_container_ids=[CONTAINER_ID])
    cell_table          = pd.DataFrame(cell_specs_raw)
    container_ids       = list(cell_table["cell_specimen_id"])
    n_matched           = len(container_ids)
    session_ids         = list(dataset.get_cell_specimen_ids())
    session_id_to_pos   = {cid: i for i, cid in enumerate(session_ids)}
    present_mask        = np.array([cid in session_id_to_pos for cid in container_ids])
    n_present           = int(present_mask.sum())
    n_absent            = n_matched - n_present
    print(f"  Container cells: {n_matched}  /  Session cells: {len(session_ids)}  "
          f"/  Present: {n_present}  /  Absent (NaN): {n_absent}")

    session_row_for_container = np.array(
        [session_id_to_pos.get(cid, -1) for cid in container_ids]
    )

    def align_to_container(session_matrix):
        shape = (n_matched,) + session_matrix.shape[1:]
        out   = np.full(shape, np.nan, dtype=np.float32)
        for c_pos, s_row in enumerate(session_row_for_container):
            if s_row >= 0:
                out[c_pos] = session_matrix[s_row]
        return out

    # -- 4. Running speed ---------------------------------------------------
    print("\n--- Step 4: Running speed ---")
    run_timestamps, run_speed = dataset.get_running_speed()
    def mean_run_speed(t0, t1):
        m = (run_timestamps >= t0) & (run_timestamps < t1)
        return float(np.mean(run_speed[m])) if m.sum() else np.nan

    # -- 5. Per-repeat response matrix --------------------------------------
    print("\n--- Step 5: Per-repeat response matrix ---")
    X_session  = np.full((n_repeats, len(session_ids)), np.nan, dtype=np.float32)
    meta_rows  = []
    drop_trial = np.zeros(n_repeats, dtype=bool)
    movie_repeat_frames = np.zeros((n_repeats, 2), dtype=np.int64)

    for i, row in per_repeat.iterrows():
        sf, ef    = int(row["start_frame"]), int(row["end_frame"])
        onset     = float(row["start_time"])
        offset    = float(row["end_time"])
        duration  = float(row["duration_s"])
        movie_repeat_frames[i] = [sf, ef]

        n_frames_in = ef - sf + 1
        if n_frames_in <= 0:
            drop_trial[i] = True
        else:
            X_session[i, :] = events[:, sf : ef + 1].mean(axis=1).astype(np.float32)

        meta_rows.append({
            "trial_id":           i,
            "repeat_index":       int(row["repeat"]),
            "start_time":         onset,
            "end_time":           offset,
            "duration_s":         duration,
            "n_frames":           n_frames_in,
            "mean_running_speed": mean_run_speed(onset, offset),
        })

    if drop_trial.sum() > 0:
        X_session            = X_session[~drop_trial]
        meta_rows            = [r for r, d in zip(meta_rows, drop_trial) if not d]
        movie_repeat_frames  = movie_repeat_frames[~drop_trial]

    X = align_to_container(X_session.T).T
    print(f"  Response matrix: {X.shape}  ({n_present} cols with data, {n_absent} NaN)")

    # -- 6. Save -----------------------------------------------------------
    print(f"\n--- Step 6: Saving to {out_dir} ---")
    np.save(out_dir / "X_trials_neurons.npy", X)
    np.save(out_dir / "movie_repeat_frames.npy", movie_repeat_frames)
    pd.DataFrame(meta_rows).to_csv(out_dir / "trial_metadata.csv", index=False)

    # cell_info per session (records which cells were present in this session)
    # Allen-precomputed per-cell stats: keep the natural-movie-1 metrics
    # (reliability across the 10 movie repeats, peak dF/F amplitude).
    # The natural-scenes columns (pref_image_ns, image_sel_ns, reliability_ns)
    # were dropped — they're stats from the natural-scenes block in Session B
    # only and don't characterise responses to natural movie 1.
    keep_cols = ["cell_specimen_id", "area", "imaging_depth",
                 "rf_center_on_x_lsn", "rf_center_on_y_lsn",
                 "reliability_nm1", "peak_dff_nm1"]
    # Silently drop any not present in this Allen build's cell_specimens
    # schema (older AllenSDK versions may omit some fields).
    missing = [c for c in keep_cols if c not in cell_table.columns]
    if missing:
        print(f"  Note: Allen cell_specimens table missing columns: {missing} "
              f"(will be omitted from cell_info.csv)")
    keep_cols = [c for c in keep_cols if c in cell_table.columns]
    cell_info = cell_table[keep_cols].copy()
    cell_info = cell_info.set_index("cell_specimen_id").reindex(container_ids).reset_index()
    cell_info[f"present_in_session_{session_label.lower()}"] = present_mask
    cell_info = cell_info.rename(columns={
        "rf_center_on_x_lsn": "x", "rf_center_on_y_lsn": "y",
    })
    cell_info.to_csv(out_dir / "cell_info.csv", index=False)

    events_aligned = align_to_container(events)
    np.save(out_dir / "events_matrix.npy", events_aligned)

    # ---- Behavioural-state side channels (needed for the next phase) -------
    # Running speed is always available for Visual Coding 2P. Pupil tracking is
    # only available where eye tracking didn't fail; we save it when present
    # and write an empty marker file otherwise.
    np.save(out_dir / "running_speed.npy",
            np.stack([run_timestamps, run_speed], axis=0))
    print(f"  running_speed.npy:       (2, {len(run_timestamps)}) "
          f"[timestamps; speed]")

    pupil_saved = False
    for getter_name in ("get_pupil_size", "get_pupil_location"):
        if hasattr(dataset, getter_name):
            try:
                ts_p, val_p = getattr(dataset, getter_name)()
                np.save(out_dir / f"{getter_name.replace('get_', '')}.npy",
                        np.stack([ts_p, val_p], axis=0)
                        if val_p.ndim == 1 else
                        np.concatenate([ts_p[None, :], val_p], axis=0))
                print(f"  {getter_name.replace('get_', '')}.npy:  saved")
                pupil_saved = True
            except Exception as exc:    # eye tracking failed for this session
                print(f"  {getter_name} unavailable: {exc}")
    if not pupil_saved:
        (out_dir / "pupil_unavailable.flag").touch()
        print(f"  pupil tracking unavailable — wrote pupil_unavailable.flag")

    print(f"  X_trials_neurons.npy:    {X.shape}")
    print(f"  movie_repeat_frames.npy: {movie_repeat_frames.shape}")
    print(f"  trial_metadata.csv:      ({len(meta_rows)} rows)")
    print(f"  cell_info.csv:           {cell_info.shape}")
    print(f"  events_matrix.npy:       {events_aligned.shape}")

    print(f"\n  Session {session_label} complete.\n")
    return {
        "session": session_label,
        "n_repeats": len(meta_rows),
        "n_present": n_present,
        "n_absent": n_absent,
        "n_timepoints": events.shape[1],
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

results = []
for s in sessions_to_run:
    results.append(load_one_session(s))

print("=" * 65)
print("  ALL SESSIONS LOADED")
print("=" * 65)
for r in results:
    print(f"  Session {r['session']}: {r['n_repeats']} repeats, "
          f"{r['n_present']} active cells, {r['n_timepoints']} timepoints")
print(f"\n  Outputs in: {BASE_OUTPUT.resolve()}")
print(f"\n  Next: python3 scripts/03_preprocess_matrix.py --session all")
print(f"        python3 scripts/02b_intersect_active.py")
