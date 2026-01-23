# Spymaster - Streaming-First Radar Architecture (Technical Reference)

**System Role**: Expert-level guide for AI Coding Agents.
**Context**: Replay-based streaming of Databento MBO data for futures and options.
**Canonical Date**: `2026-01-06` (Single-day full scope).

---

## 1. Core Workflows (Canonical)

### Run Main Pipeline (2026-01-06)
Executes all stages effectively in order: Bronze (Ingest) -> Silver (Compute Surfaces) -> Gold (Calibration).
```bash
# Recompute all silver/gold outputs for the replay date
nohup uv run python scripts/run_pipeline_2026_01_06.py --dt 2026-01-06 > /tmp/pipeline_2026-01-06.log 2>&1 &
tail -f /tmp/pipeline_2026-01-06.log
```

### Verify System Integrity
These scripts **MUST** pass after any pipeline changes.
```bash
# 1. Futures Order Book & Radar Integrity (Validates: Snapshot valid, Best Bid < Ask, Radar alignment)
uv run python scripts/test_data_integrity.py --dt 2026-01-06

# 2. Physics & Vacuum Integrity (Validates: 0..1 normalization, Band consistency, Calibration application)
uv run python scripts/test_physics_integrity.py --dt 2026-01-06

# 3. Streaming Service Architecture (Validates: Ring buffer, Batch iteration, Arrow IPC, GEX alignment)
uv run python scripts/verify_hud_service.py
```

---

## 2. Architecture & Implementation Details

### Futures Book Engine (Canonical source)
- **Path**: `src/data_eng/stages/silver/future_mbo/book_engine.py`
- **Logic**: Applies MBO messages linearly. State persists across windows.
- **Snapshot Handling**: Handled via `F_SNAPSHOT` transition detection (Implicit Start/End).
- **Valid Flag**: `book_valid=True` ONLY when a full snapshot has been processed and we are in incremental mode.
- **Outputs**: `book_snapshot_1s`, `wall_surface_1s`, `radar_vacuum_1s` (Share exact same engine state).

### Vacuum & Physics
- **Vacuum Surface**: Computed from `wall_surface_1s` + `physics_norm_calibration`.
- **Physics Bands**: Aggregates Wall + Vacuum into AT/NEAR/MID/FAR bands. 0..1 Scores.
- **Calibration**: Gold layer built from full-day stats. Required for normalization.

### Options GEX
- **Path**: `src/data_eng/stages/silver/future_option_mbo/compute_gex_surface_1s.py`
- **Logic**: Vectorized Numba processing of Option MBO.
- **Alignment**: Joins with Futures Spot Ref explicitly.

### Streaming Service
- **Path**: `src/serving/hud_streaming.py`
- **Mechanism**: Pre-computes surfaces from Silver lake. Loads into memory cache. Simulates 1s streaming via `iter_batches`.
- **Protocol**: Arrow IPC over WebSocket / HTTP.

---

## 3. Data Lake Structure
**Root**: `lake/`
- **Bronze**: Raw MBO, Instrument Defs.
- **Silver**: Computed Surfaces (Snapshot, Wall, Radar, Vacuum, Bands, GEX).
- **Gold**: Calibration models.

---

## 4. Debugging & Maintenance
- **Debug MBO Flags**: `uv run python scripts/debug_flags.py` (Check for F_SNAPSHOT/F_LAST presence).
- **Run Radar Only**: `uv run python scripts/run_radar_only.py` (Fast iteration on radar stage).
- **Clear & Re-run**: `uv run python scripts/run_updates.py` (Targeted re-run of Snapshot+Radar stages).
