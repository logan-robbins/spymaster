"""WebSocket server for vacuum / pressure data streaming.

Standalone FastAPI application that serves vacuum / pressure metrics
via WebSocket using Arrow IPC binary encoding.

Supports both ``equity_mbo`` and ``future_mbo`` product types via
runtime configuration resolved at stream start.

Three streaming modes:
    **replay** (default): Load precomputed silver parquet, stream windows.
    **live**: Read raw .dbn file event-by-event, reconstruct book in-memory,
        compute signals incrementally, and stream windows in real time.
    **event**: Read raw .dbn file event-by-event, feed to canonical
        EventDrivenVPEngine, stream dense grid per event (or throttled).

Protocol for replay/live (per connection):
    0. JSON ``{"type": "runtime_config", ...}``  (full config block, once)
    Then per 1-second window:
    1. JSON ``{"type": "batch_start", "window_end_ts_ns": "...", ...}``
    2. JSON ``{"type": "surface_header", "surface": "snap"}``
    3. Binary: Arrow IPC bytes for snap (1 row)
    4. JSON ``{"type": "surface_header", "surface": "flow"}``
    5. Binary: Arrow IPC bytes for flow (~200 rows, per-bucket)
    6. JSON ``{"type": "surface_header", "surface": "signals"}``
    7. Binary: Arrow IPC bytes for signals (1 row, aggregated)

Protocol for event mode (per connection):
    0. JSON ``{"type": "runtime_config", ...}``  (full config block, once)
    Then per event (or throttled batch):
    1. JSON ``{"type": "grid_update", "ts_ns": ..., "event_id": ..., ...}``
    2. Binary: Arrow IPC bytes for dense grid (2K+1 rows, one per bucket)
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import VPRuntimeConfig, resolve_config
from .engine import VacuumPressureEngine, validate_silver_readiness
from .formulas import GoldSignalConfig
from .stream_pipeline import async_stream_events, async_stream_windows

logger = logging.getLogger(__name__)

# Default products.yaml location relative to this file
_DEFAULT_PRODUCTS_YAML = (
    Path(__file__).resolve().parents[1] / "data_eng" / "config" / "products.yaml"
)

# ──────────────────────────────────────────────────────────────────────
# Arrow IPC schemas
# ──────────────────────────────────────────────────────────────────────

SNAP_SCHEMA = pa.schema([
    ("window_end_ts_ns", pa.int64()),
    ("mid_price", pa.float64()),
    ("spot_ref_price_int", pa.int64()),
    ("best_bid_price_int", pa.int64()),
    ("best_ask_price_int", pa.int64()),
    ("book_valid", pa.bool_()),
])

FLOW_SCHEMA = pa.schema([
    ("rel_ticks", pa.int32()),
    ("side", pa.string()),
    ("depth_qty_end", pa.float64()),
    ("add_qty", pa.float64()),
    ("pull_qty", pa.float64()),
    ("fill_qty", pa.float64()),
    ("depth_qty_rest", pa.float64()),
    ("pull_qty_rest", pa.float64()),
    ("net_flow", pa.float64()),
    ("vacuum_intensity", pa.float64()),
    ("pressure_intensity", pa.float64()),
    ("rest_fraction", pa.float64()),
])

SIGNALS_SCHEMA = pa.schema([
    ("window_end_ts_ns", pa.int64()),
    # Existing per-window metrics
    ("vacuum_above", pa.float64()),
    ("vacuum_below", pa.float64()),
    ("resting_drain_ask", pa.float64()),
    ("resting_drain_bid", pa.float64()),
    ("flow_imbalance", pa.float64()),
    ("fill_imbalance", pa.float64()),
    ("depth_imbalance", pa.float64()),
    ("rest_depth_imbalance", pa.float64()),
    ("bid_migration_com", pa.float64()),
    ("ask_migration_com", pa.float64()),
    # Pressure and resistance fields
    ("pressure_above", pa.float64()),
    ("pressure_below", pa.float64()),
    ("resistance_above", pa.float64()),
    ("resistance_below", pa.float64()),
    # Bernoulli lift model
    ("lift_up", pa.float64()),
    ("lift_down", pa.float64()),
    ("net_lift", pa.float64()),
    ("feasibility_up", pa.float64()),
    ("feasibility_down", pa.float64()),
    ("directional_bias", pa.float64()),
    # Multi-timescale (fast ~5s)
    ("lift_5s", pa.float64()),
    ("d1_5s", pa.float64()),
    ("d2_5s", pa.float64()),
    ("proj_5s", pa.float64()),
    ("dir_5s", pa.int64()),
    # Multi-timescale (medium ~15s)
    ("lift_15s", pa.float64()),
    ("d1_15s", pa.float64()),
    ("d2_15s", pa.float64()),
    ("proj_15s", pa.float64()),
    ("dir_15s", pa.int64()),
    # Multi-timescale (slow ~60s)
    ("lift_60s", pa.float64()),
    ("d1_60s", pa.float64()),
    ("d2_60s", pa.float64()),
    ("proj_60s", pa.float64()),
    ("dir_60s", pa.int64()),
    # Cross-timescale confidence, alerts, regime
    ("cross_confidence", pa.float64()),
    ("projection_coherence", pa.float64()),
    ("alert_flags", pa.int64()),
    ("regime", pa.string()),
    # Deterministic directional event machine outputs
    ("event_state", pa.string()),
    ("event_direction", pa.string()),
    ("event_strength", pa.float64()),
    ("event_confidence", pa.float64()),
    # Backward compat (mapped from new model)
    ("composite", pa.float64()),
    ("composite_smooth", pa.float64()),
    ("d1_composite", pa.float64()),
    ("d2_composite", pa.float64()),
    ("d3_composite", pa.float64()),
    ("d1_smooth", pa.float64()),
    ("d2_smooth", pa.float64()),
    ("d3_smooth", pa.float64()),
    ("wtd_slope", pa.float64()),
    ("wtd_projection", pa.float64()),
    ("wtd_projection_500ms", pa.float64()),
    ("wtd_deriv_conf", pa.float64()),
    ("z_composite_raw", pa.float64()),
    ("z_composite_smooth", pa.float64()),
    ("confidence", pa.float64()),
    ("strength", pa.float64()),
    ("strength_smooth", pa.float64()),
])


# ──────────────────────────────────────────────────────────────────────
# Event-driven dense grid Arrow IPC schema
# ──────────────────────────────────────────────────────────────────────

GRID_SCHEMA = pa.schema([
    ("k", pa.int32()),
    ("pressure_variant", pa.float64()),
    ("vacuum_variant", pa.float64()),
    ("resistance_variant", pa.float64()),
    ("add_mass", pa.float64()),
    ("pull_mass", pa.float64()),
    ("fill_mass", pa.float64()),
    ("rest_depth", pa.float64()),
    ("v_add", pa.float64()),
    ("v_pull", pa.float64()),
    ("v_fill", pa.float64()),
    ("v_rest_depth", pa.float64()),
    ("a_add", pa.float64()),
    ("a_pull", pa.float64()),
    ("a_fill", pa.float64()),
    ("a_rest_depth", pa.float64()),
    ("j_add", pa.float64()),
    ("j_pull", pa.float64()),
    ("j_fill", pa.float64()),
    ("j_rest_depth", pa.float64()),
    ("last_event_id", pa.int64()),
])
"""Arrow schema for the dense per-bucket grid emitted by EventDrivenVPEngine.

Each grid update contains exactly 2K+1 rows (one per bucket from -K to +K).
All fields are non-nullable by construction (engine guarantees G3: no NaN/Inf).
"""


# ──────────────────────────────────────────────────────────────────────
# Arrow IPC serialisation
# ──────────────────────────────────────────────────────────────────────


def _df_to_arrow_ipc(df: pd.DataFrame, schema: pa.Schema) -> bytes:
    """Convert DataFrame to Arrow IPC stream bytes.

    Handles missing columns (filled with type-appropriate zero) and
    NaN values in integer columns (filled with 0).

    Args:
        df: Input DataFrame.
        schema: Target Arrow schema.

    Returns:
        Arrow IPC stream bytes.
    """
    if df.empty:
        arrays = [pa.array([], type=f.type) for f in schema]
        table = pa.Table.from_arrays(arrays, schema=schema)
    else:
        df = df.copy()
        # Ensure every schema column exists
        for field in schema:
            if field.name not in df.columns:
                if pa.types.is_float64(field.type):
                    df[field.name] = 0.0
                elif pa.types.is_int64(field.type) or pa.types.is_int32(field.type):
                    df[field.name] = 0
                elif pa.types.is_boolean(field.type):
                    df[field.name] = False
                elif pa.types.is_string(field.type):
                    df[field.name] = ""

        # Select schema columns in order
        col_names = [f.name for f in schema]
        df = df[[c for c in col_names if c in df.columns]]

        # NaN handling for integer target types
        for field in schema:
            if field.name in df.columns:
                if pa.types.is_int64(field.type) or pa.types.is_int32(field.type):
                    df[field.name] = df[field.name].fillna(0)
                elif pa.types.is_float64(field.type):
                    df[field.name] = df[field.name].astype("float64")

        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _grid_to_arrow_ipc(grid_dict: Dict[str, Any]) -> bytes:
    """Convert an EventDrivenVPEngine grid dict to Arrow IPC stream bytes.

    Extracts the bucket list from the grid_dict and serializes each
    bucket as one row in a table matching ``GRID_SCHEMA``. Uses direct
    columnar construction from the bucket dicts for efficiency (no
    pandas intermediate).

    Args:
        grid_dict: Output from ``EventDrivenVPEngine.update()``.
            Must contain ``buckets`` key with list of bucket dicts.

    Returns:
        Arrow IPC stream bytes with 2K+1 rows.
    """
    buckets = grid_dict["buckets"]
    n = len(buckets)

    # Build columnar arrays directly from bucket dicts
    arrays = []
    for field in GRID_SCHEMA:
        if field.name == "k":
            arr = pa.array(
                [b["k"] for b in buckets], type=pa.int32(),
            )
        elif field.name == "last_event_id":
            arr = pa.array(
                [b["last_event_id"] for b in buckets], type=pa.int64(),
            )
        else:
            arr = pa.array(
                [b[field.name] for b in buckets], type=pa.float64(),
            )
        arrays.append(arr)

    table = pa.Table.from_arrays(arrays, schema=GRID_SCHEMA)

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, GRID_SCHEMA) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


# ──────────────────────────────────────────────────────────────────────
# FastAPI application factory
# ──────────────────────────────────────────────────────────────────────


def create_app(
    lake_root: Path | None = None,
    products_yaml_path: Path | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        lake_root: Path to the lake directory.  Defaults to ``backend/lake``.
        products_yaml_path: Path to products.yaml. Defaults to
            ``backend/src/data_eng/config/products.yaml``.

    Returns:
        Configured FastAPI instance.
    """
    if lake_root is None:
        lake_root = Path(__file__).resolve().parents[2] / "lake"

    if products_yaml_path is None:
        products_yaml_path = _DEFAULT_PRODUCTS_YAML

    engine = VacuumPressureEngine(lake_root)

    app = FastAPI(
        title="Vacuum Pressure Stream Server",
        version="2.0.0",
        description=(
            "Real-time vacuum / pressure detection for equity and futures "
            "order flow with runtime instrument configuration."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "vacuum-pressure"}

    @app.websocket("/v1/vacuum-pressure/stream")
    async def vacuum_pressure_stream(
        websocket: WebSocket,
        product_type: str = "equity_mbo",
        symbol: str = "QQQ",
        dt: str = "2026-02-06",
        speed: float = 1.0,
        skip_minutes: int = 5,
        mode: str = "replay",
        start_time: str | None = None,
        pre_smooth_span: int | None = None,
        d1_span: int | None = None,
        d2_span: int | None = None,
        d3_span: int | None = None,
        w_d1: float | None = None,
        w_d2: float | None = None,
        w_d3: float | None = None,
        projection_horizon_s: float | None = None,
        fast_projection_horizon_s: float | None = None,
        smooth_zscore_window: int | None = None,
    ) -> None:
        """Stream vacuum / pressure data per 1-second window.

        Query params:
            product_type: Product type (``equity_mbo`` or ``future_mbo``).
            symbol: Instrument symbol.
            dt: Date (YYYY-MM-DD).
            speed: Replay speed multiplier.
            skip_minutes: Minutes of data to skip from start.
            mode: ``replay`` (default, from silver parquet) or
                ``live`` (from raw .dbn, incremental computation).
        """
        await websocket.accept()

        if mode not in ("replay", "live", "event"):
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Invalid mode '{mode}'. Must be 'replay', 'live', or 'event'.",
            }))
            await websocket.close(code=1008, reason="Invalid mode")
            return

        # Resolve runtime config
        try:
            config = resolve_config(product_type, symbol, products_yaml_path)
        except (ValueError, FileNotFoundError) as exc:
            logger.error("Config resolution failed: %s", exc)
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(exc),
            }))
            await websocket.close(code=1008, reason=str(exc))
            return

        # Resolve optional gold-layer smoothing/projection runtime overrides
        gold_kwargs: dict[str, int | float] = {}
        if pre_smooth_span is not None:
            gold_kwargs["pre_smooth_span"] = pre_smooth_span
        if d1_span is not None:
            gold_kwargs["d1_span"] = d1_span
        if d2_span is not None:
            gold_kwargs["d2_span"] = d2_span
        if d3_span is not None:
            gold_kwargs["d3_span"] = d3_span
        if w_d1 is not None:
            gold_kwargs["w_d1"] = w_d1
        if w_d2 is not None:
            gold_kwargs["w_d2"] = w_d2
        if w_d3 is not None:
            gold_kwargs["w_d3"] = w_d3
        if projection_horizon_s is not None:
            gold_kwargs["projection_horizon_s"] = projection_horizon_s
        if fast_projection_horizon_s is not None:
            gold_kwargs["fast_projection_horizon_s"] = fast_projection_horizon_s
        if smooth_zscore_window is not None:
            gold_kwargs["smooth_zscore_window"] = smooth_zscore_window

        try:
            gold_signal_config = GoldSignalConfig(**gold_kwargs)
            gold_signal_config.validate()
        except ValueError as exc:
            logger.error("Gold signal config validation failed: %s", exc)
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(exc),
            }))
            await websocket.close(code=1008, reason=str(exc))
            return

        logger.info(
            "VP stream connected: mode=%s product_type=%s symbol=%s dt=%s "
            "speed=%.1f skip=%d config_version=%s gold_cfg=%s",
            mode, config.product_type, config.symbol, dt,
            speed, skip_minutes, config.config_version, vars(gold_signal_config),
        )

        if mode == "event":
            await _stream_event_driven(
                websocket, lake_root, config, dt, speed, start_time,
            )
        elif mode == "live":
            await _stream_live(
                websocket, lake_root, config, dt, speed, skip_minutes,
                gold_signal_config, start_time,
            )
        else:
            await _stream_replay(
                websocket, engine, lake_root, config, dt, speed, skip_minutes,
                gold_signal_config,
            )

    return app


async def _stream_replay(
    websocket: WebSocket,
    engine: VacuumPressureEngine,
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    speed: float,
    skip_minutes: int,
    gold_signal_config: GoldSignalConfig,
) -> None:
    """Stream from precomputed silver parquet (original replay mode)."""
    # Readiness check
    try:
        row_counts = validate_silver_readiness(lake_root, config, dt)
    except FileNotFoundError as exc:
        logger.error("Silver readiness check failed: %s", exc)
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(exc),
        }))
        await websocket.close(code=1008, reason="Silver data not ready")
        return

    logger.info(
        "VP replay stream ready: config=%s row_counts=%s",
        json.dumps(config.to_dict()),
        json.dumps(row_counts),
    )

    try:
        # First control message: full runtime config
        await websocket.send_text(json.dumps({
            "type": "runtime_config",
            **config.to_dict(),
            "gold_signal_config": vars(gold_signal_config),
        }))

        last_ts: int | None = None
        skip_count = skip_minutes * 60
        skipped = 0

        for wid, batch in engine.iter_windows(
            config,
            dt,
            gold_signal_config=gold_signal_config,
        ):
            if skipped < skip_count:
                skipped += 1
                continue

            # Simulate real-time delay
            if last_ts is not None:
                delta_s = (wid - last_ts) / 1_000_000_000.0
                wait = delta_s / speed
                if wait > 0:
                    await asyncio.sleep(wait)
            last_ts = wid

            await _send_window_batch(websocket, config, wid, batch)

    except WebSocketDisconnect:
        logger.info("VP replay stream client disconnected")
    except Exception as exc:
        logger.error("VP replay stream error: %s", exc, exc_info=True)
    finally:
        logger.info("VP replay stream disconnected")


async def _stream_live(
    websocket: WebSocket,
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    speed: float,
    skip_minutes: int,
    gold_signal_config: GoldSignalConfig,
    start_time: str | None = None,
) -> None:
    """Stream from raw .dbn file with incremental computation."""
    from .formulas import compute_per_bucket_scores

    logger.info(
        "VP live stream starting: config=%s start_time=%s",
        json.dumps(config.to_dict()), start_time,
    )

    try:
        # First control message: full runtime config
        await websocket.send_text(json.dumps({
            "type": "runtime_config",
            **config.to_dict(),
            "gold_signal_config": vars(gold_signal_config),
        }))

        window_count = 0
        async for wid, signals, snap_dict, flow_df in async_stream_windows(
            lake_root=lake_root,
            config=config,
            dt=dt,
            gold_config=gold_signal_config,
            speed=speed,
            skip_minutes=skip_minutes,
            start_time=start_time,
        ):
            window_count += 1

            # Build DataFrames matching the replay mode's format
            # Snap: single-row DataFrame
            snap_df = pd.DataFrame([{
                "window_end_ts_ns": snap_dict.get("window_end_ts_ns", wid),
                "mid_price": snap_dict.get("mid_price", 0.0),
                "spot_ref_price_int": snap_dict.get("spot_ref_price_int", 0),
                "best_bid_price_int": snap_dict.get("best_bid_price_int", 0),
                "best_ask_price_int": snap_dict.get("best_ask_price_int", 0),
                "book_valid": snap_dict.get("book_valid", False),
            }])

            # Flow: enrich with per-bucket scores for the frontend heatmap
            if not flow_df.empty:
                flow_enriched = compute_per_bucket_scores(
                    flow_df, config.bucket_size_dollars
                )
            else:
                flow_enriched = flow_df

            # Signals: single-row DataFrame
            signals_df = pd.DataFrame([signals])

            batch = {
                "snap": snap_df,
                "flow": flow_enriched,
                "signals": signals_df,
            }

            await _send_window_batch(websocket, config, wid, batch)

            if window_count % 600 == 0:
                logger.info(
                    "VP live stream: %d windows sent (ts=%d)",
                    window_count, wid,
                )

    except WebSocketDisconnect:
        logger.info("VP live stream client disconnected")
    except Exception as exc:
        logger.error("VP live stream error: %s", exc, exc_info=True)
    finally:
        logger.info("VP live stream disconnected (%d windows sent)", window_count)


async def _send_window_batch(
    websocket: WebSocket,
    config: VPRuntimeConfig,
    wid: int,
    batch: dict[str, pd.DataFrame],
) -> None:
    """Send a single window's data over WebSocket (shared by both modes)."""
    # Determine available surfaces
    surfaces: list[str] = []
    if not batch["snap"].empty:
        surfaces.append("snap")
    if not batch["flow"].empty:
        surfaces.append("flow")
    if not batch["signals"].empty:
        surfaces.append("signals")

    _log_window_snapshot(config=config, wid=wid, batch=batch, surfaces=surfaces)

    # batch_start
    await websocket.send_text(json.dumps({
        "type": "batch_start",
        "window_end_ts_ns": str(wid),
        "surfaces": surfaces,
        "bucket_size": config.bucket_size_dollars,
        "tick_size": config.rel_tick_size,
    }))

    # snap
    if "snap" in surfaces:
        await websocket.send_text(json.dumps({
            "type": "surface_header",
            "surface": "snap",
        }))
        await websocket.send_bytes(
            _df_to_arrow_ipc(batch["snap"], SNAP_SCHEMA)
        )

    # flow
    if "flow" in surfaces:
        await websocket.send_text(json.dumps({
            "type": "surface_header",
            "surface": "flow",
        }))
        await websocket.send_bytes(
            _df_to_arrow_ipc(batch["flow"], FLOW_SCHEMA)
        )

    # signals
    if "signals" in surfaces:
        await websocket.send_text(json.dumps({
            "type": "surface_header",
            "surface": "signals",
        }))
        await websocket.send_bytes(
            _df_to_arrow_ipc(batch["signals"], SIGNALS_SCHEMA)
        )


def _log_window_snapshot(
    config: VPRuntimeConfig,
    wid: int,
    batch: dict[str, pd.DataFrame],
    surfaces: list[str],
) -> None:
    """Emit per-window snapshot diagnostics for pause-and-debug workflows."""
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            if pd.isna(value):
                return default
            return int(value)
        except Exception:
            return default

    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    payload: dict[str, Any] = {
        "window_end_ts_ns": int(wid),
        "product_type": config.product_type,
        "symbol": config.symbol,
        "surfaces": surfaces,
    }

    if "snap" in surfaces and not batch["snap"].empty:
        snap_row = batch["snap"].iloc[0]
        payload["snap"] = {
            "mid_price": _safe_float(snap_row.get("mid_price", 0.0), 0.0),
            "best_bid_price_int": _safe_int(snap_row.get("best_bid_price_int", 0), 0),
            "best_ask_price_int": _safe_int(snap_row.get("best_ask_price_int", 0), 0),
            "book_valid": bool(snap_row.get("book_valid", False)),
        }

    if "signals" in surfaces and not batch["signals"].empty:
        sig_row = batch["signals"].iloc[0]
        payload["signals"] = {
            "regime": str(sig_row.get("regime", "")),
            "net_lift": _safe_float(
                sig_row.get("net_lift", sig_row.get("composite", 0.0)),
                0.0,
            ),
            "event_state": str(sig_row.get("event_state", "")),
            "event_direction": str(sig_row.get("event_direction", "")),
            "dir_5s": _safe_int(sig_row.get("dir_5s", 0), 0),
            "dir_15s": _safe_int(sig_row.get("dir_15s", 0), 0),
            "dir_60s": _safe_int(sig_row.get("dir_60s", 0), 0),
        }

    try:
        logger.info("VP_WINDOW %s", json.dumps(payload, separators=(",", ":")))
    except Exception as exc:
        logger.warning("Failed to serialize VP_WINDOW payload: %s", exc)


# ──────────────────────────────────────────────────────────────────────
# Event-driven streaming handler (mode=event)
# ──────────────────────────────────────────────────────────────────────


async def _stream_event_driven(
    websocket: WebSocket,
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    speed: float,
    start_time: str | None = None,
    throttle_ms: float = 100,
) -> None:
    """Stream dense grid from EventDrivenVPEngine over WebSocket.

    Protocol per update:
        1. JSON ``grid_update`` header with scalar metadata.
        2. Binary Arrow IPC with 2K+1 rows (one per bucket).

    The EventDrivenVPEngine replaces both StreamingBookAdapter and
    IncrementalSignalEngine. It processes every MBO event, maintains
    its own internal order book, and computes per-bucket derivative
    chains and force variants.

    Args:
        websocket: Active WebSocket connection.
        lake_root: Path to the lake directory.
        config: Resolved VP runtime config.
        dt: Date string (YYYY-MM-DD).
        speed: Replay speed multiplier. 0 = fire-hose.
        start_time: When to start emitting, as "HH:MM" in ET.
        throttle_ms: Minimum ms between updates (event-time).
            Default 100 = max ~10 updates/sec.
    """
    logger.info(
        "VP event stream starting: config=%s start_time=%s throttle_ms=%.0f",
        json.dumps(config.to_dict()), start_time, throttle_ms,
    )

    grid_count = 0

    try:
        # First control message: full runtime config
        await websocket.send_text(json.dumps({
            "type": "runtime_config",
            **config.to_dict(),
            "mode": "event",
            "grid_schema_fields": [f.name for f in GRID_SCHEMA],
            "grid_rows": 2 * config.grid_max_ticks + 1,
        }))

        async for grid in async_stream_events(
            lake_root=lake_root,
            config=config,
            dt=dt,
            speed=speed,
            start_time=start_time,
            throttle_ms=throttle_ms,
        ):
            grid_count += 1

            # JSON header with scalar metadata
            await websocket.send_text(json.dumps({
                "type": "grid_update",
                "ts_ns": str(grid["ts_ns"]),
                "event_id": grid["event_id"],
                "mid_price": grid["mid_price"],
                "spot_ref_price_int": str(grid["spot_ref_price_int"]),
                "best_bid_price_int": str(grid["best_bid_price_int"]),
                "best_ask_price_int": str(grid["best_ask_price_int"]),
                "book_valid": grid["book_valid"],
            }))

            # Binary: Arrow IPC with dense grid (2K+1 rows)
            await websocket.send_bytes(_grid_to_arrow_ipc(grid))

            if grid_count % 1000 == 0:
                logger.info(
                    "VP event stream: %d grids sent (event_id=%d, "
                    "mid=%.2f, spot=%s)",
                    grid_count, grid["event_id"],
                    grid["mid_price"],
                    grid["spot_ref_price_int"],
                )

    except WebSocketDisconnect:
        logger.info("VP event stream client disconnected")
    except Exception as exc:
        logger.error("VP event stream error: %s", exc, exc_info=True)
    finally:
        logger.info(
            "VP event stream disconnected (%d grids sent)", grid_count,
        )
