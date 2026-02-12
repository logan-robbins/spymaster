"""WebSocket server for vacuum / pressure data streaming.

Standalone FastAPI application that serves vacuum / pressure metrics
via WebSocket using Arrow IPC binary encoding.

Supports both ``equity_mbo`` and ``future_mbo`` product types via
runtime configuration resolved at stream start.

Protocol (per connection):
    0. JSON ``{"type": "runtime_config", ...}``  (full config block, once)
    Then per 1-second window:
    1. JSON ``{"type": "batch_start", "window_end_ts_ns": "...", ...}``
    2. JSON ``{"type": "surface_header", "surface": "snap"}``
    3. Binary: Arrow IPC bytes for snap (1 row)
    4. JSON ``{"type": "surface_header", "surface": "flow"}``
    5. Binary: Arrow IPC bytes for flow (~200 rows, per-bucket)
    6. JSON ``{"type": "surface_header", "surface": "signals"}``
    7. Binary: Arrow IPC bytes for signals (1 row, aggregated)
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import VPRuntimeConfig, resolve_config
from .engine import VacuumPressureEngine, validate_silver_readiness
from .formulas import GoldSignalConfig

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
        """
        await websocket.accept()

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
            "VP stream connected: product_type=%s symbol=%s dt=%s "
            "speed=%.1f skip=%d config_version=%s gold_cfg=%s",
            config.product_type, config.symbol, dt,
            speed, skip_minutes, config.config_version, vars(gold_signal_config),
        )

        # Readiness check (4.7)
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

        # Structured startup log (4.7)
        logger.info(
            "VP stream ready: config=%s row_counts=%s",
            json.dumps(config.to_dict()),
            json.dumps(row_counts),
        )

        try:
            # First control message: full runtime config (4.4)
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

                # Determine available surfaces
                surfaces: list[str] = []
                if not batch["snap"].empty:
                    surfaces.append("snap")
                if not batch["flow"].empty:
                    surfaces.append("flow")
                if not batch["signals"].empty:
                    surfaces.append("signals")

                # batch_start -- includes legacy fields for transition window
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

        except WebSocketDisconnect:
            logger.info("VP stream client disconnected")
        except Exception as exc:
            logger.error("VP stream error: %s", exc, exc_info=True)
        finally:
            logger.info("VP stream disconnected")

    return app
