"""WebSocket server for canonical fixed-bin vacuum-pressure streaming."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import VPRuntimeConfig, resolve_config
from .stream_pipeline import ProducerLatencyConfig, async_stream_events  # noqa: F401

logger = logging.getLogger(__name__)

_DEFAULT_PRODUCTS_YAML = (
    Path(__file__).resolve().parents[1] / "data_eng" / "config" / "products.yaml"
)

_BASE_GRID_FIELDS: List[tuple[str, pa.DataType]] = [
    ("k", pa.int32()),
    ("pressure_variant", pa.float64()),
    ("vacuum_variant", pa.float64()),
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
    ("spectrum_score", pa.float64()),
    ("spectrum_state_code", pa.int8()),
    ("last_event_id", pa.int64()),
]


def _grid_schema(config: VPRuntimeConfig) -> pa.Schema:
    fields = [pa.field(name, dtype) for name, dtype in _BASE_GRID_FIELDS]
    for horizon_ms in config.projection_horizons_ms:
        fields.append(pa.field(f"proj_score_h{horizon_ms}", pa.float64()))
    return pa.schema(fields)


def _grid_to_arrow_ipc(grid_dict: Dict[str, Any], schema: pa.Schema) -> bytes:
    buckets = grid_dict["buckets"]

    arrays = []
    for field in schema:
        name = field.name
        if pa.types.is_int8(field.type):
            arr = pa.array([int(b[name]) for b in buckets], type=pa.int8())
        elif pa.types.is_int32(field.type):
            arr = pa.array([int(b[name]) for b in buckets], type=pa.int32())
        elif pa.types.is_int64(field.type):
            arr = pa.array([int(b[name]) for b in buckets], type=pa.int64())
        else:
            arr = pa.array([float(b[name]) for b in buckets], type=pa.float64())
        arrays.append(arr)

    table = pa.Table.from_arrays(arrays, schema=schema)

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _et_hhmm_to_utc_ns(dt: str, hhmm: str) -> int:
    """Convert HH:MM ET on dt into UTC nanoseconds."""
    import pandas as pdt

    ts_utc = pdt.Timestamp(f"{dt} {hhmm}:00", tz="America/New_York").tz_convert("UTC")
    return int(ts_utc.value)


def create_app(
    lake_root: Path | None = None,
    products_yaml_path: Path | None = None,
    perf_latency_jsonl: Path | None = None,
    perf_window_start_et: str | None = None,
    perf_window_end_et: str | None = None,
    perf_summary_every_bins: int = 200,
) -> FastAPI:
    """Create the FastAPI app for canonical fixed-bin streaming."""
    if lake_root is None:
        lake_root = Path(__file__).resolve().parents[2] / "lake"
    if products_yaml_path is None:
        products_yaml_path = _DEFAULT_PRODUCTS_YAML
    if perf_summary_every_bins <= 0:
        raise ValueError(f"perf_summary_every_bins must be > 0, got {perf_summary_every_bins}")
    if perf_latency_jsonl is None and (perf_window_start_et is not None or perf_window_end_et is not None):
        raise ValueError("perf_window_start_et/perf_window_end_et require perf_latency_jsonl")

    app = FastAPI(
        title="Vacuum Pressure Stream Server",
        version="4.0.0",
        description=(
            "Canonical fixed-bin in-memory dense-grid vacuum-pressure streaming from event feed"
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
        product_type: str = "future_mbo",
        symbol: str = "MNQH6",
        dt: str = "2026-02-06",
        start_time: str | None = None,
    ) -> None:
        """Stream fixed-bin dense-grid updates from the canonical event engine."""
        await websocket.accept()

        try:
            config = resolve_config(product_type, symbol, products_yaml_path)
            schema = _grid_schema(config)
            producer_latency_cfg = None
            if perf_latency_jsonl is not None:
                window_start_ns = (
                    _et_hhmm_to_utc_ns(dt, perf_window_start_et)
                    if perf_window_start_et is not None
                    else None
                )
                window_end_ns = (
                    _et_hhmm_to_utc_ns(dt, perf_window_end_et)
                    if perf_window_end_et is not None
                    else None
                )
                producer_latency_cfg = ProducerLatencyConfig(
                    output_path=perf_latency_jsonl,
                    window_start_ns=window_start_ns,
                    window_end_ns=window_end_ns,
                    summary_every_bins=perf_summary_every_bins,
                )
        except (ValueError, FileNotFoundError) as exc:
            logger.error("VP stream setup failed: %s", exc)
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(exc),
            }))
            await websocket.close(code=1008, reason=str(exc))
            return

        logger.info(
            "VP fixed-bin stream connected: product_type=%s symbol=%s dt=%s start_time=%s radius=%d cell_width_ms=%d cfg=%s",
            config.product_type,
            config.symbol,
            dt,
            start_time,
            config.grid_radius_ticks,
            config.cell_width_ms,
            config.config_version,
        )
        if producer_latency_cfg is not None:
            logger.info(
                "VP producer latency telemetry active: output=%s window_start_ns=%s window_end_ns=%s summary_every_bins=%d",
                producer_latency_cfg.output_path,
                producer_latency_cfg.window_start_ns,
                producer_latency_cfg.window_end_ns,
                producer_latency_cfg.summary_every_bins,
            )

        await _stream_live_dense_grid(
            websocket=websocket,
            lake_root=lake_root,
            config=config,
            schema=schema,
            dt=dt,
            start_time=start_time,
            producer_latency_config=producer_latency_cfg,
        )

    return app


async def _stream_live_dense_grid(
    websocket: WebSocket,
    lake_root: Path,
    config: VPRuntimeConfig,
    schema: pa.Schema,
    dt: str,
    start_time: str | None,
    producer_latency_config: ProducerLatencyConfig | None = None,
) -> None:
    """Send fixed-bin dense-grid updates over websocket."""
    grid_count = 0

    try:
        await websocket.send_text(json.dumps({
            "type": "runtime_config",
            **config.to_dict(),
            "mode": "pre_prod",
            "deployment_stage": "pre_prod",
            "stream_format": "dense_grid",
            "grid_schema_fields": [f.name for f in schema],
            "grid_rows": 2 * config.grid_radius_ticks + 1,
        }))

        async for grid in async_stream_events(
            lake_root=lake_root,
            config=config,
            dt=dt,
            start_time=start_time,
            producer_latency_config=producer_latency_config,
        ):
            grid_count += 1

            await websocket.send_text(json.dumps({
                "type": "grid_update",
                "ts_ns": str(grid["ts_ns"]),
                "bin_seq": grid["bin_seq"],
                "bin_start_ns": str(grid["bin_start_ns"]),
                "bin_end_ns": str(grid["bin_end_ns"]),
                "bin_event_count": grid["bin_event_count"],
                "event_id": grid["event_id"],
                "mid_price": grid["mid_price"],
                "spot_ref_price_int": str(grid["spot_ref_price_int"]),
                "best_bid_price_int": str(grid["best_bid_price_int"]),
                "best_ask_price_int": str(grid["best_ask_price_int"]),
                "book_valid": grid["book_valid"],
            }))
            await websocket.send_bytes(_grid_to_arrow_ipc(grid, schema))

            if grid_count % 1000 == 0:
                logger.info(
                    "VP fixed-bin dense-grid: %d updates sent (event_id=%d)",
                    grid_count,
                    grid["event_id"],
                )

    except WebSocketDisconnect:
        logger.info("VP stream client disconnected")
    except Exception as exc:
        logger.error("VP stream error: %s", exc, exc_info=True)
    finally:
        logger.info("VP stream ended (%d fixed-bin updates sent)", grid_count)
