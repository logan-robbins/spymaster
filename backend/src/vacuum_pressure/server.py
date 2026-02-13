"""WebSocket server for live vacuum-pressure dense-grid streaming.

Canonical runtime path:
    ingest (.dbn adapter for now) -> EventDrivenVPEngine (in-memory) -> dense grid

No replay/silver/window branches are supported in this server.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import pyarrow as pa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import VPRuntimeConfig, resolve_config
from .stream_pipeline import DEFAULT_GRID_TICKS, async_stream_events

logger = logging.getLogger(__name__)

_DEFAULT_PRODUCTS_YAML = (
    Path(__file__).resolve().parents[1] / "data_eng" / "config" / "products.yaml"
)

GRID_SCHEMA = pa.schema([
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
    ("last_event_id", pa.int64()),
])


def _with_live_grid(config: VPRuntimeConfig) -> VPRuntimeConfig:
    """Return runtime config with canonical live grid radius baked in."""
    if DEFAULT_GRID_TICKS > config.grid_max_ticks:
        raise ValueError(
            f"default live grid K={DEFAULT_GRID_TICKS} exceeds configured max "
            f"{config.grid_max_ticks} "
            f"for {config.product_type}/{config.symbol}"
        )
    return replace(
        config,
        grid_max_ticks=DEFAULT_GRID_TICKS,
        config_version=f"{config.config_version}:k{DEFAULT_GRID_TICKS}",
    )


def _grid_to_arrow_ipc(grid_dict: Dict[str, Any]) -> bytes:
    """Convert EventDrivenVPEngine grid output to Arrow IPC bytes."""
    buckets = grid_dict["buckets"]

    arrays = []
    for field in GRID_SCHEMA:
        if field.name == "k":
            arr = pa.array([b["k"] for b in buckets], type=pa.int32())
        elif field.name == "last_event_id":
            arr = pa.array([b["last_event_id"] for b in buckets], type=pa.int64())
        else:
            arr = pa.array([b[field.name] for b in buckets], type=pa.float64())
        arrays.append(arr)

    table = pa.Table.from_arrays(arrays, schema=GRID_SCHEMA)

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, GRID_SCHEMA) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def create_app(
    lake_root: Path | None = None,
    products_yaml_path: Path | None = None,
) -> FastAPI:
    """Create the FastAPI app for live dense-grid streaming."""
    if lake_root is None:
        lake_root = Path(__file__).resolve().parents[2] / "lake"
    if products_yaml_path is None:
        products_yaml_path = _DEFAULT_PRODUCTS_YAML

    app = FastAPI(
        title="Vacuum Pressure Live Stream Server",
        version="3.0.0",
        description=(
            "Live in-memory dense-grid vacuum-pressure streaming from event feed "
            "(DBN replay adapter today, live subscription adapter later)."
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
        return {"status": "ok", "service": "vacuum-pressure-live"}

    @app.websocket("/v1/vacuum-pressure/stream")
    async def vacuum_pressure_stream(
        websocket: WebSocket,
        product_type: str = "future_mbo",
        symbol: str = "MNQH6",
        dt: str = "2026-02-06",
        speed: float = 1.0,
        start_time: str | None = None,
        throttle_ms: float = 25.0,
    ) -> None:
        """Stream dense-grid updates from the canonical live event engine.

        Query params:
            product_type: ``equity_mbo`` or ``future_mbo``.
            symbol: instrument symbol.
            dt: date in YYYY-MM-DD.
            speed: replay speed multiplier for DBN adapter (0=firehose).
            start_time: optional emit start HH:MM ET (warmup processed in-memory).
            throttle_ms: minimum event-time spacing between emitted updates.
        """
        await websocket.accept()

        try:
            if speed < 0:
                raise ValueError(f"speed must be >= 0, got {speed}")
            if throttle_ms < 0:
                raise ValueError(f"throttle_ms must be >= 0, got {throttle_ms}")

            base_config = resolve_config(product_type, symbol, products_yaml_path)
            config = _with_live_grid(base_config)
        except (ValueError, FileNotFoundError) as exc:
            logger.error("Live VP stream setup failed: %s", exc)
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(exc),
            }))
            await websocket.close(code=1008, reason=str(exc))
            return

        logger.info(
            "VP live stream connected: product_type=%s symbol=%s dt=%s speed=%.2f start_time=%s K=%d throttle_ms=%.1f cfg=%s",
            config.product_type,
            config.symbol,
            dt,
            speed,
            start_time,
            config.grid_max_ticks,
            throttle_ms,
            config.config_version,
        )

        await _stream_live_dense_grid(
            websocket=websocket,
            lake_root=lake_root,
            config=config,
            dt=dt,
            speed=speed,
            start_time=start_time,
            throttle_ms=throttle_ms,
        )

    return app


async def _stream_live_dense_grid(
    websocket: WebSocket,
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
    speed: float,
    start_time: str | None,
    throttle_ms: float,
) -> None:
    """Send dense-grid updates over websocket from live event pipeline."""
    grid_count = 0

    try:
        await websocket.send_text(json.dumps({
            "type": "runtime_config",
            **config.to_dict(),
            "mode": "live",
            "stream_format": "dense_grid",
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
            await websocket.send_bytes(_grid_to_arrow_ipc(grid))

            if grid_count % 1000 == 0:
                logger.info(
                    "VP live dense-grid: %d updates sent (event_id=%d)",
                    grid_count,
                    grid["event_id"],
                )

    except WebSocketDisconnect:
        logger.info("VP live stream client disconnected")
    except Exception as exc:
        logger.error("VP live stream error: %s", exc, exc_info=True)
    finally:
        logger.info("VP live stream ended (%d dense-grid updates sent)", grid_count)
