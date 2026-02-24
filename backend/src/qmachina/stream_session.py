"""Stream session setup and execution for qMachina websocket transport."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from .config import RuntimeConfig, build_config_from_mapping, resolve_config
from .serving_registry import ResolvedServing, ServingRegistry
from .stream_contract import (
    build_grid_update_payload,
    build_runtime_config_payload,
    grid_schema,
    grid_to_arrow_ipc,
)
from ..models.vacuum_pressure.stream_pipeline import ProducerLatencyConfig, async_stream_events

logger = logging.getLogger(__name__)


def _et_hhmm_to_utc_ns(dt: str, hhmm: str) -> int:
    """Convert HH:MM ET on dt into UTC nanoseconds."""
    import pandas as pdt

    ts_utc = pdt.Timestamp(f"{dt} {hhmm}:00", tz="America/New_York").tz_convert("UTC")
    return int(ts_utc.value)


@dataclass(frozen=True)
class PreparedStreamSession:
    """Resolved stream session inputs for one websocket connection."""

    resolved_serving: ResolvedServing
    config: RuntimeConfig
    schema: pa.Schema
    fields: "list | None"
    dt: str
    start_time: str | None
    producer_latency_config: ProducerLatencyConfig | None


def prepare_stream_session(
    *,
    serving_registry: ServingRegistry,
    serving: str,
    perf_latency_jsonl: Path | None,
    perf_window_start_et: str | None,
    perf_window_end_et: str | None,
    perf_summary_every_bins: int,
) -> PreparedStreamSession:
    """Resolve serving and build deterministic runtime stream session config."""
    resolved_serving = serving_registry.resolve(serving)
    serving_spec = resolved_serving.spec
    runtime_snapshot = serving_spec.runtime_snapshot
    if "product_type" not in runtime_snapshot:
        raise ValueError(
            "Published serving runtime snapshot is missing required key: product_type"
        )
    if "symbol" not in runtime_snapshot:
        raise ValueError(
            "Published serving runtime snapshot is missing required key: symbol"
        )
    dt = serving_spec.stream_dt()
    start_time = serving_spec.stream_start_time()
    product_type = str(runtime_snapshot["product_type"])
    symbol = str(runtime_snapshot["symbol"])
    resolve_config(product_type, symbol)
    config = build_config_from_mapping(
        runtime_snapshot,
        source=f"serving_runtime_snapshot:{resolved_serving.serving_id}",
    )
    # Build fields from runtime_snapshot stream_schema if present
    raw_stream_schema = runtime_snapshot.get("stream_schema")
    if raw_stream_schema:
        from .serving_config import StreamFieldSpec
        fields = [StreamFieldSpec(**f) for f in raw_stream_schema]
    else:
        import warnings
        warnings.warn(
            "stream_schema missing from runtime_snapshot; using legacy VP field list",
            DeprecationWarning,
            stacklevel=2,
        )
        fields = None
    schema = grid_schema(fields)
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
    return PreparedStreamSession(
        resolved_serving=resolved_serving,
        config=config,
        schema=schema,
        fields=fields,
        dt=dt,
        start_time=start_time,
        producer_latency_config=producer_latency_cfg,
    )


async def run_stream_session(
    *,
    websocket: WebSocket,
    lake_root: Path,
    session: PreparedStreamSession,
) -> None:
    """Send fixed-bin dense-grid updates over websocket."""
    grid_count = 0
    resolved_serving = session.resolved_serving
    config = session.config
    schema = session.schema

    logger.info(
        "Stream connected: serving=%s serving_id=%s product_type=%s symbol=%s dt=%s start_time=%s radius=%d cell_width_ms=%d projection_horizons_bins=%s cfg=%s",
        resolved_serving.alias or resolved_serving.serving_id,
        resolved_serving.serving_id,
        config.product_type,
        config.symbol,
        session.dt,
        session.start_time,
        config.grid_radius_ticks,
        config.cell_width_ms,
        list(config.projection_horizons_bins),
        config.config_version,
    )
    if session.producer_latency_config is not None:
        logger.info(
            "Producer latency telemetry active: output=%s window_start_ns=%s window_end_ns=%s summary_every_bins=%d",
            session.producer_latency_config.output_path,
            session.producer_latency_config.window_start_ns,
            session.producer_latency_config.window_end_ns,
            session.producer_latency_config.summary_every_bins,
        )

    try:
        from ..models.vacuum_pressure.stream_pipeline import _build_vp_model_config
        runtime_config_payload = build_runtime_config_payload(
            config,
            schema,
            fields=session.fields,
            model_config=_build_vp_model_config(config),
            resolved_serving=resolved_serving,
        )
        await websocket.send_text(json.dumps(runtime_config_payload))

        async for grid in async_stream_events(
            lake_root=lake_root,
            config=config,
            dt=session.dt,
            start_time=session.start_time,
            producer_latency_config=session.producer_latency_config,
        ):
            grid_count += 1

            await websocket.send_text(json.dumps(build_grid_update_payload(grid)))
            await websocket.send_bytes(grid_to_arrow_ipc(grid, schema))

            if grid_count % 1000 == 0:
                logger.info(
                    "Fixed-bin dense-grid: %d updates sent (event_id=%d)",
                    grid_count,
                    grid["event_id"],
                )
    except WebSocketDisconnect:
        logger.info("Stream client disconnected")
    except Exception as exc:
        logger.error("Stream error: %s", exc, exc_info=True)
    finally:
        logger.info("Stream ended (%d fixed-bin updates sent)", grid_count)
