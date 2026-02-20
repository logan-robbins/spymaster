"""WebSocket server for canonical fixed-bin vacuum-pressure streaming."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    VPRuntimeConfig,
    build_config_with_overrides,
    parse_projection_horizons_bins_override,
    resolve_config,
)
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
    ("bid_depth", pa.float64()),
    ("ask_depth", pa.float64()),
    ("v_add", pa.float64()),
    ("v_pull", pa.float64()),
    ("v_fill", pa.float64()),
    ("v_rest_depth", pa.float64()),
    ("v_bid_depth", pa.float64()),
    ("v_ask_depth", pa.float64()),
    ("a_add", pa.float64()),
    ("a_pull", pa.float64()),
    ("a_fill", pa.float64()),
    ("a_rest_depth", pa.float64()),
    ("a_bid_depth", pa.float64()),
    ("a_ask_depth", pa.float64()),
    ("j_add", pa.float64()),
    ("j_pull", pa.float64()),
    ("j_fill", pa.float64()),
    ("j_rest_depth", pa.float64()),
    ("j_bid_depth", pa.float64()),
    ("j_ask_depth", pa.float64()),
    ("spectrum_score", pa.float64()),
    ("spectrum_state_code", pa.int8()),
    ("best_ask_move_ticks", pa.int32()),
    ("best_bid_move_ticks", pa.int32()),
    ("ask_reprice_sign", pa.int8()),
    ("bid_reprice_sign", pa.int8()),
    ("perm_microstate_id", pa.int8()),
    ("perm_state5_code", pa.int8()),
    ("chase_up_flag", pa.int8()),
    ("chase_down_flag", pa.int8()),
    ("last_event_id", pa.int64()),
]


def _grid_schema(_config: VPRuntimeConfig) -> pa.Schema:
    return pa.schema([pa.field(name, dtype) for name, dtype in _BASE_GRID_FIELDS])


def _grid_to_arrow_ipc(grid_dict: Dict[str, Any], schema: pa.Schema) -> bytes:
    record_batch = pa.RecordBatch.from_pylist(grid_dict["buckets"], schema=schema)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(record_batch)
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
    projection_horizons_bins_override: str | None = None,
    projection_use_cubic: bool = False,
    projection_cubic_scale: float = 1.0 / 6.0,
    projection_damping_lambda: float = 0.0,
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
    if projection_cubic_scale < 0.0:
        raise ValueError(f"projection_cubic_scale must be >= 0, got {projection_cubic_scale}")
    if projection_damping_lambda < 0.0:
        raise ValueError(
            f"projection_damping_lambda must be >= 0, got {projection_damping_lambda}"
        )
    default_projection_horizons_bins = parse_projection_horizons_bins_override(
        projection_horizons_bins_override
    )

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
        projection_horizons_bins: str | None = None,
        perm_runtime_enabled: bool | None = None,
        perm_center_exclusion_radius: int | None = None,
        perm_spatial_decay_power: float | None = None,
        perm_zscore_window_bins: int | None = None,
        perm_zscore_min_periods: int | None = None,
        perm_tanh_scale: float | None = None,
        perm_d1_weight: float | None = None,
        perm_d2_weight: float | None = None,
        perm_d3_weight: float | None = None,
        perm_bull_pressure_weight: float | None = None,
        perm_bull_vacuum_weight: float | None = None,
        perm_bear_pressure_weight: float | None = None,
        perm_bear_vacuum_weight: float | None = None,
        perm_mixed_weight: float | None = None,
        perm_enable_weighted_blend: bool | None = None,
    ) -> None:
        """Stream fixed-bin dense-grid updates from the canonical event engine."""
        await websocket.accept()

        try:
            config = resolve_config(product_type, symbol, products_yaml_path)
            request_projection_horizons_bins = (
                parse_projection_horizons_bins_override(projection_horizons_bins)
                if projection_horizons_bins is not None
                else default_projection_horizons_bins
            )
            if request_projection_horizons_bins is not None:
                config = build_config_with_overrides(
                    config,
                    {
                        "projection_horizons_bins": list(request_projection_horizons_bins),
                    },
                )
            perm_overrides: dict[str, Any] = {}
            if perm_runtime_enabled is not None:
                perm_overrides["perm_runtime_enabled"] = perm_runtime_enabled
            if perm_center_exclusion_radius is not None:
                perm_overrides["perm_center_exclusion_radius"] = perm_center_exclusion_radius
            if perm_spatial_decay_power is not None:
                perm_overrides["perm_spatial_decay_power"] = perm_spatial_decay_power
            if perm_zscore_window_bins is not None:
                perm_overrides["perm_zscore_window_bins"] = perm_zscore_window_bins
            if perm_zscore_min_periods is not None:
                perm_overrides["perm_zscore_min_periods"] = perm_zscore_min_periods
            if perm_tanh_scale is not None:
                perm_overrides["perm_tanh_scale"] = perm_tanh_scale
            if perm_d1_weight is not None:
                perm_overrides["perm_d1_weight"] = perm_d1_weight
            if perm_d2_weight is not None:
                perm_overrides["perm_d2_weight"] = perm_d2_weight
            if perm_d3_weight is not None:
                perm_overrides["perm_d3_weight"] = perm_d3_weight
            if perm_bull_pressure_weight is not None:
                perm_overrides["perm_bull_pressure_weight"] = perm_bull_pressure_weight
            if perm_bull_vacuum_weight is not None:
                perm_overrides["perm_bull_vacuum_weight"] = perm_bull_vacuum_weight
            if perm_bear_pressure_weight is not None:
                perm_overrides["perm_bear_pressure_weight"] = perm_bear_pressure_weight
            if perm_bear_vacuum_weight is not None:
                perm_overrides["perm_bear_vacuum_weight"] = perm_bear_vacuum_weight
            if perm_mixed_weight is not None:
                perm_overrides["perm_mixed_weight"] = perm_mixed_weight
            if perm_enable_weighted_blend is not None:
                perm_overrides["perm_enable_weighted_blend"] = perm_enable_weighted_blend
            if perm_overrides:
                config = build_config_with_overrides(config, perm_overrides)
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
            "VP fixed-bin stream connected: product_type=%s symbol=%s dt=%s start_time=%s radius=%d cell_width_ms=%d projection_horizons_bins=%s perm_runtime_enabled=%s cfg=%s",
            config.product_type,
            config.symbol,
            dt,
            start_time,
            config.grid_radius_ticks,
            config.cell_width_ms,
            list(config.projection_horizons_bins),
            config.perm_runtime_enabled,
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
            projection_use_cubic=projection_use_cubic,
            projection_cubic_scale=projection_cubic_scale,
            projection_damping_lambda=projection_damping_lambda,
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
    projection_use_cubic: bool = False,
    projection_cubic_scale: float = 1.0 / 6.0,
    projection_damping_lambda: float = 0.0,
) -> None:
    """Send fixed-bin dense-grid updates over websocket."""
    grid_count = 0

    try:
        await websocket.send_text(json.dumps({
            "type": "runtime_config",
            **config.to_dict(),
            "runtime_model": {
                "name": "perm_derivative",
                "enabled": config.perm_runtime_enabled,
                "center_exclusion_radius": config.perm_center_exclusion_radius,
                "spatial_decay_power": config.perm_spatial_decay_power,
                "zscore_window_bins": config.perm_zscore_window_bins,
                "zscore_min_periods": config.perm_zscore_min_periods,
                "tanh_scale": config.perm_tanh_scale,
                "d1_weight": config.perm_d1_weight,
                "d2_weight": config.perm_d2_weight,
                "d3_weight": config.perm_d3_weight,
                "bull_pressure_weight": config.perm_bull_pressure_weight,
                "bull_vacuum_weight": config.perm_bull_vacuum_weight,
                "bear_pressure_weight": config.perm_bear_pressure_weight,
                "bear_vacuum_weight": config.perm_bear_vacuum_weight,
                "mixed_weight": config.perm_mixed_weight,
                "enable_weighted_blend": config.perm_enable_weighted_blend,
            },
            "projection_model": {
                "use_cubic": projection_use_cubic,
                "cubic_scale": projection_cubic_scale,
                "damping_lambda": projection_damping_lambda,
            },
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
            projection_use_cubic=projection_use_cubic,
            projection_cubic_scale=projection_cubic_scale,
            projection_damping_lambda=projection_damping_lambda,
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
                "runtime_model_name": grid.get("runtime_model_name"),
                "runtime_model_score": grid.get("runtime_model_score"),
                "runtime_model_ready": grid.get("runtime_model_ready"),
                "runtime_model_sample_count": grid.get("runtime_model_sample_count"),
                "runtime_model_base": grid.get("runtime_model_base"),
                "runtime_model_d1": grid.get("runtime_model_d1"),
                "runtime_model_d2": grid.get("runtime_model_d2"),
                "runtime_model_d3": grid.get("runtime_model_d3"),
                "runtime_model_z1": grid.get("runtime_model_z1"),
                "runtime_model_z2": grid.get("runtime_model_z2"),
                "runtime_model_z3": grid.get("runtime_model_z3"),
                "runtime_model_bull_intensity": grid.get("runtime_model_bull_intensity"),
                "runtime_model_bear_intensity": grid.get("runtime_model_bear_intensity"),
                "runtime_model_mixed_intensity": grid.get("runtime_model_mixed_intensity"),
                "runtime_model_dominant_state5_code": grid.get("runtime_model_dominant_state5_code"),
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
