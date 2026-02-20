"""WebSocket server for canonical fixed-bin vacuum-pressure streaming."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import pyarrow as pa
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    VPRuntimeConfig,
    build_config_with_overrides,
    parse_projection_horizons_bins_override,
    resolve_config,
)
from .serving_config import ServingSpec
from .stream_pipeline import ProducerLatencyConfig, async_stream_events  # noqa: F401

logger = logging.getLogger(__name__)

_BASE_GRID_FIELDS: list[tuple[str, pa.DataType]] = [
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
    ("composite", pa.float64()),
    ("composite_d1", pa.float64()),
    ("composite_d2", pa.float64()),
    ("composite_d3", pa.float64()),
    ("flow_score", pa.float64()),
    ("flow_state_code", pa.int8()),
    ("best_ask_move_ticks", pa.int32()),
    ("best_bid_move_ticks", pa.int32()),
    ("ask_reprice_sign", pa.int8()),
    ("bid_reprice_sign", pa.int8()),
    ("microstate_id", pa.int8()),
    ("state5_code", pa.int8()),
    ("chase_up_flag", pa.int8()),
    ("chase_down_flag", pa.int8()),
    ("last_event_id", pa.int64()),
]


def _grid_schema(_config: VPRuntimeConfig) -> pa.Schema:
    return pa.schema([pa.field(name, dtype) for name, dtype in _BASE_GRID_FIELDS])


def _grid_to_arrow_ipc(grid_dict: dict[str, Any], schema: pa.Schema) -> bytes:
    record_batch = pa.RecordBatch.from_pylist(grid_dict["buckets"], schema=schema)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(record_batch)
    return sink.getvalue().to_pybytes()


def _safe_float(v: Any) -> float | None:
    """Convert to float, returning None for NaN/None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None


def _safe_int(v: Any) -> int | None:
    """Convert to int, returning None for NaN/None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if pd.isna(f) else int(f)
    except (ValueError, TypeError):
        return None


def _et_hhmm_to_utc_ns(dt: str, hhmm: str) -> int:
    """Convert HH:MM ET on dt into UTC nanoseconds."""
    import pandas as pdt

    ts_utc = pdt.Timestamp(f"{dt} {hhmm}:00", tz="America/New_York").tz_convert("UTC")
    return int(ts_utc.value)


def create_app(
    lake_root: Path | None = None,
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

    # ---- Experiment Browser REST API ----

    _manifest_cache: dict[str, dict[str, Any]] = {}

    def _load_manifest(dataset_id: str) -> dict[str, Any] | None:
        """Load and cache manifest.json for a dataset from vp_immutable."""
        if dataset_id in _manifest_cache:
            return _manifest_cache[dataset_id]
        manifest_path = lake_root / "research" / "vp_immutable" / dataset_id / "manifest.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path) as f:
            manifest = json.load(f)
        _manifest_cache[dataset_id] = manifest
        return manifest

    _SIGNAL_PARAM_TO_WS: dict[str, str] = {
        "zscore_window_bins": "state_model_zscore_window_bins",
        "zscore_min_periods": "state_model_zscore_min_periods",
        "tanh_scale": "state_model_tanh_scale",
        "d1_weight": "state_model_d1_weight",
        "d2_weight": "state_model_d2_weight",
        "d3_weight": "state_model_d3_weight",
        "center_exclusion_radius": "state_model_center_exclusion_radius",
        "spatial_decay_power": "state_model_spatial_decay_power",
        "bull_pressure_weight": "state_model_bull_pressure_weight",
        "bull_vacuum_weight": "state_model_bull_vacuum_weight",
        "bear_pressure_weight": "state_model_bear_pressure_weight",
        "bear_vacuum_weight": "state_model_bear_vacuum_weight",
        "mixed_weight": "state_model_mixed_weight",
        "enable_weighted_blend": "state_model_enable_weighted_blend",
    }
    _SIGNAL_PARAM_IGNORED: set[str] = {"cell_width_ms"}

    def _build_streaming_url(
        dataset_id: str,
        signal_name: str,
        signal_params: dict[str, Any],
    ) -> tuple[str | None, bool]:
        """Build a vacuum-pressure.html URL from run metadata.

        Returns (streaming_url, can_stream). Only derivative signals
        map to the state-model query params.
        """
        if signal_name != "derivative":
            return None, False
        if not isinstance(signal_params, dict):
            return None, False

        manifest = _load_manifest(dataset_id)
        if manifest is None:
            return None, False

        src = manifest.get("source_manifest", {})
        if not isinstance(src, dict):
            src = {}
        spec = manifest.get("spec", {})
        if not isinstance(spec, dict):
            spec = {}

        product_type = (
            src.get("product_type")
            or manifest.get("product_type")
            or spec.get("product_type")
            or "future_mbo"
        )
        symbol = (
            src.get("symbol")
            or manifest.get("symbol")
            or spec.get("symbol")
            or ""
        )
        dt = (
            src.get("dt")
            or manifest.get("dt")
            or spec.get("dt")
            or ""
        )
        start_et = (
            src.get("capture_start_et")
            or src.get("stream_start_time_hhmm")
            or manifest.get("start_time")
            or spec.get("start_time")
            or ""
        )
        if ":" not in start_et and len(start_et) == 4:
            start_et = f"{start_et[:2]}:{start_et[2:]}"
        elif ":" in start_et and len(start_et) > 5:
            start_et = start_et[:5]
        if not symbol or not dt or not start_et:
            return None, False

        params: dict[str, Any] = {
            "product_type": product_type,
            "symbol": symbol,
            "dt": dt,
            "start_time": start_et,
        }

        unknown_params = sorted(
            key
            for key in signal_params.keys()
            if key not in _SIGNAL_PARAM_TO_WS and key not in _SIGNAL_PARAM_IGNORED
        )
        if unknown_params:
            logger.error(
                "Cannot build derivative streaming URL for dataset=%s: unmapped signal params=%s",
                dataset_id,
                unknown_params,
            )
            return None, False

        for harness_key, ws_key in _SIGNAL_PARAM_TO_WS.items():
            if harness_key in signal_params:
                params[ws_key] = signal_params[harness_key]

        return f"/vacuum-pressure.html?{urlencode(params)}", True

    def _results_db():
        from ..experiment_harness.results_db import ResultsDB
        return ResultsDB(lake_root / "research" / "vp_harness" / "results")

    @app.get("/v1/experiments/runs")
    async def experiment_runs(
        signal: str | None = Query(None),
        dataset_id: str | None = Query(None),
        sort: str = Query("tp_rate"),
        min_signals: int = Query(5),
        top_n: int = Query(50),
    ) -> dict:
        """Return ranked experiment results with streaming URLs."""
        db = _results_db()
        best_df = db.query_best(signal=signal, dataset_id=dataset_id, min_signals=min_signals)

        if best_df.empty:
            return {"runs": [], "filters": {"signals": [], "datasets": []}}

        meta_df = db.query_runs()

        signal_params_col = "signal_params_json"
        if signal_params_col in meta_df.columns:
            params_lookup = dict(
                zip(meta_df["run_id"], meta_df[signal_params_col], strict=False)
            )
        else:
            params_lookup = {}
        timestamp_lookup: dict[str, str] = {}
        if "timestamp_utc" in meta_df.columns:
            timestamp_lookup = {
                str(run_id): str(ts)
                for run_id, ts in zip(
                    meta_df["run_id"], meta_df["timestamp_utc"], strict=False
                )
            }

        if sort in best_df.columns:
            ascending = sort not in ("tp_rate", "mean_pnl_ticks", "events_per_hour", "n_signals")
            best_df = best_df.sort_values(sort, ascending=ascending)

        best_df = best_df.head(top_n)

        runs = []
        for _, row in best_df.iterrows():
            run_id = row.get("run_id", "")
            sig_name = row.get("signal_name", "")
            raw_params = params_lookup.get(run_id, "{}")
            try:
                sig_params = json.loads(raw_params) if isinstance(raw_params, str) else raw_params
            except (json.JSONDecodeError, TypeError):
                sig_params = {}

            ds_id = row.get("dataset_id", "")
            streaming_url, can_stream = _build_streaming_url(ds_id, sig_name, sig_params)

            runs.append({
                "run_id": run_id,
                "signal_name": sig_name,
                "dataset_id": ds_id,
                "experiment_name": row.get("experiment_name", ""),
                "signal_params_json": raw_params if isinstance(raw_params, str) else json.dumps(raw_params),
                "threshold": _safe_float(row.get("threshold")),
                "cooldown_bins": _safe_int(row.get("cooldown_bins")),
                "tp_rate": _safe_float(row.get("tp_rate")),
                "sl_rate": _safe_float(row.get("sl_rate")),
                "timeout_rate": _safe_float(row.get("timeout_rate")),
                "n_signals": _safe_int(row.get("n_signals")),
                "mean_pnl_ticks": _safe_float(row.get("mean_pnl_ticks")),
                "events_per_hour": _safe_float(row.get("events_per_hour")),
                "eval_tp_ticks": _safe_int(row.get("eval_tp_ticks")),
                "eval_sl_ticks": _safe_int(row.get("eval_sl_ticks")),
                "timestamp_utc": timestamp_lookup.get(str(run_id), ""),
                "streaming_url": streaming_url,
                "can_stream": can_stream,
            })

        all_signals = (
            sorted(meta_df["signal_name"].astype(str).unique().tolist())
            if "signal_name" in meta_df.columns
            else []
        )
        all_datasets = (
            sorted(meta_df["dataset_id"].astype(str).unique().tolist())
            if "dataset_id" in meta_df.columns
            else []
        )

        return {
            "runs": runs,
            "filters": {
                "signals": all_signals,
                "datasets": all_datasets,
            },
        }

    @app.get("/v1/experiments/runs/{run_id}/detail")
    async def experiment_run_detail(run_id: str) -> dict:
        """Return full detail for one experiment run."""
        db = _results_db()
        meta_df = db.query_runs(run_id=run_id)
        if meta_df.empty:
            return {"error": f"Run {run_id} not found"}

        meta_row = meta_df.iloc[0].to_dict()
        for k, v in meta_row.items():
            if isinstance(v, float) and pd.isna(v):
                meta_row[k] = None

        raw_params = meta_row.get("signal_params_json", "{}")
        try:
            sig_params = json.loads(raw_params) if isinstance(raw_params, str) else raw_params
        except (json.JSONDecodeError, TypeError):
            sig_params = {}

        runs_path = lake_root / "research" / "vp_harness" / "results" / "runs.parquet"
        threshold_results = []
        if runs_path.exists():
            runs_df = pd.read_parquet(runs_path)
            run_rows = runs_df[runs_df["run_id"] == run_id]
            for _, r in run_rows.iterrows():
                row_dict = {}
                for c in r.index:
                    val = r[c]
                    if isinstance(val, float) and pd.isna(val):
                        row_dict[c] = None
                    elif hasattr(val, "item"):
                        row_dict[c] = val.item()
                    else:
                        row_dict[c] = val
                threshold_results.append(row_dict)

        ds_id = meta_row.get("dataset_id", "")
        sig_name = meta_row.get("signal_name", "")
        streaming_url, can_stream = _build_streaming_url(ds_id, sig_name, sig_params)

        meta_row["signal_params"] = sig_params
        for k, v in meta_row.items():
            if hasattr(v, "item"):
                meta_row[k] = v.item()

        return {
            "meta": meta_row,
            "threshold_results": threshold_results,
            "streaming_url": streaming_url,
            "can_stream": can_stream,
        }

    @app.websocket("/v1/vacuum-pressure/stream")
    async def vacuum_pressure_stream(
        websocket: WebSocket,
        product_type: str = "future_mbo",
        symbol: str = "MNQH6",
        dt: str = "2026-02-06",
        start_time: str | None = None,
        serving: str | None = None,
        projection_horizons_bins: str | None = None,
        state_model_enabled: bool | None = None,
        state_model_center_exclusion_radius: int | None = None,
        state_model_spatial_decay_power: float | None = None,
        state_model_zscore_window_bins: int | None = None,
        state_model_zscore_min_periods: int | None = None,
        state_model_tanh_scale: float | None = None,
        state_model_d1_weight: float | None = None,
        state_model_d2_weight: float | None = None,
        state_model_d3_weight: float | None = None,
        state_model_bull_pressure_weight: float | None = None,
        state_model_bull_vacuum_weight: float | None = None,
        state_model_bear_pressure_weight: float | None = None,
        state_model_bear_vacuum_weight: float | None = None,
        state_model_mixed_weight: float | None = None,
        state_model_enable_weighted_blend: bool | None = None,
    ) -> None:
        """Stream fixed-bin dense-grid updates from the canonical event engine."""
        await websocket.accept()

        try:
            config = resolve_config(product_type, symbol)
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

            # --- Serving spec overrides (applied before explicit URL params) ---
            serving_spec: ServingSpec | None = None
            if serving is not None:
                serving_spec = ServingSpec.load_by_name(serving, lake_root)
                serving_overrides = serving_spec.to_runtime_overrides()
                if serving_overrides:
                    config = build_config_with_overrides(config, serving_overrides)
                logger.info(
                    "VP serving spec loaded: name=%s pipeline=%s",
                    serving_spec.name,
                    serving_spec.pipeline,
                )

            # --- Explicit URL params (take precedence over serving) ---
            overrides: dict[str, Any] = {}
            if state_model_enabled is not None:
                overrides["state_model_enabled"] = state_model_enabled
            if state_model_center_exclusion_radius is not None:
                overrides["state_model_center_exclusion_radius"] = state_model_center_exclusion_radius
            if state_model_spatial_decay_power is not None:
                overrides["state_model_spatial_decay_power"] = state_model_spatial_decay_power
            if state_model_zscore_window_bins is not None:
                overrides["state_model_zscore_window_bins"] = state_model_zscore_window_bins
            if state_model_zscore_min_periods is not None:
                overrides["state_model_zscore_min_periods"] = state_model_zscore_min_periods
            if state_model_tanh_scale is not None:
                overrides["state_model_tanh_scale"] = state_model_tanh_scale
            if state_model_d1_weight is not None:
                overrides["state_model_d1_weight"] = state_model_d1_weight
            if state_model_d2_weight is not None:
                overrides["state_model_d2_weight"] = state_model_d2_weight
            if state_model_d3_weight is not None:
                overrides["state_model_d3_weight"] = state_model_d3_weight
            if state_model_bull_pressure_weight is not None:
                overrides["state_model_bull_pressure_weight"] = state_model_bull_pressure_weight
            if state_model_bull_vacuum_weight is not None:
                overrides["state_model_bull_vacuum_weight"] = state_model_bull_vacuum_weight
            if state_model_bear_pressure_weight is not None:
                overrides["state_model_bear_pressure_weight"] = state_model_bear_pressure_weight
            if state_model_bear_vacuum_weight is not None:
                overrides["state_model_bear_vacuum_weight"] = state_model_bear_vacuum_weight
            if state_model_mixed_weight is not None:
                overrides["state_model_mixed_weight"] = state_model_mixed_weight
            if state_model_enable_weighted_blend is not None:
                overrides["state_model_enable_weighted_blend"] = state_model_enable_weighted_blend
            if overrides:
                config = build_config_with_overrides(config, overrides)
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
            "VP fixed-bin stream connected: product_type=%s symbol=%s dt=%s start_time=%s radius=%d cell_width_ms=%d projection_horizons_bins=%s state_model_enabled=%s cfg=%s",
            config.product_type,
            config.symbol,
            dt,
            start_time,
            config.grid_radius_ticks,
            config.cell_width_ms,
            list(config.projection_horizons_bins),
            config.state_model_enabled,
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
            serving_spec=serving_spec,
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
    serving_spec: ServingSpec | None = None,
) -> None:
    """Send fixed-bin dense-grid updates over websocket."""
    grid_count = 0

    try:
        runtime_config_payload: dict[str, Any] = {
            "type": "runtime_config",
            **config.to_dict(),
            "state_model": {
                "name": "derivative",
                "enabled": config.state_model_enabled,
                "center_exclusion_radius": config.state_model_center_exclusion_radius,
                "spatial_decay_power": config.state_model_spatial_decay_power,
                "zscore_window_bins": config.state_model_zscore_window_bins,
                "zscore_min_periods": config.state_model_zscore_min_periods,
                "tanh_scale": config.state_model_tanh_scale,
                "d1_weight": config.state_model_d1_weight,
                "d2_weight": config.state_model_d2_weight,
                "d3_weight": config.state_model_d3_weight,
                "bull_pressure_weight": config.state_model_bull_pressure_weight,
                "bull_vacuum_weight": config.state_model_bull_vacuum_weight,
                "bear_pressure_weight": config.state_model_bear_pressure_weight,
                "bear_vacuum_weight": config.state_model_bear_vacuum_weight,
                "mixed_weight": config.state_model_mixed_weight,
                "enable_weighted_blend": config.state_model_enable_weighted_blend,
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
        }
        if serving_spec is not None:
            runtime_config_payload["serving"] = serving_spec.to_runtime_config_json()
        await websocket.send_text(json.dumps(runtime_config_payload))

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
                "state_model_name": grid.get("state_model_name"),
                "state_model_score": grid.get("state_model_score"),
                "state_model_ready": grid.get("state_model_ready"),
                "state_model_sample_count": grid.get("state_model_sample_count"),
                "state_model_base": grid.get("state_model_base"),
                "state_model_d1": grid.get("state_model_d1"),
                "state_model_d2": grid.get("state_model_d2"),
                "state_model_d3": grid.get("state_model_d3"),
                "state_model_z1": grid.get("state_model_z1"),
                "state_model_z2": grid.get("state_model_z2"),
                "state_model_z3": grid.get("state_model_z3"),
                "state_model_bull_intensity": grid.get("state_model_bull_intensity"),
                "state_model_bear_intensity": grid.get("state_model_bear_intensity"),
                "state_model_mixed_intensity": grid.get("state_model_mixed_intensity"),
                "state_model_dominant_state5_code": grid.get("state_model_dominant_state5_code"),
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
