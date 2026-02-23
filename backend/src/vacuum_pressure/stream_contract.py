"""Wire contracts and serialization for VP streaming."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from .config import VPRuntimeConfig

if TYPE_CHECKING:
    from .serving_registry import ResolvedServing


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


def grid_schema(_config: VPRuntimeConfig) -> pa.Schema:
    """Return Arrow schema for dense-grid binary frames."""
    return pa.schema([pa.field(name, dtype) for name, dtype in _BASE_GRID_FIELDS])


def grid_to_arrow_ipc(grid_dict: dict[str, Any], schema: pa.Schema) -> bytes:
    """Serialize one dense grid payload to Arrow IPC stream bytes."""
    grid_cols = grid_dict.get("grid_cols")
    if not isinstance(grid_cols, dict):
        raise KeyError("grid_dict must contain 'grid_cols' mapping")

    arrays: list[pa.Array] = []
    n_rows: int | None = None
    for field in schema:
        name = field.name
        if name not in grid_cols:
            raise KeyError(f"grid_cols missing required field '{name}'")
        arr = pa.array(grid_cols[name], type=field.type)
        if n_rows is None:
            n_rows = len(arr)
        elif len(arr) != n_rows:
            raise ValueError(
                f"grid_cols field '{name}' has length {len(arr)} != {n_rows}"
            )
        arrays.append(arr)

    record_batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(record_batch)
    return sink.getvalue().to_pybytes()


def build_runtime_config_payload(
    config: VPRuntimeConfig,
    schema: pa.Schema,
    *,
    resolved_serving: "ResolvedServing | None" = None,
) -> dict[str, Any]:
    """Build runtime_config control payload."""
    payload: dict[str, Any] = {
        "type": "runtime_config",
        **config.to_dict(),
        "state_model": {
            "name": "derivative",
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
        },
        "mode": "pre_prod",
        "deployment_stage": "pre_prod",
        "stream_format": "dense_grid",
        "grid_schema_fields": [f.name for f in schema],
        "grid_rows": 2 * config.grid_radius_ticks + 1,
        "effective_config_hash": config.config_version,
    }
    if resolved_serving is not None:
        payload["serving"] = resolved_serving.spec.to_runtime_config_json(
            serving_name=resolved_serving.alias or resolved_serving.serving_id
        )
    return payload


def build_grid_update_payload(grid: dict[str, Any]) -> dict[str, Any]:
    """Build per-bin grid_update control payload."""
    return {
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
    }
