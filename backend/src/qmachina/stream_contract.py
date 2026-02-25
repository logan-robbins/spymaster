"""Wire contracts and serialization for qMachina streaming."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from .config import RuntimeConfig
from .gold_config import GoldFeatureConfig
from .serving_config import StreamFieldSpec, VisualizationConfig
from .stage_schema import SILVER_FLOAT_COLS, SILVER_INT_COL_DTYPES

if TYPE_CHECKING:
    from .serving_registry import ResolvedServing


_DTYPE_MAP = {"int8": pa.int8(), "int32": pa.int32(), "int64": pa.int64(), "float64": pa.float64()}

_INT_DTYPE_TO_PA: dict[Any, pa.DataType] = {
    np.int8: pa.int8(),
    np.int32: pa.int32(),
    np.int64: pa.int64(),
}

# Canonical silver wire schema: all silver columns in stable order.
_SILVER_SCHEMA_FIELDS: list[tuple[str, pa.DataType]] = (
    [("k", pa.int32())]
    + [(col, pa.float64()) for col in SILVER_FLOAT_COLS]
    + [
        (col, _INT_DTYPE_TO_PA[dtype])
        for col, dtype in SILVER_INT_COL_DTYPES.items()
        if col != "k"
    ]
)


def grid_schema(fields: "list[StreamFieldSpec] | None" = None) -> pa.Schema:
    """Return Arrow schema for dense-grid binary frames.

    Args:
        fields: Optional list of StreamFieldSpec from serving config.
            When provided, the schema is built from those specs.
            When None, uses the canonical silver schema.

    Returns:
        PyArrow Schema for Arrow IPC serialization.
    """
    if fields:
        return pa.schema([pa.field(f.name, _DTYPE_MAP[f.dtype]) for f in fields])
    return pa.schema([pa.field(name, dtype) for name, dtype in _SILVER_SCHEMA_FIELDS])


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
    config: RuntimeConfig,
    schema: pa.Schema,
    fields: "list[StreamFieldSpec] | None" = None,
    *,
    model_config: dict | None = None,
    resolved_serving: "ResolvedServing | None" = None,
) -> dict[str, Any]:
    """Build runtime_config control payload."""
    payload: dict[str, Any] = {
        "type": "runtime_config",
        **config.to_dict(),
        "stream_format": "dense_grid",
        "grid_schema_fields": (
            [{"name": f.name, "dtype": f.dtype, "role": f.role} for f in fields]
            if fields
            else [{"name": f.name, "dtype": str(f.type)} for f in schema]
        ),
        "grid_rows": 2 * config.grid_radius_ticks + 1,
    }
    if model_config is not None:
        payload["model_config"] = model_config
    if resolved_serving is not None:
        payload["serving"] = resolved_serving.spec.to_runtime_config_json(
            serving_name=resolved_serving.alias or resolved_serving.serving_id
        )

    # Gold feature config — fixes train/serve parity gap
    gold_cfg = GoldFeatureConfig.from_runtime_config(config)
    payload["gold_config"] = gold_cfg.model_dump()

    # Visualization and display metadata — prefer runtime_snapshot dict (registered serving);
    # fall back to direct attribute access for test mocks.
    viz_raw = None
    display_name = ""
    if resolved_serving is not None:
        snapshot = getattr(resolved_serving.spec, "runtime_snapshot", None)
        if isinstance(snapshot, dict):
            viz_raw = snapshot.get("visualization")
            display_name = snapshot.get("display_name", "") or ""
        if viz_raw is None:
            viz_attr = getattr(resolved_serving.spec, "visualization", None)
            if viz_attr is not None:
                viz_raw = viz_attr.model_dump()
        if not display_name:
            display_name = getattr(resolved_serving.spec, "display_name", "") or ""

    payload["display_name"] = display_name
    payload["visualization"] = viz_raw if viz_raw is not None else VisualizationConfig.default_heatmap().model_dump()

    # Gold DSL lineage fields (optional, present when serving spec has DSL binding)
    gold_dsl_spec_id = None
    gold_dsl_hash = None
    if resolved_serving is not None:
        snapshot = resolved_serving.spec.runtime_snapshot
        if isinstance(snapshot, dict):
            gold_dsl_spec_id = snapshot.get("gold_dsl_spec_id")
            gold_dsl_hash = snapshot.get("gold_dsl_hash")
        # Fall back to direct attribute access for ServingSpec (non-published)
        if gold_dsl_spec_id is None:
            gold_dsl_spec_id = getattr(resolved_serving.spec, "gold_dsl_spec_id", None)
        if gold_dsl_hash is None:
            gold_dsl_hash = getattr(resolved_serving.spec, "gold_dsl_hash", None)

    if gold_dsl_spec_id is not None:
        payload["gold_dsl_spec_id"] = gold_dsl_spec_id
    if gold_dsl_hash is not None:
        payload["gold_dsl_hash"] = gold_dsl_hash

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
