from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.vacuum_pressure.config import VPRuntimeConfig
from src.vacuum_pressure.server import _grid_schema, _grid_to_arrow_ipc


def _test_config() -> VPRuntimeConfig:
    return VPRuntimeConfig(
        product_type="future_mbo",
        symbol="TESTH6",
        symbol_root="TEST",
        price_scale=1e-9,
        tick_size=1e-9,
        bucket_size_dollars=1e-9,
        rel_tick_size=1e-9,
        grid_radius_ticks=2,
        cell_width_ms=100,
        n_absolute_ticks=32,
        flow_windows=(2, 4),
        flow_rollup_weights=(1.0, 1.0),
        flow_derivative_weights=(0.55, 0.30, 0.15),
        flow_tanh_scale=3.0,
        flow_neutral_threshold=0.15,
        flow_zscore_window_bins=8,
        flow_zscore_min_periods=2,
        projection_horizons_bins=(1, 3),
        projection_horizons_ms=(100, 300),
        contract_multiplier=1.0,
        qty_unit="contracts",
        price_decimals=2,
        config_version="test",
    )


def test_grid_to_arrow_ipc_serializes_expected_rows() -> None:
    schema = _grid_schema(_test_config())
    grid = {
        "grid_cols": {
            "k": [-1, 0],
            "pressure_variant": [0.1, 0.2],
            "vacuum_variant": [-0.2, -0.1],
            "add_mass": [1.0, 1.5],
            "pull_mass": [2.0, 2.5],
            "fill_mass": [3.0, 3.5],
            "rest_depth": [4.0, 4.5],
            "bid_depth": [2.5, 2.0],
            "ask_depth": [1.5, 2.5],
            "v_add": [5.0, 5.5],
            "v_pull": [6.0, 6.5],
            "v_fill": [7.0, 7.5],
            "v_rest_depth": [8.0, 8.5],
            "v_bid_depth": [5.0, 3.5],
            "v_ask_depth": [3.0, 5.0],
            "a_add": [9.0, 9.5],
            "a_pull": [10.0, 10.5],
            "a_fill": [11.0, 11.5],
            "a_rest_depth": [12.0, 12.5],
            "a_bid_depth": [7.0, 5.5],
            "a_ask_depth": [5.0, 7.0],
            "j_add": [13.0, 13.5],
            "j_pull": [14.0, 14.5],
            "j_fill": [15.0, 15.5],
            "j_rest_depth": [16.0, 16.5],
            "j_bid_depth": [9.0, 7.5],
            "j_ask_depth": [7.0, 9.0],
            "composite": [0.0, 0.0],
            "composite_d1": [0.0, 0.0],
            "composite_d2": [0.0, 0.0],
            "composite_d3": [0.0, 0.0],
            "flow_score": [-0.25, 0.5],
            "flow_state_code": [-1, 1],
            "best_ask_move_ticks": [1, 1],
            "best_bid_move_ticks": [1, 1],
            "ask_reprice_sign": [1, 1],
            "bid_reprice_sign": [1, 1],
            "microstate_id": [8, 8],
            "state5_code": [2, 0],
            "chase_up_flag": [1, 1],
            "chase_down_flag": [0, 0],
            "last_event_id": [101, 102],
        },
    }

    payload = _grid_to_arrow_ipc(grid, schema)
    table = pa.ipc.open_stream(pa.py_buffer(payload)).read_all()

    assert table.schema.equals(schema)
    assert table.num_rows == 2
    assert table["k"].to_pylist() == [-1, 0]
    assert table["flow_state_code"].to_pylist() == [-1, 1]
    assert table["last_event_id"].to_pylist() == [101, 102]
