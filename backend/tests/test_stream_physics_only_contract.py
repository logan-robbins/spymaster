"""Tests enforcing that the live stream is physics-only.

Scored outputs (flow_score, flow_state_code) and per-bin state model
computation results (state_model_*) must not appear in the stream.
Parameters for frontend-side computation are still sent once via
runtime_config, but the backend does not compute them per-bin.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Tuple

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.qmachina.config import RuntimeConfig
from src.qmachina.stream_contract import (
    build_grid_update_payload,
    build_runtime_config_payload,
    grid_schema,
)
from src.models.vacuum_pressure.stream_pipeline import stream_events

MBOEvent = Tuple[int, str, str, int, int, int, int]


def _test_config() -> RuntimeConfig:
    return RuntimeConfig(
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
        projection_horizons_bins=(1,),
        projection_horizons_ms=(100,),
        contract_multiplier=1.0,
        qty_unit="contracts",
        price_decimals=2,
        config_version="test",
    )


def _minimal_events() -> list[MBOEvent]:
    return [
        (1_000_000_000, "A", "B", 100, 10, 1, 0),
        (1_000_000_000, "A", "A", 101, 12, 2, 0),
        (1_120_000_000, "M", "A", 102, 12, 2, 0),
        (1_130_000_000, "M", "B", 101, 10, 1, 0),
    ]


def _iter_for(events: list[MBOEvent]):
    def _fake_iter(
        _lake_root: Path,
        _product_type: str,
        _symbol: str,
        _dt: str,
        skip_to_ns: int = 0,
    ) -> Iterator[MBOEvent]:
        for ev in events:
            if ev[0] >= skip_to_ns:
                yield ev

    return _fake_iter


# ---------------------------------------------------------------------------
# Arrow IPC schema
# ---------------------------------------------------------------------------


def test_arrow_schema_excludes_flow_score() -> None:
    schema = grid_schema()
    field_names = [f.name for f in schema]
    assert "flow_score" not in field_names


def test_arrow_schema_excludes_flow_state_code() -> None:
    schema = grid_schema()
    field_names = [f.name for f in schema]
    assert "flow_state_code" not in field_names


# ---------------------------------------------------------------------------
# grid_update payload
# ---------------------------------------------------------------------------


def _minimal_grid_dict() -> dict:
    """Minimal grid dict without state model outputs."""
    return {
        "ts_ns": 1_100_000_000,
        "bin_seq": 1,
        "bin_start_ns": 1_000_000_000,
        "bin_end_ns": 1_100_000_000,
        "bin_event_count": 2,
        "event_id": 10,
        "mid_price": 100.5,
        "spot_ref_price_int": 100,
        "best_bid_price_int": 99,
        "best_ask_price_int": 101,
        "book_valid": True,
        "ask_reprice_sign": 1,
        "bid_reprice_sign": 1,
        "microstate_id": 8,
        "chase_up_flag": 1,
        "chase_down_flag": 0,
    }


def test_grid_update_payload_excludes_all_state_model_fields() -> None:
    payload = build_grid_update_payload(_minimal_grid_dict())
    state_model_keys = [k for k in payload if k.startswith("state_model")]
    assert state_model_keys == [], f"Unexpected state_model keys: {state_model_keys}"


# ---------------------------------------------------------------------------
# stream_events grid output
# ---------------------------------------------------------------------------


def test_stream_events_grid_cols_exclude_flow_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.models.vacuum_pressure.stream_pipeline.iter_mbo_events",
        _iter_for(_minimal_events()),
    )
    grids = list(
        stream_events(
            lake_root=Path("/tmp"),
            config=_test_config(),
            dt="2026-02-06",
            start_time=None,
        )
    )
    assert len(grids) >= 1
    for grid in grids:
        assert "flow_score" not in grid["grid_cols"], (
            f"flow_score found in grid_cols at bin_seq={grid['bin_seq']}"
        )


def test_stream_events_grid_cols_exclude_flow_state_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.models.vacuum_pressure.stream_pipeline.iter_mbo_events",
        _iter_for(_minimal_events()),
    )
    grids = list(
        stream_events(
            lake_root=Path("/tmp"),
            config=_test_config(),
            dt="2026-02-06",
            start_time=None,
        )
    )
    assert len(grids) >= 1
    for grid in grids:
        assert "flow_state_code" not in grid["grid_cols"], (
            f"flow_state_code found in grid_cols at bin_seq={grid['bin_seq']}"
        )


def test_stream_events_grid_excludes_state_model_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.models.vacuum_pressure.stream_pipeline.iter_mbo_events",
        _iter_for(_minimal_events()),
    )
    grids = list(
        stream_events(
            lake_root=Path("/tmp"),
            config=_test_config(),
            dt="2026-02-06",
            start_time=None,
        )
    )
    assert len(grids) >= 1
    for grid in grids:
        state_model_keys = [k for k in grid if k.startswith("state_model")]
        assert state_model_keys == [], (
            f"Unexpected state_model keys at bin_seq={grid['bin_seq']}: {state_model_keys}"
        )


# ---------------------------------------------------------------------------
# Gold field exclusion: pressure_variant, composite*, state5_code must NOT
# appear in the live stream (they are computed by gold_builder / frontend).
# ---------------------------------------------------------------------------

_GOLD_FIELDS = [
    "pressure_variant",
    "vacuum_variant",
    "composite",
    "composite_d1",
    "composite_d2",
    "composite_d3",
    "state5_code",
]


@pytest.mark.parametrize("field", _GOLD_FIELDS)
def test_arrow_schema_excludes_gold_field(field: str) -> None:
    schema = grid_schema()
    field_names = [f.name for f in schema]
    assert field not in field_names, f"Gold field '{field}' found in Arrow wire schema"


@pytest.mark.parametrize("field", _GOLD_FIELDS)
def test_stream_events_grid_cols_exclude_gold_field(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    monkeypatch.setattr(
        "src.models.vacuum_pressure.stream_pipeline.iter_mbo_events",
        _iter_for(_minimal_events()),
    )
    grids = list(
        stream_events(
            lake_root=Path("/tmp"),
            config=_test_config(),
            dt="2026-02-06",
            start_time=None,
        )
    )
    assert len(grids) >= 1
    for grid in grids:
        assert field not in grid["grid_cols"], (
            f"Gold field '{field}' found in stream grid_cols at bin_seq={grid['bin_seq']}"
        )
