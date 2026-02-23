from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.vacuum_pressure.config import VPRuntimeConfig
from src.vacuum_pressure.event_engine import AbsoluteTickEngine
from src.vacuum_pressure.spectrum import IndependentCellSpectrum
from src.vacuum_pressure.stream_pipeline import stream_events

MBOEvent = Tuple[int, str, str, int, int, int, int]


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
        n_absolute_ticks=20,
        flow_windows=(2, 4),
        flow_rollup_weights=(1.0, 1.0),
        flow_derivative_weights=(0.55, 0.30, 0.15),
        flow_tanh_scale=3.0,
        flow_neutral_threshold=0.15,
        flow_zscore_window_bins=8,
        flow_zscore_min_periods=2,
        projection_horizons_bins=(1, 2),
        projection_horizons_ms=(100, 200),
        contract_multiplier=1.0,
        qty_unit="contracts",
        price_decimals=2,
        config_version="test",
    )


def _synthetic_events() -> list[MBOEvent]:
    return [
        (1_000_000_000, "A", "B", 100, 10, 1, 0),
        (1_050_000_000, "A", "A", 101, 12, 2, 0),
        (1_120_000_000, "A", "B", 100, 5, 3, 0),
        (1_250_000_000, "C", "B", 100, 0, 1, 0),
    ]


def _fake_iter_mbo_events(
    _lake_root: Path,
    _product_type: str,
    _symbol: str,
    _dt: str,
    skip_to_ns: int = 0,
) -> Iterator[MBOEvent]:
    for ev in _synthetic_events():
        if ev[0] >= skip_to_ns:
            yield ev


# ---- AbsoluteTickEngine unit tests ----


def test_absolute_tick_engine_basic_lifecycle() -> None:
    """Engine initializes, processes events, and produces valid state."""
    engine = AbsoluteTickEngine(n_ticks=20, tick_int=1)

    # Add bid and ask to establish BBO and anchor
    engine.update(
        ts_ns=1_000_000_000, action="A", side="B",
        price_int=100, size=10, order_id=1, flags=0,
    )
    engine.update(
        ts_ns=1_000_000_000, action="A", side="A",
        price_int=101, size=12, order_id=2, flags=0,
    )

    assert engine.event_count == 2
    assert engine.best_bid_price_int == 100
    assert engine.best_ask_price_int == 101
    assert engine.book_valid is True
    assert engine.anchor_tick_idx >= 0
    assert engine.spot_ref_price_int > 0


def test_absolute_tick_engine_incremental_bbo() -> None:
    """BBO is updated incrementally without full scan."""
    engine = AbsoluteTickEngine(n_ticks=20, tick_int=1)

    engine.update(ts_ns=100, action="A", side="B", price_int=10, size=5, order_id=1, flags=0)
    engine.update(ts_ns=100, action="A", side="A", price_int=15, size=5, order_id=2, flags=0)
    assert engine.best_bid_price_int == 10
    assert engine.best_ask_price_int == 15

    # Better bid
    engine.update(ts_ns=200, action="A", side="B", price_int=11, size=3, order_id=3, flags=0)
    assert engine.best_bid_price_int == 11

    # Better ask
    engine.update(ts_ns=200, action="A", side="A", price_int=14, size=3, order_id=4, flags=0)
    assert engine.best_ask_price_int == 14

    # Cancel best bid — should scan for next best
    engine.update(ts_ns=300, action="C", side="B", price_int=11, size=0, order_id=3, flags=0)
    assert engine.best_bid_price_int == 10

    # Cancel best ask
    engine.update(ts_ns=300, action="C", side="A", price_int=14, size=0, order_id=4, flags=0)
    assert engine.best_ask_price_int == 15


def test_absolute_tick_engine_no_grid_shift() -> None:
    """Anchor never changes after establishment. No grid shift."""
    engine = AbsoluteTickEngine(n_ticks=100, tick_int=1)

    # Establish anchor
    engine.update(ts_ns=100, action="A", side="B", price_int=50, size=10, order_id=1, flags=0)
    engine.update(ts_ns=100, action="A", side="A", price_int=51, size=10, order_id=2, flags=0)
    anchor = engine.anchor_tick_idx

    # Move BBO significantly
    engine.update(ts_ns=200, action="A", side="B", price_int=60, size=10, order_id=3, flags=0)
    engine.update(ts_ns=200, action="A", side="A", price_int=61, size=10, order_id=4, flags=0)

    # Anchor should not change
    assert engine.anchor_tick_idx == anchor


def test_absolute_tick_engine_derivatives_update() -> None:
    """Derivatives are updated at touched ticks after 2+ events with dt > 0."""
    engine = AbsoluteTickEngine(n_ticks=100, tick_int=1)

    # Establish anchor (both bid and ask needed)
    engine.update(ts_ns=1_000_000_000, action="A", side="B", price_int=50, size=10, order_id=1, flags=0)
    engine.update(ts_ns=1_000_000_000, action="A", side="A", price_int=51, size=10, order_id=2, flags=0)

    # First post-anchor event at price 50 — sets last_ts_ns for this tick
    engine.update(ts_ns=2_000_000_000, action="A", side="B", price_int=50, size=5, order_id=3, flags=0)

    # Second post-anchor event at price 50 with dt > 0 — computes derivatives
    engine.update(ts_ns=3_000_000_000, action="A", side="B", price_int=50, size=5, order_id=4, flags=0)

    idx = engine._price_to_idx(50)
    assert idx is not None
    arrays = engine.grid_snapshot_arrays()
    assert arrays["v_add"][idx] > 0.0
    assert arrays["pressure_variant"][idx] > 0.0


def test_absolute_tick_engine_rest_depth() -> None:
    """rest_depth reflects current book depth at each price level."""
    engine = AbsoluteTickEngine(n_ticks=20, tick_int=1)

    engine.update(ts_ns=100, action="A", side="B", price_int=10, size=5, order_id=1, flags=0)
    engine.update(ts_ns=100, action="A", side="A", price_int=11, size=8, order_id=2, flags=0)

    idx_10 = engine._price_to_idx(10)
    idx_11 = engine._price_to_idx(11)
    assert idx_10 is not None and idx_11 is not None

    assert engine._rest_depth[idx_10] == 5.0
    assert engine._rest_depth[idx_11] == 8.0

    # Cancel partial
    engine._orders[1].qty = 5  # ensure known state
    engine.update(ts_ns=200, action="C", side="B", price_int=10, size=0, order_id=1, flags=0)
    assert engine._rest_depth[idx_10] == 0.0


def test_absolute_tick_engine_book_serialization() -> None:
    """Export/import preserves book state and anchor."""
    engine = AbsoluteTickEngine(n_ticks=20, tick_int=1)

    engine.update(ts_ns=100, action="A", side="B", price_int=10, size=5, order_id=1, flags=0)
    engine.update(ts_ns=100, action="A", side="A", price_int=11, size=8, order_id=2, flags=0)

    data = engine.export_book_state()

    engine2 = AbsoluteTickEngine(n_ticks=20, tick_int=1)
    engine2.import_book_state(data)

    assert engine2.order_count == engine.order_count
    assert engine2.best_bid_price_int == engine.best_bid_price_int
    assert engine2.best_ask_price_int == engine.best_ask_price_int
    assert engine2.anchor_tick_idx == engine.anchor_tick_idx


def test_absolute_tick_engine_window_snapshot() -> None:
    """spot_to_idx maps correctly for serve-time windowing."""
    engine = AbsoluteTickEngine(n_ticks=100, tick_int=1)

    engine.update(ts_ns=100, action="A", side="B", price_int=50, size=10, order_id=1, flags=0)
    engine.update(ts_ns=100, action="A", side="A", price_int=51, size=10, order_id=2, flags=0)

    spot_int = engine.spot_ref_price_int
    center_idx = engine.spot_to_idx(spot_int)
    assert center_idx is not None

    # Window of ±5 around spot
    radius = 5
    start = center_idx - radius
    end = center_idx + radius + 1
    assert start >= 0
    assert end <= engine.n_ticks

    arrays = engine.grid_snapshot_arrays()
    window = arrays["rest_depth"][start:end]
    assert window.shape == (2 * radius + 1,)


# ---- Spectrum tests (unchanged) ----


def test_independent_cell_spectrum_kernel_independence() -> None:
    kernel = IndependentCellSpectrum(
        n_cells=3,
        windows=[2, 4],
        rollup_weights=[1.0, 1.0],
        derivative_weights=[0.55, 0.30, 0.15],
        tanh_scale=3.0,
        neutral_threshold=0.15,
        zscore_window_bins=8,
        zscore_min_periods=2,
        projection_horizons_ms=[100, 200],
        default_dt_s=0.1,
    )

    base_p = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    base_v = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    _ = kernel.update(100_000_000, base_p, base_v)
    out_a = kernel.update(200_000_000, base_p, base_v)

    changed_p = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    out_b = kernel.update(300_000_000, changed_p, base_v)

    assert np.isfinite(out_b.score).all()
    assert np.all(np.abs(out_b.score) <= 1.0 + 1e-9)

    assert out_a.score[0] == pytest.approx(out_b.score[0], abs=1e-9)
    assert out_a.score[2] == pytest.approx(out_b.score[2], abs=1e-9)


def test_independent_cell_spectrum_state_mapping_consistent() -> None:
    kernel = IndependentCellSpectrum(
        n_cells=3,
        windows=[2, 4],
        rollup_weights=[1.0, 1.0],
        derivative_weights=[0.55, 0.30, 0.15],
        tanh_scale=3.0,
        neutral_threshold=0.15,
        zscore_window_bins=8,
        zscore_min_periods=2,
        projection_horizons_ms=[100],
        default_dt_s=0.1,
    )

    for i in range(1, 8):
        p = np.array([1.0 + i, 1.0, 1.0], dtype=np.float64)
        v = np.array([1.0, 1.0 + i, 1.0], dtype=np.float64)
        out = kernel.update(i * 100_000_000, p, v)

    threshold = 0.15
    score = out.score
    state = out.state_code

    assert np.all(state[score >= threshold] == 1)
    assert np.all(state[score <= -threshold] == -1)
    neutral_mask = np.abs(score) < threshold
    assert np.all(state[neutral_mask] == 0)


# ---- Pipeline integration tests ----


def test_stream_events_emits_fixed_bins_with_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.vacuum_pressure.stream_pipeline.iter_mbo_events",
        _fake_iter_mbo_events,
    )

    config = _test_config()
    grids = list(
        stream_events(
            lake_root=Path("/tmp"),
            config=config,
            dt="2026-02-06",
            start_time=None,
        )
    )

    assert len(grids) == 3
    assert [int(g["bin_seq"]) for g in grids] == [0, 1, 2]
    assert [int(g["bin_event_count"]) for g in grids] == [2, 1, 1]

    expected_width_ns = config.cell_width_ms * 1_000_000
    for g in grids:
        assert int(g["bin_end_ns"]) - int(g["bin_start_ns"]) == expected_width_ns


def test_stream_events_bucket_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.vacuum_pressure.stream_pipeline.iter_mbo_events",
        _fake_iter_mbo_events,
    )

    config = _test_config()
    grids = list(
        stream_events(
            lake_root=Path("/tmp"),
            config=config,
            dt="2026-02-06",
            start_time=None,
        )
    )

    expected_rows = 2 * config.grid_radius_ticks + 1
    for g in grids:
        cols = g["grid_cols"]
        assert len(cols["k"]) == expected_rows
        assert "flow_score" not in cols
        assert "flow_state_code" not in cols
        composite_d1 = np.asarray(cols["composite_d1"], dtype=np.float64)
        assert np.all(np.isfinite(composite_d1))
