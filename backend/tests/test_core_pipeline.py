from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.vacuum_pressure.config import VPRuntimeConfig
from src.vacuum_pressure.core_pipeline import stream_core_events

MBOEvent = Tuple[int, str, str, int, int, int, int]


def _test_config(n_absolute_ticks: int = 20) -> VPRuntimeConfig:
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
        n_absolute_ticks=n_absolute_ticks,
        spectrum_windows=(2, 4),
        spectrum_rollup_weights=(1.0, 1.0),
        spectrum_derivative_weights=(0.55, 0.30, 0.15),
        spectrum_tanh_scale=3.0,
        spectrum_threshold_neutral=0.15,
        zscore_window_bins=8,
        zscore_min_periods=2,
        projection_horizons_bins=(1, 2),
        projection_horizons_ms=(100, 200),
        contract_multiplier=1.0,
        qty_unit="contracts",
        price_decimals=2,
        config_version="test",
    )


def _events_core_only_bids() -> list[MBOEvent]:
    return [
        (1_000_000_000, "A", "B", 100, 10, 1, 0),
        (1_050_000_000, "A", "B", 101, 12, 2, 0),
        (1_120_000_000, "A", "B", 100, 5, 3, 0),
        (1_250_000_000, "C", "B", 100, 0, 1, 0),
    ]


def _events_out_of_range() -> list[MBOEvent]:
    return [
        (1_000_000_000, "A", "B", 100, 10, 1, 0),  # anchor=100
        (1_100_000_000, "A", "B", 200, 5, 2, 0),   # outside [90..109] when n=20
    ]


def _events_soft_reanchor_event_gate() -> list[MBOEvent]:
    step_ns = 110_000_000  # 110ms => one emitted bin per event
    t0 = 1_000_000_000
    return [
        (t0 + 0 * step_ns, "A", "B", 100, 10, 1, 0),  # initial anchor near 100
        (t0 + 1 * step_ns, "A", "A", 101, 10, 2, 0),  # establish initial BBO
        (t0 + 2 * step_ns, "C", "B", 100, 0, 1, 0),   # clear old BBO
        (t0 + 3 * step_ns, "C", "A", 101, 0, 2, 0),
        (t0 + 4 * step_ns, "A", "B", 200, 7, 3, 0),   # out-of-range under old anchor
        (t0 + 5 * step_ns, "A", "A", 201, 7, 4, 0),   # out-of-range under old anchor
        (t0 + 6 * step_ns, "A", "B", 200, 5, 5, 0),   # processed after soft re-anchor
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


def test_stream_core_events_emits_full_grid_without_radius_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.vacuum_pressure.core_pipeline.iter_mbo_events",
        _iter_for(_events_core_only_bids()),
    )

    config = _test_config(n_absolute_ticks=20)
    grids = list(
        stream_core_events(
            lake_root=Path("/tmp"),
            config=config,
            dt="2026-02-06",
            start_time=None,
            fail_on_out_of_range=True,
        )
    )

    assert len(grids) == 3
    expected_width_ns = config.cell_width_ms * 1_000_000

    for grid in grids:
        assert "buckets" not in grid
        assert int(grid["n_rows"]) == config.n_absolute_ticks
        assert int(grid["bin_end_ns"]) - int(grid["bin_start_ns"]) == expected_width_ns
        cols = grid["columns"]
        assert isinstance(cols, dict)
        assert set(cols.keys()) >= {"pressure_variant", "vacuum_variant", "last_event_id"}
        assert all(arr.shape == (config.n_absolute_ticks,) for arr in cols.values())
        assert np.isfinite(cols["pressure_variant"]).all()
        assert np.isfinite(cols["vacuum_variant"]).all()

    # Anchor is seeded from first price event (100)
    first_grid = grids[0]
    assert int(first_grid["anchor_tick_idx"]) == 100
    assert int(first_grid["tick_abs_start"]) == 100 - (config.n_absolute_ticks // 2)


def test_stream_core_events_fail_fast_on_out_of_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.vacuum_pressure.core_pipeline.iter_mbo_events",
        _iter_for(_events_out_of_range()),
    )

    config = _test_config(n_absolute_ticks=20)
    with pytest.raises(ValueError, match="outside configured absolute grid"):
        list(
            stream_core_events(
                lake_root=Path("/tmp"),
                config=config,
                dt="2026-02-06",
                start_time=None,
                fail_on_out_of_range=True,
            )
        )


def test_stream_core_events_default_tolerant_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        "src.vacuum_pressure.core_pipeline.iter_mbo_events",
        _iter_for(_events_out_of_range()),
    )

    config = _test_config(n_absolute_ticks=20)
    with caplog.at_level("WARNING", logger="src.vacuum_pressure.event_engine"):
        grids = list(
            stream_core_events(
                lake_root=Path("/tmp"),
                config=config,
                dt="2026-02-06",
                start_time=None,
            )
        )

    assert len(grids) > 0
    assert any("skipping depth update" in rec.getMessage() for rec in caplog.records)


def test_stream_core_events_soft_reanchors_once_after_event_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.vacuum_pressure.core_pipeline.iter_mbo_events",
        _iter_for(_events_soft_reanchor_event_gate()),
    )
    monkeypatch.setattr(
        "src.vacuum_pressure.core_pipeline._SOFT_REANCHOR_AFTER_EVENT_COUNT",
        5,
    )

    config = _test_config(n_absolute_ticks=20)
    grids = list(
        stream_core_events(
            lake_root=Path("/tmp"),
            config=config,
            dt="2026-02-06",
            start_time=None,
        )
    )

    anchors = [int(grid["anchor_tick_idx"]) for grid in grids]
    assert anchors[0] == 100
    assert 201 in anchors
    assert len(set(anchors)) == 2
