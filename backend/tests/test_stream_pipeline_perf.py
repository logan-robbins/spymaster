from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Iterator, Tuple

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.vacuum_pressure.config import VPRuntimeConfig, build_config_with_overrides
from src.vacuum_pressure.stream_pipeline import (
    ProducerLatencyConfig,
    async_stream_events,
    stream_events,
)

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
        n_absolute_ticks=32,
        spectrum_windows=(2, 4),
        spectrum_rollup_weights=(1.0, 1.0),
        spectrum_derivative_weights=(0.55, 0.30, 0.15),
        spectrum_tanh_scale=3.0,
        spectrum_threshold_neutral=0.15,
        zscore_window_bins=8,
        zscore_min_periods=2,
        projection_horizons_bins=(1,),
        projection_horizons_ms=(100,),
        contract_multiplier=1.0,
        qty_unit="contracts",
        price_decimals=2,
        config_version="test",
    )


def _events_with_gap() -> list[MBOEvent]:
    return [
        (1_000_000_000, "A", "B", 100, 10, 1, 0),
        (1_050_000_000, "A", "A", 101, 12, 2, 0),
        (1_250_000_000, "C", "B", 100, 0, 1, 0),
    ]


def _events_with_gap_after_activity() -> list[MBOEvent]:
    return [
        (1_000_000_000, "A", "B", 100, 10, 1, 0),
        (1_050_000_000, "A", "A", 101, 12, 2, 0),
        (1_150_000_000, "A", "B", 100, 5, 3, 0),
        (1_180_000_000, "A", "B", 100, 2, 4, 0),
        (1_350_000_000, "C", "B", 100, 0, 3, 0),
    ]


def _events_with_up_chase_repricing() -> list[MBOEvent]:
    return [
        (1_000_000_000, "A", "B", 100, 10, 1, 0),
        (1_000_000_000, "A", "A", 101, 12, 2, 0),
        (1_120_000_000, "M", "A", 102, 12, 2, 0),
        (1_130_000_000, "M", "B", 101, 10, 1, 0),
    ]


def _bucket_by_k(grid: dict, k: int) -> dict:
    for row in grid["buckets"]:
        if int(row["k"]) == k:
            return row
    raise AssertionError(f"missing bucket row for k={k}")


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


def _collect_async_grids(
    config: VPRuntimeConfig,
    output_path: Path,
    *,
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
) -> list[dict]:
    latency_cfg = ProducerLatencyConfig(
        output_path=output_path,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        summary_every_bins=2,
    )

    async def _run() -> list[dict]:
        grids: list[dict] = []
        async for grid in async_stream_events(
            lake_root=Path("/tmp"),
            config=config,
            dt="2026-02-06",
            start_time=None,
            producer_latency_config=latency_cfg,
        ):
            grids.append(grid)
        return grids

    return asyncio.run(_run())


def test_stream_events_capture_producer_timing_includes_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.vacuum_pressure.stream_pipeline.iter_mbo_events",
        _iter_for(_events_with_gap()),
    )

    grids = list(
        stream_events(
            lake_root=Path("/tmp"),
            config=_test_config(),
            dt="2026-02-06",
            start_time=None,
            capture_producer_timing=True,
        )
    )

    assert len(grids) == 3
    saw_empty_bin = False

    for grid in grids:
        perf = grid["_producer_perf"]
        assert isinstance(perf["grid_ready_wall_ns"], int)

        if int(grid["bin_event_count"]) == 0:
            saw_empty_bin = True
            assert perf["bin_first_ingest_wall_ns"] is None
            assert perf["bin_last_ingest_wall_ns"] is None
        else:
            assert isinstance(perf["bin_first_ingest_wall_ns"], int)
            assert isinstance(perf["bin_last_ingest_wall_ns"], int)
            assert perf["bin_first_ingest_wall_ns"] <= perf["bin_last_ingest_wall_ns"]
            assert perf["bin_last_ingest_wall_ns"] <= perf["grid_ready_wall_ns"]

    assert saw_empty_bin


def test_stream_events_empty_bin_applies_passive_decay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.vacuum_pressure.stream_pipeline.iter_mbo_events",
        _iter_for(_events_with_gap_after_activity()),
    )

    grids = list(
        stream_events(
            lake_root=Path("/tmp"),
            config=_test_config(),
            dt="2026-02-06",
            start_time=None,
        )
    )

    # Expect bins: [2 events], [2 events], [0 events], [1 event].
    assert len(grids) == 4
    assert int(grids[0]["bin_event_count"]) == 2
    assert int(grids[1]["bin_event_count"]) == 2
    assert int(grids[2]["bin_event_count"]) == 0

    active_row = max(
        grids[1]["buckets"],
        key=lambda row: float(row["add_mass"]),
    )
    assert float(active_row["add_mass"]) > 0.0
    k = int(active_row["k"])

    b1 = _bucket_by_k(grids[1], k)
    b2 = _bucket_by_k(grids[2], k)

    # No event touched this bucket in bin 2, but passive decay should apply.
    assert int(b1["last_event_id"]) == int(b2["last_event_id"])
    assert float(b2["add_mass"]) < float(b1["add_mass"])
    assert float(b2["v_add"]) < float(b1["v_add"])
    assert float(b2["pressure_variant"]) < float(b1["pressure_variant"])


def test_stream_events_emits_permutation_labels_for_upward_chase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.vacuum_pressure.stream_pipeline.iter_mbo_events",
        _iter_for(_events_with_up_chase_repricing()),
    )

    grids = list(
        stream_events(
            lake_root=Path("/tmp"),
            config=_test_config(),
            dt="2026-02-06",
            start_time=None,
        )
    )
    assert len(grids) >= 2
    second = grids[1]
    assert int(second["ask_reprice_sign"]) == 1
    assert int(second["bid_reprice_sign"]) == 1
    assert int(second["perm_microstate_id"]) == 8
    assert int(second["chase_up_flag"]) == 1
    assert second["runtime_model_name"] == "perm_derivative"
    assert isinstance(second["runtime_model_score"], float)
    assert isinstance(second["runtime_model_ready"], bool)
    assert isinstance(second["runtime_model_sample_count"], int)
    assert isinstance(second["runtime_model_dominant_state5_code"], int)

    above = _bucket_by_k(second, 1)
    below = _bucket_by_k(second, -1)
    assert int(above["perm_state5_code"]) == 2  # bullish vacuum above spot
    assert int(below["perm_state5_code"]) == 1  # bullish pressure below spot


def test_async_stream_events_writes_latency_jsonl_and_hides_internal_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.vacuum_pressure.stream_pipeline.iter_mbo_events",
        _iter_for(_events_with_gap()),
    )

    output_path = tmp_path / "latency.jsonl"
    grids = _collect_async_grids(_test_config(), output_path)

    assert len(grids) == 3
    assert all("_producer_perf" not in grid for grid in grids)

    records = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(records) == 3

    for rec in records:
        assert rec["queue_block_us"] is not None
        assert rec["queue_block_us"] >= 0.0
        if rec["bin_event_count"] == 0:
            assert rec["first_ingest_to_queue_put_done_us"] is None
            assert rec["last_ingest_to_queue_put_done_us"] is None
        else:
            assert rec["first_ingest_to_queue_put_done_us"] is not None
            assert rec["last_ingest_to_queue_put_done_us"] is not None
            assert rec["first_ingest_to_queue_put_done_us"] >= 0.0
            assert rec["last_ingest_to_queue_put_done_us"] >= 0.0


def test_async_stream_events_latency_window_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.vacuum_pressure.stream_pipeline.iter_mbo_events",
        _iter_for(_events_with_gap()),
    )

    output_path = tmp_path / "latency_window.jsonl"
    _collect_async_grids(
        _test_config(),
        output_path,
        window_start_ns=1_100_000_000,
        window_end_ns=1_200_000_000,
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(records) == 1
    assert records[0]["bin_seq"] == 1


def test_stream_events_can_disable_runtime_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.vacuum_pressure.stream_pipeline.iter_mbo_events",
        _iter_for(_events_with_gap()),
    )
    config = build_config_with_overrides(_test_config(), {"perm_runtime_enabled": False})
    grids = list(
        stream_events(
            lake_root=Path("/tmp"),
            config=config,
            dt="2026-02-06",
            start_time=None,
        )
    )
    assert len(grids) == 3
    assert "runtime_model_name" not in grids[0]
