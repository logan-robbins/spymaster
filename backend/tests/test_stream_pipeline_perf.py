from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Iterator, Tuple

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.vacuum_pressure.config import VPRuntimeConfig
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
