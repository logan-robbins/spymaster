from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.qmachina.config import RuntimeConfig
from src.qmachina.async_stream_wrapper import ProducerLatencyConfig
from src.models.vacuum_pressure.stream_pipeline import (
    async_stream_events,
    stream_events,
)

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
    cols = grid["grid_cols"]
    k_arr = np.asarray(cols["k"], dtype=np.int32)
    matches = np.where(k_arr == int(k))[0]
    if matches.size == 0:
        raise AssertionError(f"missing bucket row for k={k}")
    idx = int(matches[0])
    return {name: cols[name][idx] for name in cols.keys()}


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
    config: RuntimeConfig,
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
        "src.models.vacuum_pressure.stream_pipeline.iter_mbo_events",
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
        "src.models.vacuum_pressure.stream_pipeline.iter_mbo_events",
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

    cols = grids[1]["grid_cols"]
    add_mass = np.asarray(cols["add_mass"], dtype=np.float64)
    k_vals = np.asarray(cols["k"], dtype=np.int32)
    active_idx = int(np.argmax(add_mass))
    active_row = {"add_mass": add_mass[active_idx], "k": k_vals[active_idx]}
    assert float(active_row["add_mass"]) > 0.0
    k = int(active_row["k"])

    b1 = _bucket_by_k(grids[1], k)
    b2 = _bucket_by_k(grids[2], k)

    # No event touched this bucket in bin 2, but passive decay should apply.
    assert int(b1["last_event_id"]) == int(b2["last_event_id"])
    assert float(b2["add_mass"]) < float(b1["add_mass"])
    assert float(b2["v_add"]) < float(b1["v_add"])
    # pressure_variant is a gold field computed offline — not in the silver stream


def test_stream_events_emits_permutation_labels_for_upward_chase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.models.vacuum_pressure.stream_pipeline.iter_mbo_events",
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
    assert int(second["microstate_id"]) == 8
    assert int(second["chase_up_flag"]) == 1

    # state5_code is a gold field computed offline — not in the silver stream.
    # Verify that BBO permutation labels ARE emitted (they are silver fields).
    above = _bucket_by_k(second, 1)
    below = _bucket_by_k(second, -1)
    assert "ask_reprice_sign" in above  # silver permutation label is present
    assert "ask_reprice_sign" in below
    assert "state5_code" not in above  # gold field not in stream


def test_async_stream_events_writes_latency_jsonl_and_hides_internal_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.models.vacuum_pressure.stream_pipeline.iter_mbo_events",
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
        "src.models.vacuum_pressure.stream_pipeline.iter_mbo_events",
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
