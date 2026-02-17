from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

import pandas as pd
import pyarrow.parquet as pq
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

import scripts.cache_vp_output as cache_vp_output
from src.vacuum_pressure.config import VPRuntimeConfig

def _test_config() -> VPRuntimeConfig:
    return VPRuntimeConfig(
        product_type="future_mbo",
        symbol="TESTH6",
        symbol_root="TEST",
        price_scale=1e-9,
        tick_size=1e-9,
        bucket_size_dollars=1e-9,
        rel_tick_size=1e-9,
        grid_radius_ticks=1,
        cell_width_ms=100,
        n_absolute_ticks=16,
        spectrum_windows=(2, 4),
        spectrum_rollup_weights=(1.0, 1.0),
        spectrum_derivative_weights=(0.55, 0.30, 0.15),
        spectrum_tanh_scale=3.0,
        spectrum_threshold_neutral=0.15,
        zscore_window_bins=8,
        zscore_min_periods=2,
        projection_horizons_ms=(100, 250),
        contract_multiplier=1.0,
        qty_unit="contracts",
        price_decimals=2,
        config_version="test",
    )


def _make_bucket_row(k: int, base: float, event_id: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "k": k,
        "spectrum_state_code": -1 if k < 0 else 1,
        "last_event_id": event_id + k,
    }
    for idx, field_name in enumerate(cache_vp_output._BUCKET_FLOAT_FIELDS):
        row[field_name] = base + float(idx + 1) / 100.0
    row["proj_score_h100"] = base + 0.5
    row["proj_score_h250"] = base + 0.75
    return row


def _make_grid(ts_ns: int, seq: int, event_id: int) -> Dict[str, Any]:
    return {
        "ts_ns": ts_ns,
        "bin_seq": seq,
        "bin_start_ns": ts_ns - 100,
        "bin_end_ns": ts_ns,
        "bin_event_count": 17 + seq,
        "event_id": event_id,
        "mid_price": 123.45 + seq,
        "spot_ref_price_int": 100_000 + seq,
        "best_bid_price_int": 99_900 + seq,
        "best_ask_price_int": 100_100 + seq,
        "book_valid": True,
        "buckets": [
            _make_bucket_row(-1, 0.1 * (seq + 1), event_id),
            _make_bucket_row(0, 0.2 * (seq + 1), event_id),
        ],
    }


def _iter_grids(grids: list[Dict[str, Any]]):
    def _fake_stream_events(
        *,
        lake_root: Path,
        config: VPRuntimeConfig,
        dt: str,
        start_time: str | None = None,
        projection_use_cubic: bool = False,
        projection_cubic_scale: float = 1.0 / 6.0,
        projection_damping_lambda: float = 0.0,
    ) -> Iterator[Dict[str, Any]]:
        del lake_root
        del config
        del dt
        del start_time
        del projection_use_cubic
        del projection_cubic_scale
        del projection_damping_lambda
        yield from grids

    return _fake_stream_events


def test_capture_stream_output_writes_windowed_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    grids = [
        _make_grid(1_000, 0, 10),
        _make_grid(2_000, 1, 20),
        _make_grid(3_000, 2, 30),
    ]
    monkeypatch.setattr(cache_vp_output, "stream_events", _iter_grids(grids))

    output_dir = tmp_path / "capture"
    summary = cache_vp_output.capture_stream_output(
        lake_root=tmp_path,
        config=_test_config(),
        dt="2026-02-06",
        stream_start_time_hhmm="09:25",
        capture_start_ns=1_500,
        capture_end_ns=3_500,
        output_dir=output_dir,
        flush_bins_every=1,
    )

    assert summary["rows"]["bins"] == 2
    assert summary["rows"]["buckets"] == 4
    assert summary["first_bin_ts_ns"] == 2_000
    assert summary["last_bin_ts_ns"] == 3_000

    bins_tbl = pq.read_table(output_dir / "bins.parquet")
    bins = bins_tbl.to_pydict()
    assert bins["ts_ns"] == [2_000, 3_000]
    assert bins["bin_seq"] == [1, 2]
    assert bins["book_valid"] == [True, True]

    buckets_tbl = pq.read_table(output_dir / "buckets.parquet")
    buckets = buckets_tbl.to_pydict()
    assert buckets["bin_seq"] == [1, 1, 2, 2]
    assert buckets["k"] == [-1, 0, -1, 0]
    assert "proj_score_h100" in buckets
    assert "proj_score_h250" in buckets
    assert len(buckets["proj_score_h100"]) == 4

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"]["bins"] == 2
    assert manifest["rows"]["buckets"] == 4
    assert manifest["projection_horizons_ms"] == [100, 250]


def test_capture_stream_output_fails_when_window_has_no_bins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cache_vp_output, "stream_events", _iter_grids([_make_grid(1_000, 0, 10)]))

    output_dir = tmp_path / "capture_empty"
    with pytest.raises(RuntimeError, match="No emitted bins captured"):
        cache_vp_output.capture_stream_output(
            lake_root=tmp_path,
            config=_test_config(),
            dt="2026-02-06",
            stream_start_time_hhmm="09:25",
            capture_start_ns=2_000,
            capture_end_ns=3_000,
            output_dir=output_dir,
        )

    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []


def test_stream_start_hhmm_requires_minute_boundary() -> None:
    ts = pd.Timestamp("2026-02-06 09:25:30", tz="America/New_York")
    with pytest.raises(ValueError, match="minute boundary"):
        cache_vp_output._stream_start_hhmm(ts)
