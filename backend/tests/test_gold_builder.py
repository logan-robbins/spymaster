"""Tests for the offline gold feature builder.

Verifies VP force computation, state5 lookup, scoring integration, and
idempotency of generate_gold_dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.experiment_harness.dataset_registry import DatasetPaths
from src.experiment_harness.gold_builder import (
    _compute_state5_series,
    _compute_force_block_for_k_series,
    build_gold_grid,
    generate_gold_dataset,
)
from src.qmachina.gold_config import GoldFeatureConfig
from src.qmachina.stage_schema import GOLD_COLS


def _make_gold_cfg() -> GoldFeatureConfig:
    return GoldFeatureConfig(
        c1_v_add=1.0,
        c2_v_rest_pos=0.5,
        c3_a_add=0.2,
        c4_v_pull=1.0,
        c5_v_fill=0.5,
        c6_v_rest_neg=0.3,
        c7_a_pull=0.2,
        flow_windows=[2, 4],
        flow_rollup_weights=[0.6, 0.4],
        flow_derivative_weights=[0.55, 0.30, 0.15],
        flow_tanh_scale=3.0,
        flow_neutral_threshold=0.15,
        flow_zscore_window_bins=8,
        flow_zscore_min_periods=2,
    )


def _write_minimal_silver(root: Path, n_bins: int = 5, k_range: int = 3) -> DatasetPaths:
    """Write bins.parquet + grid_clean.parquet, return DatasetPaths."""
    root.mkdir(parents=True, exist_ok=True)
    k_vals = np.arange(-k_range, k_range + 1, dtype=np.int32)

    bins_rows = []
    for i in range(n_bins):
        bins_rows.append({
            "bin_seq": i,
            "ts_ns": int(1e9 + i * 100_000_000),
            "bin_start_ns": int(1e9 + i * 100_000_000),
            "bin_end_ns": int(1e9 + (i + 1) * 100_000_000),
            "mid_price": 100.0 + i * 0.25,
        })
    bins_df = pd.DataFrame(bins_rows)
    bins_path = root / "bins.parquet"
    bins_df.to_parquet(bins_path, index=False)

    rng = np.random.default_rng(42)
    rows = []
    for bin_seq in range(n_bins):
        for k in k_vals:
            row = {
                "bin_seq": int(bin_seq),
                "k": int(k),
                "last_event_id": int(bin_seq * 100 + k + k_range),
                "ask_reprice_sign": int(rng.integers(-1, 2)),
                "bid_reprice_sign": int(rng.integers(-1, 2)),
                "microstate_id": 0,
                "chase_up_flag": 0,
                "chase_down_flag": 0,
                "best_ask_move_ticks": 0,
                "best_bid_move_ticks": 0,
                "v_add": float(rng.normal(0.5, 0.1)),
                "v_pull": float(rng.normal(0.3, 0.1)),
                "v_fill": float(rng.normal(0.1, 0.05)),
                "v_rest_depth": float(rng.normal(0.0, 0.2)),
                "v_bid_depth": float(rng.normal(0.2, 0.1)),
                "v_ask_depth": float(rng.normal(0.2, 0.1)),
                "a_add": float(rng.normal(0.0, 0.05)),
                "a_pull": float(rng.normal(0.0, 0.05)),
                "a_fill": float(rng.normal(0.0, 0.02)),
                "a_rest_depth": float(rng.normal(0.0, 0.05)),
                "a_bid_depth": float(rng.normal(0.0, 0.05)),
                "a_ask_depth": float(rng.normal(0.0, 0.05)),
                "add_mass": float(rng.random()),
                "pull_mass": float(rng.random()),
                "fill_mass": float(rng.random()),
                "rest_depth": float(rng.random()),
                "bid_depth": float(rng.random()),
                "ask_depth": float(rng.random()),
                "j_add": 0.0,
                "j_pull": 0.0,
                "j_fill": 0.0,
                "j_rest_depth": 0.0,
                "j_bid_depth": 0.0,
                "j_ask_depth": 0.0,
            }
            rows.append(row)

    grid_df = pd.DataFrame(rows)
    silver_path = root / "grid_clean.parquet"
    grid_df.to_parquet(silver_path, index=False)

    manifest_path = root / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    gold_path = root / "gold_grid.parquet"  # does not exist yet
    return DatasetPaths(
        bins_parquet=bins_path,
        grid_clean_parquet=silver_path,
        gold_grid_parquet=gold_path,
        dataset_id="test",
    )


# ---------------------------------------------------------------------------
# _compute_state5_series
# ---------------------------------------------------------------------------

def test_state5_above_spot_bull_vacuum() -> None:
    # k > 0, ask=1, bid=1 -> BULL_VACUUM (2)
    k = np.array([1], dtype=np.int32)
    ask = np.array([1], dtype=np.int8)
    bid = np.array([1], dtype=np.int8)
    result = _compute_state5_series(k, ask, bid)
    assert result[0] == 2


def test_state5_below_spot_bear_vacuum() -> None:
    # k < 0, ask=1, bid=0 -> BEAR_VACUUM (-2)
    k = np.array([-1], dtype=np.int32)
    ask = np.array([1], dtype=np.int8)
    bid = np.array([0], dtype=np.int8)
    result = _compute_state5_series(k, ask, bid)
    assert result[0] == -2


def test_state5_at_spot_is_mixed() -> None:
    # k == 0 -> MIXED (0) regardless of signs
    k = np.array([0], dtype=np.int32)
    ask = np.array([1], dtype=np.int8)
    bid = np.array([1], dtype=np.int8)
    result = _compute_state5_series(k, ask, bid)
    assert result[0] == 0


def test_state5_output_dtype() -> None:
    k = np.array([1, -1, 0], dtype=np.int32)
    ask = np.array([0, 0, 0], dtype=np.int8)
    bid = np.array([0, 0, 0], dtype=np.int8)
    result = _compute_state5_series(k, ask, bid)
    assert result.dtype == np.int8


# ---------------------------------------------------------------------------
# _compute_force_block_for_k_series
# ---------------------------------------------------------------------------

def test_vp_output_columns() -> None:
    cfg = _make_gold_cfg()
    n = 6
    rng = np.random.default_rng(0)
    silver_k = pd.DataFrame({
        "bin_seq": np.arange(n),
        "k": np.full(n, 5),
        "v_add": rng.random(n),
        "v_pull": rng.random(n),
        "v_fill": rng.random(n),
        "v_rest_depth": rng.random(n) - 0.5,
        "a_add": rng.random(n) - 0.5,
        "a_pull": rng.random(n) - 0.5,
    })
    result = _compute_force_block_for_k_series(silver_k, cfg, cell_width_s=0.1)
    for col in ["bin_seq", "k", "pressure_variant", "vacuum_variant",
                "composite", "composite_d1", "composite_d2", "composite_d3"]:
        assert col in result.columns


def test_vp_pressure_non_negative_when_all_positive() -> None:
    cfg = _make_gold_cfg()
    n = 4
    silver_k = pd.DataFrame({
        "bin_seq": np.arange(n),
        "k": np.full(n, 1),
        "v_add": np.ones(n),
        "v_pull": np.zeros(n),
        "v_fill": np.zeros(n),
        "v_rest_depth": np.ones(n),
        "a_add": np.ones(n),
        "a_pull": np.zeros(n),
    })
    result = _compute_force_block_for_k_series(silver_k, cfg, cell_width_s=0.1)
    assert (result["pressure_variant"] >= 0).all()


# ---------------------------------------------------------------------------
# build_gold_grid
# ---------------------------------------------------------------------------

def test_build_gold_grid_output_columns(tmp_path: Path) -> None:
    paths = _write_minimal_silver(tmp_path)
    cfg = _make_gold_cfg()
    gold = build_gold_grid(paths.grid_clean_parquet, paths.bins_parquet, cfg)
    for col in GOLD_COLS:
        assert col in gold.columns, f"Missing gold column: {col}"


def test_build_gold_grid_row_count(tmp_path: Path) -> None:
    n_bins, k_range = 5, 3
    paths = _write_minimal_silver(tmp_path, n_bins=n_bins, k_range=k_range)
    cfg = _make_gold_cfg()
    gold = build_gold_grid(paths.grid_clean_parquet, paths.bins_parquet, cfg)
    n_k = 2 * k_range + 1
    assert len(gold) == n_bins * n_k


def test_build_gold_grid_state5_dtype(tmp_path: Path) -> None:
    paths = _write_minimal_silver(tmp_path)
    cfg = _make_gold_cfg()
    gold = build_gold_grid(paths.grid_clean_parquet, paths.bins_parquet, cfg)
    assert gold["state5_code"].dtype == np.int8


def test_build_gold_grid_flow_score_finite(tmp_path: Path) -> None:
    paths = _write_minimal_silver(tmp_path)
    cfg = _make_gold_cfg()
    gold = build_gold_grid(paths.grid_clean_parquet, paths.bins_parquet, cfg)
    assert np.all(np.isfinite(gold["flow_score"]))


# ---------------------------------------------------------------------------
# generate_gold_dataset: idempotency
# ---------------------------------------------------------------------------

def test_generate_gold_dataset_writes_parquet(tmp_path: Path) -> None:
    paths = _write_minimal_silver(tmp_path)
    cfg = _make_gold_cfg()
    gold_path = generate_gold_dataset(paths, cfg)
    assert gold_path.exists()
    df = pd.read_parquet(gold_path)
    assert len(df) > 0


def test_generate_gold_dataset_idempotent(tmp_path: Path) -> None:
    """Second call with same config returns immediately (manifest hash match)."""
    paths = _write_minimal_silver(tmp_path)
    cfg = _make_gold_cfg()
    p1 = generate_gold_dataset(paths, cfg)
    mtime1 = p1.stat().st_mtime
    p2 = generate_gold_dataset(paths, cfg)
    mtime2 = p2.stat().st_mtime
    assert mtime1 == mtime2, "gold_grid.parquet was regenerated on second call"


def test_generate_gold_dataset_force_recomputes(tmp_path: Path) -> None:
    """force=True regenerates even when manifest hash matches."""
    paths = _write_minimal_silver(tmp_path)
    cfg = _make_gold_cfg()
    generate_gold_dataset(paths, cfg)
    # Corrupt the parquet to verify force=True regenerates it
    paths.gold_grid_parquet.write_bytes(b"corrupted")
    generate_gold_dataset(paths, cfg, force=True)
    # If forced recompute ran, the file is now a valid parquet again
    df = pd.read_parquet(paths.gold_grid_parquet)
    assert len(df) > 0


def test_generate_gold_dataset_updates_manifest(tmp_path: Path) -> None:
    paths = _write_minimal_silver(tmp_path)
    cfg = _make_gold_cfg()
    generate_gold_dataset(paths, cfg)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "gold_config_hash" in manifest
    assert manifest["gold_config_hash"] == cfg.config_hash()


def test_generate_gold_dataset_different_config_regenerates(tmp_path: Path) -> None:
    """Different config hash triggers regeneration."""
    import time
    paths = _write_minimal_silver(tmp_path)
    cfg1 = _make_gold_cfg()
    generate_gold_dataset(paths, cfg1)
    mtime1 = paths.gold_grid_parquet.stat().st_mtime

    cfg2 = cfg1.model_copy(update={"c1_v_add": 2.0})
    time.sleep(0.01)
    generate_gold_dataset(paths, cfg2)
    mtime2 = paths.gold_grid_parquet.stat().st_mtime
    assert mtime2 > mtime1, "gold_grid.parquet was NOT regenerated for new config"
