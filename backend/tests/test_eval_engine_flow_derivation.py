"""Tests for EvalEngine gold column routing and fail-fast behavior.

Gold columns (pressure_variant, vacuum_variant, composite*, state5_code,
flow_score, flow_state_code) must be loaded from gold_grid.parquet, not
derived on-the-fly from silver data. If gold_grid.parquet is absent,
load_dataset must raise KeyError immediately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.experiment_harness.dataset_registry import DatasetRegistry
from src.experiment_harness.eval_engine import EvalEngine
from src.qmachina.stage_schema import GOLD_COLS, SILVER_FLOAT_COLS


def _write_silver_dataset(root: Path, dataset_id: str) -> Path:
    """Write a minimal silver dataset (bins + grid_clean, no gold_grid)."""
    dataset_dir = root / "research" / "datasets" / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    n_bins = 3
    k_vals = np.arange(-2, 3, dtype=np.int32)

    bins_df = pd.DataFrame(
        {
            "bin_seq": [0, 1, 2],
            "ts_ns": [1_000, 2_000, 3_000],
            "mid_price": [100.0, 100.25, 100.5],
        }
    )
    bins_df.to_parquet(dataset_dir / "bins.parquet", index=False)

    rows: list[dict] = []
    for bin_seq in range(n_bins):
        for k in k_vals:
            row: dict = {
                "bin_seq": int(bin_seq),
                "k": int(k),
                "last_event_id": int(bin_seq * 100 + k + 50),
                "ask_reprice_sign": int(np.sign(k + 1)),
                "bid_reprice_sign": int(np.sign(k)),
                "microstate_id": 0,
                "chase_up_flag": 0,
                "chase_down_flag": 0,
                "best_ask_move_ticks": 0,
                "best_bid_move_ticks": 0,
            }
            for col in SILVER_FLOAT_COLS:
                row[col] = float(np.random.default_rng(bin_seq * 100 + k + 50).random())
            rows.append(row)

    grid_df = pd.DataFrame(rows)
    grid_df.to_parquet(dataset_dir / "grid_clean.parquet", index=False)
    (dataset_dir / "manifest.json").write_text("{}", encoding="utf-8")
    return dataset_dir


def _write_gold_grid(dataset_dir: Path, k_vals: np.ndarray, n_bins: int) -> None:
    """Write a minimal gold_grid.parquet alongside the silver dataset."""
    rows: list[dict] = []
    for bin_seq in range(n_bins):
        for k in k_vals:
            rows.append({
                "bin_seq": int(bin_seq),
                "k": int(k),
                "pressure_variant": float(bin_seq * 0.1 + k * 0.01),
                "vacuum_variant": float(bin_seq * 0.05 + k * 0.02),
                "composite": 0.0,
                "composite_d1": 0.0,
                "composite_d2": 0.0,
                "composite_d3": 0.0,
                "state5_code": np.int8(1),
                "flow_score": float(bin_seq * 0.2),
                "flow_state_code": np.int8(0),
            })
    gold_df = pd.DataFrame(rows)
    gold_df.to_parquet(dataset_dir / "gold_grid.parquet", index=False)


# ---------------------------------------------------------------------------
# Silver-only loading
# ---------------------------------------------------------------------------

def test_load_silver_cols_from_grid_clean(tmp_path: Path) -> None:
    """Silver columns are loaded from grid_clean.parquet without gold_grid."""
    dataset_dir = _write_silver_dataset(tmp_path, "ds_silver_only")

    registry = DatasetRegistry(tmp_path)
    engine = EvalEngine()
    result = engine.load_dataset("ds_silver_only", ["v_add", "a_pull"], registry)

    assert "v_add" in result
    assert result["v_add"].shape[0] == 3  # n_bins


# ---------------------------------------------------------------------------
# Gold col fail-fast: no gold_grid.parquet
# ---------------------------------------------------------------------------

def test_load_gold_col_without_gold_grid_raises(tmp_path: Path) -> None:
    """Loading a gold column without gold_grid.parquet raises KeyError immediately."""
    _write_silver_dataset(tmp_path, "ds_no_gold")

    registry = DatasetRegistry(tmp_path)
    engine = EvalEngine()
    with pytest.raises(KeyError, match="gold_grid.parquet"):
        engine.load_dataset("ds_no_gold", ["flow_score"], registry)


def test_load_flow_state_code_without_gold_grid_raises(tmp_path: Path) -> None:
    _write_silver_dataset(tmp_path, "ds_no_gold2")

    registry = DatasetRegistry(tmp_path)
    engine = EvalEngine()
    with pytest.raises(KeyError, match="gold_grid.parquet"):
        engine.load_dataset("ds_no_gold2", ["flow_state_code"], registry)


def test_load_pressure_variant_without_gold_grid_raises(tmp_path: Path) -> None:
    _write_silver_dataset(tmp_path, "ds_no_gold3")

    registry = DatasetRegistry(tmp_path)
    engine = EvalEngine()
    with pytest.raises(KeyError, match="gold_grid.parquet"):
        engine.load_dataset("ds_no_gold3", ["pressure_variant"], registry)


def test_load_state5_code_without_gold_grid_raises(tmp_path: Path) -> None:
    _write_silver_dataset(tmp_path, "ds_no_gold4")

    registry = DatasetRegistry(tmp_path)
    engine = EvalEngine()
    with pytest.raises(KeyError, match="gold_grid.parquet"):
        engine.load_dataset("ds_no_gold4", ["state5_code"], registry)


# ---------------------------------------------------------------------------
# Gold col loading: gold_grid.parquet present
# ---------------------------------------------------------------------------

def test_load_gold_col_with_gold_grid_succeeds(tmp_path: Path) -> None:
    """Gold columns are loaded from gold_grid.parquet when it exists."""
    k_vals = np.arange(-2, 3, dtype=np.int32)
    dataset_dir = _write_silver_dataset(tmp_path, "ds_with_gold")
    _write_gold_grid(dataset_dir, k_vals, n_bins=3)

    registry = DatasetRegistry(tmp_path)
    engine = EvalEngine()
    result = engine.load_dataset("ds_with_gold", ["flow_score", "flow_state_code"], registry)

    assert "flow_score" in result
    assert "flow_state_code" in result
    assert result["flow_score"].shape == (3, 101)  # 3 bins, 101 k values (k=-50..50)
    assert result["flow_state_code"].shape == (3, 101)


def test_load_mixed_silver_and_gold_cols(tmp_path: Path) -> None:
    """Mixed silver + gold columns merge correctly on (bin_seq, k)."""
    k_vals = np.arange(-2, 3, dtype=np.int32)
    dataset_dir = _write_silver_dataset(tmp_path, "ds_mixed")
    _write_gold_grid(dataset_dir, k_vals, n_bins=3)

    registry = DatasetRegistry(tmp_path)
    engine = EvalEngine()
    result = engine.load_dataset("ds_mixed", ["v_add", "pressure_variant"], registry)

    assert "v_add" in result
    assert "pressure_variant" in result
    assert result["v_add"].shape == result["pressure_variant"].shape


def test_all_gold_cols_in_stage_schema() -> None:
    """Verify GOLD_COLS contains all expected gold field names."""
    expected = {
        "pressure_variant", "vacuum_variant", "composite",
        "composite_d1", "composite_d2", "composite_d3",
        "state5_code", "flow_score", "flow_state_code",
    }
    assert expected.issubset(GOLD_COLS), f"Missing from GOLD_COLS: {expected - GOLD_COLS}"
