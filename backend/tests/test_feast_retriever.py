"""Tests for FeastFeatureRetriever."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.experiment_harness.dataset_registry import DatasetRegistry
from src.experiment_harness.eval_engine import EvalEngine, K_MIN, K_MAX, N_TICKS
from src.experiment_harness.feature_store.config import FeatureStoreConfig
from src.experiment_harness.feature_store.retriever import FeastFeatureRetriever
from src.experiment_harness.feature_store.writer import sync_dataset_to_feature_store


def _make_full_fixture(tmp_path: Path) -> tuple[str, Path]:
    """Create a 3-bin × 101-k dataset covering the full k range."""
    dataset_id = "test_ds_retriever_abc12345"
    dataset_dir = tmp_path / "research" / "datasets" / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    bins_df = pd.DataFrame(
        {
            "bin_seq": [0, 1, 2],
            "ts_ns": [1_000_000_000_000, 2_000_000_000_000, 3_000_000_000_000],
            "bin_start_ns": [900_000_000_000, 1_900_000_000_000, 2_900_000_000_000],
            "bin_end_ns": [1_000_000_000_000, 2_000_000_000_000, 3_000_000_000_000],
            "mid_price": [100.0, 101.0, 102.0],
            "event_id": [1, 2, 3],
            "bin_event_count": [10, 20, 30],
            "book_valid": [True, True, True],
            "best_bid_price_int": [99_000_000_000, 100_000_000_000, 101_000_000_000],
            "best_ask_price_int": [101_000_000_000, 102_000_000_000, 103_000_000_000],
            "spot_ref_price_int": [100_000_000_000, 101_000_000_000, 102_000_000_000],
        }
    )
    bins_path = dataset_dir / "bins.parquet"
    bins_df.to_parquet(bins_path, index=False)

    k_values = np.arange(K_MIN, K_MAX + 1, dtype=np.int32)
    rows = []
    for b in range(3):
        for k in k_values:
            v = float(b + int(k)) * 0.1
            a = float(b - int(k)) * 0.05
            # Simulates grid_clean filtering (exclude all-zero rows)
            if v != 0.0 or a != 0.0:
                rows.append({"bin_seq": b, "k": int(k), "velocity": v, "acceleration": a})
    grid_df = pd.DataFrame(rows)
    grid_path = dataset_dir / "grid_clean.parquet"
    grid_df.to_parquet(grid_path, index=False)

    return dataset_id, tmp_path


def _sync_fixture(tmp_path: Path) -> tuple[str, Path, DatasetRegistry]:
    dataset_id, lake_root = _make_full_fixture(tmp_path)
    config = FeatureStoreConfig(enabled=True)
    registry = DatasetRegistry(lake_root)
    paths = registry.resolve(dataset_id)
    sync_dataset_to_feature_store(dataset_id, paths, lake_root, config)
    return dataset_id, lake_root, registry


def test_load_dataset_correct_keys_and_shapes(tmp_path: Path) -> None:
    """load_dataset returns dict with correct keys and (n_bins, 101) shapes."""
    dataset_id, lake_root, registry = _sync_fixture(tmp_path)
    config = FeatureStoreConfig(enabled=True)

    retriever = FeastFeatureRetriever(lake_root, config)
    result = retriever.load_dataset(dataset_id, ["velocity"], registry)

    assert result["n_bins"] == 3
    assert result["velocity"].shape == (3, N_TICKS)
    assert "bins" in result
    assert "mid_price" in result
    assert "ts_ns" in result
    assert "k_values" in result
    assert len(result["k_values"]) == N_TICKS


def test_load_dataset_values_match_eval_engine(tmp_path: Path) -> None:
    """Feast retriever output matches direct EvalEngine.load_dataset()."""
    dataset_id, lake_root, registry = _sync_fixture(tmp_path)
    config = FeatureStoreConfig(enabled=True)

    retriever = FeastFeatureRetriever(lake_root, config)
    feast_result = retriever.load_dataset(
        dataset_id, ["velocity", "acceleration"], registry
    )

    engine = EvalEngine()
    direct_result = engine.load_dataset(
        dataset_id, ["velocity", "acceleration"], registry
    )

    np.testing.assert_array_almost_equal(
        feast_result["velocity"], direct_result["velocity"], decimal=6
    )
    np.testing.assert_array_almost_equal(
        feast_result["acceleration"], direct_result["acceleration"], decimal=6
    )
    np.testing.assert_array_equal(feast_result["ts_ns"], direct_result["ts_ns"])
    np.testing.assert_array_almost_equal(
        feast_result["mid_price"], direct_result["mid_price"]
    )


def test_missing_dataset_raises_file_not_found(tmp_path: Path) -> None:
    """Loading an unregistered dataset_id raises FileNotFoundError."""
    dataset_id, lake_root, registry = _sync_fixture(tmp_path)
    config = FeatureStoreConfig(enabled=True)

    retriever = FeastFeatureRetriever(lake_root, config)
    with pytest.raises(FileNotFoundError):
        retriever.load_dataset("nonexistent_dataset_xyz", ["velocity"], registry)


def test_missing_column_raises_key_error(tmp_path: Path) -> None:
    """Requesting an unknown column raises KeyError."""
    dataset_id, lake_root, registry = _sync_fixture(tmp_path)
    config = FeatureStoreConfig(enabled=True)

    retriever = FeastFeatureRetriever(lake_root, config)
    with pytest.raises(KeyError):
        retriever.load_dataset(dataset_id, ["nonexistent_column_xyz"], registry)


def test_uninitialised_repo_raises_file_not_found(tmp_path: Path) -> None:
    """FeastFeatureRetriever raises FileNotFoundError if repo is not initialised."""
    config = FeatureStoreConfig(enabled=True)
    with pytest.raises(FileNotFoundError):
        FeastFeatureRetriever(tmp_path, config)
