"""Tests for sync_dataset_to_feature_store()."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.experiment_harness.dataset_registry import DatasetPaths
from src.experiment_harness.feature_store.config import FeatureStoreConfig
from src.experiment_harness.feature_store.writer import sync_dataset_to_feature_store


def _make_fixture(tmp_path: Path) -> tuple[str, DatasetPaths]:
    """Create a minimal 3-bin × 5-k dataset fixture."""
    dataset_id = "test_ds_writer_abc12345"
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

    k_values = [-2, -1, 0, 1, 2]
    rows = []
    for b in range(3):
        for k in k_values:
            rows.append(
                {
                    "bin_seq": b,
                    "k": k,
                    "velocity": float(b + k) * 0.1,
                    "acceleration": float(b - k) * 0.05,
                }
            )
    grid_df = pd.DataFrame(rows)
    grid_path = dataset_dir / "grid_clean.parquet"
    grid_df.to_parquet(grid_path, index=False)

    paths = DatasetPaths(
        bins_parquet=bins_path,
        grid_clean_parquet=grid_path,
        gold_grid_parquet=dataset_dir / "gold_grid.parquet",
        dataset_id=dataset_id,
    )
    return dataset_id, paths


def test_sync_writes_parquets_with_extra_columns(tmp_path: Path) -> None:
    """Synced parquets have event_timestamp and dataset_id columns."""
    dataset_id, paths = _make_fixture(tmp_path)
    config = FeatureStoreConfig(enabled=True)

    sync_dataset_to_feature_store(dataset_id, paths, tmp_path, config)

    bins_out = (
        tmp_path / "research" / "feature_store" / "offline" / "bins" / f"{dataset_id}.parquet"
    )
    grid_out = (
        tmp_path / "research" / "feature_store" / "offline" / "grid" / f"{dataset_id}.parquet"
    )
    assert bins_out.exists()
    assert grid_out.exists()

    bins_df = pd.read_parquet(bins_out)
    assert "event_timestamp" in bins_df.columns
    assert "dataset_id" in bins_df.columns
    assert (bins_df["dataset_id"] == dataset_id).all()

    grid_df = pd.read_parquet(grid_out)
    assert "event_timestamp" in grid_df.columns
    assert "dataset_id" in grid_df.columns
    assert (grid_df["dataset_id"] == dataset_id).all()


def test_sync_creates_feast_repo_files(tmp_path: Path) -> None:
    """Sync creates feature_store.yaml and registry.db."""
    dataset_id, paths = _make_fixture(tmp_path)
    config = FeatureStoreConfig(enabled=True)

    sync_dataset_to_feature_store(dataset_id, paths, tmp_path, config)

    fs_root = tmp_path / "research" / "feature_store"
    assert (fs_root / "feature_store.yaml").exists()
    assert (fs_root / "registry.db").exists()


def test_sync_idempotent(tmp_path: Path) -> None:
    """Calling sync twice does not raise or duplicate rows."""
    dataset_id, paths = _make_fixture(tmp_path)
    config = FeatureStoreConfig(enabled=True)

    sync_dataset_to_feature_store(dataset_id, paths, tmp_path, config)
    sync_dataset_to_feature_store(dataset_id, paths, tmp_path, config)  # second call is no-op

    bins_out = (
        tmp_path / "research" / "feature_store" / "offline" / "bins" / f"{dataset_id}.parquet"
    )
    bins_df = pd.read_parquet(bins_out)
    assert len(bins_df) == 3  # exactly the original 3 bins, not doubled


def test_sync_event_timestamps_match_ts_ns(tmp_path: Path) -> None:
    """event_timestamp in synced parquet corresponds to the original ts_ns."""
    import pandas as pd

    dataset_id, paths = _make_fixture(tmp_path)
    config = FeatureStoreConfig(enabled=True)

    sync_dataset_to_feature_store(dataset_id, paths, tmp_path, config)

    bins_out = (
        tmp_path / "research" / "feature_store" / "offline" / "bins" / f"{dataset_id}.parquet"
    )
    bins_df = pd.read_parquet(bins_out)

    expected_ts = pd.to_datetime(bins_df["ts_ns"], unit="ns", utc=True)
    pd.testing.assert_series_equal(
        bins_df["event_timestamp"].reset_index(drop=True),
        expected_ts.reset_index(drop=True),
        check_names=False,
    )
