"""Feast-backed feature retriever for the experiment harness."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import FeatureStoreConfig

logger = logging.getLogger(__name__)

K_MIN: int = -50
K_MAX: int = 50
N_TICKS: int = 101
_DERIVABLE_FLOW_COLUMNS: frozenset[str] = frozenset({"flow_score", "flow_state_code"})
_FLOW_SUPPORT_COLUMNS: tuple[str, ...] = ("composite_d1", "composite_d2", "composite_d3")


class FeastFeatureRetriever:
    """Retrieves features from the Feast offline feature store.

    Provides the same dict contract as ``EvalEngine.load_dataset()``.

    Args:
        lake_root: Root path of the data lake.
        config: Feature store configuration.

    Raises:
        FileNotFoundError: If the Feast repo (feature_store.yaml) is not found.
    """

    def __init__(self, lake_root: Path, config: FeatureStoreConfig) -> None:
        from feast import FeatureStore

        self._fs_root = Path(lake_root) / "research" / "feature_store"
        self._offline_bins = self._fs_root / "offline" / "bins"
        self._offline_grid = self._fs_root / "offline" / "grid"

        yaml_path = self._fs_root / "feature_store.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Feast feature store not initialized at {self._fs_root}. "
                "Run 'generate' with feature_store.enabled: true first."
            )
        self._store = FeatureStore(repo_path=str(self._fs_root))

    def load_dataset(
        self,
        dataset_id: str,
        columns: list[str],
        registry: Any,
    ) -> dict[str, Any]:
        """Load a dataset from the Feast offline store.

        Returns the same dict structure as ``EvalEngine.load_dataset()``.

        Args:
            dataset_id: Dataset identifier string.
            columns: Grid column names to load and pivot.
            registry: DatasetRegistry instance (used for flow scoring config).

        Returns:
            Dict with bins, mid_price, ts_ns, n_bins, k_values, and
            (n_bins, 101) float64 arrays for each requested column.

        Raises:
            FileNotFoundError: If the dataset is not in the feature store.
            KeyError: If a requested column is unknown and not derivable.
        """
        bins_parquet = self._offline_bins / f"{dataset_id}.parquet"
        if not bins_parquet.exists():
            raise FileNotFoundError(
                f"Dataset '{dataset_id}' not found in feature store at {bins_parquet}"
            )

        # Load bins directly (no Feast overhead for simple retrieval)
        fs_bins_df = pd.read_parquet(bins_parquet)
        fs_bins_df = fs_bins_df.sort_values("bin_seq").reset_index(drop=True)
        event_timestamps = fs_bins_df["event_timestamp"].values
        bins_df = fs_bins_df.drop(
            columns=[c for c in ["event_timestamp", "dataset_id"] if c in fs_bins_df.columns]
        )
        n_bins = len(bins_df)
        mid_price = bins_df["mid_price"].values.astype(np.float64)
        ts_ns = bins_df["ts_ns"].values.astype(np.int64)

        # Validate columns against grid parquet schema
        grid_parquet = self._offline_grid / f"{dataset_id}.parquet"
        grid_schema_cols = set(pq.read_schema(str(grid_parquet)).names)
        grid_schema_cols -= {"event_timestamp", "dataset_id", "bin_seq", "k"}

        missing_cols = [c for c in columns if c not in grid_schema_cols]
        missing_flow_cols = [c for c in missing_cols if c in _DERIVABLE_FLOW_COLUMNS]
        missing_unsupported = [c for c in missing_cols if c not in _DERIVABLE_FLOW_COLUMNS]
        if missing_unsupported:
            raise KeyError(
                f"Dataset '{dataset_id}' missing required columns: {missing_unsupported}"
            )

        # Build list of columns to fetch from Feast
        feast_cols: list[str] = [c for c in columns if c not in _DERIVABLE_FLOW_COLUMNS]
        if missing_flow_cols:
            missing_support = [
                c for c in _FLOW_SUPPORT_COLUMNS if c not in grid_schema_cols
            ]
            if missing_support:
                raise KeyError(
                    f"Dataset '{dataset_id}' missing columns needed to derive flow fields: "
                    f"{missing_support}"
                )
            for sc in _FLOW_SUPPORT_COLUMNS:
                if sc not in feast_cols:
                    feast_cols.append(sc)

        # Build entity_df for grid (n_bins × 101 rows)
        # Feast v0.60 supports only a single join key per entity, so we use a
        # composite string key.  bin_seq and k are carried along in entity_df
        # so they survive the join and are available for pivoting.
        k_values = np.arange(K_MIN, K_MAX + 1, dtype=np.int32)
        bin_seq_repeated = np.repeat(bins_df["bin_seq"].values, N_TICKS).astype(np.int32)
        k_tiled = np.tile(k_values, n_bins)
        grid_keys = (
            dataset_id
            + "__"
            + bin_seq_repeated.astype(str)
            + "__"
            + k_tiled.astype(str)
        )
        entity_df = pd.DataFrame(
            {
                "grid_key": grid_keys,
                "bin_seq": bin_seq_repeated,
                "k": k_tiled,
                "event_timestamp": np.repeat(event_timestamps, N_TICKS),
            }
        )

        # Retrieve features from Feast
        grid_df = self._store.get_historical_features(
            entity_df=entity_df,
            features=[f"grid_view:{col}" for col in feast_cols],
        ).to_df()

        # Handle derivable flow columns
        if missing_flow_cols:
            from ...models.vacuum_pressure.scoring import score_dataset
            from ..eval_engine import _resolve_flow_scoring_config

            paths = registry.resolve(dataset_id)
            scoring_cfg = _resolve_flow_scoring_config(paths.grid_clean_parquet.parent)
            support_mask = grid_df[list(_FLOW_SUPPORT_COLUMNS)].notna().all(axis=1)
            if support_mask.any():
                score_input = grid_df.loc[
                    support_mask,
                    ["bin_seq", "k"] + list(_FLOW_SUPPORT_COLUMNS),
                ].copy()
                scored_df = score_dataset(score_input, scoring_cfg, N_TICKS)
                if "flow_score" in missing_flow_cols:
                    grid_df.loc[support_mask, "flow_score"] = scored_df[
                        "flow_score"
                    ].values
                if "flow_state_code" in missing_flow_cols:
                    grid_df.loc[support_mask, "flow_state_code"] = scored_df[
                        "flow_state_code"
                    ].values

        # Pivot each column to (n_bins, 101) — shared with EvalEngine.load_dataset
        from ..eval_engine import pivot_grid_to_arrays

        result: dict[str, Any] = {
            "bins": bins_df,
            "mid_price": mid_price,
            "ts_ns": ts_ns,
            "n_bins": n_bins,
            "k_values": k_values,
        }

        pivoted = pivot_grid_to_arrays(grid_df, bins_df, columns, fillna_value=0.0)
        result.update(pivoted)

        logger.info(
            "Feast retrieval: dataset '%s', %d bins, %d columns",
            dataset_id,
            n_bins,
            len(columns),
        )
        return result
