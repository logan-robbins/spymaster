"""Feast feature store writer: syncs immutable datasets to the offline store."""
from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import FeatureStoreConfig

logger = logging.getLogger(__name__)

_FEAST_FEATURE_STORE_YAML = """\
project: {project}
provider: local
registry: registry.db
offline_store:
  type: file
entity_key_serialization_version: 3
"""

_EXCLUDE_FROM_SCHEMA = frozenset({"dataset_id", "bin_seq", "k", "event_timestamp"})


def _arrow_to_feast_type(arrow_type: pa.DataType):
    """Map an Arrow DataType to the corresponding Feast ValueType."""
    from feast.types import Bool, Float32, Float64, Int32, Int64, String

    if pa.types.is_float64(arrow_type):
        return Float64
    if pa.types.is_float32(arrow_type):
        return Float32
    if pa.types.is_int64(arrow_type):
        return Int64
    if (
        pa.types.is_int32(arrow_type)
        or pa.types.is_int16(arrow_type)
        or pa.types.is_int8(arrow_type)
    ):
        return Int32
    if pa.types.is_boolean(arrow_type):
        return Bool
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return String
    return Float64  # default fallback


def _build_schema_fields(schema: pa.Schema, exclude: frozenset[str]) -> list:
    """Build Feast Field list from an Arrow schema, excluding specified column names."""
    from feast import Field

    fields = []
    for arrow_field in schema:
        if arrow_field.name in exclude:
            continue
        feast_type = _arrow_to_feast_type(arrow_field.type)
        fields.append(Field(name=arrow_field.name, dtype=feast_type))
    return fields


def _ensure_feast_repo(
    fs_root: Path,
    offline_bins_dir: Path,
    offline_grid_dir: Path,
    config: FeatureStoreConfig,
) -> None:
    """Initialize the Feast repo directory and apply entities/views.

    Writes feature_store.yaml if not present, reads schema from the first
    available parquet file in each offline directory, and applies entities
    and feature views to the Feast registry.

    Feast (as of v0.60) only supports a single join key per entity, so we use
    composite string keys: ``bins_key = "{dataset_id}__{bin_seq}"`` and
    ``grid_key = "{dataset_id}__{bin_seq}__{k}"``.
    """
    from feast import Entity, FeatureStore, FeatureView, FileSource

    yaml_path = fs_root / "feature_store.yaml"
    if not yaml_path.exists():
        fs_root.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(
            _FEAST_FEATURE_STORE_YAML.format(project=config.project),
            encoding="utf-8",
        )
        logger.info("Wrote feature_store.yaml at %s", yaml_path)

    grid_files = sorted(offline_grid_dir.glob("*.parquet"))
    if not grid_files:
        raise FileNotFoundError(f"No grid parquet files in {offline_grid_dir}")
    grid_schema = pq.read_schema(grid_files[0])

    bins_files = sorted(offline_bins_dir.glob("*.parquet"))
    if not bins_files:
        raise FileNotFoundError(f"No bins parquet files in {offline_bins_dir}")
    bins_schema = pq.read_schema(bins_files[0])

    # Single composite join key per entity (feast ≤ v0.60 limitation)
    bins_entity = Entity(name="bins_entity", join_keys=["bins_key"])
    grid_entity = Entity(name="grid_entity", join_keys=["grid_key"])

    bins_exclude = frozenset({"bins_key", "dataset_id", "bin_seq", "event_timestamp"})
    grid_exclude = frozenset({"grid_key", "dataset_id", "bin_seq", "k", "event_timestamp"})

    bins_view = FeatureView(
        name="bins_view",
        entities=[bins_entity],
        schema=_build_schema_fields(bins_schema, bins_exclude),
        source=FileSource(
            path=str(offline_bins_dir),
            timestamp_field="event_timestamp",
        ),
        ttl=timedelta(days=3650),
    )
    grid_view = FeatureView(
        name="grid_view",
        entities=[grid_entity],
        schema=_build_schema_fields(grid_schema, grid_exclude),
        source=FileSource(
            path=str(offline_grid_dir),
            timestamp_field="event_timestamp",
        ),
        ttl=timedelta(days=3650),
    )

    store = FeatureStore(repo_path=str(fs_root))
    store.apply([bins_entity, grid_entity, bins_view, grid_view])
    logger.info("Applied Feast entities and feature views")


def sync_dataset_to_feature_store(
    dataset_id: str,
    paths: object,
    lake_root: Path,
    config: FeatureStoreConfig,
) -> None:
    """Sync an immutable dataset to the Feast offline feature store.

    Idempotent: if the dataset is already synced, returns immediately.

    Args:
        dataset_id: Dataset identifier string.
        paths: DatasetPaths with bins_parquet and grid_clean_parquet attributes.
        lake_root: Root path of the data lake.
        config: Feature store configuration.
    """
    fs_root = Path(lake_root) / "research" / "feature_store"
    offline_bins_dir = fs_root / "offline" / "bins"
    offline_grid_dir = fs_root / "offline" / "grid"

    bins_out = offline_bins_dir / f"{dataset_id}.parquet"
    if bins_out.exists():
        logger.info("Dataset '%s' already in feature store; skipping sync", dataset_id)
        return

    offline_bins_dir.mkdir(parents=True, exist_ok=True)
    offline_grid_dir.mkdir(parents=True, exist_ok=True)

    # --- Bins ---
    bins_df = pd.read_parquet(paths.bins_parquet)
    bins_df["dataset_id"] = dataset_id
    bins_df["event_timestamp"] = pd.to_datetime(bins_df["ts_ns"], unit="ns", utc=True)
    # Single composite join key (feast v0.60 supports only one join key per entity)
    bins_df["bins_key"] = dataset_id + "__" + bins_df["bin_seq"].astype(str)
    bins_df.to_parquet(bins_out, index=False)
    logger.info("Wrote bins feature store parquet: %s", bins_out)

    # --- Grid ---
    grid_df = pd.read_parquet(paths.grid_clean_parquet)
    bin_ts = bins_df[["bin_seq", "event_timestamp"]].set_index("bin_seq")
    grid_df["dataset_id"] = dataset_id
    grid_df["event_timestamp"] = grid_df["bin_seq"].map(bin_ts["event_timestamp"])
    # Single composite join key
    grid_df["grid_key"] = (
        dataset_id + "__" + grid_df["bin_seq"].astype(str) + "__" + grid_df["k"].astype(str)
    )
    grid_out = offline_grid_dir / f"{dataset_id}.parquet"
    grid_df.to_parquet(grid_out, index=False)
    logger.info("Wrote grid feature store parquet: %s", grid_out)

    _ensure_feast_repo(fs_root, offline_bins_dir, offline_grid_dir, config)
    logger.info("Feature store sync complete for dataset '%s'", dataset_id)
