"""Dataset registry for resolving experiment dataset IDs to file paths.

Searches two locations in priority order:
1. ``datasets/{id}/`` -- curated baseline datasets (read-only).
2. ``harness/generated_grids/{id}/`` -- grids regenerated from raw .dbn with
   non-default grid variant parameters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved file paths for a single dataset.

    Attributes:
        bins_parquet: Path to the bins.parquet file (time bins with mid_price, ts_ns).
        grid_clean_parquet: Path to the grid_clean.parquet file (k x bin_seq grid columns).
        dataset_id: The string identifier for this dataset.
    """

    bins_parquet: Path
    grid_clean_parquet: Path
    dataset_id: str


class DatasetRegistry:
    """Registry that resolves dataset IDs to their on-disk file paths.

    Searches ``{lake_root}/research/datasets/{id}/`` first, then
    ``{lake_root}/research/harness/generated_grids/{id}/``.

    Args:
        lake_root: Root path of the data lake (e.g. ``backend/lake``).
    """

    def __init__(self, lake_root: Path) -> None:
        self._lake_root = Path(lake_root)
        self._immutable_root = self._lake_root / "research" / "datasets"
        self._generated_root = self._lake_root / "research" / "harness" / "generated_grids"

    def resolve(self, dataset_id: str) -> DatasetPaths:
        """Resolve a dataset ID to its file paths.

        Checks immutable datasets first, then generated grids.

        Args:
            dataset_id: Unique string identifier for the dataset
                (e.g. ``mnqh6_20260206_0925_1025``).

        Returns:
            DatasetPaths with resolved bin and grid parquet paths.

        Raises:
            FileNotFoundError: If the dataset is not found in either location.
        """
        # Check immutable datasets first (curated baselines)
        immutable_dir = self._immutable_root / dataset_id
        bins_path = immutable_dir / "bins.parquet"
        if bins_path.exists():
            grid_path = immutable_dir / "grid_clean.parquet"
            if not grid_path.exists():
                raise FileNotFoundError(
                    f"Dataset '{dataset_id}' has bins.parquet but missing "
                    f"grid_clean.parquet in {immutable_dir}"
                )
            logger.debug("Resolved dataset '%s' from immutable: %s", dataset_id, immutable_dir)
            return DatasetPaths(
                bins_parquet=bins_path,
                grid_clean_parquet=grid_path,
                dataset_id=dataset_id,
            )

        # Check generated grids (from grid variant sweeps)
        generated_dir = self._generated_root / dataset_id
        bins_path = generated_dir / "bins.parquet"
        if bins_path.exists():
            grid_path = generated_dir / "grid_clean.parquet"
            if not grid_path.exists():
                raise FileNotFoundError(
                    f"Dataset '{dataset_id}' has bins.parquet but missing "
                    f"grid_clean.parquet in {generated_dir}"
                )
            logger.debug("Resolved dataset '%s' from generated: %s", dataset_id, generated_dir)
            return DatasetPaths(
                bins_parquet=bins_path,
                grid_clean_parquet=grid_path,
                dataset_id=dataset_id,
            )

        raise FileNotFoundError(
            f"Dataset '{dataset_id}' not found. Searched:\n"
            f"  - {self._immutable_root / dataset_id}\n"
            f"  - {self._generated_root / dataset_id}"
        )

    def list_datasets(self) -> list[str]:
        """List all available dataset IDs across both locations.

        Scans immutable and generated grid directories for subdirectories
        containing ``bins.parquet``.

        Returns:
            Sorted list of unique dataset ID strings.
        """
        dataset_ids: set[str] = set()

        for root_dir in (self._immutable_root, self._generated_root):
            if not root_dir.exists():
                continue
            for child in root_dir.iterdir():
                if child.is_dir() and (child / "bins.parquet").exists():
                    dataset_ids.add(child.name)

        return sorted(dataset_ids)

    def register_generated(self, variant_id: str, path: Path) -> None:
        """Register a generated grid dataset.

        Currently a no-op -- generated grids are auto-discovered by
        ``resolve()`` based on directory convention. This method exists
        as a hook for future explicit registration workflows.

        Args:
            variant_id: The variant identifier string.
            path: Path to the generated dataset directory.
        """
        logger.debug(
            "register_generated called for variant '%s' at %s (no-op)",
            variant_id,
            path,
        )
