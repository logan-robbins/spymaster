from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    """A logical dataset mapped to a physical location and a schema contract."""

    key: str
    path: str  # relative to lake_root
    format: str  # csv (demo) / parquet (prod)
    partition_keys: List[str]
    contract: str  # relative path to contract file


@dataclass(frozen=True)
class AppConfig:
    lake_root: Path
    datasets: Dict[str, DatasetSpec]

    def dataset(self, key: str) -> DatasetSpec:
        try:
            return self.datasets[key]
        except KeyError as e:
            raise KeyError(f"Unknown dataset key: {key}") from e


def load_config(repo_root: Path, config_path: Path) -> AppConfig:
    """Load config/datasets.yaml."""

    raw = yaml.safe_load(config_path.read_text())
    lake_root = repo_root / raw["lake_root"]

    ds: Dict[str, DatasetSpec] = {}
    for key, spec in raw["datasets"].items():
        ds[key] = DatasetSpec(
            key=key,
            path=spec["path"],
            format=spec["format"],
            partition_keys=list(spec["partition_keys"]),
            contract=spec["contract"],
        )

    return AppConfig(lake_root=lake_root, datasets=ds)
