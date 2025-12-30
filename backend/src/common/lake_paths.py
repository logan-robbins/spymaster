"""
Data lake path helpers.

Provides hive-style partition directory helpers (e.g., version=..., date=...).
Used to keep canonical output locations consistent across pipeline runs.
"""

from __future__ import annotations

import re
from pathlib import Path


_ILLEGAL_PARTITION_CHARS = re.compile(r"[\\/]")
_WHITESPACE = re.compile(r"\s+")


def sanitize_partition_value(value: str) -> str:
    """Sanitize a hive partition value for safe filesystem usage."""
    if value is None:
        raise ValueError("Partition value cannot be None")
    val = str(value).strip()
    if not val:
        raise ValueError("Partition value cannot be empty")
    val = _ILLEGAL_PARTITION_CHARS.sub("_", val)
    val = _WHITESPACE.sub("_", val)
    return val


def version_partition(version: str) -> str:
    return f"version={sanitize_partition_value(version)}"


def date_partition(date: str) -> str:
    return f"date={sanitize_partition_value(date)}"


def silver_features_root(data_root: str | Path) -> Path:
    return Path(data_root) / "silver" / "features"


def silver_state_root(data_root: str | Path) -> Path:
    return Path(data_root) / "silver" / "state"


def gold_episodes_root(data_root: str | Path) -> Path:
    return Path(data_root) / "gold" / "episodes"


def gold_indices_root(data_root: str | Path) -> Path:
    return Path(data_root) / "gold" / "indices"


def canonical_signals_dir(data_root: str | Path, dataset: str, version: str) -> Path:
    return silver_features_root(data_root) / dataset / version_partition(version)


def canonical_state_dir(data_root: str | Path, dataset: str, version: str) -> Path:
    return silver_state_root(data_root) / dataset / version_partition(version)


def canonical_episodes_dir(data_root: str | Path, dataset: str, version: str) -> Path:
    return gold_episodes_root(data_root) / dataset / version_partition(version)


def canonical_indices_dir(data_root: str | Path, dataset: str, version: str) -> Path:
    return gold_indices_root(data_root) / dataset / version_partition(version)
