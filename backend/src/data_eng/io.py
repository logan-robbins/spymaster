from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import AppConfig, DatasetSpec
from .contracts import contract_hash


@dataclass(frozen=True)
class PartitionRef:
    dataset_key: str
    dir: Path

    @property
    def data_file(self) -> Path:
        return self.dir / "part-00000.csv"

    @property
    def manifest_file(self) -> Path:
        return self.dir / "_MANIFEST.json"

    @property
    def success_file(self) -> Path:
        return self.dir / "_SUCCESS"


def partition_ref(cfg: AppConfig, dataset_key: str, symbol: str, dt: str) -> PartitionRef:
    spec = cfg.dataset(dataset_key)
    path = spec.path.format(symbol=symbol)
    partition_path = cfg.lake_root / path / f"dt={dt}"
    
    return PartitionRef(
        dataset_key=dataset_key,
        dir=partition_path,
    )


def is_partition_complete(ref: PartitionRef) -> bool:
    return ref.success_file.exists()


def read_partition_csv(ref: PartitionRef) -> pd.DataFrame:
    if not ref.data_file.exists():
        raise FileNotFoundError(f"Missing data file: {ref.data_file}")
    return pd.read_csv(ref.data_file)


def sha256_file(path: Path) -> str:
    return contract_hash(path)  # same implementation (sha256 of text)


def read_manifest_hash(ref: PartitionRef) -> str:
    if not ref.manifest_file.exists():
        raise FileNotFoundError(f"Missing manifest: {ref.manifest_file}")
    return sha256_file(ref.manifest_file)


def write_partition_csv(
    cfg: AppConfig,
    dataset_key: str,
    symbol: str,
    dt: str,
    df: pd.DataFrame,
    contract_path: Path,
    inputs: Optional[List[Dict[str, Any]]] = None,
    stage: str = "",
) -> PartitionRef:
    """Idempotent-ish partition writer (overwrite partition dir atomically)."""

    out = partition_ref(cfg, dataset_key, symbol, dt)
    out.dir.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = out.dir.parent / f".tmp_{uuid.uuid4().hex}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Write data
    df.to_csv(tmp_dir / "part-00000.csv", index=False)

    # Write manifest
    payload = {
        "dataset": dataset_key,
        "dt": dt,
        "row_count": int(df.shape[0]),
        "contract_path": str(contract_path.as_posix()),
        "contract_sha256": contract_hash(contract_path),
        "inputs": inputs or [],
        "stage": stage,
    }
    (tmp_dir / "_MANIFEST.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    # Completion marker
    (tmp_dir / "_SUCCESS").write_text("")

    # Atomic-ish commit
    if out.dir.exists():
        shutil.rmtree(out.dir)
    os.replace(tmp_dir, out.dir)

    return out
