from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import AppConfig, DatasetSpec
from .contracts import contract_hash


@dataclass(frozen=True)
class PartitionRef:
    dataset_key: str
    dir: Path
    format: str

    @property
    def manifest_file(self) -> Path:
        return self.dir / "_MANIFEST.json"

    @property
    def success_file(self) -> Path:
        return self.dir / "_SUCCESS"

    def list_data_files(self) -> List[Path]:
        if not self.dir.exists():
            return []
        
        if self.format == "parquet":
            return sorted(self.dir.glob("part-*.parquet"))
        elif self.format == "csv":
            return sorted(self.dir.glob("part-*.csv"))
        elif self.format == "dbn":
            return sorted(self.dir.glob("part-*.dbn"))
        else:
            raise ValueError(f"Unsupported format: {self.format}")


def partition_ref(
    cfg: AppConfig, 
    dataset_key: str, 
    symbol: str, 
    dt: str,
    hour: Optional[str] = None
) -> PartitionRef:
    spec = cfg.dataset(dataset_key)
    path = spec.path.format(symbol=symbol)
    
    if hour is not None:
        partition_path = cfg.lake_root / path / f"dt={dt}" / f"hour={hour}"
    else:
        partition_path = cfg.lake_root / path / f"dt={dt}"
    
    return PartitionRef(
        dataset_key=dataset_key,
        dir=partition_path,
        format=spec.format,
    )


def is_partition_complete(ref: PartitionRef) -> bool:
    return ref.success_file.exists()


def read_partition(ref: PartitionRef) -> pd.DataFrame:
    data_files = ref.list_data_files()
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in: {ref.dir}")
    
    dfs = []
    for file_path in data_files:
        if ref.format == "parquet":
            dfs.append(pd.read_parquet(file_path))
        elif ref.format == "csv":
            dfs.append(pd.read_csv(file_path))
        elif ref.format == "dbn":
            import databento as db
            store = db.DBNStore.from_file(str(file_path))
            dfs.append(store.to_df().reset_index())
        else:
            raise ValueError(f"Unsupported format: {ref.format}")
    
    if len(dfs) == 1:
        return dfs[0]
    
    return pd.concat(dfs, ignore_index=True)


def sha256_file(path: Path) -> str:
    return contract_hash(path)


def read_manifest_hash(ref: PartitionRef) -> str:
    if not ref.manifest_file.exists():
        raise FileNotFoundError(f"Missing manifest: {ref.manifest_file}")
    return sha256_file(ref.manifest_file)


def write_partition(
    cfg: AppConfig,
    dataset_key: str,
    symbol: str,
    dt: str,
    df: pd.DataFrame,
    contract_path: Path,
    inputs: Optional[List[Dict[str, Any]]] = None,
    stage: str = "",
    hour: Optional[str] = None,
) -> PartitionRef:
    """Atomic partition writer supporting parquet, csv, and dbn formats."""

    out = partition_ref(cfg, dataset_key, symbol, dt, hour)
    spec = cfg.dataset(dataset_key)
    
    out.dir.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = out.dir.parent / f".tmp_{uuid.uuid4().hex}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    if spec.format == "parquet":
        data_file = tmp_dir / "part-00000.parquet"
        df_sorted = df.sort_values('ts_event_ns') if 'ts_event_ns' in df.columns else df
        table = pa.Table.from_pandas(df_sorted, preserve_index=False)
        pq.write_table(
            table,
            data_file,
            compression='zstd',
            compression_level=3
        )
    elif spec.format == "csv":
        data_file = tmp_dir / "part-00000.csv"
        df.to_csv(data_file, index=False)
    elif spec.format == "dbn":
        raise NotImplementedError("DBN writing not implemented - use Databento API for ingestion")
    else:
        raise ValueError(f"Unsupported format: {spec.format}")

    payload = {
        "dataset": dataset_key,
        "dt": dt,
        "hour": hour,
        "row_count": int(df.shape[0]),
        "contract_path": str(contract_path.as_posix()),
        "contract_sha256": contract_hash(contract_path),
        "inputs": inputs or [],
        "stage": stage,
    }
    (tmp_dir / "_MANIFEST.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    (tmp_dir / "_SUCCESS").write_text("")

    if out.dir.exists():
        shutil.rmtree(out.dir)
    os.replace(tmp_dir, out.dir)

    return out
