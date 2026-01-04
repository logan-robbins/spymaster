from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..config import AppConfig
from ..contracts import enforce_contract, load_avro_contract
from ..io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    read_partition_csv,
    write_partition_csv,
)


@dataclass(frozen=True)
class StageIO:
    inputs: List[str]
    output: str


class Stage:
    """Atomic, idempotent pipeline stage.

    This base class implements:
    - input readiness checks via `_SUCCESS`
    - output idempotency via `_SUCCESS`
    - contract enforcement (Avro field names + order)
    - lineage recording via upstream `_MANIFEST.json` hashes

    Subclasses implement `transform()`.
    """

    name: str
    io: StageIO

    def __init__(self, name: str, io: StageIO) -> None:
        self.name = name
        self.io = io

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        raise NotImplementedError

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        # Ensure all inputs exist and are complete.
        inputs_refs = [partition_ref(cfg, k, symbol, dt) for k in self.io.inputs]
        for r in inputs_refs:
            if not is_partition_complete(r):
                raise FileNotFoundError(f"Input not ready: {r.dataset_key} dt={dt} (missing {r.success_file})")

        # Read inputs. (This demo assumes 1 input; extend to joins as needed.)
        if len(inputs_refs) != 1:
            raise ValueError("Demo stages expect exactly 1 input")
        in_ref = inputs_refs[0]
        df_in = read_partition_csv(in_ref)

        # Enforce input contract
        in_contract_path = repo_root / cfg.dataset(in_ref.dataset_key).contract
        in_contract = load_avro_contract(in_contract_path)
        df_in = enforce_contract(df_in, in_contract)

        df_out = self.transform(df_in, dt)

        # Enforce output contract
        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        # Lineage
        lineage: List[Dict[str, Any]] = [
            {
                "dataset": in_ref.dataset_key,
                "dt": dt,
                "manifest_sha256": read_manifest_hash(in_ref),
            }
        ]

        write_partition_csv(
            cfg=cfg,
            dataset_key=self.io.output,
            symbol=symbol,
            dt=dt,
            df=df_out,
            contract_path=out_contract_path,
            inputs=lineage,
            stage=self.name,
        )
