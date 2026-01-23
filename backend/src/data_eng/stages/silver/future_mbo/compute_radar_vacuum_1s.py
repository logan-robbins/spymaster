from __future__ import annotations

from pathlib import Path
import pandas as pd

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    write_partition,
)
from .book_engine import RADAR_COLUMNS, compute_futures_surfaces_1s_from_batches
from .mbo_batches import iter_mbo_batches


class SilverComputeRadarVacuum1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_radar_vacuum_1s",
            io=StageIO(
                inputs=["bronze.future_mbo.mbo"],
                output="silver.future_mbo.radar_vacuum_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        _, _, df_out = compute_futures_surfaces_1s_from_batches(
            iter_mbo_batches(cfg, repo_root, symbol, dt)
        )

        if df_out.empty:
            df_out = pd.DataFrame(columns=RADAR_COLUMNS)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        mbo_ref = partition_ref(cfg, "bronze.future_mbo.mbo", symbol, dt)
        lineage = [{"dataset": mbo_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(mbo_ref)}]

        write_partition(
            cfg=cfg,
            dataset_key=self.io.output,
            symbol=symbol,
            dt=dt,
            df=df_out,
            contract_path=out_contract_path,
            inputs=lineage,
            stage=self.name,
        )
