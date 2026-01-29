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
from .book_engine import (
    DEPTH_FLOW_COLUMNS,
    SNAP_COLUMNS,
    compute_equity_surfaces_1s_from_batches,
)
from .mbo_batches import first_hour_window_ns, iter_mbo_batches


class SilverComputeEquityBookStates1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_equity_book_states_1s",
            io=StageIO(
                inputs=["bronze.equity_mbo.mbo"],
                output=[
                    "silver.equity_mbo.book_snapshot_1s",
                    "silver.equity_mbo.depth_and_flow_1s",
                ],
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        outputs = self.io.output
        if isinstance(outputs, str):
            outputs = [outputs]

        ref_snap = partition_ref(cfg, "silver.equity_mbo.book_snapshot_1s", symbol, dt)
        ref_flow = partition_ref(cfg, "silver.equity_mbo.depth_and_flow_1s", symbol, dt)

        if is_partition_complete(ref_snap) and is_partition_complete(ref_flow):
            return

        # Warmup to capture resting liquidity before the output window.
        WARMUP_NS = 3600_000_000_000 * 6  # 6 hours

        df_snap, df_flow, _ = compute_equity_surfaces_1s_from_batches(
            iter_mbo_batches(cfg, repo_root, symbol, dt, start_buffer_ns=WARMUP_NS),
            compute_depth_flow=True,
        )

        start_ns, _ = first_hour_window_ns(dt)
        df_snap = df_snap[df_snap["window_end_ts_ns"] >= start_ns]
        df_flow = df_flow[df_flow["window_end_ts_ns"] >= start_ns]

        contract_path_snap = repo_root / cfg.dataset("silver.equity_mbo.book_snapshot_1s").contract
        contract_path_flow = repo_root / cfg.dataset("silver.equity_mbo.depth_and_flow_1s").contract

        contract_snap = load_avro_contract(contract_path_snap)
        contract_flow = load_avro_contract(contract_path_flow)

        df_snap = enforce_contract(df_snap, contract_snap)
        df_flow = enforce_contract(df_flow, contract_flow)

        mbo_ref = partition_ref(cfg, "bronze.equity_mbo.mbo", symbol, dt)
        lineage = [{"dataset": mbo_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(mbo_ref)}]

        write_partition(
            cfg=cfg,
            dataset_key="silver.equity_mbo.book_snapshot_1s",
            symbol=symbol,
            dt=dt,
            df=df_snap,
            contract_path=contract_path_snap,
            inputs=lineage,
            stage=self.name,
        )

        write_partition(
            cfg=cfg,
            dataset_key="silver.equity_mbo.depth_and_flow_1s",
            symbol=symbol,
            dt=dt,
            df=df_flow,
            contract_path=contract_path_flow,
            inputs=lineage,
            stage=self.name,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        _, df_flow, _ = compute_equity_surfaces_1s_from_batches(
            [df],
            compute_depth_flow=True,
        )
        if df_flow.empty:
            df_flow = pd.DataFrame(columns=DEPTH_FLOW_COLUMNS)
        return df_flow
