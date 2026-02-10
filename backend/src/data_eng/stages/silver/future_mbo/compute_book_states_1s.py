from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...base import Stage, StageIO
from ....config import AppConfig, ProductConfig
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
    compute_futures_surfaces_1s_from_batches,
)
from .mbo_batches import first_hour_window_ns, iter_mbo_batches


class SilverComputeBookStates1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_book_states_1s",
            io=StageIO(
                inputs=["bronze.future_mbo.mbo"],
                output=[
                    "silver.future_mbo.book_snapshot_1s",
                    "silver.future_mbo.depth_and_flow_1s",
                ],
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str, product: ProductConfig | None = None) -> None:
        outputs = self.io.output
        if isinstance(outputs, str):
            outputs = [outputs]

        ref_snap = partition_ref(cfg, "silver.future_mbo.book_snapshot_1s", symbol, dt)
        ref_flow = partition_ref(cfg, "silver.future_mbo.depth_and_flow_1s", symbol, dt)

        # Basic check: if both exist, skip. Ideally we check both separate completeness.
        if is_partition_complete(ref_snap) and is_partition_complete(ref_flow):
            return

        # Warmup: 15 hours from output window start (09:30 ET = 14:30 UTC)
        # reaches back before 00:00 UTC, which is where the bronze futures
        # window now begins (capturing the Databento daily MBO snapshot with
        # F_SNAPSHOT=32).  This ensures the book engine consumes ALL bronze
        # data from the daily snapshot forward.
        WARMUP_NS = 3600_000_000_000 * 15  # 15 hours

        # Compute BOTH snapshot and depth/flow in one pass
        df_snap, df_flow, _ = compute_futures_surfaces_1s_from_batches(
            iter_mbo_batches(cfg, repo_root, symbol, dt, start_buffer_ns=WARMUP_NS),
            compute_depth_flow=True,
            product=product,
        )

        # Filter out warmup data from output
        start_ns, _ = first_hour_window_ns(dt)
        df_snap = df_snap[df_snap["window_end_ts_ns"] >= start_ns]
        df_flow = df_flow[df_flow["window_end_ts_ns"] >= start_ns]

        contract_path_snap = repo_root / cfg.dataset("silver.future_mbo.book_snapshot_1s").contract
        contract_path_flow = repo_root / cfg.dataset("silver.future_mbo.depth_and_flow_1s").contract

        contract_snap = load_avro_contract(contract_path_snap)
        contract_flow = load_avro_contract(contract_path_flow)

        df_snap = enforce_contract(df_snap, contract_snap)
        df_flow = enforce_contract(df_flow, contract_flow)

        mbo_ref = partition_ref(cfg, "bronze.future_mbo.mbo", symbol, dt)
        lineage = [{"dataset": mbo_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(mbo_ref)}]

        # Write Snapshot
        write_partition(
            cfg=cfg,
            dataset_key="silver.future_mbo.book_snapshot_1s",
            symbol=symbol,
            dt=dt,
            df=df_snap,
            contract_path=contract_path_snap,
            inputs=lineage,
            stage=self.name,
        )

        # Write DepthFlow
        write_partition(
            cfg=cfg,
            dataset_key="silver.future_mbo.depth_and_flow_1s",
            symbol=symbol,
            dt=dt,
            df=df_flow,
            contract_path=contract_path_flow,
            inputs=lineage,
            stage=self.name,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # NOTE: This stage produces multiple outputs, so transform is ambiguous.
        # We typically implement transform for the primary output or return a generic result.
        # For simple verification, we can return the flow dataframe or tuple.
        # However, Base Stage.transform expects one DF. 
        # We'll return df_flow for now as it is the superset of complexity.
        _, df_flow, _ = compute_futures_surfaces_1s_from_batches(
            [df],
            compute_depth_flow=True,
        )
        if df_flow.empty:
            df_flow = pd.DataFrame(columns=DEPTH_FLOW_COLUMNS)
        return df_flow
