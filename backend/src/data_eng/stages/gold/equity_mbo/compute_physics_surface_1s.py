from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ...base import Stage, StageIO
from ....config import AppConfig, ProductConfig
from ....contracts import enforce_contract, load_avro_contract
from ....filters.gold_strict_filters import apply_gold_strict_filters
from ....io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    read_partition,
    write_partition,
)

logger = logging.getLogger(__name__)

EPS_QTY = 1.0


class GoldComputeEquityPhysicsSurface1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_compute_equity_physics_surface_1s",
            io=StageIO(
                inputs=[
                    "silver.equity_mbo.book_snapshot_1s",
                    "silver.equity_mbo.depth_and_flow_1s",
                ],
                output="gold.equity_mbo.physics_surface_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str, product: ProductConfig | None = None) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        snap_key = "silver.equity_mbo.book_snapshot_1s"
        flow_key = "silver.equity_mbo.depth_and_flow_1s"

        snap_ref = partition_ref(cfg, snap_key, symbol, dt)
        flow_ref = partition_ref(cfg, flow_key, symbol, dt)

        for ref in (snap_ref, flow_ref):
            if not is_partition_complete(ref):
                raise FileNotFoundError(f"Input not ready: {ref.dataset_key} dt={dt}")

        snap_contract = load_avro_contract(repo_root / cfg.dataset(snap_key).contract)
        flow_contract = load_avro_contract(repo_root / cfg.dataset(flow_key).contract)

        df_snap = enforce_contract(read_partition(snap_ref), snap_contract)
        df_flow = enforce_contract(read_partition(flow_ref), flow_contract)

        # Apply gold strict filters for institutional-grade data
        original_snap_count = len(df_snap)
        original_flow_count = len(df_flow)
        
        df_snap, snap_stats = apply_gold_strict_filters(df_snap, product_type="equities", return_stats=True)
        df_flow, flow_stats = apply_gold_strict_filters(df_flow, product_type="equities", return_stats=True)
        
        if snap_stats.get("total_filtered", 0) > 0 or flow_stats.get("total_filtered", 0) > 0:
            logger.info(
                f"Gold filters for {dt}: snap={snap_stats.get('total_filtered', 0)}/{original_snap_count}, "
                f"flow={flow_stats.get('total_filtered', 0)}/{original_flow_count}"
            )

        df_out = self.transform(df_snap, df_flow)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {"dataset": snap_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(snap_ref)},
            {"dataset": flow_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(flow_ref)},
        ]

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

    def transform(
        self,
        df_snap: pd.DataFrame,
        df_flow: pd.DataFrame,
    ) -> pd.DataFrame:
        if df_snap.empty or df_flow.empty:
            return pd.DataFrame(
                columns=[
                    "window_end_ts_ns",
                    "event_ts_ns",
                    "spot_ref_price_int",
                    "rel_ticks",
                    "rel_ticks_side",
                    "side",
                    "add_intensity",
                    "fill_intensity",
                    "pull_intensity",
                    "liquidity_velocity",
                ]
            )

        df = df_flow.copy()

        depth_start = df["depth_qty_start"].astype(float).to_numpy()
        add_qty = df["add_qty"].astype(float).to_numpy()
        fill_qty = df["fill_qty"].astype(float).to_numpy()
        pull_qty = df["pull_qty"].astype(float).to_numpy()

        denom = depth_start + EPS_QTY

        df["add_intensity"] = add_qty / denom
        df["fill_intensity"] = fill_qty / denom
        df["pull_intensity"] = pull_qty / denom
        df["liquidity_velocity"] = df["add_intensity"] - df["pull_intensity"] - df["fill_intensity"]

        df["event_ts_ns"] = df["window_end_ts_ns"]

        return df[
            [
                "window_end_ts_ns",
                "event_ts_ns",
                "spot_ref_price_int",
                "rel_ticks",
                "rel_ticks_side",
                "side",
                "add_intensity",
                "fill_intensity",
                "pull_intensity",
                "liquidity_velocity",
            ]
        ]
