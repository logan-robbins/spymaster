from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ...base import Stage, StageIO
from ....config import AppConfig
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


class GoldComputeEquityOptionPhysicsSurface1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_compute_equity_option_physics_surface_1s",
            io=StageIO(
                inputs=["silver.equity_option_cmbp_1.depth_and_flow_1s"],
                output="gold.equity_option_cmbp_1.physics_surface_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        flow_key = "silver.equity_option_cmbp_1.depth_and_flow_1s"
        flow_ref = partition_ref(cfg, flow_key, symbol, dt)
        if not is_partition_complete(flow_ref):
            raise FileNotFoundError(f"Input not ready: {flow_ref.dataset_key} dt={dt}")

        flow_contract = load_avro_contract(repo_root / cfg.dataset(flow_key).contract)
        df_flow = enforce_contract(read_partition(flow_ref), flow_contract)

        # Apply gold strict filters for institutional-grade data
        original_flow_count = len(df_flow)
        df_flow, flow_stats = apply_gold_strict_filters(df_flow, product_type="equity_options", return_stats=True)
        if flow_stats.get("total_filtered", 0) > 0:
            logger.info(
                f"Gold filters for {dt}: flow={flow_stats.get('total_filtered', 0)}/{original_flow_count}"
            )

        df_out = self.transform(df_flow)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
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

    def transform(self, df_flow: pd.DataFrame) -> pd.DataFrame:
        if df_flow.empty:
            return pd.DataFrame(
                columns=[
                    "window_end_ts_ns",
                    "event_ts_ns",
                    "spot_ref_price_int",
                    "strike_price_int",
                    "strike_points",
                    "rel_ticks",
                    "right",
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
                "strike_price_int",
                "strike_points",
                "rel_ticks",
                "right",
                "side",
                "add_intensity",
                "fill_intensity",
                "pull_intensity",
                "liquidity_velocity",
            ]
        ]
