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
    read_partition,
    write_partition,
)

EPS_QTY = 1.0


class GoldComputePhysicsSurface1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_compute_physics_surface_1s",
            io=StageIO(
                inputs=[
                    "silver.future_mbo.book_snapshot_1s",
                    "silver.future_mbo.depth_and_flow_1s",
                    "gold.hud.physics_norm_calibration",
                ],
                output="gold.future_mbo.physics_surface_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        snap_key = "silver.future_mbo.book_snapshot_1s"
        flow_key = "silver.future_mbo.depth_and_flow_1s"
        cal_key = "gold.hud.physics_norm_calibration"

        snap_ref = partition_ref(cfg, snap_key, symbol, dt)
        flow_ref = partition_ref(cfg, flow_key, symbol, dt)
        cal_ref = partition_ref(cfg, cal_key, symbol, dt)

        for ref in (snap_ref, flow_ref, cal_ref):
            if not is_partition_complete(ref):
                raise FileNotFoundError(f"Input not ready: {ref.dataset_key} dt={dt}")

        snap_contract = load_avro_contract(repo_root / cfg.dataset(snap_key).contract)
        flow_contract = load_avro_contract(repo_root / cfg.dataset(flow_key).contract)
        cal_contract = load_avro_contract(repo_root / cfg.dataset(cal_key).contract)

        df_snap = enforce_contract(read_partition(snap_ref), snap_contract)
        df_flow = enforce_contract(read_partition(flow_ref), flow_contract)
        df_cal = enforce_contract(read_partition(cal_ref), cal_contract)

        df_out = self.transform(df_snap, df_flow, df_cal)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {"dataset": snap_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(snap_ref)},
            {"dataset": flow_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(flow_ref)},
            {"dataset": cal_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(cal_ref)},
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
        df_cal: pd.DataFrame,
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

        # -----------------------------
        # Symmetrical Mechanics
        # -----------------------------
        df = df_flow.copy()

        # Inputs
        depth_start = df["depth_qty_start"].astype(float).to_numpy()
        
        # Quantities
        add_qty = df["add_qty"].astype(float).to_numpy()
        fill_qty = df["fill_qty"].astype(float).to_numpy()
        pull_qty_total = df["pull_qty_total"].astype(float).to_numpy()
        
        # Intensities = Qty / (Depth + Epsilon)
        # We use depth_start as the baseline for "Impact relative to what was there"
        
        denom = depth_start + EPS_QTY
        
        df["add_intensity"] = add_qty / denom
        df["fill_intensity"] = fill_qty / denom
        df["pull_intensity"] = pull_qty_total / denom

        # Net Velocity (Add - Pull - Fill)
        # True net change in liquidity at this level
        # Positive = building (adds outpace removals)
        # Negative = eroding (pulls + fills outpace adds)
        df["liquidity_velocity"] = df["add_intensity"] - df["pull_intensity"] - df["fill_intensity"]

        df["event_ts_ns"] = df["window_end_ts_ns"] 

        return df[[
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
        ]]


