from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
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
                    "side",
                    "pull_intensity_rest",
                    "pull_add_log",
                    "wall_strength_log",
                    "wall_erosion",
                    "vacuum_score",
                    "physics_score",
                    "physics_score_signed",
                ]
            )

        cal = _load_calibration(df_cal)

        # -----------------------------
        # 1. Compute Base Metrics (Previously Vacuum Logic)
        # -----------------------------
        df = df_flow.copy()

        # Inputs
        depth_start = df["depth_qty_start"].astype(float).to_numpy()
        depth_rest = df["depth_qty_rest"].astype(float).to_numpy()
        pull_qty_rest = df["pull_qty_rest"].astype(float).to_numpy()
        add_qty = df["add_qty"].astype(float).to_numpy()
        d1_depth = df["d1_depth_qty"].astype(float).to_numpy()

        # Core Metrics calculation
        df["pull_intensity_rest"] = pull_qty_rest / (depth_start + EPS_QTY)
        df["pull_add_log"] = np.log((pull_qty_rest + EPS_QTY) / (add_qty + EPS_QTY))
        df["wall_strength_log"] = np.log(depth_rest + 1.0)
        df["wall_erosion"] = np.maximum(-d1_depth, 0.0)

        # Derivatives? 
        # The schema in JSON removed explicit d1/d2/d3 columns from the surface definition
        # except implicitly if needed for vacuum score. 
        # But wait, vacuum_score calculation (line 324 in json) uses 'd2_pull_add_log' if we kept it?
        # Actually in the *Refactored JSON*, I removed 'd2_pull_add_log' from input list of vacuum_score 
        # and from the stage output. Let's double check.
        # Checking my REPLACE step... yes I removed d1/d2/d3 from transformation_steps and built_from.
        # So I do not need to compute them for the output dataframe.

        # -----------------------------
        # 2. Compute "Vacuum" Score
        # -----------------------------
        # vacuum_score = Mean(Norm(pull_add), Norm(pull_intensity), Norm(erosion))
        # Wait, the JSON says vacuum_score is built from:
        # pull_add_log, pull_intensity_rest, wall_erosion, gold.hud.physics_norm_calibration
        
        log1p_pull_intensity = np.log1p(df["pull_intensity_rest"].to_numpy())
        log1p_erosion_norm = np.log1p(df["wall_erosion"].to_numpy() / (depth_start + EPS_QTY))
        
        n1 = _norm(df["pull_add_log"].to_numpy(), cal["pull_add_log"])
        n2 = _norm(log1p_pull_intensity, cal["log1p_pull_intensity_rest"])
        n3 = _norm(log1p_erosion_norm, cal["log1p_erosion_norm"])
        
        # d2_pull_add_log was removed from the list, so we average 3 components?
        # Or did I keep d2 in the JSON?
        # Looking at Step 52 diff: "d2_pull_add_log" was removed from "vacuum_score" built_from.
        # So we use 3 components.
        df["vacuum_score"] = (n1 + n2 + n3) / 3.0
        
        # -----------------------------
        # 3. Compute "Physics" Score (Composite)
        # -----------------------------
        # physics_score = 0.50*vacuum + 0.35*erosion + 0.15*(1-strength)
        
        # We need normalized strength and erosion for this formula
        wall_strength_norm = _norm(df["wall_strength_log"].to_numpy(), cal["wall_strength_log"])
        wall_erosion_norm = n3 # Helper var above
        
        df["physics_score"] = (
            0.50 * df["vacuum_score"]
            + 0.35 * wall_erosion_norm
            + 0.15 * (1.0 - wall_strength_norm)
        )
        
        # ADDED for Bands downstream
        df["wall_strength_norm"] = wall_strength_norm
        df["wall_erosion_norm"] = wall_erosion_norm

        # -----------------------------
        # 4. Compute Signed Score
        # -----------------------------
        side = df["side"].astype(str)
        rel = df["rel_ticks"].astype(int)
        score = df["physics_score"].astype(float)

        is_ask = (side == "A") | (side == "ask")
        is_bid = (side == "B") | (side == "bid")

        signed_score = np.zeros_like(score)
        mask_pos = is_ask & (rel > 0)
        mask_neg = is_bid & (rel < 0)

        signed_score[mask_pos] = score[mask_pos]
        signed_score[mask_neg] = -score[mask_neg]

        df["physics_score_signed"] = signed_score

        df["event_ts_ns"] = df["window_end_ts_ns"] 

        return df[[
            "window_end_ts_ns", 
            "event_ts_ns", 
            "spot_ref_price_int", 
            "rel_ticks", 
            "side", 
            "pull_intensity_rest",
            "pull_add_log",
            # "wall_strength_log",  <-- Replaced with Norm
            # "wall_erosion",       <-- Replaced with Norm
            "wall_strength_norm",
            "wall_erosion_norm",
            "vacuum_score",
            "physics_score", 
            "physics_score_signed"
        ]]


def _load_calibration(df_cal: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    required = {
        "pull_add_log",
        "log1p_pull_intensity_rest",
        "log1p_erosion_norm",
        "wall_strength_log",
    }
    cal: Dict[str, Tuple[float, float]] = {}
    for row in df_cal.itertuples(index=False):
        cal[str(row.metric_name)] = (float(row.q05), float(row.q95))
    
    missing = required.difference(cal.keys())
    # Partial calibration or strict?
    # Strict is safer for consistency.
    if missing:
        # Note: If previous calibration didn't have these keys, we might need to run calibration once.
        # But let's assume calibration file is up to date or will be regenerated.
        raise ValueError(f"Missing calibration metrics: {sorted(missing)}")

    for name, (lo, hi) in cal.items():
        if hi <= lo:
            if lo == hi:
                hi = lo + 1.0
            else:
                # Flat or inverted, just fix
                hi = lo + 1.0
        cal[name] = (lo, hi)
    return cal


def _norm(values: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)
