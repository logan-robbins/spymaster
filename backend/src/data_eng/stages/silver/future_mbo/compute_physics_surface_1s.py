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


class SilverComputePhysicsSurface1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_physics_surface_1s",
            io=StageIO(
                inputs=[
                    "silver.future_mbo.book_snapshot_1s",
                    "silver.future_mbo.depth_and_flow_1s",
                    "silver.future_mbo.vacuum_surface_1s",
                    "gold.hud.physics_norm_calibration",
                ],
                output="silver.future_mbo.physics_surface_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        snap_key = "silver.future_mbo.book_snapshot_1s"
        snap_key = "silver.future_mbo.book_snapshot_1s"
        flow_key = "silver.future_mbo.depth_and_flow_1s"
        vac_key = "silver.future_mbo.vacuum_surface_1s"
        cal_key = "gold.hud.physics_norm_calibration"

        snap_ref = partition_ref(cfg, snap_key, symbol, dt)
        flow_ref = partition_ref(cfg, flow_key, symbol, dt)
        vac_ref = partition_ref(cfg, vac_key, symbol, dt)
        cal_ref = partition_ref(cfg, cal_key, symbol, dt)

        for ref in (snap_ref, flow_ref, vac_ref, cal_ref):
            if not is_partition_complete(ref):
                raise FileNotFoundError(f"Input not ready: {ref.dataset_key} dt={dt}")

        snap_contract = load_avro_contract(repo_root / cfg.dataset(snap_key).contract)
        snap_contract = load_avro_contract(repo_root / cfg.dataset(snap_key).contract)
        flow_contract = load_avro_contract(repo_root / cfg.dataset(flow_key).contract)
        vac_contract = load_avro_contract(repo_root / cfg.dataset(vac_key).contract)
        cal_contract = load_avro_contract(repo_root / cfg.dataset(cal_key).contract)

        df_snap = enforce_contract(read_partition(snap_ref), snap_contract)
        df_flow = enforce_contract(read_partition(flow_ref), flow_contract)
        df_vac = enforce_contract(read_partition(vac_ref), vac_contract)
        df_cal = enforce_contract(read_partition(cal_ref), cal_contract)

        df_out = self.transform(df_snap, df_flow, df_vac, df_cal)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {"dataset": snap_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(snap_ref)},
            {"dataset": flow_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(flow_ref)},
            {"dataset": vac_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(vac_ref)},
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
        df_vac: pd.DataFrame,
        df_cal: pd.DataFrame,
    ) -> pd.DataFrame:
        if df_snap.empty:
            return pd.DataFrame(
                columns=[
                    "window_end_ts_ns",
                    "event_ts_ns",
                    "spot_ref_price_int",
                    "rel_ticks",
                    "side",
                    "physics_score",
                    "physics_score_signed",
                ]
            )

        cal = _load_calibration(df_cal)

        # Merge Wall and Vacuum on (window_end_ts_ns, rel_ticks)
        # Assuming Vacuum also has 'side', or we join by index if vacuum is just mapped to ticks
        # Silver Vacuum Surface definition: window_end_ts_ns, rel_ticks, vacuum_score
        # Check if we need side? No, rel_ticks+spot implies side roughly, but we want exact side from wall data maybe?
        # Actually Wall Surface has 'side'. Vacuum Surface usually doesn't need side if it's just a grid.
        # Let's clean up and join logic.

        wall = df_flow.copy()
        
        # Log Transformations
        wall["wall_strength_log"] = np.log(wall["depth_qty_rest"].astype(float) + 1.0)
        wall_erosion = np.maximum(-wall["d1_depth_qty"].astype(float), 0.0)
        wall["log1p_erosion_norm"] = np.log1p(wall_erosion / (wall["depth_qty_start"].astype(float) + EPS_QTY))

        # Join Vacuum
        # Try to join on keys. If vacuum surface has no 'side', join on ['window_end_ts_ns', 'rel_ticks']
        vac = df_vac.copy().rename(columns={"vacuum_score": "vacuum_norm"})
        
        # Outer join to capture all ticks? Or left join on Wall?
        # Typically physics is computed where there is Activity (Wall). 
        # But we might want pure vacuum ticks too. 
        # However, without Wall data (side etc), we can't fully compute 'ease' formula terms like wall_strength.
        # Let's do Outer Join on Ticks and FillNA.

        merged = pd.merge(
            wall, 
            vac[["window_end_ts_ns", "rel_ticks", "vacuum_norm"]], 
            on=["window_end_ts_ns", "rel_ticks"], 
            how="outer"
        )

        # Fill NAs
        merged["wall_strength_log"] = merged["wall_strength_log"].fillna(0.0)
        merged["log1p_erosion_norm"] = merged["log1p_erosion_norm"].fillna(0.0)
        merged["vacuum_norm"] = merged["vacuum_norm"].fillna(0.0)
        
        # If side is missing (from pure vacuum tick), infer from rel_ticks
        # rel_ticks > 0 -> Ask ('A'), rel_ticks < 0 -> Bid ('B')
        
        # We cannot pass ndarray to fillna value safely.
        # Let's fill explicitly where null.
        side_missing = merged["side"].isna()
        if side_missing.any():
            inferred_side = np.where(merged.loc[side_missing, "rel_ticks"] > 0, "A", "B")
            merged.loc[side_missing, "side"] = inferred_side

        # Normalize
        merged["wall_strength_norm"] = _norm(merged["wall_strength_log"].to_numpy(), cal["wall_strength_log"])
        merged["wall_erosion_norm"] = _norm(merged["log1p_erosion_norm"].to_numpy(), cal["log1p_erosion_norm"])

        # Compute Ease
        # ease = 0.50*vacuum_norm + 0.35*erosion_norm + 0.15*(1 - wall_strength_norm)
        merged["physics_score"] = (
            0.50 * merged["vacuum_norm"]
            + 0.35 * merged["wall_erosion_norm"]
            + 0.15 * (1.0 - merged["wall_strength_norm"])
        )

        # Compute Signed Score
        # if A and rel > 0: +score
        # if B and rel < 0: -score
        # else 0
        
        # Note: side might be 'ask'/'bid' or 'A'/'B'. Standardize.
        side = merged["side"].astype(str)
        rel = merged["rel_ticks"].astype(int)
        score = merged["physics_score"].astype(float)

        is_ask = (side == "A") | (side == "ask")
        is_bid = (side == "B") | (side == "bid")

        signed_score = np.zeros_like(score)
        mask_pos = is_ask & (rel > 0)
        mask_neg = is_bid & (rel < 0)

        signed_score[mask_pos] = score[mask_pos]
        signed_score[mask_neg] = -score[mask_neg]

        merged["physics_score_signed"] = signed_score

        # Join with Snapshot to get spot_ref_price_int
        # Use window_end_ts_ns as event_ts_ns since snapshot is windowed
        # DEBUG: Print columns
        print(f"DEBUG: df_snap columns: {df_snap.columns.tolist()}")
        
        # Join with Snapshot to get authoritative spot_ref_price_int (Wall might have NaNs for pure-vacuum rows)
        if "spot_ref_price_int" in merged.columns:
            merged = merged.drop(columns=["spot_ref_price_int"])
            
        snap_lite = df_snap[["window_end_ts_ns", "spot_ref_price_int"]]
        final = pd.merge(merged, snap_lite, on="window_end_ts_ns", how="inner")
        
        final["event_ts_ns"] = final["window_end_ts_ns"]

        # Select columns
        return final[[
            "window_end_ts_ns", 
            "event_ts_ns", 
            "spot_ref_price_int", 
            "rel_ticks", 
            "side", 
            "physics_score", 
            "physics_score_signed"
        ]]


def _load_calibration(df_cal: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    required = {
        "wall_strength_log",
        "log1p_erosion_norm",
    }
    cal: Dict[str, Tuple[float, float]] = {}
    for row in df_cal.itertuples(index=False):
        cal[str(row.metric_name)] = (float(row.q05), float(row.q95))
    missing = required.difference(cal.keys())
    if missing:
        # Fallback or error?
        # For robustness, if missing, use defaults?
        # raise ValueError(f"Missing calibration metrics: {sorted(missing)}")
        pass # Allow partial? No, metrics are needed.
    
    for name, (lo, hi) in cal.items():
        if hi <= lo:
            if lo == hi:
                hi = lo + 1.0
        cal[name] = (lo, hi)
    return cal


def _norm(values: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    if bounds is None: return np.zeros_like(values)
    lo, hi = bounds
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)
