from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d

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


class SilverComputeBucketRadar1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_bucket_radar_1s",
            io=StageIO(
                inputs=[
                    "silver.future_mbo.wall_surface_1s",
                    "silver.future_mbo.vacuum_surface_1s",
                    "silver.future_option_mbo.gex_surface_1s",
                    "gold.hud.physics_norm_calibration",
                ],
                output="silver.future_mbo.bucket_radar_surface_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        wall_key = "silver.future_mbo.wall_surface_1s"
        vac_key = "silver.future_mbo.vacuum_surface_1s"
        gex_key = "silver.future_option_mbo.gex_surface_1s"
        cal_key = "gold.hud.physics_norm_calibration"

        wall_ref = partition_ref(cfg, wall_key, symbol, dt)
        vac_ref = partition_ref(cfg, vac_key, symbol, dt)
        gex_ref = partition_ref(cfg, gex_key, symbol, dt)
        cal_ref = partition_ref(cfg, cal_key, symbol, dt)

        # Check inputs (GEX is optional-ish, but contract enforces readiness if listed)
        # If GEX is missing (e.g. no options data), we might want to proceed with zeros?
        # But standard pipeline expects strict dependencies.
        for ref in (wall_ref, vac_ref, cal_ref):
            if not is_partition_complete(ref):
                raise FileNotFoundError(f"Input not ready: {ref.dataset_key} dt={dt}")
        
        # GEX might handle different symbol? Usually same root symbol 'ES'.
        # If GEX not ready, we can warn or fail. 
        # Assuming it exists for now as per task.
        if not is_partition_complete(gex_ref):
             # Try fallback or fail? Fail fast as per rules.
             raise FileNotFoundError(f"Input not ready: {gex_ref.dataset_key} dt={dt}")

        wall_contract = load_avro_contract(repo_root / cfg.dataset(wall_key).contract)
        vac_contract = load_avro_contract(repo_root / cfg.dataset(vac_key).contract)
        gex_contract = load_avro_contract(repo_root / cfg.dataset(gex_key).contract)
        cal_contract = load_avro_contract(repo_root / cfg.dataset(cal_key).contract)

        df_wall = enforce_contract(read_partition(wall_ref), wall_contract)
        df_vac = enforce_contract(read_partition(vac_ref), vac_contract)
        df_gex = enforce_contract(read_partition(gex_ref), gex_contract)
        df_cal = enforce_contract(read_partition(cal_ref), cal_contract)

        df_out = self.transform(df_wall, df_vac, df_gex, df_cal)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {"dataset": wall_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(wall_ref)},
            {"dataset": vac_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(vac_ref)},
            {"dataset": gex_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(gex_ref)},
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
        df_wall: pd.DataFrame,
        df_vac: pd.DataFrame,
        df_gex: pd.DataFrame,
        df_cal: pd.DataFrame,
    ) -> pd.DataFrame:
        if df_wall.empty:
            return pd.DataFrame(
                columns=[
                    "window_start_ts_ns",
                    "window_end_ts_ns",
                    "bucket_rel",
                    "blocked_level",
                    "cavitation",
                    "gex_stiffness",
                    "mobility",
                    "spot_ref_price_int"
                ]
            )

        cal = _load_calibration(df_cal)
        
        # 1. Prepare Wall (Mass)
        # Compute normalized wall strength
        wall = df_wall.copy()
        wall["wall_strength_log"] = np.log(wall["depth_qty_rest"].astype(float) + 1.0)
        wall["mass"] = _norm(wall["wall_strength_log"].to_numpy(), cal.get("wall_strength_log", (0, 10)))
        
        # Need spot_ref_price_int for output
        # Wall has spot_ref_price_int
        
        # 2. Prepare Vacuum
        vac = df_vac.copy()
        # Vacuum surface already has 'vacuum_score' 0..1
        vac = vac.rename(columns={"vacuum_score": "vacuum"})
        
        # 3. Prepare GEX
        gex = df_gex.copy()
        # Gex has 'gex_abs', need to normalize.
        # Use simple global norm or calibration if available?
        # Let's use robust scaling 0..1 based on observation or just percentile if available.
        # For now, simplistic normalization: clip(gex_abs / 2e9, 0, 1) or similar?
        # Actually, let's look for 'gex_abs' in calibration?
        # If not present, use a heuristic max.
        gex_max = cal.get("gex_abs", (0, 100000))[1]
        gex["stiffness"] = np.clip(gex["gex_abs"] / (gex_max + 1.0), 0.0, 1.0)

        # 4. Join Wall + Vacuum on (window_end_ts_ns, rel_ticks)
        # First ensure rel_ticks in wall/vacuum are compatible.
        
        # Merge Wall and Vacuum
        wv = pd.merge(
            wall[["window_start_ts_ns", "window_end_ts_ns", "spot_ref_price_int", "rel_ticks", "mass"]],
            vac[["window_end_ts_ns", "rel_ticks", "vacuum"]],
            on=["window_end_ts_ns", "rel_ticks"],
            how="outer"
        )
        wv["mass"] = wv["mass"].fillna(0.0)
        wv["vacuum"] = wv["vacuum"].fillna(0.0)
        
        # Fill missing grouping cols
        # If created by vacuum-only tick, might miss spot_ref?
        # Usually wall is dense enough or we forward fill?
        # Better: grouping by window to fill spot_ref
        
        # Propagate spot_ref and start_ts per window
        metadata = wall.groupby("window_end_ts_ns")[["window_start_ts_ns", "spot_ref_price_int"]].first().reset_index()
        wv = wv.drop(columns=["window_start_ts_ns", "spot_ref_price_int"], errors="ignore")
        wv = pd.merge(wv, metadata, on="window_end_ts_ns", how="inner") # Dropping orphan vacuum ticks if any
        
        # 5. Bucketization
        # bucket_rel = floor(rel_ticks / 2)
        # BUCKET_TICKS = 2
        
        # We need to map GEX to buckets too.
        # GEX rel_ticks are multiples of 20.
        # GEX rel_ticks=20 -> bucket_rel=10.
        
        wv["bucket_rel"] = np.floor(wv["rel_ticks"] / 2).astype(int)
        
        # Aggregation:
        # For Mass: Max or Mean? "Solid boxes". Mean seems safer for aliasing.
        # For Vacuum: Mean.
        
        bucketed = wv.groupby(["window_end_ts_ns", "bucket_rel"]).agg({
            "mass": "mean",
            "vacuum": "mean",
            "window_start_ts_ns": "first",
            "spot_ref_price_int": "first"
        }).reset_index()
        
        # 6. Merge GEX
        # GEX: bucket_rel = floor(rel_ticks / 2).
        gex["bucket_rel"] = np.floor(gex["rel_ticks"] / 2).astype(int)
        
        # GEX is sparse. We want to apply it to the bucket, maybe smooth it later.
        # Merge left onto bucketed (or outer?)
        # We only care about buckets near spot?
        # Let's do Left join onto bucketed set (which defines 'active' space from wall/vac).
        # OR should GEX force empty buckets to exist?
        # GEX barriers exist even if no liquidity. So Outer join.
        
        merged = pd.merge(
            bucketed,
            gex[["window_end_ts_ns", "bucket_rel", "stiffness"]],
            on=["window_end_ts_ns", "bucket_rel"],
            how="outer"
        )
        merged["mass"] = merged["mass"].fillna(0.0)
        merged["vacuum"] = merged["vacuum"].fillna(0.0)
        merged["stiffness"] = merged["stiffness"].fillna(0.0)
        
        # Restore metadata for GEX-only rows
        # If GEX introduced new rows, we need window_start_ts/spot_ref.
        # We can join metadata again.
        
        # Optimization: do this join once.
        merged = merged.drop(columns=["window_start_ts_ns", "spot_ref_price_int"], errors="ignore")
        merged = pd.merge(merged, metadata, on="window_end_ts_ns", how="inner")
        
        # 7. Compute Blockedness and Cavitation
        # R0 = clamp(mass + stiffness, 0, 1)
        # R_eff = R0 * (1 - vacuum)^gamma  (gamma=1)
        # BlockedLevel = round(5 * R_eff)
        # Cavitation = vacuum^1.5
        
        r0 = np.clip(merged["mass"] + merged["stiffness"], 0.0, 1.0)
        r_eff = r0 * (1.0 - merged["vacuum"]) # gamma=1
        
        merged["blocked_level"] = np.round(5 * r_eff).astype(int)
        merged["cavitation"] = np.power(merged["vacuum"], 1.5)
        
        # GEX stiffness output
        merged["gex_stiffness"] = merged["stiffness"]
        
        # Mobility: 1 - R_eff? Or the more complex physics score?
        # UPDATE.md: Mobility = 1 - R_eff
        merged["mobility"] = 1.0 - r_eff
        
        return merged[[
            "window_start_ts_ns",
            "window_end_ts_ns",
            "bucket_rel",
            "blocked_level",
            "cavitation",
            "gex_stiffness",
            "mobility",
            "spot_ref_price_int"
        ]]


def _load_calibration(df_cal: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    cal: Dict[str, Tuple[float, float]] = {}
    for row in df_cal.itertuples(index=False):
        cal[str(row.metric_name)] = (float(row.q05), float(row.q95))
    
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
