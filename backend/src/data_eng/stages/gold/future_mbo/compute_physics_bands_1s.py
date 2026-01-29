from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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

BANDS: List[Tuple[str, int, int]] = [
    ("at", 0, 2),
    ("near", 3, 5),
    ("mid", 6, 14),
    ("far", 15, 20),
]


class GoldComputePhysicsBands1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_compute_physics_bands_1s",
            io=StageIO(
                inputs=[
                    "silver.future_mbo.book_snapshot_1s",
                    "gold.future_mbo.physics_surface_1s",
                    "gold.hud.physics_norm_calibration",
                ],
                output="gold.future_mbo.physics_bands_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        snap_key = "silver.future_mbo.book_snapshot_1s"
        surf_key = "gold.future_mbo.physics_surface_1s"
        cal_key = "gold.hud.physics_norm_calibration"

        snap_ref = partition_ref(cfg, snap_key, symbol, dt)
        surf_ref = partition_ref(cfg, surf_key, symbol, dt)
        cal_ref = partition_ref(cfg, cal_key, symbol, dt)

        for ref in (snap_ref, surf_ref, cal_ref):
            if not is_partition_complete(ref):
                raise FileNotFoundError(f"Input not ready: {ref.dataset_key} dt={dt}")

        snap_contract = load_avro_contract(repo_root / cfg.dataset(snap_key).contract)
        surf_contract = load_avro_contract(repo_root / cfg.dataset(surf_key).contract)
        cal_contract = load_avro_contract(repo_root / cfg.dataset(cal_key).contract)

        df_snap = enforce_contract(read_partition(snap_ref), snap_contract)
        df_surf = enforce_contract(read_partition(surf_ref), surf_contract)
        df_cal = enforce_contract(read_partition(cal_ref), cal_contract)

        df_out = self.transform(df_snap, df_surf, df_cal)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {"dataset": snap_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(snap_ref)},
            {"dataset": surf_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(surf_ref)},
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
        df_surf: pd.DataFrame,
        df_cal: pd.DataFrame,
    ) -> pd.DataFrame:
        if df_snap.empty or df_surf.empty:
            return pd.DataFrame(
                columns=[
                    "window_start_ts_ns",
                    "window_end_ts_ns",
                    "spot_ref_price_int",
                    "mid_price",
                    "mid_price_int",
                    # Add Intensity
                    "above_at_add_intensity", "above_near_add_intensity", "above_mid_add_intensity", "above_far_add_intensity",
                    "below_at_add_intensity", "below_near_add_intensity", "below_mid_add_intensity", "below_far_add_intensity",
                    # Fill Intensity
                    "above_at_fill_intensity", "above_near_fill_intensity", "above_mid_fill_intensity", "above_far_fill_intensity",
                    "below_at_fill_intensity", "below_near_fill_intensity", "below_mid_fill_intensity", "below_far_fill_intensity",
                    # Pull Intensity
                    "above_at_pull_intensity", "above_near_pull_intensity", "above_mid_pull_intensity", "above_far_pull_intensity",
                    "below_at_pull_intensity", "below_near_pull_intensity", "below_mid_pull_intensity", "below_far_pull_intensity",
                    # Liquidity Velocity
                    "above_at_liquidity_velocity", "above_near_liquidity_velocity", "above_mid_liquidity_velocity", "above_far_liquidity_velocity",
                    "below_at_liquidity_velocity", "below_near_liquidity_velocity", "below_mid_liquidity_velocity", "below_far_liquidity_velocity",
                    # Wall Strength
                    "above_at_wall_strength", "above_near_wall_strength", "above_mid_wall_strength", "above_far_wall_strength",
                    "below_at_wall_strength", "below_near_wall_strength", "below_mid_wall_strength", "below_far_wall_strength",
                ]
            )

        # -----------------------------
        # Aggregation
        # -----------------------------
        # We group by Band -> Mean(Intensity)
        
        surf = _band_filter(df_surf.copy())
        
        metrics = ["add_intensity", "fill_intensity", "pull_intensity", "liquidity_velocity", "wall_strength"]
        
        agg = (
            surf.groupby(["window_end_ts_ns", "direction", "band"], as_index=False)[metrics]
            .mean()
        )
        
        # Pivot
        wide = agg.pivot_table(
            index="window_end_ts_ns",
            columns=["direction", "band"],
            values=metrics,
            fill_value=0.0,
        )
        wide.columns = [
            f"{direction}_{band}_{metric}"
            for metric, direction, band in wide.columns.to_flat_index()
        ]
        wide = wide.reset_index()

        base = df_snap[
            ["window_start_ts_ns", "window_end_ts_ns", "spot_ref_price_int", "mid_price", "mid_price_int"]
        ].copy()

        out = base.merge(wide, on="window_end_ts_ns", how="left")
        out = out.fillna(0.0)
        out = _ensure_columns_mechanics(out)

        return out[
            [
                "window_start_ts_ns",
                "window_end_ts_ns",
                "spot_ref_price_int",
                "mid_price",
                "mid_price_int",
                # Add
                "above_at_add_intensity", "above_near_add_intensity", "above_mid_add_intensity", "above_far_add_intensity",
                "below_at_add_intensity", "below_near_add_intensity", "below_mid_add_intensity", "below_far_add_intensity",
                # Fill
                "above_at_fill_intensity", "above_near_fill_intensity", "above_mid_fill_intensity", "above_far_fill_intensity",
                "below_at_fill_intensity", "below_near_fill_intensity", "below_mid_fill_intensity", "below_far_fill_intensity",
                # Pull
                "above_at_pull_intensity", "above_near_pull_intensity", "above_mid_pull_intensity", "above_far_pull_intensity",
                "below_at_pull_intensity", "below_near_pull_intensity", "below_mid_pull_intensity", "below_far_pull_intensity",
                # Liquidity Velocity
                "above_at_liquidity_velocity", "above_near_liquidity_velocity", "above_mid_liquidity_velocity", "above_far_liquidity_velocity",
                "below_at_liquidity_velocity", "below_near_liquidity_velocity", "below_mid_liquidity_velocity", "below_far_liquidity_velocity",
                # Strength
                "above_at_wall_strength", "above_near_wall_strength", "above_mid_wall_strength", "above_far_wall_strength",
                "below_at_wall_strength", "below_near_wall_strength", "below_mid_wall_strength", "below_far_wall_strength",
            ]
        ]


def _band_filter(df: pd.DataFrame) -> pd.DataFrame:
    rel_ticks = df["rel_ticks"].astype(int).to_numpy()
    side = df["side"].astype(str).to_numpy()
    direction = np.where((side == "A") & (rel_ticks > 0), "above", "")
    direction = np.where((side == "B") & (rel_ticks < 0), "below", direction)
    abs_ticks = np.abs(rel_ticks)

    band = np.full(len(df), "", dtype=object)
    for name, lo, hi in BANDS:
        mask = (abs_ticks >= lo) & (abs_ticks <= hi)
        band[mask] = name

    df = df.copy()
    df["direction"] = direction
    df["band"] = band
    df = df[(df["direction"] != "") & (df["band"] != "")]
    return df


def _ensure_columns_mechanics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["add_intensity", "fill_intensity", "pull_intensity", "liquidity_velocity", "wall_strength"]
    targets = []
    for direction in ("above", "below"):
        for band, _, _ in BANDS:
            for metric in metrics:
                targets.append(f"{direction}_{band}_{metric}")
                
    for col in targets:
        if col not in df.columns:
            df[col] = 0.0
    return df
