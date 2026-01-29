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
                    "above_at_wall_strength_norm",
                    "above_at_wall_erosion_norm",
                    "above_at_vacuum_norm",
                    "above_at_ease",
                    "above_near_wall_strength_norm",
                    "above_near_wall_erosion_norm",
                    "above_near_vacuum_norm",
                    "above_near_ease",
                    "above_mid_wall_strength_norm",
                    "above_mid_wall_erosion_norm",
                    "above_mid_vacuum_norm",
                    "above_mid_ease",
                    "above_far_wall_strength_norm",
                    "above_far_wall_erosion_norm",
                    "above_far_vacuum_norm",
                    "above_far_ease",
                    "below_at_wall_strength_norm",
                    "below_at_wall_erosion_norm",
                    "below_at_vacuum_norm",
                    "below_at_ease",
                    "below_near_wall_strength_norm",
                    "below_near_wall_erosion_norm",
                    "below_near_vacuum_norm",
                    "below_near_ease",
                    "below_mid_wall_strength_norm",
                    "below_mid_wall_erosion_norm",
                    "below_mid_vacuum_norm",
                    "below_mid_ease",
                    "below_far_wall_strength_norm",
                    "below_far_wall_erosion_norm",
                    "below_far_vacuum_norm",
                    "below_far_ease",
                    "above_score",
                    "below_score",
                    "vacuum_total_score",
                ]
            )

        # NOTE: df_surf already has normalized metrics [0, 1]
        # wall_strength_norm, wall_erosion_norm, vacuum_score (vacuum_norm)
        
        # We process Surface -> Bands by grouping and taking mean.
        
        # Filter into bands
        # We need rel_ticks, side from df_surf
        surf = _band_filter(df_surf.copy())
        
        # Rename for clarity/aggregation
        # vacuum_score -> vacuum_norm
        surf = surf.rename(columns={"vacuum_score": "vacuum_norm"})
        
        # Aggregate
        # Group by Time, Direction, Band and Mean the NORMALIZED metrics.
        # This is a behavior change from Silver (which mean'd Log metrics then normalized),
        # but Architecturally preferred for Gold layer: "Average Physics" vs "Physics of Average".
        agg = (
            surf.groupby(["window_end_ts_ns", "direction", "band"], as_index=False)[
                ["wall_strength_norm", "wall_erosion_norm", "vacuum_norm"]
            ]
            .mean()
        )
        
        # Compute Ease on the Aggregated Band
        agg["ease"] = (
            0.50 * agg["vacuum_norm"]
            + 0.35 * agg["wall_erosion_norm"]
            + 0.15 * (1.0 - agg["wall_strength_norm"])
        )

        wide = agg.pivot_table(
            index="window_end_ts_ns",
            columns=["direction", "band"],
            values=["wall_strength_norm", "wall_erosion_norm", "vacuum_norm", "ease"],
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
        out = _ensure_columns(out)

        out["above_score"] = 0.60 * out["above_at_ease"] + 0.40 * out["above_near_ease"]
        out["below_score"] = 0.60 * out["below_at_ease"] + 0.40 * out["below_near_ease"]
        out["vacuum_total_score"] = (out["above_score"] + out["below_score"]) / 2.0

        return out[
            [
                "window_start_ts_ns",
                "window_end_ts_ns",
                "spot_ref_price_int",
                "mid_price",
                "mid_price_int",
                "above_at_wall_strength_norm",
                "above_at_wall_erosion_norm",
                "above_at_vacuum_norm",
                "above_at_ease",
                "above_near_wall_strength_norm",
                "above_near_wall_erosion_norm",
                "above_near_vacuum_norm",
                "above_near_ease",
                "above_mid_wall_strength_norm",
                "above_mid_wall_erosion_norm",
                "above_mid_vacuum_norm",
                "above_mid_ease",
                "above_far_wall_strength_norm",
                "above_far_wall_erosion_norm",
                "above_far_vacuum_norm",
                "above_far_ease",
                "below_at_wall_strength_norm",
                "below_at_wall_erosion_norm",
                "below_at_vacuum_norm",
                "below_at_ease",
                "below_near_wall_strength_norm",
                "below_near_wall_erosion_norm",
                "below_near_vacuum_norm",
                "below_near_ease",
                "below_mid_wall_strength_norm",
                "below_mid_wall_erosion_norm",
                "below_mid_vacuum_norm",
                "below_mid_ease",
                "below_far_wall_strength_norm",
                "below_far_wall_erosion_norm",
                "below_far_vacuum_norm",
                "below_far_ease",
                "above_score",
                "below_score",
                "vacuum_total_score",
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


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    targets = []
    for direction in ("above", "below"):
        for band, _, _ in BANDS:
            targets.extend(
                [
                    f"{direction}_{band}_wall_strength_norm",
                    f"{direction}_{band}_wall_erosion_norm",
                    f"{direction}_{band}_vacuum_norm",
                    f"{direction}_{band}_ease",
                ]
            )
    for col in targets:
        if col not in df.columns:
            df[col] = 0.0
    return df
