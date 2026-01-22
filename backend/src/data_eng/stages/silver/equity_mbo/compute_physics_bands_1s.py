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

EPS_QTY = 1.0

BANDS: List[Tuple[str, int, int]] = [
    ("at", 0, 2),
    ("near", 3, 5),
    ("mid", 6, 14),
    ("far", 15, 20),
]


class SilverComputeEquityPhysicsBands1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_equity_physics_bands_1s",
            io=StageIO(
                inputs=[
                    "silver.equity_mbo.book_snapshot_1s",
                    "silver.equity_mbo.wall_surface_1s",
                    "silver.equity_mbo.vacuum_surface_1s",
                    "gold.equity_mbo.physics_norm_calibration",
                ],
                output="silver.equity_mbo.physics_bands_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        snap_key = "silver.equity_mbo.book_snapshot_1s"
        wall_key = "silver.equity_mbo.wall_surface_1s"
        vac_key = "silver.equity_mbo.vacuum_surface_1s"
        cal_key = "gold.equity_mbo.physics_norm_calibration"

        snap_ref = partition_ref(cfg, snap_key, symbol, dt)
        wall_ref = partition_ref(cfg, wall_key, symbol, dt)
        vac_ref = partition_ref(cfg, vac_key, symbol, dt)
        cal_ref = partition_ref(cfg, cal_key, symbol, dt)

        for ref in (snap_ref, wall_ref, vac_ref, cal_ref):
            if not is_partition_complete(ref):
                raise FileNotFoundError(f"Input not ready: {ref.dataset_key} dt={dt}")

        snap_contract = load_avro_contract(repo_root / cfg.dataset(snap_key).contract)
        wall_contract = load_avro_contract(repo_root / cfg.dataset(wall_key).contract)
        vac_contract = load_avro_contract(repo_root / cfg.dataset(vac_key).contract)
        cal_contract = load_avro_contract(repo_root / cfg.dataset(cal_key).contract)

        df_snap = enforce_contract(read_partition(snap_ref), snap_contract)
        df_wall = enforce_contract(read_partition(wall_ref), wall_contract)
        df_vac = enforce_contract(read_partition(vac_ref), vac_contract)
        df_cal = enforce_contract(read_partition(cal_ref), cal_contract)

        df_out = self.transform_multi(df_snap, df_wall, df_vac, df_cal)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {"dataset": snap_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(snap_ref)},
            {"dataset": wall_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(wall_ref)},
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

    def transform_multi(
        self,
        df_snap: pd.DataFrame,
        df_wall: pd.DataFrame,
        df_vac: pd.DataFrame,
        df_cal: pd.DataFrame,
    ) -> pd.DataFrame:
        if df_snap.empty:
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
                    "above_near_wall_strength_norm",
                    "above_near_wall_erosion_norm",
                    "above_near_vacuum_norm",
                    "above_mid_wall_strength_norm",
                    "above_mid_wall_erosion_norm",
                    "above_mid_vacuum_norm",
                    "above_far_wall_strength_norm",
                    "above_far_wall_erosion_norm",
                    "above_far_vacuum_norm",
                    "below_at_wall_strength_norm",
                    "below_at_wall_erosion_norm",
                    "below_at_vacuum_norm",
                    "below_near_wall_strength_norm",
                    "below_near_wall_erosion_norm",
                    "below_near_vacuum_norm",
                    "below_mid_wall_strength_norm",
                    "below_mid_wall_erosion_norm",
                    "below_mid_vacuum_norm",
                    "below_far_wall_strength_norm",
                    "below_far_wall_erosion_norm",
                    "below_far_vacuum_norm",
                    "above_score",
                    "below_score",
                    "vacuum_total_score",
                ]
            )

        cal = _load_calibration(df_cal)

        wall = df_wall.copy()
        wall["wall_strength_log"] = np.log(wall["depth_qty_rest"].astype(float) + 1.0)
        wall_erosion = np.maximum(-wall["d1_depth_qty"].astype(float), 0.0)
        wall["log1p_erosion_norm"] = np.log1p(wall_erosion / (wall["depth_qty_start"].astype(float) + EPS_QTY))

        wall = _band_filter(wall)
        vac = _band_filter(df_vac.copy())

        wall_agg = (
            wall.groupby(["window_end_ts_ns", "direction", "band"], as_index=False)[
                ["wall_strength_log", "log1p_erosion_norm"]
            ]
            .mean()
        )
        vac_agg = (
            vac.groupby(["window_end_ts_ns", "direction", "band"], as_index=False)[["vacuum_score"]]
            .mean()
            .rename(columns={"vacuum_score": "vacuum_norm"})
        )

        agg = wall_agg.merge(vac_agg, on=["window_end_ts_ns", "direction", "band"], how="outer")
        agg["wall_strength_log"] = agg["wall_strength_log"].fillna(0.0)
        agg["log1p_erosion_norm"] = agg["log1p_erosion_norm"].fillna(0.0)
        agg["vacuum_norm"] = agg["vacuum_norm"].fillna(0.0)

        agg["wall_strength_norm"] = _norm(agg["wall_strength_log"].to_numpy(), cal["wall_strength_log"])
        agg["wall_erosion_norm"] = _norm(agg["log1p_erosion_norm"].to_numpy(), cal["log1p_erosion_norm"])
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
                "above_near_wall_strength_norm",
                "above_near_wall_erosion_norm",
                "above_near_vacuum_norm",
                "above_mid_wall_strength_norm",
                "above_mid_wall_erosion_norm",
                "above_mid_vacuum_norm",
                "above_far_wall_strength_norm",
                "above_far_wall_erosion_norm",
                "above_far_vacuum_norm",
                "below_at_wall_strength_norm",
                "below_at_wall_erosion_norm",
                "below_at_vacuum_norm",
                "below_near_wall_strength_norm",
                "below_near_wall_erosion_norm",
                "below_near_vacuum_norm",
                "below_mid_wall_strength_norm",
                "below_mid_wall_erosion_norm",
                "below_mid_vacuum_norm",
                "below_far_wall_strength_norm",
                "below_far_wall_erosion_norm",
                "below_far_vacuum_norm",
                "above_score",
                "below_score",
                "vacuum_total_score",
            ]
        ]


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
        raise ValueError(f"Missing calibration metrics: {sorted(missing)}")
    for name, (lo, hi) in cal.items():
        if hi <= lo:
            raise ValueError(f"Invalid calibration bounds for {name}: {lo} {hi}")
    return cal


def _norm(values: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _band_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["direction"] = np.where(df["side"] == "A", "above", "below")
    df["abs_ticks"] = df["rel_ticks"].abs()
    df["band"] = ""
    for name, lo, hi in BANDS:
        mask = (df["abs_ticks"] >= lo) & (df["abs_ticks"] <= hi)
        df.loc[mask, "band"] = name
    return df.loc[df["band"] != ""].copy()


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "above_at_wall_strength_norm": 0.0,
        "above_at_wall_erosion_norm": 0.0,
        "above_at_vacuum_norm": 0.0,
        "above_near_wall_strength_norm": 0.0,
        "above_near_wall_erosion_norm": 0.0,
        "above_near_vacuum_norm": 0.0,
        "above_mid_wall_strength_norm": 0.0,
        "above_mid_wall_erosion_norm": 0.0,
        "above_mid_vacuum_norm": 0.0,
        "above_far_wall_strength_norm": 0.0,
        "above_far_wall_erosion_norm": 0.0,
        "above_far_vacuum_norm": 0.0,
        "below_at_wall_strength_norm": 0.0,
        "below_at_wall_erosion_norm": 0.0,
        "below_at_vacuum_norm": 0.0,
        "below_near_wall_strength_norm": 0.0,
        "below_near_wall_erosion_norm": 0.0,
        "below_near_vacuum_norm": 0.0,
        "below_mid_wall_strength_norm": 0.0,
        "below_mid_wall_erosion_norm": 0.0,
        "below_mid_vacuum_norm": 0.0,
        "below_far_wall_strength_norm": 0.0,
        "below_far_wall_erosion_norm": 0.0,
        "below_far_vacuum_norm": 0.0,
        "above_at_ease": 0.0,
        "above_near_ease": 0.0,
        "above_mid_ease": 0.0,
        "above_far_ease": 0.0,
        "below_at_ease": 0.0,
        "below_near_ease": 0.0,
        "below_mid_ease": 0.0,
        "below_far_ease": 0.0,
    }
    for key, value in cols.items():
        if key not in df.columns:
            df[key] = value
    return df
