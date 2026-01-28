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


class SilverComputeVacuumSurface1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_vacuum_surface_1s",
            io=StageIO(
                inputs=[
                    "silver.future_mbo.depth_and_flow_1s",
                    "gold.hud.physics_norm_calibration",
                ],
                output="silver.future_mbo.vacuum_surface_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        flow_key = "silver.future_mbo.depth_and_flow_1s"
        cal_key = "gold.hud.physics_norm_calibration"

        flow_ref = partition_ref(cfg, flow_key, symbol, dt)
        if not is_partition_complete(flow_ref):
            raise FileNotFoundError(f"Input not ready: {flow_key} dt={dt}")

        cal_ref = partition_ref(cfg, cal_key, symbol, dt)
        if not is_partition_complete(cal_ref):
            raise FileNotFoundError(f"Input not ready: {cal_key} dt={dt}")

        flow_contract = load_avro_contract(repo_root / cfg.dataset(flow_key).contract)
        cal_contract = load_avro_contract(repo_root / cfg.dataset(cal_key).contract)

        df_flow = enforce_contract(read_partition(flow_ref), flow_contract)
        df_cal = enforce_contract(read_partition(cal_ref), cal_contract)

        df_out = self.transform_multi(df_flow, df_cal)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
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

    def transform_multi(self, df_flow: pd.DataFrame, df_cal: pd.DataFrame) -> pd.DataFrame:
        if df_flow.empty:
            return pd.DataFrame(
                columns=[
                    "window_start_ts_ns",
                    "window_end_ts_ns",
                    "price_int",
                    "side",
                    "spot_ref_price_int",
                    "rel_ticks",
                    "pull_intensity_rest",
                    "pull_add_log",
                    "d1_pull_add_log",
                    "d2_pull_add_log",
                    "d3_pull_add_log",
                    "wall_strength_log",
                    "wall_erosion",
                    "vacuum_score",
                    "window_valid",
                ]
            )

        cal = _load_calibration(df_cal)

        depth_start = df_flow["depth_qty_start"].astype(float).to_numpy()
        depth_rest = df_flow["depth_qty_rest"].astype(float).to_numpy()
        pull_qty_rest = df_flow["pull_qty_rest"].astype(float).to_numpy()
        add_qty = df_flow["add_qty"].astype(float).to_numpy()
        d1_depth = df_flow["d1_depth_qty"].astype(float).to_numpy()

        pull_intensity_rest = pull_qty_rest / (depth_start + EPS_QTY)
        pull_add_log = np.log((pull_qty_rest + EPS_QTY) / (add_qty + EPS_QTY))
        wall_erosion = np.maximum(-d1_depth, 0.0)
        wall_strength_log = np.log(depth_rest + 1.0)

        df = df_flow[
            [
                "window_start_ts_ns",
                "window_end_ts_ns",
                "price_int",
                "side",
                "spot_ref_price_int",
                "rel_ticks",
                "window_valid",
            ]
        ].copy()

        df["pull_intensity_rest"] = pull_intensity_rest
        df["pull_add_log"] = pull_add_log
        df["wall_strength_log"] = wall_strength_log
        df["wall_erosion"] = wall_erosion
        df["depth_qty_start"] = depth_start

        df = df.sort_values(["side", "price_int", "window_end_ts_ns"])
        group = df.groupby(["side", "price_int"])["pull_add_log"]
        df["d1_pull_add_log"] = group.diff().fillna(0.0)
        df["d2_pull_add_log"] = df.groupby(["side", "price_int"])["d1_pull_add_log"].diff().fillna(0.0)
        df["d3_pull_add_log"] = df.groupby(["side", "price_int"])["d2_pull_add_log"].diff().fillna(0.0)

        log1p_pull_intensity = np.log1p(df["pull_intensity_rest"].to_numpy())
        erosion_norm = np.log1p(df["wall_erosion"].to_numpy() / (df["depth_qty_start"].to_numpy() + EPS_QTY))

        n1 = _norm(df["pull_add_log"].to_numpy(), cal["pull_add_log"])
        n2 = _norm(log1p_pull_intensity, cal["log1p_pull_intensity_rest"])
        n3 = _norm(erosion_norm, cal["log1p_erosion_norm"])
        n4 = _norm(df["d2_pull_add_log"].to_numpy(), cal["d2_pull_add_log"])

        df["vacuum_score"] = (n1 + n2 + n3 + n4) / 4.0

        df = df.drop(columns=["depth_qty_start"])

        return df[
            [
                "window_start_ts_ns",
                "window_end_ts_ns",
                "price_int",
                "side",
                "spot_ref_price_int",
                "rel_ticks",
                "pull_intensity_rest",
                "pull_add_log",
                "d1_pull_add_log",
                "d2_pull_add_log",
                "d3_pull_add_log",
                "wall_strength_log",
                "wall_erosion",
                "vacuum_score",
                "window_valid",
            ]
        ]


def _load_calibration(df_cal: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    required = {
        "pull_add_log",
        "log1p_pull_intensity_rest",
        "log1p_erosion_norm",
        "d2_pull_add_log",
    }
    cal: Dict[str, Tuple[float, float]] = {}
    for row in df_cal.itertuples(index=False):
        cal[str(row.metric_name)] = (float(row.q05), float(row.q95))
    missing = required.difference(cal.keys())
    if missing:
        raise ValueError(f"Missing calibration metrics: {sorted(missing)}")
    for name, (lo, hi) in cal.items():
        if hi <= lo:
            # Handle zero variance or inverted bounds
            # If constant (lo == hi), expand to avoid div/0
            if lo == hi:
                hi = lo + 1.0
            else:
                # Actual inversion is bad, swap? or error.
                # Assuming just flat distribution here.
                # If inverted, data is broken.
                raise ValueError(f"Invalid calibration bounds for {name}: {lo} {hi}")
        cal[name] = (lo, hi)
    return cal


def _norm(values: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)
