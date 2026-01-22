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
from ....utils import session_window_ns

LOOKBACK_SESSIONS = 2
SESSION_WINDOW = "06:00-16:00_ETC_GMT+5"
EPS_QTY = 1.0


class GoldBuildEquityPhysicsNormCalibration(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_build_equity_physics_norm_calibration",
            io=StageIO(
                inputs=["silver.equity_mbo.wall_surface_1s"],
                output="gold.equity_mbo.physics_norm_calibration",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        wall_key = "silver.equity_mbo.wall_surface_1s"

        wall_dates = _list_dates(cfg, wall_key, symbol)
        eligible = sorted(d for d in set(wall_dates) if d <= dt)

        if len(eligible) < LOOKBACK_SESSIONS:
            raise ValueError(f"Insufficient sessions for calibration: {len(eligible)} < {LOOKBACK_SESSIONS}")

        lookback = eligible[-LOOKBACK_SESSIONS:]

        wall_contract = load_avro_contract(repo_root / cfg.dataset(wall_key).contract)

        wall_metrics: Dict[str, List[np.ndarray]] = {
            "pull_add_log": [],
            "log1p_pull_intensity_rest": [],
            "log1p_erosion_norm": [],
            "d2_pull_add_log": [],
            "wall_strength_log": [],
        }

        lineage = []

        for session_date in lookback:
            wall_ref = partition_ref(cfg, wall_key, symbol, session_date)
            if not is_partition_complete(wall_ref):
                raise FileNotFoundError(f"Missing partition: {wall_ref.dataset_key} dt={session_date}")

            df_wall = enforce_contract(read_partition(wall_ref), wall_contract)
            start_ns, end_ns = session_window_ns(session_date)
            df_wall = df_wall.loc[
                (df_wall["window_end_ts_ns"] >= start_ns) & (df_wall["window_end_ts_ns"] < end_ns)
            ].copy()
            if df_wall.empty:
                raise ValueError(f"Empty calibration window for {session_date}")

            _append_wall_metrics(df_wall, wall_metrics)

            lineage.append(
                {
                    "dataset": wall_ref.dataset_key,
                    "dt": session_date,
                    "manifest_sha256": read_manifest_hash(wall_ref),
                }
            )

        rows = []
        for name, chunks in wall_metrics.items():
            values = np.concatenate(chunks) if chunks else np.array([], dtype=float)
            if values.size == 0:
                raise ValueError(f"No values for metric {name}")
            q05 = float(np.quantile(values, 0.05))
            q95 = float(np.quantile(values, 0.95))
            rows.append(_row(name, q05, q95, dt))

        df_out = pd.DataFrame(rows)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

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


def _list_dates(cfg: AppConfig, dataset_key: str, symbol: str) -> List[str]:
    base = cfg.lake_root / cfg.dataset(dataset_key).path.format(symbol=symbol)
    if not base.exists():
        return []
    dates = []
    for entry in base.glob("dt=*"):
        if entry.is_dir():
            dates.append(entry.name.split("=")[1])
    return sorted(dates)


def _append_wall_metrics(df_wall: pd.DataFrame, out: Dict[str, List[np.ndarray]]) -> None:
    df_wall = df_wall.sort_values(["side", "price_int", "window_end_ts_ns"]).copy()

    depth_start = df_wall["depth_qty_start"].astype(float).to_numpy()
    depth_rest = df_wall["depth_qty_rest"].astype(float).to_numpy()
    pull_qty_rest = df_wall["pull_qty_rest"].astype(float).to_numpy()
    add_qty = df_wall["add_qty"].astype(float).to_numpy()
    d1_depth = df_wall["d1_depth_qty"].astype(float).to_numpy()

    pull_add_log = np.log((pull_qty_rest + EPS_QTY) / (add_qty + EPS_QTY))
    pull_intensity = pull_qty_rest / (depth_start + EPS_QTY)
    wall_erosion = np.maximum(-d1_depth, 0.0)
    log1p_erosion_norm = np.log1p(wall_erosion / (depth_start + EPS_QTY))

    df_wall["pull_add_log"] = pull_add_log
    df_wall["d1_pull_add_log"] = df_wall.groupby(["side", "price_int"])["pull_add_log"].diff().fillna(0.0)
    df_wall["d2_pull_add_log"] = df_wall.groupby(["side", "price_int"])["d1_pull_add_log"].diff().fillna(0.0)

    out["pull_add_log"].append(pull_add_log)
    out["log1p_pull_intensity_rest"].append(np.log1p(pull_intensity))
    out["log1p_erosion_norm"].append(log1p_erosion_norm)
    out["d2_pull_add_log"].append(df_wall["d2_pull_add_log"].to_numpy())
    out["wall_strength_log"].append(np.log(depth_rest + 1.0))


def _row(name: str, q05: float, q95: float, dt: str) -> Dict[str, object]:
    return {
        "metric_name": name,
        "q05": q05,
        "q95": q95,
        "lookback_sessions": LOOKBACK_SESSIONS,
        "session_window": SESSION_WINDOW,
        "asof_dt": dt,
    }
