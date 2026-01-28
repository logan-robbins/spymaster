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

LOOKBACK_SESSIONS = 1
SESSION_WINDOW = "09:30-10:30_ETC_GMT+5"
EPS_QTY = 1.0


class GoldBuildHudPhysicsNormCalibration(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_build_hud_physics_norm_calibration",
            io=StageIO(
                inputs=[
                    "silver.future_mbo.wall_surface_1s",
                    "silver.future_option_mbo.gex_surface_1s",
                    "silver.future_option_mbo.book_wall_1s",
                    "bronze.shared.instrument_definitions",
                ],
                output="gold.hud.physics_norm_calibration",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        wall_key = "silver.future_mbo.wall_surface_1s"
        gex_key = "silver.future_option_mbo.gex_surface_1s"
        flow_key = "silver.future_option_mbo.book_wall_1s"
        def_key = "bronze.shared.instrument_definitions"

        wall_dates = _list_dates(cfg, wall_key, symbol)
        gex_dates = _list_dates(cfg, gex_key, symbol)
        flow_dates = _list_dates(cfg, flow_key, symbol)
        def_dates = _list_dates(cfg, def_key, symbol)
        eligible = sorted(
            d
            for d in set(wall_dates).intersection(gex_dates, flow_dates, def_dates)
            if d <= dt
        )

        if len(eligible) < LOOKBACK_SESSIONS:
            raise ValueError(f"Insufficient sessions for calibration: {len(eligible)} < {LOOKBACK_SESSIONS}")

        lookback = eligible[-LOOKBACK_SESSIONS:]

        wall_contract = load_avro_contract(repo_root / cfg.dataset(wall_key).contract)
        gex_contract = load_avro_contract(repo_root / cfg.dataset(gex_key).contract)
        flow_contract = load_avro_contract(repo_root / cfg.dataset(flow_key).contract)
        def_contract = load_avro_contract(repo_root / cfg.dataset(def_key).contract)

        wall_metrics: Dict[str, List[np.ndarray]] = {
            "pull_add_log": [],
            "log1p_pull_intensity_rest": [],
            "log1p_erosion_norm": [],
            "d2_pull_add_log": [],
            "wall_strength_log": [],
        }
        flow_metrics: Dict[str, List[np.ndarray]] = {
            "flow_abs": [],
            "flow_reinforce": [],
            "pull_rest_intensity": [],
        }
        gex_values: List[np.ndarray] = []

        lineage = []

        for session_date in lookback:
            wall_ref = partition_ref(cfg, wall_key, symbol, session_date)
            gex_ref = partition_ref(cfg, gex_key, symbol, session_date)
            flow_ref = partition_ref(cfg, flow_key, symbol, session_date)
            def_ref = partition_ref(cfg, def_key, symbol, session_date)

            for ref in (wall_ref, gex_ref, flow_ref, def_ref):
                if not is_partition_complete(ref):
                    raise FileNotFoundError(f"Missing partition: {ref.dataset_key} dt={session_date}")

            df_wall = enforce_contract(read_partition(wall_ref), wall_contract)
            df_gex = enforce_contract(read_partition(gex_ref), gex_contract)
            df_flow = enforce_contract(read_partition(flow_ref), flow_contract)
            df_def = enforce_contract(read_partition(def_ref), def_contract)

            start_ns, end_ns = _rth_window_ns(session_date)

            df_wall = df_wall.loc[
                (df_wall["window_end_ts_ns"] >= start_ns) & (df_wall["window_end_ts_ns"] < end_ns)
            ].copy()
            df_gex = df_gex.loc[
                (df_gex["window_end_ts_ns"] >= start_ns) & (df_gex["window_end_ts_ns"] < end_ns)
            ].copy()
            df_flow = df_flow.loc[
                (df_flow["window_end_ts_ns"] >= start_ns) & (df_flow["window_end_ts_ns"] < end_ns)
            ].copy()

            if df_wall.empty or df_gex.empty:
                raise ValueError(f"Empty calibration window for {session_date}")

            _append_wall_metrics(df_wall, wall_metrics)
            gex_values.append(df_gex["gex_abs"].astype(float).to_numpy())
            _append_flow_metrics(df_flow, df_def, df_gex, flow_metrics, session_date)

            lineage.append(
                {
                    "dataset": wall_ref.dataset_key,
                    "dt": session_date,
                    "manifest_sha256": read_manifest_hash(wall_ref),
                }
            )
            lineage.append(
                {
                    "dataset": gex_ref.dataset_key,
                    "dt": session_date,
                    "manifest_sha256": read_manifest_hash(gex_ref),
                }
            )
            lineage.append(
                {
                    "dataset": flow_ref.dataset_key,
                    "dt": session_date,
                    "manifest_sha256": read_manifest_hash(flow_ref),
                }
            )
            lineage.append(
                {
                    "dataset": def_ref.dataset_key,
                    "dt": session_date,
                    "manifest_sha256": read_manifest_hash(def_ref),
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

        gex_all = np.concatenate(gex_values) if gex_values else np.array([], dtype=float)
        if gex_all.size == 0:
            raise ValueError("No values for metric gex_abs")
        rows.append(_row("gex_abs", float(np.quantile(gex_all, 0.05)), float(np.quantile(gex_all, 0.95)), dt))

        for name, chunks in flow_metrics.items():
            values = np.concatenate(chunks) if chunks else np.array([], dtype=float)
            if values.size == 0:
                raise ValueError(f"No values for metric {name}")
            rows.append(_row(name, float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95)), dt))

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


def _rth_window_ns(session_date: str) -> Tuple[int, int]:
    start = pd.Timestamp(f"{session_date} 09:30:00", tz="Etc/GMT+5")
    end = pd.Timestamp(f"{session_date} 10:30:00", tz="Etc/GMT+5")
    return int(start.tz_convert("UTC").value), int(end.tz_convert("UTC").value)


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


def _append_flow_metrics(
    df_flow: pd.DataFrame,
    df_def: pd.DataFrame,
    df_gex: pd.DataFrame,
    out: Dict[str, List[np.ndarray]],
    session_date: str,
) -> None:
    if df_flow.empty or df_def.empty or df_gex.empty:
        return

    defs = _load_definitions(df_def, session_date)
    if defs.empty:
        return

    df_join = df_flow.merge(defs[["instrument_id", "strike_price_int"]], on="instrument_id", how="inner")
    if df_join.empty:
        return

    agg = df_join.groupby(["window_end_ts_ns", "strike_price_int"], as_index=False).agg(
        add_qty_sum=("add_qty", "sum"),
        pull_qty_sum=("pull_qty", "sum"),
        fill_qty_sum=("fill_qty", "sum"),
        pull_rest_qty_sum=("pull_rest_qty", "sum"),
        depth_total_sum=("depth_total", "sum"),
    )

    agg["flow_abs"] = agg["add_qty_sum"] + agg["pull_qty_sum"] + agg["fill_qty_sum"]
    agg["flow_reinforce"] = agg["add_qty_sum"] - agg["pull_qty_sum"] - agg["fill_qty_sum"]
    agg["pull_rest_intensity"] = agg["pull_rest_qty_sum"] / (agg["depth_total_sum"] + EPS_QTY)

    grid = df_gex[["window_end_ts_ns", "strike_price_int"]].drop_duplicates()
    aligned = grid.merge(agg, on=["window_end_ts_ns", "strike_price_int"], how="left").fillna(0.0)

    out["flow_abs"].append(aligned["flow_abs"].astype(float).to_numpy())
    out["flow_reinforce"].append(aligned["flow_reinforce"].astype(float).to_numpy())
    out["pull_rest_intensity"].append(aligned["pull_rest_intensity"].astype(float).to_numpy())


def _load_definitions(df_def: pd.DataFrame, session_date: str) -> pd.DataFrame:
    required = {"instrument_id", "instrument_class", "strike_price", "expiration"}
    missing = required.difference(df_def.columns)
    if missing:
        raise ValueError(f"Missing definition columns: {sorted(missing)}")
    df_def = df_def.sort_values("ts_event").groupby("instrument_id", as_index=False).last()
    df_def = df_def.loc[df_def["instrument_class"].isin({"C", "P"})].copy()
    exp_dates = (
        pd.to_datetime(df_def["expiration"].astype("int64"), utc=True)
        .dt.tz_convert("Etc/GMT+5")
        .dt.date.astype(str)
    )
    df_def = df_def.loc[exp_dates == session_date].copy()
    df_def["instrument_id"] = df_def["instrument_id"].astype("int64")
    df_def["strike_price_int"] = df_def["strike_price"].astype("int64")
    return df_def[["instrument_id", "strike_price_int"]]


def _row(name: str, q05: float, q95: float, dt: str) -> Dict[str, object]:
    return {
        "metric_name": name,
        "q05": q05,
        "q95": q95,
        "lookback_sessions": LOOKBACK_SESSIONS,
        "session_window": SESSION_WINDOW,
        "asof_dt": dt,
    }
