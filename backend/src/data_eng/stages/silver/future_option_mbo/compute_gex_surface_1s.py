from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from .options_book_engine import OptionsBookEngine
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
from ..future_mbo.mbo_batches import first_hour_window_ns

PRICE_SCALE = 1e-9
WINDOW_NS = 1_000_000_000
TICK_SIZE = 0.25
TICK_INT = int(round(TICK_SIZE / PRICE_SCALE))
GEX_STRIKE_STEP_POINTS = 5
GEX_MAX_STRIKE_OFFSETS = 12
RISK_FREE_RATE = 0.05
CONTRACT_MULTIPLIER = 50
EPS_GEX = 1e-6
EPS_QTY = 1.0
DEFAULT_UNDERLYING = "ES"


class SilverComputeGexSurface1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_gex_surface_1s",
            io=StageIO(
                inputs=["bronze.future_option_mbo.mbo"],
                output="silver.future_option_mbo.gex_surface_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_key_gex = "silver.future_option_mbo.gex_surface_1s"
        out_key_wall = "silver.future_option_mbo.book_wall_1s"
        out_key_flow = "silver.future_option_mbo.book_flow_1s"
        out_key_gex_flow = "silver.future_option_mbo.gex_flow_surface_1s"

        ref_gex = partition_ref(cfg, out_key_gex, symbol, dt)
        ref_wall = partition_ref(cfg, out_key_wall, symbol, dt)
        ref_flow = partition_ref(cfg, out_key_flow, symbol, dt)
        ref_gex_flow = partition_ref(cfg, out_key_gex_flow, symbol, dt)

        if (
            is_partition_complete(ref_gex)
            and is_partition_complete(ref_wall)
            and is_partition_complete(ref_flow)
            and is_partition_complete(ref_gex_flow)
        ):
            return

        mbo_key = "bronze.future_option_mbo.mbo"
        snap_key = "silver.future_mbo.book_snapshot_1s"
        stat_key = "silver.future_option.statistics_clean"
        def_key = "bronze.shared.instrument_definitions"

        mbo_ref = partition_ref(cfg, mbo_key, symbol, dt)
        snap_ref = partition_ref(cfg, snap_key, symbol, dt)
        stat_ref = partition_ref(cfg, stat_key, symbol, dt)
        def_ref = partition_ref(cfg, def_key, symbol, dt)

        for ref in (mbo_ref, snap_ref, stat_ref, def_ref):
            if not is_partition_complete(ref):
                raise FileNotFoundError(f"Missing partition: {ref.dataset_key} dt={dt}")

        mbo_contract = load_avro_contract(repo_root / cfg.dataset(mbo_key).contract)
        snap_contract = load_avro_contract(repo_root / cfg.dataset(snap_key).contract)
        stat_contract = load_avro_contract(repo_root / cfg.dataset(stat_key).contract)
        def_contract = load_avro_contract(repo_root / cfg.dataset(def_key).contract)

        df_mbo = enforce_contract(read_partition(mbo_ref), mbo_contract)
        df_snap = enforce_contract(read_partition(snap_ref), snap_contract)
        df_stat = enforce_contract(read_partition(stat_ref), stat_contract)
        df_def = enforce_contract(read_partition(def_ref), def_contract)

        start_ns, end_ns = first_hour_window_ns(dt)
        df_mbo = df_mbo.loc[(df_mbo["ts_event"] >= start_ns) & (df_mbo["ts_event"] < end_ns)].copy()
        df_snap = df_snap.loc[
            (df_snap["window_end_ts_ns"] >= start_ns) & (df_snap["window_end_ts_ns"] < end_ns)
        ].copy()

        df_gex, df_wall, df_flow, defs = self.transform_multi(df_mbo, df_snap, df_stat, df_def, dt)

        gex_contract_path = repo_root / cfg.dataset(out_key_gex).contract
        wall_contract_path = repo_root / cfg.dataset(out_key_wall).contract
        flow_contract_path = repo_root / cfg.dataset(out_key_flow).contract
        gex_flow_contract_path = repo_root / cfg.dataset(out_key_gex_flow).contract

        gex_contract = load_avro_contract(gex_contract_path)
        wall_contract = load_avro_contract(wall_contract_path)
        flow_contract = load_avro_contract(flow_contract_path)

        if df_gex.empty:
            df_gex = _empty_df(gex_contract.fields)
        if df_wall.empty:
            df_wall = _empty_df(wall_contract.fields)
        if df_flow.empty:
            df_flow = _empty_df(flow_contract.fields)

        df_gex = enforce_contract(df_gex, gex_contract)
        df_wall = enforce_contract(df_wall, wall_contract)
        df_flow = enforce_contract(df_flow, flow_contract)

        lineage = [
            {"dataset": mbo_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(mbo_ref)},
            {"dataset": snap_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(snap_ref)},
            {"dataset": stat_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(stat_ref)},
            {"dataset": def_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(def_ref)},
        ]

        if not is_partition_complete(ref_gex):
            write_partition(
                cfg=cfg,
                dataset_key=out_key_gex,
                symbol=symbol,
                dt=dt,
                df=df_gex,
                contract_path=gex_contract_path,
                inputs=lineage,
                stage=self.name,
            )

        if not is_partition_complete(ref_wall):
            write_partition(
                cfg=cfg,
                dataset_key=out_key_wall,
                symbol=symbol,
                dt=dt,
                df=df_wall,
                contract_path=wall_contract_path,
                inputs=lineage,
                stage=self.name,
            )

        if not is_partition_complete(ref_flow):
            write_partition(
                cfg=cfg,
                dataset_key=out_key_flow,
                symbol=symbol,
                dt=dt,
                df=df_flow,
                contract_path=flow_contract_path,
                inputs=lineage,
                stage=self.name,
            )

        if not is_partition_complete(ref_gex_flow):
            cal_key = "gold.hud.physics_norm_calibration"
            cal_ref = partition_ref(cfg, cal_key, symbol, dt)
            if not is_partition_complete(cal_ref):
                print(
                    f"WARN: Missing {cal_key} dt={dt}. Skipping gex_flow_surface_1s until calibration is available."
                )
                return
            cal_contract = load_avro_contract(repo_root / cfg.dataset(cal_key).contract)
            df_cal = enforce_contract(read_partition(cal_ref), cal_contract)

            gex_flow_contract = load_avro_contract(gex_flow_contract_path)
            try:
                df_gex_flow = self.transform_gex_flow(df_wall, defs, df_snap, df_cal)
            except ValueError as exc:
                print(f"WARN: {exc}. Skipping gex_flow_surface_1s until calibration is refreshed.")
                return
            if df_gex_flow.empty:
                df_gex_flow = _empty_df(gex_flow_contract.fields)
            df_gex_flow = enforce_contract(df_gex_flow, gex_flow_contract)

            flow_lineage = lineage + [
                {"dataset": cal_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(cal_ref)}
            ]

            write_partition(
                cfg=cfg,
                dataset_key=out_key_gex_flow,
                symbol=symbol,
                dt=dt,
                df=df_gex_flow,
                contract_path=gex_flow_contract_path,
                inputs=flow_lineage,
                stage=self.name,
            )

    def transform_multi(
        self,
        df_mbo: pd.DataFrame,
        df_snap: pd.DataFrame,
        df_stat: pd.DataFrame,
        df_def: pd.DataFrame,
        dt: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if df_mbo.empty or df_snap.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        defs = _load_definitions(df_def, dt)
        oi_map = _load_open_interest(df_stat)
        defs = defs.loc[defs["option_symbol"].isin(oi_map.keys())].copy()
        if defs.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        eligible_ids = set(defs["instrument_id"].astype(int).tolist())
        df_mbo = df_mbo.loc[df_mbo["instrument_id"].isin(eligible_ids)].copy()
        if df_mbo.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        engine = OptionsBookEngine()
        df_flow_raw, df_bbo = engine.process_batch(df_mbo)

        df_wall = pd.DataFrame()
        df_flow = pd.DataFrame()
        if not df_flow_raw.empty:
            df_flow_raw = df_flow_raw.copy()
            df_wall = df_flow_raw[
                [
                    "window_end_ts_ns",
                    "instrument_id",
                    "side",
                    "price_int",
                    "depth_total",
                    "add_qty",
                    "pull_qty",
                    "pull_rest_qty",
                    "fill_qty",
                ]
            ].copy()
            df_flow = df_flow_raw[
                [
                    "window_end_ts_ns",
                    "instrument_id",
                    "side",
                    "price_int",
                    "add_qty",
                    "pull_qty",
                    "fill_qty",
                ]
            ].copy()

        df_gex = pd.DataFrame()
        if not df_bbo.empty:
            df_mids = df_bbo.copy()
            df_mids["mid_price"] = df_mids["mid_price_int"] * PRICE_SCALE
            df_mids = df_mids.merge(df_snap[["window_end_ts_ns", "spot_ref_price_int"]], on="window_end_ts_ns", how="inner")
            df_mids["spot_ref_price"] = df_mids["spot_ref_price_int"] * PRICE_SCALE
            df_mids = df_mids.merge(defs, on="instrument_id", how="inner")
            df_mids["open_interest"] = df_mids["option_symbol"].map(oi_map).fillna(0.0)

            df_gex = _calc_gex_vectorized(df_mids, df_snap)

        return df_gex, df_wall, df_flow, defs

    def transform_gex_flow(
        self,
        df_wall: pd.DataFrame,
        defs: pd.DataFrame,
        df_snap: pd.DataFrame,
        df_cal: pd.DataFrame,
    ) -> pd.DataFrame:
        if df_wall.empty or df_snap.empty or defs.empty:
            return pd.DataFrame()

        cal = _load_calibration(df_cal)

        defs = defs.copy()
        defs["strike_price_int"] = defs["strike_price"].astype(np.int64)

        df_join = df_wall.merge(
            defs[["instrument_id", "strike_price_int"]],
            on="instrument_id",
            how="inner",
        )
        if df_join.empty:
            return pd.DataFrame()

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

        grid = _build_strike_grid(df_snap)
        df = grid.merge(agg, on=["window_end_ts_ns", "strike_price_int"], how="left")

        fill_cols = [
            "add_qty_sum",
            "pull_qty_sum",
            "fill_qty_sum",
            "pull_rest_qty_sum",
            "depth_total_sum",
            "flow_abs",
            "flow_reinforce",
            "pull_rest_intensity",
        ]
        for col in fill_cols:
            df[col] = df[col].fillna(0.0)

        if (df["rel_ticks"] % 20 != 0).any():
            raise ValueError("gex_flow_surface_1s rel_ticks not aligned to 20-tick grid")

        df["flow_abs_norm"] = _norm(df["flow_abs"].astype(float).to_numpy(), cal["flow_abs"])
        df["flow_reinforce_norm"] = _norm(df["flow_reinforce"].astype(float).to_numpy(), cal["flow_reinforce"])
        df["pull_rest_intensity_norm"] = _norm(
            df["pull_rest_intensity"].astype(float).to_numpy(), cal["pull_rest_intensity"]
        )

        df["window_start_ts_ns"] = df["window_end_ts_ns"] - WINDOW_NS

        return df[
            [
                "window_start_ts_ns",
                "window_end_ts_ns",
                "underlying",
                "strike_price_int",
                "spot_ref_price_int",
                "rel_ticks",
                "strike_points",
                "add_qty_sum",
                "pull_qty_sum",
                "fill_qty_sum",
                "flow_abs",
                "flow_reinforce",
                "flow_abs_norm",
                "flow_reinforce_norm",
                "pull_rest_qty_sum",
                "depth_total_sum",
                "pull_rest_intensity",
                "pull_rest_intensity_norm",
            ]
        ]


def _build_strike_grid(df_snap: pd.DataFrame) -> pd.DataFrame:
    if df_snap.empty:
        return pd.DataFrame()

    grid_base = df_snap[["window_end_ts_ns", "spot_ref_price_int"]].drop_duplicates().copy()
    grid_base["spot_ref_price"] = grid_base["spot_ref_price_int"] * PRICE_SCALE

    offsets = np.arange(-GEX_MAX_STRIKE_OFFSETS, GEX_MAX_STRIKE_OFFSETS + 1)
    grid_list = []
    for offset in offsets:
        tmp = grid_base.copy()
        strike_ref = (tmp["spot_ref_price"] / GEX_STRIKE_STEP_POINTS).round() * GEX_STRIKE_STEP_POINTS
        tmp["strike_points"] = strike_ref + offset * GEX_STRIKE_STEP_POINTS
        tmp["strike_price_int"] = (tmp["strike_points"] / PRICE_SCALE).round().astype(np.int64)
        tmp["rel_ticks"] = ((tmp["strike_price_int"] - tmp["spot_ref_price_int"]) / TICK_INT).round().astype(np.int64)
        tmp["underlying"] = DEFAULT_UNDERLYING
        grid_list.append(tmp)

    df_grid = pd.concat(grid_list, ignore_index=True)
    return df_grid.drop(columns=["spot_ref_price"], errors="ignore")


def _calc_gex_vectorized(df: pd.DataFrame, df_snap: pd.DataFrame) -> pd.DataFrame:
    spot_scale = df["spot_ref_price"]
    freq = GEX_STRIKE_STEP_POINTS
    strike_ref = (spot_scale / freq).round() * freq

    k = df["strike_price"].astype(float) * PRICE_SCALE
    limit = GEX_MAX_STRIKE_OFFSETS * freq
    mask = (k - strike_ref).abs() <= limit + 1e-9
    df = df[mask].copy()

    exp_ns = df["expiration"].astype(float)
    now_ns = df["window_end_ts_ns"].astype(float)
    t_days = (exp_ns - now_ns) / 1e9 / 86400.0

    df = df[t_days > 0].copy()
    if df.empty:
        return pd.DataFrame()

    T = t_days / 365.0

    F = df["spot_ref_price"]
    K = df["strike_price"] * PRICE_SCALE
    mid = df["mid_price"]
    is_call = df["right"] == "C"

    iv = _vectorized_iv(mid.values, F.values, K.values, T.values, is_call.values)

    d1 = (np.log(F / K) + 0.5 * iv**2 * T) / (iv * np.sqrt(T))
    gamma = np.exp(-RISK_FREE_RATE * T) * norm.pdf(d1) / (F * iv * np.sqrt(T))

    gex_val = gamma * df["open_interest"] * CONTRACT_MULTIPLIER

    df["gex_val"] = gex_val
    df["is_call"] = is_call

    df_grid = _build_strike_grid(df_snap)
    if df_grid.empty:
        return pd.DataFrame()

    raw_strike_float = df["strike_price"].astype(float) * PRICE_SCALE
    binned_strike = (raw_strike_float / freq).round() * freq
    df["grid_strike_int"] = (binned_strike / PRICE_SCALE).round().astype(np.int64)

    grp = (
        df.groupby(["window_end_ts_ns", "grid_strike_int", "is_call"])["gex_val"]
        .sum()
        .reset_index()
        .rename(columns={"grid_strike_int": "strike_price_int"})
    )

    piv = grp.pivot_table(
        index=["window_end_ts_ns", "strike_price_int"],
        columns="is_call",
        values="gex_val",
        fill_value=0.0,
    )
    piv.columns = ["put_val", "call_val"]
    piv = piv.reset_index()

    res = df_grid.merge(piv, on=["window_end_ts_ns", "strike_price_int"], how="left").fillna(0.0)

    res["window_start_ts_ns"] = res["window_end_ts_ns"] - WINDOW_NS
    res["gex_call_abs"] = res["call_val"]
    res["gex_put_abs"] = res["put_val"]
    res["gex_abs"] = res["gex_call_abs"] + res["gex_put_abs"]
    res["gex"] = res["gex_call_abs"] - res["gex_put_abs"]
    res["gex_imbalance_ratio"] = res["gex"] / (res["gex_abs"] + EPS_GEX)

    res = res.sort_values(["strike_price_int", "window_end_ts_ns"])

    res["d1_gex_abs"] = res.groupby("strike_price_int")["gex_abs"].diff().fillna(0.0)
    res["d2_gex_abs"] = res.groupby("strike_price_int")["d1_gex_abs"].diff().fillna(0.0)
    res["d3_gex_abs"] = res.groupby("strike_price_int")["d2_gex_abs"].diff().fillna(0.0)

    res["d1_gex"] = res.groupby("strike_price_int")["gex"].diff().fillna(0.0)
    res["d2_gex"] = res.groupby("strike_price_int")["d1_gex"].diff().fillna(0.0)
    res["d3_gex"] = res.groupby("strike_price_int")["d2_gex"].diff().fillna(0.0)

    res["d1_gex_imbalance_ratio"] = res.groupby("strike_price_int")["gex_imbalance_ratio"].diff().fillna(0.0)
    res["d2_gex_imbalance_ratio"] = res.groupby("strike_price_int")["d1_gex_imbalance_ratio"].diff().fillna(0.0)
    res["d3_gex_imbalance_ratio"] = res.groupby("strike_price_int")["d2_gex_imbalance_ratio"].diff().fillna(0.0)

    res["underlying_spot_ref"] = res["spot_ref_price_int"] * PRICE_SCALE

    drop_cols = ["call_val", "put_val"]
    res = res.drop(columns=drop_cols, errors="ignore")

    return res[
        [
            "window_start_ts_ns",
            "window_end_ts_ns",
            "underlying",
            "strike_price_int",
            "underlying_spot_ref",
            "spot_ref_price_int",
            "rel_ticks",
            "strike_points",
            "gex_call_abs",
            "gex_put_abs",
            "gex_abs",
            "gex",
            "gex_imbalance_ratio",
            "d1_gex_abs",
            "d2_gex_abs",
            "d3_gex_abs",
            "d1_gex",
            "d2_gex",
            "d3_gex",
            "d1_gex_imbalance_ratio",
            "d2_gex_imbalance_ratio",
            "d3_gex_imbalance_ratio",
        ]
    ]


def _vectorized_iv(price, F, K, T, is_call):
    sigma = np.full_like(price, 0.5)
    sqrt_T = np.sqrt(T)
    disc = np.exp(-RISK_FREE_RATE * T)

    for _ in range(3):
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        vega = F * norm.pdf(d1) * sqrt_T * disc
        vega = np.where(vega < 1e-6, 1e-6, vega)

        call_p = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
        put_p = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

        model_p = np.where(is_call, call_p, put_p)
        diff = model_p - price

        sigma = sigma - diff / vega
        sigma = np.clip(sigma, 0.001, 5.0)

    return sigma


def _load_definitions(df_def: pd.DataFrame, session_date: str) -> pd.DataFrame:
    required = {"instrument_id", "instrument_class", "underlying", "strike_price", "expiration", "raw_symbol"}
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
    df_def["strike_price"] = df_def["strike_price"].astype("int64")
    df_def["expiration"] = df_def["expiration"].astype("int64")
    df_def["underlying"] = df_def["underlying"].astype(str)
    df_def["option_symbol"] = df_def["raw_symbol"].astype(str)
    df_def["right"] = df_def["instrument_class"].astype(str)
    return df_def[["instrument_id", "option_symbol", "underlying", "right", "strike_price", "expiration"]]


def _load_open_interest(df_stat: pd.DataFrame) -> Dict[str, float]:
    required = {"option_symbol", "open_interest"}
    missing = required.difference(df_stat.columns)
    if missing:
        raise ValueError(f"Missing statistics columns: {sorted(missing)}")
    oi = df_stat.set_index("option_symbol")["open_interest"].astype(float).to_dict()
    return {k: v for k, v in oi.items() if v > 0}


def _load_calibration(df_cal: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    required = {"flow_abs", "flow_reinforce", "pull_rest_intensity"}
    cal: Dict[str, Tuple[float, float]] = {}
    for row in df_cal.itertuples(index=False):
        cal[str(row.metric_name)] = (float(row.q05), float(row.q95))
    missing = required.difference(cal.keys())
    if missing:
        raise ValueError(f"Missing calibration metrics: {sorted(missing)}")
    for name, (lo, hi) in cal.items():
        if hi <= lo:
            if lo == hi:
                hi = lo + 1.0
            else:
                raise ValueError(f"Invalid calibration bounds for {name}: {lo} {hi}")
        cal[name] = (lo, hi)
    return cal


def _norm(values: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)
