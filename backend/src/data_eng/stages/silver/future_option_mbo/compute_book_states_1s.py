from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

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
STRIKE_STEP_POINTS = 5.0
STRIKE_STEP_INT = int(round(STRIKE_STEP_POINTS / PRICE_SCALE))
MAX_STRIKE_OFFSETS = 10  # +/- $50 around spot
RIGHTS = ("C", "P")
SIDES = ("A", "B")


class SilverComputeOptionBookStates1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_option_book_states_1s",
            io=StageIO(
                inputs=["bronze.future_option_mbo.mbo", "silver.future_mbo.book_snapshot_1s"],
                output=[
                    "silver.future_option_mbo.book_snapshot_1s",
                    "silver.future_option_mbo.depth_and_flow_1s",
                ],
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_snap_key = "silver.future_option_mbo.book_snapshot_1s"
        out_flow_key = "silver.future_option_mbo.depth_and_flow_1s"

        ref_snap = partition_ref(cfg, out_snap_key, symbol, dt)
        ref_flow = partition_ref(cfg, out_flow_key, symbol, dt)

        if is_partition_complete(ref_snap) and is_partition_complete(ref_flow):
            return

        mbo_key = "bronze.future_option_mbo.mbo"
        fut_snap_key = "silver.future_mbo.book_snapshot_1s"

        mbo_ref = partition_ref(cfg, mbo_key, symbol, dt)
        fut_snap_ref = partition_ref(cfg, fut_snap_key, symbol, dt)

        for ref in (mbo_ref, fut_snap_ref):
            if not is_partition_complete(ref):
                raise FileNotFoundError(f"Missing partition: {ref.dataset_key} dt={dt}")

        mbo_contract = load_avro_contract(repo_root / cfg.dataset(mbo_key).contract)
        fut_snap_contract = load_avro_contract(repo_root / cfg.dataset(fut_snap_key).contract)

        df_mbo = enforce_contract(read_partition(mbo_ref), mbo_contract)
        df_fut_snap = enforce_contract(read_partition(fut_snap_ref), fut_snap_contract)

        start_ns, end_ns = first_hour_window_ns(dt)
        df_mbo = df_mbo.loc[(df_mbo["ts_event"] >= start_ns) & (df_mbo["ts_event"] < end_ns)].copy()
        df_fut_snap = df_fut_snap.loc[
            (df_fut_snap["window_end_ts_ns"] >= start_ns) & (df_fut_snap["window_end_ts_ns"] < end_ns)
        ].copy()

        df_snap, df_flow = self.transform(df_mbo, df_fut_snap)

        snap_contract_path = repo_root / cfg.dataset(out_snap_key).contract
        flow_contract_path = repo_root / cfg.dataset(out_flow_key).contract

        snap_contract = load_avro_contract(snap_contract_path)
        flow_contract = load_avro_contract(flow_contract_path)

        if df_snap.empty:
            df_snap = _empty_df(snap_contract.fields)
        if df_flow.empty:
            df_flow = _empty_df(flow_contract.fields)

        df_snap = enforce_contract(df_snap, snap_contract)
        df_flow = enforce_contract(df_flow, flow_contract)

        lineage = [
            {"dataset": mbo_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(mbo_ref)},
            {"dataset": fut_snap_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(fut_snap_ref)},
        ]

        if not is_partition_complete(ref_snap):
            write_partition(
                cfg=cfg,
                dataset_key=out_snap_key,
                symbol=symbol,
                dt=dt,
                df=df_snap,
                contract_path=snap_contract_path,
                inputs=lineage,
                stage=self.name,
            )

        if not is_partition_complete(ref_flow):
            write_partition(
                cfg=cfg,
                dataset_key=out_flow_key,
                symbol=symbol,
                dt=dt,
                df=df_flow,
                contract_path=flow_contract_path,
                inputs=lineage,
                stage=self.name,
            )

    def transform(self, df_mbo: pd.DataFrame, df_fut_snap: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df_mbo.empty or df_fut_snap.empty:
            return pd.DataFrame(), pd.DataFrame()

        defs = _build_defs(df_mbo)
        if defs.empty:
            return pd.DataFrame(), pd.DataFrame()

        engine = OptionsBookEngine()
        df_flow_raw, df_bbo = engine.process_batch(df_mbo)

        df_snap = _build_option_snapshots(df_bbo, defs, df_fut_snap)
        df_flow = _build_option_flow_surface(df_flow_raw, defs, df_fut_snap)

        return df_snap, df_flow


def _build_defs(df_mbo: pd.DataFrame) -> pd.DataFrame:
    required = {"instrument_id", "strike", "right"}
    missing = required.difference(df_mbo.columns)
    if missing:
        raise ValueError(f"Missing option definition columns in MBO: {sorted(missing)}")

    df_defs = df_mbo[["instrument_id", "strike", "right"]].drop_duplicates().copy()
    df_defs = df_defs.loc[df_defs["right"].isin(RIGHTS)].copy()
    df_defs["instrument_id"] = df_defs["instrument_id"].astype("int64")
    df_defs["strike_price_int"] = df_defs["strike"].astype("int64")
    df_defs["right"] = df_defs["right"].astype(str)
    return df_defs[["instrument_id", "strike_price_int", "right"]]


def _round_to_nearest_strike_int(price_int: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(price_int, dtype="int64")
    return ((values + STRIKE_STEP_INT // 2) // STRIKE_STEP_INT) * STRIKE_STEP_INT


def _build_option_snapshots(
    df_bbo: pd.DataFrame,
    defs: pd.DataFrame,
    df_fut_snap: pd.DataFrame,
) -> pd.DataFrame:
    if df_bbo.empty:
        return pd.DataFrame()

    df = df_bbo.merge(defs, on="instrument_id", how="inner")
    if df.empty:
        return pd.DataFrame()

    df = df.merge(df_fut_snap[["window_end_ts_ns", "spot_ref_price_int"]], on="window_end_ts_ns", how="inner")
    if df.empty:
        return pd.DataFrame()

    df["window_start_ts_ns"] = df["window_end_ts_ns"] - WINDOW_NS
    df["mid_price"] = df["mid_price_int"].astype(float) * PRICE_SCALE
    df["book_valid"] = True

    return df[
        [
            "window_start_ts_ns",
            "window_end_ts_ns",
            "instrument_id",
            "right",
            "strike_price_int",
            "bid_price_int",
            "ask_price_int",
            "mid_price",
            "mid_price_int",
            "spot_ref_price_int",
            "book_valid",
        ]
    ]


def _build_option_flow_surface(
    df_flow_raw: pd.DataFrame,
    defs: pd.DataFrame,
    df_fut_snap: pd.DataFrame,
) -> pd.DataFrame:
    if df_flow_raw.empty:
        return pd.DataFrame()

    df = df_flow_raw.merge(defs, on="instrument_id", how="inner")
    if df.empty:
        return pd.DataFrame()

    df = df.merge(
        df_fut_snap[["window_end_ts_ns", "spot_ref_price_int"]],
        on="window_end_ts_ns",
        how="inner",
    )
    if df.empty:
        return pd.DataFrame()

    df["strike_price_int"] = df["strike_price_int"].astype("int64")
    df["spot_ref_price_int"] = df["spot_ref_price_int"].astype("int64")
    df["strike_ref_price_int"] = _round_to_nearest_strike_int(df["spot_ref_price_int"])

    strike_delta = df["strike_price_int"] - df["strike_ref_price_int"]
    if (strike_delta % STRIKE_STEP_INT != 0).any():
        raise ValueError("Option strikes not aligned to $5 grid")
    df["rel_strike"] = (strike_delta // STRIKE_STEP_INT).astype(int)

    df = df.loc[df["rel_strike"].between(-MAX_STRIKE_OFFSETS, MAX_STRIKE_OFFSETS)].copy()
    if df.empty:
        return pd.DataFrame()

    tick_delta = df["strike_price_int"] - df["spot_ref_price_int"]
    if (tick_delta % TICK_INT != 0).any():
        raise ValueError("Option strikes not aligned to $0.25 tick grid")
    df["rel_ticks"] = (tick_delta // TICK_INT).astype(int)
    df["strike_points"] = df["strike_price_int"].astype(float) * PRICE_SCALE

    grouped = df.groupby(
        ["window_end_ts_ns", "spot_ref_price_int", "strike_price_int", "right", "side", "rel_ticks", "strike_points"],
        as_index=False,
    ).agg(
        depth_qty_end=("depth_total", "sum"),
        depth_qty_rest=("depth_rest", "sum"),
        depth_qty_start=("depth_start", "sum"),  # Use tracked start depth from engine
        add_qty=("add_qty", "sum"),
        pull_qty=("pull_qty", "sum"),
        pull_qty_rest=("pull_rest_qty", "sum"),
        fill_qty=("fill_qty", "sum"),
    )

    # depth_qty_start is now tracked directly in the engine, no formula calculation needed
    # The accounting identity should hold: depth_qty_start + add_qty - pull_qty - fill_qty = depth_qty_end

    grid = _build_strike_grid(df_fut_snap)
    if grid.empty:
        return pd.DataFrame()

    grid = _expand_grid(grid)

    df_out = grid.merge(
        grouped,
        on=["window_end_ts_ns", "spot_ref_price_int", "strike_price_int", "right", "side", "rel_ticks", "strike_points"],
        how="left",
    )

    fill_cols = [
        "depth_qty_end",
        "depth_qty_rest",
        "add_qty",
        "pull_qty",
        "pull_qty_rest",
        "fill_qty",
        "depth_qty_start",
    ]
    for col in fill_cols:
        df_out[col] = df_out[col].fillna(0.0)

    df_out["window_start_ts_ns"] = df_out["window_end_ts_ns"] - WINDOW_NS
    df_out["window_valid"] = df_out["spot_ref_price_int"].astype("int64") > 0

    return df_out[
        [
            "window_start_ts_ns",
            "window_end_ts_ns",
            "strike_price_int",
            "strike_points",
            "right",
            "side",
            "spot_ref_price_int",
            "rel_ticks",
            "depth_qty_start",
            "depth_qty_end",
            "add_qty",
            "pull_qty",
            "pull_qty_rest",
            "fill_qty",
            "depth_qty_rest",
            "window_valid",
        ]
    ]


def _build_strike_grid(df_fut_snap: pd.DataFrame) -> pd.DataFrame:
    if df_fut_snap.empty:
        return pd.DataFrame()

    grid_base = df_fut_snap[["window_end_ts_ns", "spot_ref_price_int"]].drop_duplicates().copy()
    grid_base["spot_ref_price_int"] = grid_base["spot_ref_price_int"].astype("int64")
    grid_base["strike_ref_price_int"] = _round_to_nearest_strike_int(grid_base["spot_ref_price_int"])

    offsets = np.arange(-MAX_STRIKE_OFFSETS, MAX_STRIKE_OFFSETS + 1)
    grid_list: List[pd.DataFrame] = []
    for offset in offsets:
        tmp = grid_base.copy()
        tmp["strike_price_int"] = tmp["strike_ref_price_int"] + offset * STRIKE_STEP_INT
        tmp["strike_points"] = tmp["strike_price_int"].astype(float) * PRICE_SCALE
        tick_delta = tmp["strike_price_int"] - tmp["spot_ref_price_int"]
        tmp["rel_ticks"] = (tick_delta // TICK_INT).astype("int64")
        grid_list.append(
            tmp[["window_end_ts_ns", "spot_ref_price_int", "strike_price_int", "strike_points", "rel_ticks"]]
        )

    return pd.concat(grid_list, ignore_index=True)


def _expand_grid(grid: pd.DataFrame) -> pd.DataFrame:
    right_df = pd.DataFrame({"right": list(RIGHTS), "_key": 1})
    side_df = pd.DataFrame({"side": list(SIDES), "_key": 1})
    grid = grid.copy()
    grid["_key"] = 1
    grid = grid.merge(right_df, on="_key", how="inner").merge(side_df, on="_key", how="inner")
    return grid.drop(columns=["_key"], errors="ignore")


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)
