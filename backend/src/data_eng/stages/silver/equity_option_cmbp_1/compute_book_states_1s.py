from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .cmbp1_book_engine import Cmbp1BookEngine
from ...base import Stage, StageIO
from ....config import AppConfig, ProductConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    read_partition,
    write_partition,
)
from ..equity_mbo.mbo_batches import first_hour_window_ns

PRICE_SCALE = 1e-9
WINDOW_NS = 1_000_000_000
UNDERLYING_BUCKET_SIZE = 0.50
UNDERLYING_BUCKET_INT = int(round(UNDERLYING_BUCKET_SIZE / PRICE_SCALE))
STRIKE_STEP_POINTS = 1.0
MAX_STRIKE_OFFSETS = 25  # +/- $25 around spot
RIGHTS = ("C", "P")
SIDES = ("A", "B")


class SilverComputeEquityOptionBookStates1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_equity_option_book_states_1s",
            io=StageIO(
                inputs=["bronze.equity_option_cmbp_1.cmbp_1", "silver.equity_mbo.book_snapshot_1s"],
                output=[
                    "silver.equity_option_cmbp_1.book_snapshot_1s",
                    "silver.equity_option_cmbp_1.depth_and_flow_1s",
                ],
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str, product: ProductConfig | None = None) -> None:
        out_snap_key = "silver.equity_option_cmbp_1.book_snapshot_1s"
        out_flow_key = "silver.equity_option_cmbp_1.depth_and_flow_1s"

        ref_snap = partition_ref(cfg, out_snap_key, symbol, dt)
        ref_flow = partition_ref(cfg, out_flow_key, symbol, dt)

        if is_partition_complete(ref_snap) and is_partition_complete(ref_flow):
            return

        cmbp_key = "bronze.equity_option_cmbp_1.cmbp_1"
        eq_snap_key = "silver.equity_mbo.book_snapshot_1s"

        cmbp_ref = partition_ref(cfg, cmbp_key, symbol, dt)
        eq_snap_ref = partition_ref(cfg, eq_snap_key, symbol, dt)

        for ref in (cmbp_ref, eq_snap_ref):
            if not is_partition_complete(ref):
                raise FileNotFoundError(f"Missing partition: {ref.dataset_key} dt={dt}")

        cmbp_contract = load_avro_contract(repo_root / cfg.dataset(cmbp_key).contract)
        eq_snap_contract = load_avro_contract(repo_root / cfg.dataset(eq_snap_key).contract)

        df_cmbp = enforce_contract(read_partition(cmbp_ref), cmbp_contract)
        df_eq_snap = enforce_contract(read_partition(eq_snap_ref), eq_snap_contract)

        start_ns, end_ns = first_hour_window_ns(dt)
        df_cmbp = df_cmbp.loc[(df_cmbp["ts_event"] >= start_ns) & (df_cmbp["ts_event"] < end_ns)].copy()
        df_eq_snap = df_eq_snap.loc[
            (df_eq_snap["window_end_ts_ns"] >= start_ns) & (df_eq_snap["window_end_ts_ns"] < end_ns)
        ].copy()

        df_snap, df_flow = self.transform(df_cmbp, df_eq_snap)

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
            {"dataset": cmbp_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(cmbp_ref)},
            {"dataset": eq_snap_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(eq_snap_ref)},
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

    def transform(self, df_cmbp: pd.DataFrame, df_eq_snap: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df_cmbp.empty or df_eq_snap.empty:
            return pd.DataFrame(), pd.DataFrame()

        defs = _build_defs(df_cmbp)
        if defs.empty:
            return pd.DataFrame(), pd.DataFrame()

        engine = Cmbp1BookEngine()
        df_flow_raw, df_bbo = engine.process_batch(df_cmbp)

        df_snap = _build_option_snapshots(df_bbo, defs, df_eq_snap)
        df_flow = _build_option_flow_surface(df_flow_raw, defs, df_eq_snap)

        return df_snap, df_flow


def _build_defs(df_cmbp: pd.DataFrame) -> pd.DataFrame:
    required = {"instrument_id", "strike", "right"}
    missing = required.difference(df_cmbp.columns)
    if missing:
        raise ValueError(f"Missing option definition columns in cmbp_1: {sorted(missing)}")

    df_defs = df_cmbp[["instrument_id", "strike", "right"]].drop_duplicates().copy()
    df_defs = df_defs.loc[df_defs["right"].isin(RIGHTS)].copy()
    df_defs["instrument_id"] = df_defs["instrument_id"].astype("int64")
    df_defs["strike_price_int"] = df_defs["strike"].astype("int64")
    df_defs["right"] = df_defs["right"].astype(str)
    return df_defs[["instrument_id", "strike_price_int", "right"]]


def _build_option_snapshots(
    df_bbo: pd.DataFrame,
    defs: pd.DataFrame,
    df_eq_snap: pd.DataFrame,
) -> pd.DataFrame:
    if df_bbo.empty:
        return pd.DataFrame()

    df = df_bbo.merge(defs, on="instrument_id", how="inner")
    if df.empty:
        return pd.DataFrame()

    df = df.merge(df_eq_snap[["window_end_ts_ns", "spot_ref_price_int"]], on="window_end_ts_ns", how="inner")
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
    df_eq_snap: pd.DataFrame,
) -> pd.DataFrame:
    if df_flow_raw.empty:
        return pd.DataFrame()

    df = df_flow_raw.merge(defs, on="instrument_id", how="inner")
    if df.empty:
        return pd.DataFrame()

    df = df.merge(
        df_eq_snap[["window_end_ts_ns", "spot_ref_price_int"]],
        on="window_end_ts_ns",
        how="inner",
    )
    if df.empty:
        return pd.DataFrame()

    spot_ref_price = df["spot_ref_price_int"].astype(float) * PRICE_SCALE
    strike_price = df["strike_price_int"].astype(float) * PRICE_SCALE
    offsets = np.round((strike_price - spot_ref_price) / STRIKE_STEP_POINTS).astype(int)
    offsets = np.clip(offsets, -MAX_STRIKE_OFFSETS, MAX_STRIKE_OFFSETS)

    df["rel_ticks"] = offsets * 2
    df["strike_points"] = spot_ref_price + offsets * STRIKE_STEP_POINTS
    df["strike_price_int"] = (df["strike_points"] / PRICE_SCALE).round().astype("int64")

    grouped = df.groupby(
        ["window_end_ts_ns", "spot_ref_price_int", "strike_price_int", "right", "side", "rel_ticks", "strike_points"],
        as_index=False,
    ).agg(
        depth_qty_end=("depth_total", "sum"),
        add_qty=("add_qty", "sum"),
        pull_qty=("pull_qty", "sum"),
        pull_qty_rest=("pull_rest_qty", "sum"),
        fill_qty=("fill_qty", "sum"),
    )

    grouped["depth_qty_start"] = (
        grouped["depth_qty_end"]
        - grouped["add_qty"]
        + grouped["pull_qty"]
        + grouped["fill_qty"]
    )
    grouped["depth_qty_start"] = grouped["depth_qty_start"].clip(lower=0.0)

    grid = _build_strike_grid(df_eq_snap)
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
        "add_qty",
        "pull_qty",
        "pull_qty_rest",
        "fill_qty",
        "depth_qty_start",
    ]
    for col in fill_cols:
        df_out[col] = df_out[col].fillna(0.0)

    if (df_out["rel_ticks"] % 2 != 0).any():
        raise ValueError("Equity option flow surface rel_ticks not aligned to $1 grid")

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
            "window_valid",
        ]
    ]


def _build_strike_grid(df_eq_snap: pd.DataFrame) -> pd.DataFrame:
    if df_eq_snap.empty:
        return pd.DataFrame()

    grid_base = df_eq_snap[["window_end_ts_ns", "spot_ref_price_int"]].drop_duplicates().copy()
    grid_base["spot_ref_price"] = grid_base["spot_ref_price_int"].astype(float) * PRICE_SCALE

    offsets = np.arange(-MAX_STRIKE_OFFSETS, MAX_STRIKE_OFFSETS + 1)
    grid_list: List[pd.DataFrame] = []
    for offset in offsets:
        tmp = grid_base.copy()
        tmp["strike_points"] = tmp["spot_ref_price"] + offset * STRIKE_STEP_POINTS
        tmp["strike_price_int"] = (tmp["strike_points"] / PRICE_SCALE).round().astype("int64")
        tmp["rel_ticks"] = offset * 2
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
