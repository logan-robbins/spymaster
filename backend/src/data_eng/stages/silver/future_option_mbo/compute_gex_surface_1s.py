from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

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

PRICE_SCALE = 1e-9
WINDOW_NS = 1_000_000_000
GEX_STRIKE_STEP_POINTS = 5
GEX_MAX_STRIKE_OFFSETS = 12
RISK_FREE_RATE = 0.05
CONTRACT_MULTIPLIER = 50
EPS_GEX = 1e-6

ACTION_ADD = "A"
ACTION_CANCEL = "C"
ACTION_MODIFY = "M"
ACTION_CLEAR = "R"
ACTION_FILL = "F"
ACTION_TRADE = "T"

F_SNAPSHOT = 128
F_LAST = 256


@dataclass
class OrderState:
    side: str
    price_int: int
    qty: int


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
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
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

        df_out = self.transform_multi(df_mbo, df_snap, df_stat, df_def, symbol, dt)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {"dataset": mbo_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(mbo_ref)},
            {"dataset": snap_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(snap_ref)},
            {"dataset": stat_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(stat_ref)},
            {"dataset": def_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(def_ref)},
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
        df_mbo: pd.DataFrame,
        df_snap: pd.DataFrame,
        df_stat: pd.DataFrame,
        df_def: pd.DataFrame,
        symbol: str,
        dt: str,
    ) -> pd.DataFrame:
        if df_mbo.empty or df_snap.empty:
            return pd.DataFrame()

        defs = _load_definitions(df_def, dt)
        oi_map = _load_open_interest(df_stat)

        defs = defs.loc[defs["option_symbol"].isin(oi_map.keys())].copy()
        if defs.empty:
            return pd.DataFrame()

        eligible_ids = set(defs["instrument_id"].astype(int).tolist())
        df_mbo = df_mbo.loc[df_mbo["instrument_id"].isin(eligible_ids)].copy()
        if df_mbo.empty:
            return pd.DataFrame()

        df_mbo = df_mbo.sort_values(["ts_event", "sequence"], ascending=[True, True])
        df_mbo["window_id"] = (df_mbo["ts_event"] // WINDOW_NS).astype("int64")

        spot_map = df_snap.set_index("window_end_ts_ns")["spot_ref_price_int"].to_dict()

        books = _init_books(eligible_ids)
        last_mid: Dict[int, float] = {}

        rows: List[Dict[str, object]] = []

        curr_window = None
        window_start_ts = 0
        window_end_ts = 0

        for row in df_mbo.itertuples(index=False):
            ts = int(row.ts_event)
            window_id = int(row.window_id)
            if curr_window is None:
                curr_window = window_id
                window_start_ts = curr_window * WINDOW_NS
                window_end_ts = window_start_ts + WINDOW_NS

            if window_id != curr_window:
                _emit_gex_window(
                    rows,
                    books,
                    last_mid,
                    defs,
                    oi_map,
                    spot_map,
                    window_start_ts,
                    window_end_ts,
                    symbol,
                )
                curr_window = window_id
                window_start_ts = curr_window * WINDOW_NS
                window_end_ts = window_start_ts + WINDOW_NS

            _apply_mbo_row(books, last_mid, row)

        if curr_window is not None:
            _emit_gex_window(
                rows,
                books,
                last_mid,
                defs,
                oi_map,
                spot_map,
                window_start_ts,
                window_end_ts,
                symbol,
            )

        df_out = pd.DataFrame(rows)
        if df_out.empty:
            return df_out

        df_out = df_out.sort_values(["strike_price_int", "window_end_ts_ns"])
        grouped_abs = df_out.groupby("strike_price_int")["gex_abs"]
        grouped = df_out.groupby("strike_price_int")["gex"]
        grouped_ratio = df_out.groupby("strike_price_int")["gex_imbalance_ratio"]

        df_out["d1_gex_abs"] = grouped_abs.diff().fillna(0.0)
        df_out["d2_gex_abs"] = df_out.groupby("strike_price_int")["d1_gex_abs"].diff().fillna(0.0)
        df_out["d3_gex_abs"] = df_out.groupby("strike_price_int")["d2_gex_abs"].diff().fillna(0.0)

        df_out["d1_gex"] = grouped.diff().fillna(0.0)
        df_out["d2_gex"] = df_out.groupby("strike_price_int")["d1_gex"].diff().fillna(0.0)
        df_out["d3_gex"] = df_out.groupby("strike_price_int")["d2_gex"].diff().fillna(0.0)

        df_out["d1_gex_imbalance_ratio"] = grouped_ratio.diff().fillna(0.0)
        df_out["d2_gex_imbalance_ratio"] = df_out.groupby("strike_price_int")["d1_gex_imbalance_ratio"].diff().fillna(0.0)
        df_out["d3_gex_imbalance_ratio"] = df_out.groupby("strike_price_int")["d2_gex_imbalance_ratio"].diff().fillna(0.0)

        return df_out


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


def _init_books(instrument_ids: set[int]) -> Dict[int, Dict[str, object]]:
    books = {}
    for iid in instrument_ids:
        books[iid] = {
            "orders": {},
            "bid": {},
            "ask": {},
            "valid": True,
            "in_snapshot": False,
        }
    return books


def _apply_mbo_row(books: Dict[int, Dict[str, object]], last_mid: Dict[int, float], row) -> None:
    iid = int(row.instrument_id)
    book = books.get(iid)
    if book is None:
        return

    action = row.action
    side = row.side
    price = int(row.price)
    size = int(row.size)
    oid = int(row.order_id)
    flags = int(row.flags)

    if action == ACTION_CLEAR:
        book["orders"].clear()
        book["bid"].clear()
        book["ask"].clear()
        book["valid"] = False
        book["in_snapshot"] = bool(flags & F_SNAPSHOT)
        return

    if book["in_snapshot"] and (flags & F_LAST):
        book["in_snapshot"] = False
        book["valid"] = True

    if action == ACTION_ADD:
        book["orders"][oid] = OrderState(side=side, price_int=price, qty=size)
        _update_depth(book, side, price, size)
    elif action == ACTION_MODIFY:
        old = book["orders"].get(oid)
        if old is None:
            return
        _update_depth(book, old.side, old.price_int, -old.qty)
        new_side = old.side
        new_price = price
        new_qty = size
        book["orders"][oid] = OrderState(side=new_side, price_int=new_price, qty=new_qty)
        _update_depth(book, new_side, new_price, new_qty)
    elif action == ACTION_CANCEL:
        old = book["orders"].get(oid)
        if old is None:
            return
        _update_depth(book, old.side, old.price_int, -old.qty)
        del book["orders"][oid]
    elif action == ACTION_FILL:
        old = book["orders"].get(oid)
        if old is None:
            return
        fill_qty = size
        _update_depth(book, old.side, old.price_int, -fill_qty)
        remaining = old.qty - fill_qty
        if remaining <= 0:
            del book["orders"][oid]
        else:
            old.qty = remaining

    bid = _best_price(book["bid"], is_bid=True)
    ask = _best_price(book["ask"], is_bid=False)
    if bid > 0 and ask > 0:
        last_mid[iid] = (bid + ask) * 0.5 * PRICE_SCALE


def _update_depth(book: Dict[str, object], side: str, price: int, delta: int) -> None:
    depth = book["bid"] if side == "B" else book["ask"]
    depth[price] = depth.get(price, 0) + delta
    if depth[price] <= 0:
        del depth[price]


def _best_price(depth: Dict[int, int], is_bid: bool) -> int:
    if not depth:
        return 0
    return max(depth.keys()) if is_bid else min(depth.keys())


def _emit_gex_window(
    rows: List[Dict[str, object]],
    books: Dict[int, Dict[str, object]],
    last_mid: Dict[int, float],
    defs: pd.DataFrame,
    oi_map: Dict[str, float],
    spot_map: Dict[int, int],
    window_start_ts: int,
    window_end_ts: int,
    symbol: str,
) -> None:
    spot_ref_int = int(spot_map.get(window_end_ts, 0))
    if spot_ref_int <= 0:
        return
    spot_ref = spot_ref_int * PRICE_SCALE

    strike_ref = round(spot_ref / GEX_STRIKE_STEP_POINTS) * GEX_STRIKE_STEP_POINTS
    strike_points = [
        strike_ref + i * GEX_STRIKE_STEP_POINTS
        for i in range(-GEX_MAX_STRIKE_OFFSETS, GEX_MAX_STRIKE_OFFSETS + 1)
    ]
    strike_ints = [int(round(s / PRICE_SCALE)) for s in strike_points]
    strike_int_set = set(strike_ints)

    gex_call = {s: 0.0 for s in strike_ints}
    gex_put = {s: 0.0 for s in strike_ints}

    for row in defs.itertuples(index=False):
        strike_points_int = int(row.strike_price)
        if strike_points_int not in strike_int_set:
            continue
        strike_price = strike_points_int * PRICE_SCALE

        exp_ns = int(row.expiration)
        if exp_ns <= window_end_ts:
            continue

        dte_days = (exp_ns - window_end_ts) / 1e9 / 86400.0
        if dte_days <= 0:
            continue

        oi = float(oi_map.get(row.option_symbol, 0.0))
        if oi <= 0:
            continue

        book = books.get(int(row.instrument_id))
        if book is None or not book["valid"]:
            continue

        mid = last_mid.get(int(row.instrument_id), 0.0)
        if mid <= 0:
            continue

        T = dte_days / 365.0
        iv = _implied_vol(mid, spot_ref, strike_price, T, row.right)
        gamma = _black76_gamma(spot_ref, strike_price, T, iv)
        gex_val = gamma * oi * CONTRACT_MULTIPLIER

        if row.right == "C":
            gex_call[strike_points_int] += gex_val
        else:
            gex_put[strike_points_int] += gex_val

    for strike_int, strike in zip(strike_ints, strike_points):
        call_abs = gex_call[strike_int]
        put_abs = gex_put[strike_int]
        gex_abs = call_abs + put_abs
        gex = call_abs - put_abs
        imbalance = gex / (gex_abs + EPS_GEX)
        rows.append(
            {
                "window_start_ts_ns": window_start_ts,
                "window_end_ts_ns": window_end_ts,
                "underlying": symbol,
                "strike_price_int": strike_int,
                "underlying_spot_ref": spot_ref,
                "strike_points": strike,
                "gex_call_abs": call_abs,
                "gex_put_abs": put_abs,
                "gex_abs": gex_abs,
                "gex": gex,
                "gex_imbalance_ratio": imbalance,
                "d1_gex_abs": 0.0,
                "d2_gex_abs": 0.0,
                "d3_gex_abs": 0.0,
                "d1_gex": 0.0,
                "d2_gex": 0.0,
                "d3_gex": 0.0,
                "d1_gex_imbalance_ratio": 0.0,
                "d2_gex_imbalance_ratio": 0.0,
                "d3_gex_imbalance_ratio": 0.0,
            }
        )


def _black76_price(F: float, K: float, T: float, sigma: float, right: str) -> float:
    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    vol_sqrt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    disc = math.exp(-RISK_FREE_RATE * T)
    if right == "C":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def _implied_vol(price: float, F: float, K: float, T: float, right: str) -> float:
    if price <= 0 or F <= 0 or K <= 0 or T <= 0:
        return 0.0

    disc = math.exp(-RISK_FREE_RATE * T)
    intrinsic = max(F - K, 0.0) if right == "C" else max(K - F, 0.0)
    if price <= disc * intrinsic:
        return 0.0

    def func(sig: float) -> float:
        return _black76_price(F, K, T, sig, right) - price

    low = 1e-6
    high = 5.0
    f_low = func(low)
    f_high = func(high)
    if f_low * f_high > 0:
        return 0.0
    try:
        return float(brentq(func, low, high, maxiter=200))
    except Exception:
        return 0.0


def _black76_gamma(F: float, K: float, T: float, sigma: float) -> float:
    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    vol_sqrt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    disc = math.exp(-RISK_FREE_RATE * T)
    return disc * norm.pdf(d1) / (F * vol_sqrt)
