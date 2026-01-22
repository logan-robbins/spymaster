from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

PRICE_SCALE = 1e-9
TICK_SIZE = 0.01
TICK_INT = int(round(TICK_SIZE / PRICE_SCALE))
WINDOW_NS = 1_000_000_000
REST_NS = 500_000_000
EPS_QTY = 1.0
HUD_MAX_TICKS = 600

ACTION_ADD = "A"
ACTION_CANCEL = "C"
ACTION_MODIFY = "M"
ACTION_CLEAR = "R"
ACTION_TRADE = "T"
ACTION_FILL = "F"

PULL_ACTIONS = {ACTION_CANCEL, ACTION_MODIFY}


@dataclass
class OrderState:
    side: str
    price_int: int
    qty: int
    ts_enter_price: int


class SilverComputeEquitySnapshotAndWall1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_equity_snapshot_and_wall_1s",
            io=StageIO(
                inputs=["bronze.equity_mbo.mbo"],
                output="silver.equity_mbo.book_snapshot_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_key_snap = "silver.equity_mbo.book_snapshot_1s"
        out_key_wall = "silver.equity_mbo.wall_surface_1s"

        ref_snap = partition_ref(cfg, out_key_snap, symbol, dt)
        ref_wall = partition_ref(cfg, out_key_wall, symbol, dt)

        if is_partition_complete(ref_snap) and is_partition_complete(ref_wall):
            return

        input_key = self.io.inputs[0]
        in_ref = partition_ref(cfg, input_key, symbol, dt)
        if not is_partition_complete(in_ref):
            raise FileNotFoundError(f"Input not ready: {input_key} dt={dt}")

        in_contract_path = repo_root / cfg.dataset(input_key).contract
        in_contract = load_avro_contract(in_contract_path)
        df_in = read_partition(in_ref)
        df_in = enforce_contract(df_in, in_contract)

        df_snap, df_wall = self.transform(df_in, dt)

        contract_snap_path = repo_root / cfg.dataset(out_key_snap).contract
        contract_wall_path = repo_root / cfg.dataset(out_key_wall).contract
        contract_snap = load_avro_contract(contract_snap_path)
        contract_wall = load_avro_contract(contract_wall_path)

        df_snap = enforce_contract(df_snap, contract_snap)
        df_wall = enforce_contract(df_wall, contract_wall)

        manifest = read_manifest_hash(in_ref)
        lineage = [{"dataset": in_ref.dataset_key, "dt": dt, "manifest_sha256": manifest}]

        if not is_partition_complete(ref_snap):
            write_partition(
                cfg=cfg,
                dataset_key=out_key_snap,
                symbol=symbol,
                dt=dt,
                df=df_snap,
                contract_path=contract_snap_path,
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
                contract_path=contract_wall_path,
                inputs=lineage,
                stage=self.name,
            )

    def transform(self, df: pd.DataFrame, dt: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        required_cols = {"ts_event", "action", "side", "price", "size", "order_id", "sequence"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

        df = df.sort_values(["ts_event", "sequence"], ascending=[True, True])

        ts_event = df["ts_event"].to_numpy(dtype=np.int64)
        action = df["action"].to_numpy(dtype=object)
        side = df["side"].to_numpy(dtype=object)
        price = df["price"].to_numpy(dtype=np.int64)
        size = df["size"].to_numpy(dtype=np.int64)
        order_id = df["order_id"].to_numpy(dtype=np.int64)
        flags = df["flags"].to_numpy(dtype=np.int64) if "flags" in df.columns else np.zeros(len(df), dtype=np.int64)

        orders: Dict[int, OrderState] = {}
        depth_bid: Dict[int, int] = {}
        depth_ask: Dict[int, int] = {}

        wall_accum: Dict[Tuple[int, str], Dict[str, float]] = {}

        rows_snap = []
        rows_wall = []

        curr_window_id = None
        window_start_ts = 0
        window_end_ts = 0

        book_valid = False
        last_trade_price = 0

        F_SNAPSHOT = 128
        F_LAST = 256
        in_snapshot_load = False

        for i in range(len(df)):
            ts = ts_event[i]
            act = action[i]
            oid = order_id[i]
            px = price[i]
            sz = size[i]
            sd = side[i]
            flg = flags[i]

            w_id = ts // WINDOW_NS
            if curr_window_id is None:
                curr_window_id = w_id
                window_start_ts = w_id * WINDOW_NS
                window_end_ts = window_start_ts + WINDOW_NS

            if w_id != curr_window_id:
                bb_p, bb_q = _best_level(depth_bid, is_bid=True)
                ba_p, ba_q = _best_level(depth_ask, is_bid=False)

                mid = 0.0
                mid_int = 0
                if bb_p > 0 and ba_p > 0:
                    mid = (bb_p + ba_p) / 2.0 * PRICE_SCALE
                    mid_int = int(round((bb_p + ba_p) / 2.0))

                spot_ref = last_trade_price
                if spot_ref == 0:
                    if book_valid and bb_p > 0:
                        spot_ref = bb_p

                rows_snap.append(
                    {
                        "window_start_ts_ns": window_start_ts,
                        "window_end_ts_ns": window_end_ts,
                        "best_bid_price_int": bb_p,
                        "best_bid_qty": bb_q,
                        "best_ask_price_int": ba_p,
                        "best_ask_qty": ba_q,
                        "mid_price": mid,
                        "mid_price_int": mid_int,
                        "last_trade_price_int": last_trade_price,
                        "spot_ref_price_int": spot_ref,
                        "book_valid": book_valid,
                    }
                )

                if spot_ref > 0:
                    _emit_wall_rows(
                        rows_wall,
                        wall_accum,
                        depth_bid,
                        depth_ask,
                        window_start_ts,
                        window_end_ts,
                        spot_ref,
                        book_valid,
                        ts,
                        orders,
                    )

                wall_accum.clear()
                curr_window_id = w_id
                window_start_ts = w_id * WINDOW_NS
                window_end_ts = window_start_ts + WINDOW_NS

            if act == ACTION_CLEAR:
                orders.clear()
                depth_bid.clear()
                depth_ask.clear()
                if flg & F_SNAPSHOT:
                    in_snapshot_load = True
                    book_valid = False
                else:
                    in_snapshot_load = True
                    book_valid = False
                continue

            if in_snapshot_load and (flg & F_LAST):
                in_snapshot_load = False
                book_valid = True

            if act == ACTION_ADD:
                orders[oid] = OrderState(side=sd, price_int=px, qty=sz, ts_enter_price=ts)
                _apply_depth_delta(depth_bid, depth_ask, sd, px, sz)
                _accum_wall(wall_accum, px, sd, "add_qty", sz)

            elif act == ACTION_MODIFY:
                old = orders.get(oid)
                if old:
                    _apply_depth_delta(depth_bid, depth_ask, old.side, old.price_int, -old.qty)
                    new_px = px
                    new_sz = sz
                    orders[oid] = OrderState(side=old.side, price_int=new_px, qty=new_sz, ts_enter_price=ts)
                    if new_px == old.price_int:
                        orders[oid].ts_enter_price = old.ts_enter_price
                    _apply_depth_delta(depth_bid, depth_ask, old.side, new_px, new_sz)

                    if new_px != old.price_int:
                        delta_pull = old.qty
                        _accum_pull(wall_accum, old.price_int, old.side, delta_pull, ts, old.ts_enter_price)
                        _accum_wall(wall_accum, new_px, old.side, "add_qty", new_sz)
                    else:
                        if new_sz < old.qty:
                            delta = old.qty - new_sz
                            _accum_pull(wall_accum, new_px, old.side, delta, ts, old.ts_enter_price)
                        elif new_sz > old.qty:
                            delta = new_sz - old.qty
                            _accum_wall(wall_accum, new_px, old.side, "add_qty", delta)

            elif act == ACTION_CANCEL:
                old = orders.get(oid)
                if old:
                    _apply_depth_delta(depth_bid, depth_ask, old.side, old.price_int, -old.qty)
                    _accum_pull(wall_accum, old.price_int, old.side, old.qty, ts, old.ts_enter_price)
                    del orders[oid]

            elif act == ACTION_FILL:
                old = orders.get(oid)
                if old:
                    fill_sz = sz
                    rem_sz = old.qty - fill_sz
                    _apply_depth_delta(depth_bid, depth_ask, old.side, old.price_int, -fill_sz)
                    _accum_wall(wall_accum, old.price_int, old.side, "fill_qty", fill_sz)
                    if rem_sz <= 0:
                        del orders[oid]
                    else:
                        old.qty = rem_sz

            elif act == ACTION_TRADE:
                if px > 0:
                    last_trade_price = px

        if curr_window_id is not None:
            bb_p, bb_q = _best_level(depth_bid, is_bid=True)
            ba_p, ba_q = _best_level(depth_ask, is_bid=False)
            mid = (bb_p + ba_p) * 0.5 * PRICE_SCALE if (bb_p and ba_p) else 0.0
            mid_int = int((bb_p + ba_p) * 0.5) if (bb_p and ba_p) else 0
            spot_ref = last_trade_price if last_trade_price else (bb_p if (book_valid and bb_p) else 0)

            rows_snap.append(
                {
                    "window_start_ts_ns": window_start_ts,
                    "window_end_ts_ns": window_end_ts,
                    "best_bid_price_int": bb_p,
                    "best_bid_qty": bb_q,
                    "best_ask_price_int": ba_p,
                    "best_ask_qty": ba_q,
                    "mid_price": mid,
                    "mid_price_int": mid_int,
                    "last_trade_price_int": last_trade_price,
                    "spot_ref_price_int": spot_ref,
                    "book_valid": book_valid,
                }
            )
            if spot_ref > 0:
                _emit_wall_rows(
                    rows_wall,
                    wall_accum,
                    depth_bid,
                    depth_ask,
                    window_start_ts,
                    window_end_ts,
                    spot_ref,
                    book_valid,
                    ts_event[-1],
                    orders,
                )

        df_snap_out = pd.DataFrame(rows_snap)
        df_wall_out = pd.DataFrame(rows_wall)

        if not df_wall_out.empty:
            df_wall_out = _calc_derivatives(df_wall_out)
        else:
            cols = [
                "window_start_ts_ns",
                "window_end_ts_ns",
                "price_int",
                "side",
                "spot_ref_price_int",
                "rel_ticks",
                "depth_qty_start",
                "depth_qty_end",
                "add_qty",
                "pull_qty_total",
                "depth_qty_rest",
                "pull_qty_rest",
                "fill_qty",
                "d1_depth_qty",
                "d2_depth_qty",
                "d3_depth_qty",
                "window_valid",
            ]
            df_wall_out = pd.DataFrame(columns=cols)

        return df_snap_out, df_wall_out


def _apply_depth_delta(bid_book, ask_book, side, price, delta):
    if side == "B":
        bid_book[price] = bid_book.get(price, 0) + delta
        if bid_book[price] <= 0:
            del bid_book[price]
    elif side == "A":
        ask_book[price] = ask_book.get(price, 0) + delta
        if ask_book[price] <= 0:
            del ask_book[price]


def _best_level(book, is_bid):
    if not book:
        return 0, 0
    if is_bid:
        p = max(book.keys())
    else:
        p = min(book.keys())
    return p, book[p]


def _accum_wall(accum, price, side, field, value):
    key = (price, side)
    if key not in accum:
        accum[key] = {"add_qty": 0.0, "pull_qty_total": 0.0, "pull_qty_rest": 0.0, "fill_qty": 0.0}
    accum[key][field] += float(value)


def _accum_pull(accum, price, side, qty, curr_ts, enter_ts):
    _accum_wall(accum, price, side, "pull_qty_total", qty)
    if (curr_ts - enter_ts) >= REST_NS:
        _accum_wall(accum, price, side, "pull_qty_rest", qty)


def _emit_wall_rows(
    rows,
    accum,
    depth_bid,
    depth_ask,
    start_ts,
    end_ts,
    spot_ref,
    valid,
    curr_ts,
    orders,
):
    min_p = spot_ref - HUD_MAX_TICKS * TICK_INT
    max_p = spot_ref + HUD_MAX_TICKS * TICK_INT

    active_prices = set()
    for p in depth_bid.keys():
        if min_p <= p <= max_p:
            active_prices.add((p, "B"))
    for p in depth_ask.keys():
        if min_p <= p <= max_p:
            active_prices.add((p, "A"))
    for (p, s) in accum.keys():
        if min_p <= p <= max_p:
            active_prices.add((p, s))

    res_depth = {}
    for oid, state in orders.items():
        if state.qty <= 0:
            continue
        p = state.price_int
        s = state.side
        if min_p <= p <= max_p and (curr_ts - state.ts_enter_price) >= REST_NS:
            res_depth[(p, s)] = res_depth.get((p, s), 0.0) + float(state.qty)

    for (p, s) in active_prices:
        depth_book = depth_bid if s == "B" else depth_ask
        depth_end = float(depth_book.get(p, 0))
        depth_start = depth_end

        acc = accum.get((p, s), {"add_qty": 0.0, "pull_qty_total": 0.0, "pull_qty_rest": 0.0, "fill_qty": 0.0})
        add_qty = float(acc["add_qty"])
        pull_total = float(acc["pull_qty_total"])
        fill_qty = float(acc["fill_qty"])
        depth_start = depth_end - add_qty + pull_total + fill_qty
        if depth_start < 0:
            depth_start = 0.0

        rest_qty = float(res_depth.get((p, s), 0.0))
        rel_ticks = int(round((p - spot_ref) / TICK_INT))

        rows.append(
            {
                "window_start_ts_ns": int(start_ts),
                "window_end_ts_ns": int(end_ts),
                "price_int": int(p),
                "side": str(s),
                "spot_ref_price_int": int(spot_ref),
                "rel_ticks": int(rel_ticks),
                "depth_qty_start": float(depth_start),
                "depth_qty_end": float(depth_end),
                "add_qty": float(add_qty),
                "pull_qty_total": float(pull_total),
                "depth_qty_rest": float(rest_qty),
                "pull_qty_rest": float(acc["pull_qty_rest"]),
                "fill_qty": float(fill_qty),
                "d1_depth_qty": 0.0,
                "d2_depth_qty": 0.0,
                "d3_depth_qty": 0.0,
                "window_valid": bool(valid),
            }
        )


def _calc_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["side", "price_int", "window_end_ts_ns"]).copy()
    group = df.groupby(["side", "price_int"])["depth_qty_end"]
    df["d1_depth_qty"] = group.diff().fillna(0.0)
    df["d2_depth_qty"] = df.groupby(["side", "price_int"])["d1_depth_qty"].diff().fillna(0.0)
    df["d3_depth_qty"] = df.groupby(["side", "price_int"])["d2_depth_qty"].diff().fillna(0.0)
    return df
