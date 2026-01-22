from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
PULL_ACTIONS = {"C", "M"}
DELTA_TICKS = 20
AT_TICKS = 2
NEAR_TICKS = 5
FAR_TICKS_LOW = 15
FAR_TICKS_HIGH = 20
WINDOW_NS = 1_000_000_000
REST_NS = 500_000_000
EPS_QTY = 1.0
EPS_DIST_TICKS = 1.0
TRADE_ACTION = "T"
TINY_TOL = 1e-9

BUCKET_OUT = "OUT"
ASK_ABOVE_AT = "ASK_ABOVE_AT"
ASK_ABOVE_NEAR = "ASK_ABOVE_NEAR"
ASK_ABOVE_MID = "ASK_ABOVE_MID"
ASK_ABOVE_FAR = "ASK_ABOVE_FAR"
BID_BELOW_AT = "BID_BELOW_AT"
BID_BELOW_NEAR = "BID_BELOW_NEAR"
BID_BELOW_MID = "BID_BELOW_MID"
BID_BELOW_FAR = "BID_BELOW_FAR"

ASK_INFLUENCE = {ASK_ABOVE_AT, ASK_ABOVE_NEAR, ASK_ABOVE_MID, ASK_ABOVE_FAR}
BID_INFLUENCE = {BID_BELOW_AT, BID_BELOW_NEAR, BID_BELOW_MID, BID_BELOW_FAR}

BASE_FEATURES = [
    "f1_ask_com_disp_log",
    "f1_ask_slope_convex_log",
    "f1_ask_slope_inner_log",
    "f1_ask_at_share_delta",
    "f1_ask_near_share_delta",
    "f1_ask_reprice_away_share_rest",
    "f2_ask_pull_add_log_rest",
    "f2_ask_pull_intensity_log_rest",
    "f2_ask_at_pull_share_rest",
    "f2_ask_near_pull_share_rest",
    "f3_bid_com_disp_log",
    "f3_bid_slope_convex_log",
    "f3_bid_slope_inner_log",
    "f3_bid_at_share_delta",
    "f3_bid_near_share_delta",
    "f3_bid_reprice_away_share_rest",
    "f4_bid_pull_add_log_rest",
    "f4_bid_pull_intensity_log_rest",
    "f4_bid_at_pull_share_rest",
    "f4_bid_near_pull_share_rest",
    "f5_vacuum_expansion_log",
    "f6_vacuum_decay_log",
    "f7_vacuum_total_log",
    "f8_ask_bbo_dist_ticks",
    "f9_bid_bbo_dist_ticks",
]

UP_FEATURES = [
    "u1_ask_com_disp_log",
    "u2_ask_slope_convex_log",
    "u2_ask_slope_inner_log",
    "u3_ask_at_share_decay",
    "u3_ask_near_share_decay",
    "u4_ask_reprice_away_share_rest",
    "u5_ask_pull_add_log_rest",
    "u6_ask_pull_intensity_log_rest",
    "u7_ask_at_pull_share_rest",
    "u7_ask_near_pull_share_rest",
    "u8_bid_com_approach_log",
    "u9_bid_slope_support_log",
    "u9_bid_slope_inner_log",
    "u10_bid_at_share_rise",
    "u10_bid_near_share_rise",
    "u11_bid_reprice_toward_share_rest",
    "u12_bid_add_pull_log_rest",
    "u13_bid_add_intensity_log",
    "u14_bid_far_pull_share_rest",
    "u15_up_expansion_log",
    "u16_up_flow_log",
    "u17_up_total_log",
    "u18_ask_bbo_dist_ticks",
    "u19_bid_bbo_dist_ticks",
]

DERIV_FEATURES = [f"d1_{name}" for name in BASE_FEATURES] + [
    f"d2_{name}" for name in BASE_FEATURES
] + [f"d3_{name}" for name in BASE_FEATURES]

UP_DERIV_FEATURES = [f"d1_{name}" for name in UP_FEATURES] + [
    f"d2_{name}" for name in UP_FEATURES
] + [f"d3_{name}" for name in UP_FEATURES]

OUTPUT_COLUMNS = [
    "window_start_ts_ns",
    "window_end_ts_ns",
    "spot_ref_price",
    "spot_ref_price_int",
    "approach_dir",
] + BASE_FEATURES + DERIV_FEATURES + UP_FEATURES + UP_DERIV_FEATURES


@dataclass
class OrderState:
    side: str
    price_int: int
    qty: int
    ts_enter_price: int


class SilverComputeEquityRadarVacuum1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_equity_radar_vacuum_1s",
            io=StageIO(
                inputs=["bronze.equity_mbo.mbo", "silver.equity_mbo.book_snapshot_1s"],
                output="silver.equity_mbo.radar_vacuum_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        mbo_key = "bronze.equity_mbo.mbo"
        snap_key = "silver.equity_mbo.book_snapshot_1s"

        mbo_ref = partition_ref(cfg, mbo_key, symbol, dt)
        snap_ref = partition_ref(cfg, snap_key, symbol, dt)

        if not is_partition_complete(mbo_ref):
            raise FileNotFoundError(f"Input not ready: {mbo_key} dt={dt}")
        if not is_partition_complete(snap_ref):
            raise FileNotFoundError(f"Input not ready: {snap_key} dt={dt}")

        mbo_contract = load_avro_contract(repo_root / cfg.dataset(mbo_key).contract)
        snap_contract = load_avro_contract(repo_root / cfg.dataset(snap_key).contract)

        df_mbo = enforce_contract(read_partition(mbo_ref), mbo_contract)
        df_snap = enforce_contract(read_partition(snap_ref), snap_contract)

        df_out = self.transform(df_mbo, df_snap, symbol)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {"dataset": mbo_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(mbo_ref)},
            {"dataset": snap_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(snap_ref)},
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

    def transform(self, df_mbo: pd.DataFrame, df_snap: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df_mbo.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        df_mbo = df_mbo.loc[df_mbo["symbol"] == symbol].copy()
        if df_mbo.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        df_mbo = df_mbo.loc[df_mbo["action"] != "N"].copy()
        if df_mbo.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        df_mbo = df_mbo.sort_values(["ts_event", "sequence"], ascending=[True, True])

        spot_map = dict(zip(df_snap["window_end_ts_ns"], df_snap["spot_ref_price_int"]))

        return compute_radar_vacuum_1s(df_mbo, spot_map)


def compute_radar_vacuum_1s(df: pd.DataFrame, spot_map: Dict[int, int]) -> pd.DataFrame:
    orders: Dict[int, OrderState] = {}
    accum = _new_accumulators()
    rows = []

    curr_window_id = None
    window_start_ts = 0
    window_end_ts = 0
    curr_spot_ref = 0

    start_snap: Optional[Dict[str, float]] = None

    ts_event_arr = df["ts_event"].values.astype(np.int64)
    action_arr = df["action"].values
    side_arr = df["side"].values
    price_arr = df["price"].values.astype(np.int64)
    size_arr = df["size"].values.astype(np.int64)
    oid_arr = df["order_id"].values.astype(np.int64)

    for i in range(len(df)):
        ts = ts_event_arr[i]
        act = action_arr[i]

        w_id = ts // WINDOW_NS

        if curr_window_id is None:
            curr_window_id = w_id
            window_start_ts = w_id * WINDOW_NS
            window_end_ts = window_start_ts + WINDOW_NS
            curr_spot_ref = spot_map.get(window_end_ts, 0)
            start_snap = _snapshot(orders, curr_spot_ref) if curr_spot_ref > 0 else None
        elif w_id != curr_window_id:
            if curr_spot_ref > 0 and start_snap is not None:
                end_snap = _snapshot(orders, curr_spot_ref)
                base_feats = _compute_base_features(start_snap, end_snap, accum)
                up_feats = _compute_up_features(base_feats, start_snap, accum)
                row = {
                    "window_start_ts_ns": window_start_ts,
                    "window_end_ts_ns": window_end_ts,
                    "spot_ref_price": curr_spot_ref * PRICE_SCALE,
                    "spot_ref_price_int": curr_spot_ref,
                }
                row = {**row, **base_feats, **up_feats}
                rows.append(row)

            curr_window_id = w_id
            window_start_ts = curr_window_id * WINDOW_NS
            window_end_ts = window_start_ts + WINDOW_NS
            curr_spot_ref = spot_map.get(window_end_ts, 0)
            accum = _new_accumulators()
            start_snap = _snapshot(orders, curr_spot_ref) if curr_spot_ref > 0 else None

        if act == "R":
            orders.clear()
            accum = _new_accumulators()
            continue

        oid = oid_arr[i]
        old_order = orders.get(oid)

        if curr_spot_ref <= 0:
            _track_orders_simple(orders, act, oid, side_arr[i], price_arr[i], size_arr[i], ts)
            continue

        if old_order:
            old_side = old_order.side
            old_px = old_order.price_int
            old_qty = old_order.qty
            old_bucket = _bucket_for(old_side, old_px, curr_spot_ref)
            old_ts_enter = old_order.ts_enter_price
        else:
            old_side = ""
            old_px = 0
            old_qty = 0
            old_bucket = BUCKET_OUT
            old_ts_enter = 0

        old_in_ask = old_side == "A" and old_bucket in ASK_INFLUENCE
        old_in_bid = old_side == "B" and old_bucket in BID_INFLUENCE

        new_side = old_side
        new_px = old_px
        new_qty = old_qty
        new_bucket = old_bucket
        new_ts_enter = old_ts_enter
        has_new = False

        if act == "A":
            new_side = side_arr[i]
            new_px = price_arr[i]
            new_qty = size_arr[i]
            new_bucket = _bucket_for(new_side, new_px, curr_spot_ref)
            new_ts_enter = ts
            has_new = True
            orders[oid] = OrderState(new_side, new_px, new_qty, new_ts_enter)
        elif act == "C":
            orders.pop(oid, None)
            has_new = False
        elif act == "M":
            new_px = price_arr[i]
            new_qty = size_arr[i]
            new_bucket = _bucket_for(old_side, new_px, curr_spot_ref)
            if new_px != old_px:
                new_ts_enter = ts
            has_new = True
            orders[oid] = OrderState(old_side, new_px, new_qty, new_ts_enter)
        elif act == "F":
            fill_sz = size_arr[i]
            new_qty = old_qty - fill_sz
            if new_qty > 0:
                orders[oid].qty = new_qty
                has_new = True
            else:
                orders.pop(oid, None)
                has_new = False
        elif act == "T":
            continue

        new_in_ask = has_new and new_side == "A" and new_bucket in ASK_INFLUENCE
        new_in_bid = has_new and new_side == "B" and new_bucket in BID_INFLUENCE

        q_old_ask = old_qty if old_in_ask else 0
        q_new_ask = new_qty if new_in_ask else 0
        q_old_bid = old_qty if old_in_bid else 0
        q_new_bid = new_qty if new_in_bid else 0

        delta_ask = q_new_ask - q_old_ask
        delta_bid = q_new_bid - q_old_bid

        if delta_ask > 0:
            accum["ask_add_qty"] += float(delta_ask)
        if delta_bid > 0:
            accum["bid_add_qty"] += float(delta_bid)

        if delta_ask < 0 and act in PULL_ACTIONS:
            age = ts - old_ts_enter
            if age >= REST_NS:
                pull = float(-delta_ask)
                accum["ask_pull_rest_qty"] += pull
                if old_bucket == ASK_ABOVE_AT:
                    accum["ask_pull_rest_qty_at"] += pull
                elif old_bucket == ASK_ABOVE_NEAR:
                    accum["ask_pull_rest_qty_near"] += pull

        if delta_bid < 0 and act in PULL_ACTIONS:
            age = ts - old_ts_enter
            if age >= REST_NS:
                pull = float(-delta_bid)
                accum["bid_pull_rest_qty"] += pull
                if old_bucket == BID_BELOW_AT:
                    accum["bid_pull_rest_qty_at"] += pull
                elif old_bucket == BID_BELOW_NEAR:
                    accum["bid_pull_rest_qty_near"] += pull

        if act == "M":
            if old_side == "A" and old_in_ask:
                age = ts - old_ts_enter
                if age >= REST_NS:
                    dist_old = old_px - curr_spot_ref
                    dist_new = new_px - curr_spot_ref
                    if new_px <= curr_spot_ref:
                        accum["ask_reprice_toward_rest_qty"] += float(old_qty)
                    else:
                        if dist_new > dist_old:
                            accum["ask_reprice_away_rest_qty"] += float(old_qty)
                        elif dist_new < dist_old:
                            accum["ask_reprice_toward_rest_qty"] += float(old_qty)

            if old_side == "B" and old_in_bid:
                age = ts - old_ts_enter
                if age >= REST_NS:
                    dist_old = curr_spot_ref - old_px
                    dist_new = curr_spot_ref - new_px
                    if new_px >= curr_spot_ref:
                        accum["bid_reprice_toward_rest_qty"] += float(old_qty)
                    else:
                        if dist_new > dist_old:
                            accum["bid_reprice_away_rest_qty"] += float(old_qty)
                        elif dist_new < dist_old:
                            accum["bid_reprice_toward_rest_qty"] += float(old_qty)

    if curr_window_id is not None:
        if curr_spot_ref > 0 and start_snap is not None:
            end_snap = _snapshot(orders, curr_spot_ref)
            base_feats = _compute_base_features(start_snap, end_snap, accum)
            up_feats = _compute_up_features(base_feats, start_snap, accum)
            row = {
                "window_start_ts_ns": window_start_ts,
                "window_end_ts_ns": window_end_ts,
                "spot_ref_price": curr_spot_ref * PRICE_SCALE,
                "spot_ref_price_int": curr_spot_ref,
            }
            row = {**row, **base_feats, **up_feats}
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    _add_derivatives(rows, BASE_FEATURES)
    _add_derivatives(rows, UP_FEATURES)

    df_out = pd.DataFrame(rows)
    spot_ints = df_out["spot_ref_price_int"].values
    df_out["approach_dir"] = _compute_approach_dir_simple(spot_ints)

    _validate_feature_ranges(df_out)
    return df_out[OUTPUT_COLUMNS]


def _track_orders_simple(orders, act, oid, side, price, size, ts):
    if act == "A":
        orders[oid] = OrderState(side, price, size, ts)
    elif act == "C":
        orders.pop(oid, None)
    elif act == "M":
        old = orders.get(oid)
        ts_enter = ts if (old and old.price_int != price) else (old.ts_enter_price if old else ts)
        orders[oid] = OrderState(old.side if old else side, price, size, ts_enter)
    elif act == "F":
        old = orders.get(oid)
        if old:
            rem = old.qty - size
            if rem > 0:
                old.qty = rem
            else:
                orders.pop(oid, None)


def _bucket_for(side: str, price_int: int, spot_ref: int) -> str:
    if side == "A" and price_int >= spot_ref:
        ticks = int(round((price_int - spot_ref) / TICK_INT))
        if ticks < 0 or ticks > DELTA_TICKS:
            return BUCKET_OUT
        if ticks <= AT_TICKS:
            return ASK_ABOVE_AT
        if ticks <= NEAR_TICKS:
            return ASK_ABOVE_NEAR
        if ticks >= FAR_TICKS_LOW:
            return ASK_ABOVE_FAR
        return ASK_ABOVE_MID

    if side == "B" and price_int <= spot_ref:
        ticks = int(round((spot_ref - price_int) / TICK_INT))
        if ticks < 0 or ticks > DELTA_TICKS:
            return BUCKET_OUT
        if ticks <= AT_TICKS:
            return BID_BELOW_AT
        if ticks <= NEAR_TICKS:
            return BID_BELOW_NEAR
        if ticks >= FAR_TICKS_LOW:
            return BID_BELOW_FAR
        return BID_BELOW_MID

    return BUCKET_OUT


def _snapshot(orders: Dict[int, OrderState], spot_ref: int) -> Dict[str, float]:
    d = {
        "ask_depth_total": 0.0,
        "ask_depth_at": 0.0,
        "ask_depth_near": 0.0,
        "ask_depth_far": 0.0,
        "bid_depth_total": 0.0,
        "bid_depth_at": 0.0,
        "bid_depth_near": 0.0,
        "bid_depth_far": 0.0,
    }

    ask_com_num = 0.0
    bid_com_num = 0.0
    min_ask = None
    max_bid = None

    for o in orders.values():
        bucket = _bucket_for(o.side, o.price_int, spot_ref)
        if bucket == BUCKET_OUT:
            continue

        qty = float(o.qty)
        if o.side == "A":
            d["ask_depth_total"] += qty
            ask_com_num += float(o.price_int) * qty
            if bucket == ASK_ABOVE_AT:
                d["ask_depth_at"] += qty
            elif bucket == ASK_ABOVE_NEAR:
                d["ask_depth_near"] += qty
            elif bucket == ASK_ABOVE_FAR:
                d["ask_depth_far"] += qty
            if min_ask is None or o.price_int < min_ask:
                min_ask = o.price_int
        else:
            d["bid_depth_total"] += qty
            bid_com_num += float(o.price_int) * qty
            if bucket == BID_BELOW_AT:
                d["bid_depth_at"] += qty
            elif bucket == BID_BELOW_NEAR:
                d["bid_depth_near"] += qty
            elif bucket == BID_BELOW_FAR:
                d["bid_depth_far"] += qty
            if max_bid is None or o.price_int > max_bid:
                max_bid = o.price_int

    if d["ask_depth_total"] < TINY_TOL:
        d["d_ask_ticks"] = float(DELTA_TICKS)
    else:
        com_p = ask_com_num / d["ask_depth_total"]
        d["d_ask_ticks"] = max((com_p - spot_ref) / TICK_INT, 0.0)

    if d["bid_depth_total"] < TINY_TOL:
        d["d_bid_ticks"] = float(DELTA_TICKS)
    else:
        com_p = bid_com_num / d["bid_depth_total"]
        d["d_bid_ticks"] = max((spot_ref - com_p) / TICK_INT, 0.0)

    if min_ask is None:
        d["bbo_ask_ticks"] = float(DELTA_TICKS)
    else:
        d["bbo_ask_ticks"] = max((min_ask - spot_ref) / TICK_INT, 0.0)

    if max_bid is None:
        d["bbo_bid_ticks"] = float(DELTA_TICKS)
    else:
        d["bbo_bid_ticks"] = max((spot_ref - max_bid) / TICK_INT, 0.0)

    return d


def _new_accumulators() -> Dict[str, float]:
    return {
        "ask_add_qty": 0.0,
        "ask_pull_rest_qty": 0.0,
        "ask_pull_rest_qty_at": 0.0,
        "ask_pull_rest_qty_near": 0.0,
        "ask_reprice_away_rest_qty": 0.0,
        "ask_reprice_toward_rest_qty": 0.0,
        "bid_add_qty": 0.0,
        "bid_pull_rest_qty": 0.0,
        "bid_pull_rest_qty_at": 0.0,
        "bid_pull_rest_qty_near": 0.0,
        "bid_reprice_away_rest_qty": 0.0,
        "bid_reprice_toward_rest_qty": 0.0,
    }


def _compute_base_features(start: dict, end: dict, acc: dict) -> dict:
    f1_ask_com_disp_log = math.log(
        (end["d_ask_ticks"] + EPS_DIST_TICKS) / (start["d_ask_ticks"] + EPS_DIST_TICKS)
    )
    f1_ask_slope_convex_log = math.log((end["ask_depth_far"] + EPS_QTY) / (end["ask_depth_near"] + EPS_QTY))
    f1_ask_slope_inner_log = math.log((end["ask_depth_near"] + EPS_QTY) / (end["ask_depth_at"] + EPS_QTY))
    at_share_start = start["ask_depth_at"] / (start["ask_depth_total"] + EPS_QTY)
    at_share_end = end["ask_depth_at"] / (end["ask_depth_total"] + EPS_QTY)
    f1_ask_at_share_delta = at_share_end - at_share_start
    near_share_start = start["ask_depth_near"] / (start["ask_depth_total"] + EPS_QTY)
    near_share_end = end["ask_depth_near"] / (end["ask_depth_total"] + EPS_QTY)
    f1_ask_near_share_delta = near_share_end - near_share_start

    den_ask_reprice = acc["ask_reprice_away_rest_qty"] + acc["ask_reprice_toward_rest_qty"]
    if den_ask_reprice == 0:
        f1_ask_reprice_away_share_rest = 0.5
    else:
        f1_ask_reprice_away_share_rest = acc["ask_reprice_away_rest_qty"] / (den_ask_reprice + EPS_QTY)

    f2_ask_pull_add_log_rest = math.log((acc["ask_pull_rest_qty"] + EPS_QTY) / (acc["ask_add_qty"] + EPS_QTY))
    f2_ask_pull_intensity_log_rest = math.log1p(
        acc["ask_pull_rest_qty"] / (start["ask_depth_total"] + EPS_QTY)
    )
    f2_ask_at_pull_share_rest = acc["ask_pull_rest_qty_at"] / (acc["ask_pull_rest_qty"] + EPS_QTY)
    f2_ask_near_pull_share_rest = acc["ask_pull_rest_qty_near"] / (acc["ask_pull_rest_qty"] + EPS_QTY)

    f3_bid_com_disp_log = math.log(
        (end["d_bid_ticks"] + EPS_DIST_TICKS) / (start["d_bid_ticks"] + EPS_DIST_TICKS)
    )
    f3_bid_slope_convex_log = math.log((end["bid_depth_far"] + EPS_QTY) / (end["bid_depth_near"] + EPS_QTY))
    f3_bid_slope_inner_log = math.log((end["bid_depth_near"] + EPS_QTY) / (end["bid_depth_at"] + EPS_QTY))
    at_share_start = start["bid_depth_at"] / (start["bid_depth_total"] + EPS_QTY)
    at_share_end = end["bid_depth_at"] / (end["bid_depth_total"] + EPS_QTY)
    f3_bid_at_share_delta = at_share_end - at_share_start
    near_share_start = start["bid_depth_near"] / (start["bid_depth_total"] + EPS_QTY)
    near_share_end = end["bid_depth_near"] / (end["bid_depth_total"] + EPS_QTY)
    f3_bid_near_share_delta = near_share_end - near_share_start

    den_bid_reprice = acc["bid_reprice_away_rest_qty"] + acc["bid_reprice_toward_rest_qty"]
    if den_bid_reprice == 0:
        f3_bid_reprice_away_share_rest = 0.5
    else:
        f3_bid_reprice_away_share_rest = acc["bid_reprice_away_rest_qty"] / (den_bid_reprice + EPS_QTY)

    f4_bid_pull_add_log_rest = math.log((acc["bid_pull_rest_qty"] + EPS_QTY) / (acc["bid_add_qty"] + EPS_QTY))
    f4_bid_pull_intensity_log_rest = math.log1p(
        acc["bid_pull_rest_qty"] / (start["bid_depth_total"] + EPS_QTY)
    )
    f4_bid_at_pull_share_rest = acc["bid_pull_rest_qty_at"] / (acc["bid_pull_rest_qty"] + EPS_QTY)
    f4_bid_near_pull_share_rest = acc["bid_pull_rest_qty_near"] / (acc["bid_pull_rest_qty"] + EPS_QTY)

    f5_vacuum_expansion_log = f1_ask_com_disp_log + f3_bid_com_disp_log
    f6_vacuum_decay_log = f2_ask_pull_add_log_rest + f4_bid_pull_add_log_rest
    f7_vacuum_total_log = f5_vacuum_expansion_log + f6_vacuum_decay_log

    return {
        "f1_ask_com_disp_log": float(f1_ask_com_disp_log),
        "f1_ask_slope_convex_log": float(f1_ask_slope_convex_log),
        "f1_ask_slope_inner_log": float(f1_ask_slope_inner_log),
        "f1_ask_at_share_delta": float(f1_ask_at_share_delta),
        "f1_ask_near_share_delta": float(f1_ask_near_share_delta),
        "f1_ask_reprice_away_share_rest": float(f1_ask_reprice_away_share_rest),
        "f2_ask_pull_add_log_rest": float(f2_ask_pull_add_log_rest),
        "f2_ask_pull_intensity_log_rest": float(f2_ask_pull_intensity_log_rest),
        "f2_ask_at_pull_share_rest": float(f2_ask_at_pull_share_rest),
        "f2_ask_near_pull_share_rest": float(f2_ask_near_pull_share_rest),
        "f3_bid_com_disp_log": float(f3_bid_com_disp_log),
        "f3_bid_slope_convex_log": float(f3_bid_slope_convex_log),
        "f3_bid_slope_inner_log": float(f3_bid_slope_inner_log),
        "f3_bid_at_share_delta": float(f3_bid_at_share_delta),
        "f3_bid_near_share_delta": float(f3_bid_near_share_delta),
        "f3_bid_reprice_away_share_rest": float(f3_bid_reprice_away_share_rest),
        "f4_bid_pull_add_log_rest": float(f4_bid_pull_add_log_rest),
        "f4_bid_pull_intensity_log_rest": float(f4_bid_pull_intensity_log_rest),
        "f4_bid_at_pull_share_rest": float(f4_bid_at_pull_share_rest),
        "f4_bid_near_pull_share_rest": float(f4_bid_near_pull_share_rest),
        "f5_vacuum_expansion_log": float(f5_vacuum_expansion_log),
        "f6_vacuum_decay_log": float(f6_vacuum_decay_log),
        "f7_vacuum_total_log": float(f7_vacuum_total_log),
        "f8_ask_bbo_dist_ticks": float(end["bbo_ask_ticks"]),
        "f9_bid_bbo_dist_ticks": float(end["bbo_bid_ticks"]),
    }


def _compute_up_features(base: dict, start: dict, acc: dict) -> dict:
    u1_ask_com_disp_log = -base["f1_ask_com_disp_log"]
    u2_ask_slope_convex_log = base["f1_ask_slope_convex_log"]
    u2_ask_slope_inner_log = base["f1_ask_slope_inner_log"]
    u3_ask_at_share_decay = -base["f1_ask_at_share_delta"]
    u3_ask_near_share_decay = -base["f1_ask_near_share_delta"]
    u4_ask_reprice_away_share_rest = base["f1_ask_reprice_away_share_rest"]
    u5_ask_pull_add_log_rest = base["f2_ask_pull_add_log_rest"]
    u6_ask_pull_intensity_log_rest = base["f2_ask_pull_intensity_log_rest"]
    u7_ask_at_pull_share_rest = base["f2_ask_at_pull_share_rest"]
    u7_ask_near_pull_share_rest = base["f2_ask_near_pull_share_rest"]
    u8_bid_com_approach_log = -base["f3_bid_com_disp_log"]
    u9_bid_slope_support_log = base["f3_bid_slope_convex_log"]
    u9_bid_slope_inner_log = base["f3_bid_slope_inner_log"]
    u10_bid_at_share_rise = base["f3_bid_at_share_delta"]
    u10_bid_near_share_rise = base["f3_bid_near_share_delta"]
    u11_bid_reprice_toward_share_rest = base["f3_bid_reprice_away_share_rest"]
    u12_bid_add_pull_log_rest = base["f4_bid_pull_add_log_rest"]
    u13_bid_add_intensity_log = base["f4_bid_pull_intensity_log_rest"]
    u14_bid_far_pull_share_rest = base["f4_bid_near_pull_share_rest"]
    u15_up_expansion_log = u1_ask_com_disp_log + u8_bid_com_approach_log
    u16_up_flow_log = u5_ask_pull_add_log_rest + u12_bid_add_pull_log_rest
    u17_up_total_log = u15_up_expansion_log + u16_up_flow_log
    u18_ask_bbo_dist_ticks = base["f8_ask_bbo_dist_ticks"]
    u19_bid_bbo_dist_ticks = base["f9_bid_bbo_dist_ticks"]

    return {
        "u1_ask_com_disp_log": float(u1_ask_com_disp_log),
        "u2_ask_slope_convex_log": float(u2_ask_slope_convex_log),
        "u2_ask_slope_inner_log": float(u2_ask_slope_inner_log),
        "u3_ask_at_share_decay": float(u3_ask_at_share_decay),
        "u3_ask_near_share_decay": float(u3_ask_near_share_decay),
        "u4_ask_reprice_away_share_rest": float(u4_ask_reprice_away_share_rest),
        "u5_ask_pull_add_log_rest": float(u5_ask_pull_add_log_rest),
        "u6_ask_pull_intensity_log_rest": float(u6_ask_pull_intensity_log_rest),
        "u7_ask_at_pull_share_rest": float(u7_ask_at_pull_share_rest),
        "u7_ask_near_pull_share_rest": float(u7_ask_near_pull_share_rest),
        "u8_bid_com_approach_log": float(u8_bid_com_approach_log),
        "u9_bid_slope_support_log": float(u9_bid_slope_support_log),
        "u9_bid_slope_inner_log": float(u9_bid_slope_inner_log),
        "u10_bid_at_share_rise": float(u10_bid_at_share_rise),
        "u10_bid_near_share_rise": float(u10_bid_near_share_rise),
        "u11_bid_reprice_toward_share_rest": float(u11_bid_reprice_toward_share_rest),
        "u12_bid_add_pull_log_rest": float(u12_bid_add_pull_log_rest),
        "u13_bid_add_intensity_log": float(u13_bid_add_intensity_log),
        "u14_bid_far_pull_share_rest": float(u14_bid_far_pull_share_rest),
        "u15_up_expansion_log": float(u15_up_expansion_log),
        "u16_up_flow_log": float(u16_up_flow_log),
        "u17_up_total_log": float(u17_up_total_log),
        "u18_ask_bbo_dist_ticks": float(u18_ask_bbo_dist_ticks),
        "u19_bid_bbo_dist_ticks": float(u19_bid_bbo_dist_ticks),
    }


def _add_derivatives(rows: list, feature_names: list):
    if len(rows) < 2:
        return
    for name in feature_names:
        d1 = f"d1_{name}"
        d2 = f"d2_{name}"
        d3 = f"d3_{name}"
        rows[0][d1] = 0.0
        rows[0][d2] = 0.0
        rows[0][d3] = 0.0
        if len(rows) > 1:
            rows[1][d1] = rows[1][name] - rows[0][name]
            rows[1][d2] = 0.0
            rows[1][d3] = 0.0
        for i in range(2, len(rows)):
            rows[i][d1] = rows[i][name] - rows[i - 1][name]
            rows[i][d2] = rows[i][d1] - rows[i - 1][d1]
            rows[i][d3] = rows[i][d2] - rows[i - 1][d2]


def _compute_approach_dir_simple(spot_history: np.ndarray) -> list:
    res = np.full(len(spot_history), "approach_none", dtype=object)
    if len(spot_history) < 3:
        return res.tolist()
    price = spot_history.astype(float)
    lag = np.roll(price, 2)
    trend = price - lag
    trend[:2] = 0
    trends_cat = np.where(trend > 0, "approach_up", np.where(trend < 0, "approach_down", "approach_none"))
    return trends_cat.tolist()


def _validate_feature_ranges(df: pd.DataFrame):
    for col in BASE_FEATURES + UP_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing feature {col}")
    if df[BASE_FEATURES + UP_FEATURES].isna().any().any():
        raise ValueError("NaNs in radar vacuum features")
