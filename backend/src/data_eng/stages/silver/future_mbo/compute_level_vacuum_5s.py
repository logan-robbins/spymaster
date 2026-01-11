from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
TICK_SIZE = 0.25
TICK_INT = int(round(TICK_SIZE / PRICE_SCALE))
PULL_ACTIONS = {"C", "M"}
DELTA_TICKS = 20
NEAR_TICKS = 5
FAR_TICKS_LOW = 15
FAR_TICKS_HIGH = 20
WINDOW_NS = 5_000_000_000
REST_NS = 500_000_000
EPS_QTY = 1.0
EPS_DIST_TICKS = 1.0
TRADE_ACTION = "T"

BUCKET_OUT = "OUT"
ASK_ABOVE_NEAR = "ASK_ABOVE_NEAR"
ASK_ABOVE_MID = "ASK_ABOVE_MID"
ASK_ABOVE_FAR = "ASK_ABOVE_FAR"
BID_BELOW_NEAR = "BID_BELOW_NEAR"
BID_BELOW_MID = "BID_BELOW_MID"
BID_BELOW_FAR = "BID_BELOW_FAR"

ASK_INFLUENCE = {ASK_ABOVE_NEAR, ASK_ABOVE_MID, ASK_ABOVE_FAR}
BID_INFLUENCE = {BID_BELOW_NEAR, BID_BELOW_MID, BID_BELOW_FAR}

BASE_FEATURES = [
    "f1_ask_com_disp_log",
    "f1_ask_slope_convex_log",
    "f1_ask_near_share_delta",
    "f1_ask_reprice_away_share_rest",
    "f2_ask_pull_add_log_rest",
    "f2_ask_pull_intensity_rest",
    "f2_ask_near_pull_share_rest",
    "f3_bid_com_disp_log",
    "f3_bid_slope_convex_log",
    "f3_bid_near_share_delta",
    "f3_bid_reprice_away_share_rest",
    "f4_bid_pull_add_log_rest",
    "f4_bid_pull_intensity_rest",
    "f4_bid_near_pull_share_rest",
    "f5_vacuum_expansion_log",
    "f6_vacuum_decay_log",
    "f7_vacuum_total_log",
]

UP_FEATURES = [
    "u1_ask_com_disp_log",
    "u2_ask_slope_convex_log",
    "u3_ask_near_share_decay",
    "u4_ask_reprice_away_share_rest",
    "u5_ask_pull_add_log_rest",
    "u6_ask_pull_intensity_rest",
    "u7_ask_near_pull_share_rest",
    "u8_bid_com_approach_log",
    "u9_bid_slope_support_log",
    "u10_bid_near_share_rise",
    "u11_bid_reprice_toward_share_rest",
    "u12_bid_add_pull_log_rest",
    "u13_bid_add_intensity",
    "u14_bid_far_pull_share_rest",
    "u15_up_expansion_log",
    "u16_up_flow_log",
    "u17_up_total_log",
]

DERIV_FEATURES = [f"d1_{name}" for name in BASE_FEATURES] + [f"d2_{name}" for name in BASE_FEATURES] + [
    f"d3_{name}" for name in BASE_FEATURES
]

UP_DERIV_FEATURES = [f"d1_{name}" for name in UP_FEATURES] + [f"d2_{name}" for name in UP_FEATURES] + [
    f"d3_{name}" for name in UP_FEATURES
]

OUTPUT_COLUMNS = [
    "window_start_ts_ns",
    "window_end_ts_ns",
    "P_ref",
    "P_REF_INT",
    "approach_dir",
] + BASE_FEATURES + DERIV_FEATURES + UP_FEATURES + UP_DERIV_FEATURES


@dataclass
class OrderState:
    side: str
    price_int: int
    qty: int
    bucket: str
    bucket_enter_ts: int


class SilverComputeMboLevelVacuum5s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_mbo_level_vacuum_5s",
            io=StageIO(
                inputs=["bronze.future_mbo.mbo"],
                output="silver.future_mbo.mbo_level_vacuum_5s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)

        if is_partition_complete(out_ref):
            try:
                existing = read_partition(out_ref)
                enforce_contract(existing, out_contract)
                return
            except Exception:
                pass

        input_key = self.io.inputs[0]
        in_ref = partition_ref(cfg, input_key, symbol, dt)
        if not is_partition_complete(in_ref):
            raise FileNotFoundError(f"Input not ready: {input_key} dt={dt}")

        in_contract_path = repo_root / cfg.dataset(input_key).contract
        in_contract = load_avro_contract(in_contract_path)
        df_in = read_partition(in_ref)
        df_in = enforce_contract(df_in, in_contract)

        df_out = self.transform(df_in, dt)
        df_out = enforce_contract(df_out, out_contract)

        lineage: List[Dict[str, Any]] = [
            {
                "dataset": in_ref.dataset_key,
                "dt": dt,
                "manifest_sha256": read_manifest_hash(in_ref),
            }
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

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        p_ref = _load_p_ref()
        symbol_target = df["symbol"].iloc[0]
        return compute_mbo_level_vacuum_5s(df, p_ref, symbol_target)


def compute_mbo_level_vacuum_5s(df: pd.DataFrame, p_ref: float, symbol_target: str) -> pd.DataFrame:
    required_cols = {"ts_event", "action", "side", "price", "size", "order_id", "sequence", "symbol"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.loc[df["symbol"] == symbol_target].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = df.loc[df["action"] != "N"].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = df.sort_values(["ts_event", "sequence"], ascending=[True, True])
    df["ts_event"] = df["ts_event"].astype("int64")
    df["sequence"] = df["sequence"].astype("int64")
    df["size"] = df["size"].astype("int64")
    df["order_id"] = df["order_id"].astype("int64")
    df["price"] = df["price"].astype("int64")

    p_ref_int = int(round(p_ref / PRICE_SCALE))

    orders: dict[int, OrderState] = {}
    accum = _new_accumulators()
    rows = []
    current_window_id = None
    window_invalid = False
    window_start_snapshot = None
    window_start_ts_ns = None
    window_end_ts_ns = None

    for row in df.itertuples(index=False):
        ts_event = int(row.ts_event)
        window_id = ts_event // WINDOW_NS

        if current_window_id is None:
            current_window_id = window_id
            window_invalid = False
            window_start_snapshot = _snapshot(orders, p_ref_int)
            window_start_ts_ns = current_window_id * WINDOW_NS
            window_end_ts_ns = window_start_ts_ns + WINDOW_NS
        elif window_id != current_window_id:
            if not window_invalid:
                end_snapshot = _snapshot(orders, p_ref_int)
                base_features = _compute_base_features(window_start_snapshot, end_snapshot, accum)
                up_features = _compute_up_features(base_features, window_start_snapshot, accum)
                rows.append(
                    {
                        "window_start_ts_ns": int(window_start_ts_ns),
                        "window_end_ts_ns": int(window_end_ts_ns),
                        "P_ref": float(p_ref),
                        "P_REF_INT": int(p_ref_int),
                        **base_features,
                        **up_features,
                    }
                )
            current_window_id = window_id
            window_invalid = False
            window_start_snapshot = _snapshot(orders, p_ref_int)
            accum = _new_accumulators()
            window_start_ts_ns = current_window_id * WINDOW_NS
            window_end_ts_ns = window_start_ts_ns + WINDOW_NS

        action = row.action
        if action == "R":
            orders.clear()
            accum = _new_accumulators()
            window_invalid = True
            window_start_snapshot = _snapshot(orders, p_ref_int)
            continue

        order_id = int(row.order_id)
        old = orders.get(order_id)
        if old is None and action in {"C", "M", "F", "T"}:
            continue

        if old is not None:
            old_side = old.side
            old_price_int = old.price_int
            old_qty = old.qty
            old_bucket = old.bucket
            old_bucket_enter_ts = old.bucket_enter_ts
        else:
            old_side = ""
            old_price_int = 0
            old_qty = 0
            old_bucket = BUCKET_OUT
            old_bucket_enter_ts = 0

        old_in_ask = old_side == "A" and old_bucket in ASK_INFLUENCE
        old_in_bid = old_side == "B" and old_bucket in BID_INFLUENCE

        new_order = None
        new_side = old_side
        new_price_int = old_price_int
        new_qty = old_qty
        new_bucket = old_bucket
        new_bucket_enter_ts = old_bucket_enter_ts

        if action == "A":
            new_side = row.side
            new_price_int = int(row.price)
            new_qty = int(row.size)
            new_bucket = _bucket_for(new_side, new_price_int, p_ref_int)
            new_bucket_enter_ts = ts_event
            new_order = OrderState(
                side=new_side,
                price_int=new_price_int,
                qty=new_qty,
                bucket=new_bucket,
                bucket_enter_ts=new_bucket_enter_ts,
            )
        elif action == "C":
            new_order = None
        elif action == "M":
            new_price_int = int(row.price)
            new_qty = int(row.size)
            new_bucket = _bucket_for(old_side, new_price_int, p_ref_int)
            if new_bucket != old_bucket:
                new_bucket_enter_ts = ts_event
            new_order = OrderState(
                side=old_side,
                price_int=new_price_int,
                qty=new_qty,
                bucket=new_bucket,
                bucket_enter_ts=new_bucket_enter_ts,
            )
        elif action == "F":
            new_qty = old_qty - int(row.size)
            if new_qty > 0:
                new_order = OrderState(
                    side=old_side,
                    price_int=old_price_int,
                    qty=new_qty,
                    bucket=old_bucket,
                    bucket_enter_ts=old_bucket_enter_ts,
                )
        elif action == "T":
            new_order = old
        else:
            raise ValueError(f"Unsupported action: {action}")

        new_in_ask = new_order is not None and new_order.side == "A" and new_order.bucket in ASK_INFLUENCE
        new_in_bid = new_order is not None and new_order.side == "B" and new_order.bucket in BID_INFLUENCE

        q_old_ask_zone = old_qty if old_in_ask else 0
        q_new_ask_zone = new_order.qty if new_in_ask else 0
        q_old_bid_zone = old_qty if old_in_bid else 0
        q_new_bid_zone = new_order.qty if new_in_bid else 0

        delta_ask = q_new_ask_zone - q_old_ask_zone
        delta_bid = q_new_bid_zone - q_old_bid_zone

        if delta_ask > 0:
            accum["ask_add_qty"] += float(delta_ask)
        if delta_bid > 0:
            accum["bid_add_qty"] += float(delta_bid)

        if delta_ask < 0 and action in PULL_ACTIONS:
            age_ns = ts_event - old_bucket_enter_ts
            if age_ns >= REST_NS:
                pull = float(-delta_ask)
                accum["ask_pull_rest_qty"] += pull
                if old_bucket == ASK_ABOVE_NEAR:
                    accum["ask_pull_rest_qty_near"] += pull

        if delta_bid < 0 and action in PULL_ACTIONS:
            age_ns = ts_event - old_bucket_enter_ts
            if age_ns >= REST_NS:
                pull = float(-delta_bid)
                accum["bid_pull_rest_qty"] += pull
                if old_bucket == BID_BELOW_NEAR:
                    accum["bid_pull_rest_qty_near"] += pull

        if action == "M" and old_side == "A" and old_bucket in ASK_INFLUENCE:
            age_ns = ts_event - old_bucket_enter_ts
            if age_ns >= REST_NS:
                dist_old = (old_price_int - p_ref_int) / TICK_INT
                if new_price_int <= p_ref_int:
                    accum["ask_reprice_toward_rest_qty"] += float(old_qty)
                else:
                    dist_new = (new_price_int - p_ref_int) / TICK_INT
                    if dist_new > dist_old:
                        accum["ask_reprice_away_rest_qty"] += float(old_qty)
                    elif dist_new < dist_old:
                        accum["ask_reprice_toward_rest_qty"] += float(old_qty)

        if action == "M" and old_side == "B" and old_bucket in BID_INFLUENCE:
            age_ns = ts_event - old_bucket_enter_ts
            if age_ns >= REST_NS:
                dist_old = (p_ref_int - old_price_int) / TICK_INT
                if new_price_int >= p_ref_int:
                    accum["bid_reprice_toward_rest_qty"] += float(old_qty)
                else:
                    dist_new = (p_ref_int - new_price_int) / TICK_INT
                    if dist_new > dist_old:
                        accum["bid_reprice_away_rest_qty"] += float(old_qty)
                    elif dist_new < dist_old:
                        accum["bid_reprice_toward_rest_qty"] += float(old_qty)

        if action == "A":
            orders[order_id] = new_order
        elif action == "C":
            orders.pop(order_id, None)
        elif action == "M":
            orders[order_id] = new_order
        elif action == "F":
            if new_order is None:
                orders.pop(order_id, None)
            else:
                orders[order_id] = new_order
        elif action == "T":
            continue

    if current_window_id is not None and not window_invalid:
        end_snapshot = _snapshot(orders, p_ref_int)
        base_features = _compute_base_features(window_start_snapshot, end_snapshot, accum)
        up_features = _compute_up_features(base_features, window_start_snapshot, accum)
        rows.append(
            {
                "window_start_ts_ns": int(window_start_ts_ns),
                "window_end_ts_ns": int(window_end_ts_ns),
                "P_ref": float(p_ref),
                "P_REF_INT": int(p_ref_int),
                **base_features,
                **up_features,
            }
        )

    if len(rows) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    _add_derivatives(rows, BASE_FEATURES)
    _add_derivatives(rows, UP_FEATURES)
    df_out = pd.DataFrame(rows)
    trade_ts, trade_px = _extract_trade_stream(df)
    px_end_int = _compute_px_end_int(
        trade_ts,
        trade_px,
        df_out["window_end_ts_ns"].to_numpy(dtype=np.int64),
    )
    df_out["approach_dir"] = _compute_approach_dir(px_end_int, p_ref_int)
    return df_out.loc[:, OUTPUT_COLUMNS]


def _bucket_for(side: str, price_int: int, p_ref_int: int) -> str:
    if side == "A" and price_int > p_ref_int:
        ticks = int(round((price_int - p_ref_int) / TICK_INT))
        if ticks < 1 or ticks > DELTA_TICKS:
            return BUCKET_OUT
        if ticks <= NEAR_TICKS:
            return ASK_ABOVE_NEAR
        if ticks >= FAR_TICKS_LOW:
            return ASK_ABOVE_FAR
        return ASK_ABOVE_MID

    if side == "B" and price_int < p_ref_int:
        ticks = int(round((p_ref_int - price_int) / TICK_INT))
        if ticks < 1 or ticks > DELTA_TICKS:
            return BUCKET_OUT
        if ticks <= NEAR_TICKS:
            return BID_BELOW_NEAR
        if ticks >= FAR_TICKS_LOW:
            return BID_BELOW_FAR
        return BID_BELOW_MID

    return BUCKET_OUT


def _snapshot(orders: dict[int, OrderState], p_ref_int: int) -> dict:
    ask_depth_total = 0.0
    ask_depth_near = 0.0
    ask_depth_far = 0.0
    ask_com_num = 0.0
    bid_depth_total = 0.0
    bid_depth_near = 0.0
    bid_depth_far = 0.0
    bid_com_num = 0.0

    for order in orders.values():
        if order.side == "A" and order.bucket in ASK_INFLUENCE:
            qty = float(order.qty)
            ask_depth_total += qty
            ask_com_num += float(order.price_int) * qty
            if order.bucket == ASK_ABOVE_NEAR:
                ask_depth_near += qty
            elif order.bucket == ASK_ABOVE_FAR:
                ask_depth_far += qty
        elif order.side == "B" and order.bucket in BID_INFLUENCE:
            qty = float(order.qty)
            bid_depth_total += qty
            bid_com_num += float(order.price_int) * qty
            if order.bucket == BID_BELOW_NEAR:
                bid_depth_near += qty
            elif order.bucket == BID_BELOW_FAR:
                bid_depth_far += qty

    ask_com_price_int = ask_com_num / max(ask_depth_total, EPS_QTY)
    bid_com_price_int = bid_com_num / max(bid_depth_total, EPS_QTY)

    d_ask_ticks = max((ask_com_price_int - p_ref_int) / TICK_INT, 0.0)
    d_bid_ticks = max((p_ref_int - bid_com_price_int) / TICK_INT, 0.0)

    return {
        "ask_depth_total": ask_depth_total,
        "ask_depth_near": ask_depth_near,
        "ask_depth_far": ask_depth_far,
        "ask_com_price_int": ask_com_price_int,
        "bid_depth_total": bid_depth_total,
        "bid_depth_near": bid_depth_near,
        "bid_depth_far": bid_depth_far,
        "bid_com_price_int": bid_com_price_int,
        "d_ask_ticks": d_ask_ticks,
        "d_bid_ticks": d_bid_ticks,
    }


def _compute_base_features(start: dict, end: dict, acc: dict) -> dict:
    f1_ask_com_disp_log = math.log((end["d_ask_ticks"] + EPS_DIST_TICKS) / (start["d_ask_ticks"] + EPS_DIST_TICKS))
    f1_ask_slope_convex_log = math.log((end["ask_depth_far"] + EPS_QTY) / (end["ask_depth_near"] + EPS_QTY))
    near_share_start = start["ask_depth_near"] / (start["ask_depth_total"] + EPS_QTY)
    near_share_end = end["ask_depth_near"] / (end["ask_depth_total"] + EPS_QTY)
    f1_ask_near_share_delta = near_share_end - near_share_start

    den_ask_reprice = acc["ask_reprice_away_rest_qty"] + acc["ask_reprice_toward_rest_qty"]
    if den_ask_reprice == 0:
        f1_ask_reprice_away_share_rest = 0.5
    else:
        f1_ask_reprice_away_share_rest = acc["ask_reprice_away_rest_qty"] / (den_ask_reprice + EPS_QTY)

    f2_ask_pull_add_log_rest = math.log((acc["ask_pull_rest_qty"] + EPS_QTY) / (acc["ask_add_qty"] + EPS_QTY))
    f2_ask_pull_intensity_rest = acc["ask_pull_rest_qty"] / (start["ask_depth_total"] + EPS_QTY)
    f2_ask_near_pull_share_rest = acc["ask_pull_rest_qty_near"] / (acc["ask_pull_rest_qty"] + EPS_QTY)

    f3_bid_com_disp_log = math.log((end["d_bid_ticks"] + EPS_DIST_TICKS) / (start["d_bid_ticks"] + EPS_DIST_TICKS))
    f3_bid_slope_convex_log = math.log((end["bid_depth_far"] + EPS_QTY) / (end["bid_depth_near"] + EPS_QTY))
    near_share_start = start["bid_depth_near"] / (start["bid_depth_total"] + EPS_QTY)
    near_share_end = end["bid_depth_near"] / (end["bid_depth_total"] + EPS_QTY)
    f3_bid_near_share_delta = near_share_end - near_share_start

    den_bid_reprice = acc["bid_reprice_away_rest_qty"] + acc["bid_reprice_toward_rest_qty"]
    if den_bid_reprice == 0:
        f3_bid_reprice_away_share_rest = 0.5
    else:
        f3_bid_reprice_away_share_rest = acc["bid_reprice_away_rest_qty"] / (den_bid_reprice + EPS_QTY)

    f4_bid_pull_add_log_rest = math.log((acc["bid_pull_rest_qty"] + EPS_QTY) / (acc["bid_add_qty"] + EPS_QTY))
    f4_bid_pull_intensity_rest = acc["bid_pull_rest_qty"] / (start["bid_depth_total"] + EPS_QTY)
    f4_bid_near_pull_share_rest = acc["bid_pull_rest_qty_near"] / (acc["bid_pull_rest_qty"] + EPS_QTY)

    f5_vacuum_expansion_log = f1_ask_com_disp_log + f3_bid_com_disp_log
    f6_vacuum_decay_log = f2_ask_pull_add_log_rest + f4_bid_pull_add_log_rest
    f7_vacuum_total_log = f5_vacuum_expansion_log + f6_vacuum_decay_log

    return {
        "f1_ask_com_disp_log": float(f1_ask_com_disp_log),
        "f1_ask_slope_convex_log": float(f1_ask_slope_convex_log),
        "f1_ask_near_share_delta": float(f1_ask_near_share_delta),
        "f1_ask_reprice_away_share_rest": float(f1_ask_reprice_away_share_rest),
        "f2_ask_pull_add_log_rest": float(f2_ask_pull_add_log_rest),
        "f2_ask_pull_intensity_rest": float(f2_ask_pull_intensity_rest),
        "f2_ask_near_pull_share_rest": float(f2_ask_near_pull_share_rest),
        "f3_bid_com_disp_log": float(f3_bid_com_disp_log),
        "f3_bid_slope_convex_log": float(f3_bid_slope_convex_log),
        "f3_bid_near_share_delta": float(f3_bid_near_share_delta),
        "f3_bid_reprice_away_share_rest": float(f3_bid_reprice_away_share_rest),
        "f4_bid_pull_add_log_rest": float(f4_bid_pull_add_log_rest),
        "f4_bid_pull_intensity_rest": float(f4_bid_pull_intensity_rest),
        "f4_bid_near_pull_share_rest": float(f4_bid_near_pull_share_rest),
        "f5_vacuum_expansion_log": float(f5_vacuum_expansion_log),
        "f6_vacuum_decay_log": float(f6_vacuum_decay_log),
        "f7_vacuum_total_log": float(f7_vacuum_total_log),
    }


def _compute_up_features(base: dict, start: dict, acc: dict) -> dict:
    u1_ask_com_disp_log = base["f1_ask_com_disp_log"]
    u2_ask_slope_convex_log = base["f1_ask_slope_convex_log"]
    u3_ask_near_share_decay = -base["f1_ask_near_share_delta"]
    u4_ask_reprice_away_share_rest = base["f1_ask_reprice_away_share_rest"]
    u5_ask_pull_add_log_rest = base["f2_ask_pull_add_log_rest"]
    u6_ask_pull_intensity_rest = base["f2_ask_pull_intensity_rest"]
    u7_ask_near_pull_share_rest = base["f2_ask_near_pull_share_rest"]

    u8_bid_com_approach_log = -base["f3_bid_com_disp_log"]
    u9_bid_slope_support_log = -base["f3_bid_slope_convex_log"]
    u10_bid_near_share_rise = base["f3_bid_near_share_delta"]
    u11_bid_reprice_toward_share_rest = 1.0 - base["f3_bid_reprice_away_share_rest"]
    u12_bid_add_pull_log_rest = -base["f4_bid_pull_add_log_rest"]
    u13_bid_add_intensity = acc["bid_add_qty"] / (start["bid_depth_total"] + EPS_QTY)
    u14_bid_far_pull_share_rest = 1.0 - base["f4_bid_near_pull_share_rest"]

    u15_up_expansion_log = u1_ask_com_disp_log + u8_bid_com_approach_log
    u16_up_flow_log = u5_ask_pull_add_log_rest + u12_bid_add_pull_log_rest
    u17_up_total_log = u15_up_expansion_log + u16_up_flow_log

    return {
        "u1_ask_com_disp_log": float(u1_ask_com_disp_log),
        "u2_ask_slope_convex_log": float(u2_ask_slope_convex_log),
        "u3_ask_near_share_decay": float(u3_ask_near_share_decay),
        "u4_ask_reprice_away_share_rest": float(u4_ask_reprice_away_share_rest),
        "u5_ask_pull_add_log_rest": float(u5_ask_pull_add_log_rest),
        "u6_ask_pull_intensity_rest": float(u6_ask_pull_intensity_rest),
        "u7_ask_near_pull_share_rest": float(u7_ask_near_pull_share_rest),
        "u8_bid_com_approach_log": float(u8_bid_com_approach_log),
        "u9_bid_slope_support_log": float(u9_bid_slope_support_log),
        "u10_bid_near_share_rise": float(u10_bid_near_share_rise),
        "u11_bid_reprice_toward_share_rest": float(u11_bid_reprice_toward_share_rest),
        "u12_bid_add_pull_log_rest": float(u12_bid_add_pull_log_rest),
        "u13_bid_add_intensity": float(u13_bid_add_intensity),
        "u14_bid_far_pull_share_rest": float(u14_bid_far_pull_share_rest),
        "u15_up_expansion_log": float(u15_up_expansion_log),
        "u16_up_flow_log": float(u16_up_flow_log),
        "u17_up_total_log": float(u17_up_total_log),
    }


def _add_derivatives(rows: list[dict], feature_names: list[str]) -> list[dict]:
    prev = {name: None for name in feature_names}
    prev_d1 = {name: 0.0 for name in feature_names}
    prev_d2 = {name: 0.0 for name in feature_names}

    for row in rows:
        for name in feature_names:
            if prev[name] is None:
                d1 = 0.0
                d2 = 0.0
                d3 = 0.0
            else:
                d1 = row[name] - prev[name]
                d2 = d1 - prev_d1[name]
                d3 = d2 - prev_d2[name]
            row[f"d1_{name}"] = float(d1)
            row[f"d2_{name}"] = float(d2)
            row[f"d3_{name}"] = float(d3)
            prev[name] = row[name]
            prev_d1[name] = d1
            prev_d2[name] = d2

    return rows


def _new_accumulators() -> dict:
    return {
        "ask_add_qty": 0.0,
        "ask_pull_rest_qty": 0.0,
        "ask_pull_rest_qty_near": 0.0,
        "ask_reprice_away_rest_qty": 0.0,
        "ask_reprice_toward_rest_qty": 0.0,
        "bid_add_qty": 0.0,
        "bid_pull_rest_qty": 0.0,
        "bid_pull_rest_qty_near": 0.0,
        "bid_reprice_away_rest_qty": 0.0,
        "bid_reprice_toward_rest_qty": 0.0,
    }


def _extract_trade_stream(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    trades = df.loc[df["action"] == TRADE_ACTION].copy()
    if len(trades) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    trades = trades.sort_values(["ts_event", "sequence"], ascending=[True, True])
    ts_event = trades["ts_event"].to_numpy(dtype=np.int64)
    price = trades["price"].to_numpy(dtype=np.int64)
    return ts_event, price


def _compute_px_end_int(
    trade_ts: np.ndarray,
    trade_px: np.ndarray,
    window_end_ts: np.ndarray,
) -> np.ndarray:
    px_end = np.full(window_end_ts.shape[0], np.nan, dtype=np.float64)
    if trade_ts.size == 0:
        return px_end
    idx = np.searchsorted(trade_ts, window_end_ts, side="right") - 1
    valid = idx >= 0
    px_end[valid] = trade_px[idx[valid]].astype(np.float64)
    return px_end


def _compute_approach_dir(px_end_int: np.ndarray, p_ref_int: int) -> np.ndarray:
    dist_ticks = (px_end_int - float(p_ref_int)) / float(TICK_INT)
    trend = np.full(dist_ticks.shape, np.nan, dtype=np.float64)
    if dist_ticks.shape[0] > 3:
        trend_vals = px_end_int[3:] - px_end_int[:-3]
        trend[3:] = trend_vals

    approach = np.full(dist_ticks.shape, "approach_none", dtype=object)
    valid = np.isfinite(dist_ticks) & np.isfinite(trend)
    if not np.any(valid):
        return approach

    dist_ok = np.abs(dist_ticks) <= 20
    up_mask = valid & dist_ok & (dist_ticks < 0) & (trend > 0)
    down_mask = valid & dist_ok & (dist_ticks > 0) & (trend < 0)
    approach[up_mask] = "approach_up"
    approach[down_mask] = "approach_down"
    return approach


def _load_p_ref() -> float:
    value = os.environ.get("P_REF")
    if value is None:
        raise ValueError("Missing P_REF env var")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid P_REF: {value}") from exc
