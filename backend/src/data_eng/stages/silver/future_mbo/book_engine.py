from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Tuple

import pandas as pd

PRICE_SCALE = 1e-9
TICK_SIZE = 0.25
TICK_INT = int(round(TICK_SIZE / PRICE_SCALE))
WINDOW_NS = 1_000_000_000
REST_NS = 500_000_000
HUD_MAX_TICKS = 600

F_SNAPSHOT = 128
F_LAST = 256

ACTION_ADD = "A"
ACTION_CANCEL = "C"
ACTION_MODIFY = "M"
ACTION_CLEAR = "R"
ACTION_TRADE = "T"
ACTION_FILL = "F"
ACTION_NONE = "N"

SIDE_ASK = "A"
SIDE_BID = "B"

DELTA_TICKS = 20
AT_TICKS = 2
NEAR_TICKS = 5
FAR_TICKS_LOW = 15

EPS_QTY = 1.0
EPS_DIST_TICKS = 1.0
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

SNAP_COLUMNS = [
    "window_start_ts_ns",
    "window_end_ts_ns",
    "best_bid_price_int",
    "best_bid_qty",
    "best_ask_price_int",
    "best_ask_qty",
    "mid_price",
    "mid_price_int",
    "last_trade_price_int",
    "spot_ref_price_int",
    "book_valid",
]

WALL_COLUMNS = [
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

RADAR_COLUMNS = [
    "window_start_ts_ns",
    "window_end_ts_ns",
    "spot_ref_price",
    "spot_ref_price_int",
    "approach_dir",
] + BASE_FEATURES + [f"d1_{name}" for name in BASE_FEATURES] + [f"d2_{name}" for name in BASE_FEATURES] + [f"d3_{name}" for name in BASE_FEATURES] + UP_FEATURES + [f"d1_{name}" for name in UP_FEATURES] + [f"d2_{name}" for name in UP_FEATURES] + [f"d3_{name}" for name in UP_FEATURES]


@dataclass
class OrderState:
    side: str
    price_int: int
    qty: int
    ts_enter_price: int


class ApproachDirState:
    def __init__(self) -> None:
        self.spot_history: List[int] = []

    def next_dir(self, spot_int: int) -> str:
        self.spot_history.append(spot_int)
        if len(self.spot_history) <= 3:
            return "approach_none"
        trend = self.spot_history[-1] - self.spot_history[-4]
        if trend > 0:
            return "approach_up"
        if trend < 0:
            return "approach_down"
        return "approach_none"


class RadarDerivativeState:
    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names
        self.prev: Dict[str, float | None] = {name: None for name in feature_names}
        self.prev_d1: Dict[str, float] = {name: 0.0 for name in feature_names}
        self.prev_d2: Dict[str, float] = {name: 0.0 for name in feature_names}

    def apply(self, row: Dict[str, float]) -> None:
        for name in self.feature_names:
            prev_val = self.prev[name]
            if prev_val is None:
                d1 = 0.0
                d2 = 0.0
                d3 = 0.0
            else:
                d1 = row[name] - prev_val
                d2 = d1 - self.prev_d1[name]
                d3 = d2 - self.prev_d2[name]
            row[f"d1_{name}"] = float(d1)
            row[f"d2_{name}"] = float(d2)
            row[f"d3_{name}"] = float(d3)
            self.prev[name] = row[name]
            self.prev_d1[name] = d1
            self.prev_d2[name] = d2


class FuturesBookEngine:
    def __init__(
        self,
        tick_int: int = TICK_INT,
        window_ns: int = WINDOW_NS,
        rest_ns: int = REST_NS,
        hud_max_ticks: int = HUD_MAX_TICKS,
    ) -> None:
        self.tick_int = tick_int
        self.window_ns = window_ns
        self.rest_ns = rest_ns
        self.hud_max_ticks = hud_max_ticks

        self.orders: Dict[int, OrderState] = {}
        self.depth_bid: Dict[int, int] = {}
        self.depth_ask: Dict[int, int] = {}

        self.book_valid = False
        self.snapshot_in_progress = False
        self.last_trade_price_int = 0

        self.curr_window_id: int | None = None
        self.window_start_ts = 0
        self.window_end_ts = 0
        self.window_has_snapshot = False
        self.window_events: List[Tuple[int, str, str, int, int, int]] = []
        self.window_orders_start: Dict[int, OrderState] = {}
        self.window_depth_start_bid: Dict[int, int] = {}
        self.window_depth_start_ask: Dict[int, int] = {}

        self.acc_add_bid: Dict[int, float] = {}
        self.acc_add_ask: Dict[int, float] = {}
        self.acc_pull_bid: Dict[int, float] = {}
        self.acc_pull_ask: Dict[int, float] = {}
        self.acc_pull_rest_bid: Dict[int, float] = {}
        self.acc_pull_rest_ask: Dict[int, float] = {}
        self.acc_fill_bid: Dict[int, float] = {}
        self.acc_fill_ask: Dict[int, float] = {}

        self.prev_depth_bid: Dict[int, float] = {}
        self.prev_d1_bid: Dict[int, float] = {}
        self.prev_d2_bid: Dict[int, float] = {}
        self.prev_depth_ask: Dict[int, float] = {}
        self.prev_d1_ask: Dict[int, float] = {}
        self.prev_d2_ask: Dict[int, float] = {}

        self.radar_deriv_base = RadarDerivativeState(BASE_FEATURES)
        self.radar_deriv_up = RadarDerivativeState(UP_FEATURES)
        self.approach_state = ApproachDirState()

        self.snap_rows: List[Dict[str, object]] = []
        self.wall_rows: List[Dict[str, object]] = []
        self.radar_rows: List[Dict[str, object]] = []

    def process(self, events: Iterable[Tuple[int, str, str, int, int, int, int]]) -> None:
        for ts, action, side, price, size, order_id, flags in events:
            self.apply_event(ts, action, side, price, size, order_id, flags)
        self.flush_final()

    def apply_event(self, ts: int, action: str, side: str, price: int, size: int, order_id: int, flags: int) -> None:
        window_id = ts // self.window_ns
        if self.curr_window_id is None:
            self._start_window(window_id)
        elif window_id > self.curr_window_id:
            self._flush_until(window_id)

        if action != ACTION_NONE:
            self.window_events.append((ts, action, side, price, size, order_id))

        if action == ACTION_CLEAR:
            self._clear_book(flags)
            return

        if action == ACTION_TRADE:
            if price > 0:
                self.last_trade_price_int = price
            if self.snapshot_in_progress and (flags & F_LAST):
                self.snapshot_in_progress = False
                self.book_valid = True
            return

        if action == ACTION_ADD:
            self._add_order(ts, side, price, size, order_id)
        elif action == ACTION_CANCEL:
            self._cancel_order(ts, order_id)
        elif action == ACTION_MODIFY:
            self._modify_order(ts, order_id, price, size)
        elif action == ACTION_FILL:
            self._fill_order(order_id, size)

        if self.snapshot_in_progress and (flags & F_LAST):
            self.snapshot_in_progress = False
            self.book_valid = True

    def flush_final(self) -> None:
        if self.curr_window_id is None:
            return
        self._emit_window()

    def _start_window(self, window_id: int) -> None:
        self.curr_window_id = window_id
        self.window_start_ts = window_id * self.window_ns
        self.window_end_ts = self.window_start_ts + self.window_ns
        self.window_has_snapshot = self.snapshot_in_progress
        self.window_events = []
        self.window_orders_start = _clone_orders(self.orders)
        self.window_depth_start_bid = dict(self.depth_bid)
        self.window_depth_start_ask = dict(self.depth_ask)
        self._reset_accumulators()

    def _flush_until(self, target_window_id: int) -> None:
        while self.curr_window_id is not None and self.curr_window_id < target_window_id:
            self._emit_window()
            next_window = self.curr_window_id + 1
            self._start_window(next_window)

    def _emit_window(self) -> None:
        best_bid_price, best_bid_qty = _best_level(self.depth_bid, is_bid=True)
        best_ask_price, best_ask_qty = _best_level(self.depth_ask, is_bid=False)

        mid_price = 0.0
        mid_price_int = 0
        if best_bid_price > 0 and best_ask_price > 0:
            mid_price = (best_bid_price + best_ask_price) * 0.5 * PRICE_SCALE
            mid_price_int = int(round((best_bid_price + best_ask_price) * 0.5))

        spot_ref = self.last_trade_price_int
        if spot_ref == 0 and self.book_valid and best_bid_price > 0:
            spot_ref = best_bid_price

        self.snap_rows.append(
            {
                "window_start_ts_ns": self.window_start_ts,
                "window_end_ts_ns": self.window_end_ts,
                "best_bid_price_int": best_bid_price,
                "best_bid_qty": best_bid_qty,
                "best_ask_price_int": best_ask_price,
                "best_ask_qty": best_ask_qty,
                "mid_price": mid_price,
                "mid_price_int": mid_price_int,
                "last_trade_price_int": self.last_trade_price_int,
                "spot_ref_price_int": spot_ref,
                "book_valid": self.book_valid,
            }
        )

        window_valid = self.book_valid and (not self.window_has_snapshot)

        if spot_ref > 0:
            self._emit_wall_rows(spot_ref, window_valid)
            self._emit_radar_row(spot_ref)

        self._reset_accumulators()

    def _emit_wall_rows(self, spot_ref: int, window_valid: bool) -> None:
        min_price = spot_ref - self.hud_max_ticks * self.tick_int
        max_price = spot_ref + self.hud_max_ticks * self.tick_int

        bid_prices = _collect_prices(self.depth_bid, self.acc_add_bid, self.acc_pull_bid, self.acc_fill_bid, min_price, max_price)
        ask_prices = _collect_prices(self.depth_ask, self.acc_add_ask, self.acc_pull_ask, self.acc_fill_ask, min_price, max_price)

        rest_bid = _resting_depth(self.orders, SIDE_BID, min_price, max_price, self.window_end_ts, self.rest_ns)
        rest_ask = _resting_depth(self.orders, SIDE_ASK, min_price, max_price, self.window_end_ts, self.rest_ns)

        for price in bid_prices:
            depth_end = float(self.depth_bid.get(price, 0))
            add_qty = self.acc_add_bid.get(price, 0.0)
            pull_qty = self.acc_pull_bid.get(price, 0.0)
            pull_rest = self.acc_pull_rest_bid.get(price, 0.0)
            fill_qty = self.acc_fill_bid.get(price, 0.0)
            depth_start = depth_end - add_qty + pull_qty + fill_qty
            if depth_start < 0:
                depth_start = 0.0
            rel_ticks = int(round((price - spot_ref) / self.tick_int))
            d1, d2, d3 = _depth_derivatives(price, depth_end, self.prev_depth_bid, self.prev_d1_bid, self.prev_d2_bid)
            self.wall_rows.append(
                {
                    "window_start_ts_ns": self.window_start_ts,
                    "window_end_ts_ns": self.window_end_ts,
                    "price_int": price,
                    "side": SIDE_BID,
                    "spot_ref_price_int": spot_ref,
                    "rel_ticks": rel_ticks,
                    "depth_qty_start": float(depth_start),
                    "depth_qty_end": float(depth_end),
                    "add_qty": float(add_qty),
                    "pull_qty_total": float(pull_qty),
                    "depth_qty_rest": float(rest_bid.get(price, 0.0)),
                    "pull_qty_rest": float(pull_rest),
                    "fill_qty": float(fill_qty),
                    "d1_depth_qty": float(d1),
                    "d2_depth_qty": float(d2),
                    "d3_depth_qty": float(d3),
                    "window_valid": window_valid,
                }
            )

        for price in ask_prices:
            depth_end = float(self.depth_ask.get(price, 0))
            add_qty = self.acc_add_ask.get(price, 0.0)
            pull_qty = self.acc_pull_ask.get(price, 0.0)
            pull_rest = self.acc_pull_rest_ask.get(price, 0.0)
            fill_qty = self.acc_fill_ask.get(price, 0.0)
            depth_start = depth_end - add_qty + pull_qty + fill_qty
            if depth_start < 0:
                depth_start = 0.0
            rel_ticks = int(round((price - spot_ref) / self.tick_int))
            d1, d2, d3 = _depth_derivatives(price, depth_end, self.prev_depth_ask, self.prev_d1_ask, self.prev_d2_ask)
            self.wall_rows.append(
                {
                    "window_start_ts_ns": self.window_start_ts,
                    "window_end_ts_ns": self.window_end_ts,
                    "price_int": price,
                    "side": SIDE_ASK,
                    "spot_ref_price_int": spot_ref,
                    "rel_ticks": rel_ticks,
                    "depth_qty_start": float(depth_start),
                    "depth_qty_end": float(depth_end),
                    "add_qty": float(add_qty),
                    "pull_qty_total": float(pull_qty),
                    "depth_qty_rest": float(rest_ask.get(price, 0.0)),
                    "pull_qty_rest": float(pull_rest),
                    "fill_qty": float(fill_qty),
                    "d1_depth_qty": float(d1),
                    "d2_depth_qty": float(d2),
                    "d3_depth_qty": float(d3),
                    "window_valid": window_valid,
                }
            )

    def _emit_radar_row(self, spot_ref: int) -> None:
        start_snap = _snapshot_from_depth(self.window_depth_start_bid, self.window_depth_start_ask, spot_ref)
        end_snap = _snapshot_from_depth(self.depth_bid, self.depth_ask, spot_ref)
        accum = _radar_accumulators(self.window_orders_start, self.window_events, spot_ref, self.rest_ns)

        base_feats = _compute_base_features(start_snap, end_snap, accum)
        up_feats = _compute_up_features(base_feats, start_snap, accum)

        row: Dict[str, object] = {
            "window_start_ts_ns": self.window_start_ts,
            "window_end_ts_ns": self.window_end_ts,
            "spot_ref_price": spot_ref * PRICE_SCALE,
            "spot_ref_price_int": spot_ref,
        }
        row.update(base_feats)
        row.update(up_feats)

        self.radar_deriv_base.apply(row)
        self.radar_deriv_up.apply(row)
        row["approach_dir"] = self.approach_state.next_dir(spot_ref)

        self.radar_rows.append(row)

    def _reset_accumulators(self) -> None:
        self.acc_add_bid = {}
        self.acc_add_ask = {}
        self.acc_pull_bid = {}
        self.acc_pull_ask = {}
        self.acc_pull_rest_bid = {}
        self.acc_pull_rest_ask = {}
        self.acc_fill_bid = {}
        self.acc_fill_ask = {}

    def _clear_book(self, flags: int) -> None:
        self.orders.clear()
        self.depth_bid.clear()
        self.depth_ask.clear()
        self.snapshot_in_progress = True
        self.book_valid = False
        self.window_has_snapshot = True
        if flags & F_SNAPSHOT:
            self.snapshot_in_progress = True
        return

    def _add_order(self, ts: int, side: str, price: int, size: int, order_id: int) -> None:
        if side not in (SIDE_BID, SIDE_ASK):
            return
        self.orders[order_id] = OrderState(side, price, size, ts)
        if side == SIDE_BID:
            self.depth_bid[price] = self.depth_bid.get(price, 0) + size
            self.acc_add_bid[price] = self.acc_add_bid.get(price, 0.0) + float(size)
        else:
            self.depth_ask[price] = self.depth_ask.get(price, 0) + size
            self.acc_add_ask[price] = self.acc_add_ask.get(price, 0.0) + float(size)

    def _cancel_order(self, ts: int, order_id: int) -> None:
        order = self.orders.pop(order_id, None)
        if order is None:
            return
        side = order.side
        price = order.price_int
        qty = order.qty
        if side == SIDE_BID:
            self.depth_bid[price] = self.depth_bid.get(price, 0) - qty
            if self.depth_bid[price] <= 0:
                self.depth_bid.pop(price, None)
            self.acc_pull_bid[price] = self.acc_pull_bid.get(price, 0.0) + float(qty)
            if ts - order.ts_enter_price >= self.rest_ns:
                self.acc_pull_rest_bid[price] = self.acc_pull_rest_bid.get(price, 0.0) + float(qty)
        elif side == SIDE_ASK:
            self.depth_ask[price] = self.depth_ask.get(price, 0) - qty
            if self.depth_ask[price] <= 0:
                self.depth_ask.pop(price, None)
            self.acc_pull_ask[price] = self.acc_pull_ask.get(price, 0.0) + float(qty)
            if ts - order.ts_enter_price >= self.rest_ns:
                self.acc_pull_rest_ask[price] = self.acc_pull_rest_ask.get(price, 0.0) + float(qty)

    def _modify_order(self, ts: int, order_id: int, new_price: int, new_qty: int) -> None:
        order = self.orders.get(order_id)
        if order is None:
            return
        side = order.side
        old_price = order.price_int
        old_qty = order.qty
        old_ts_enter = order.ts_enter_price

        if side == SIDE_BID:
            self.depth_bid[old_price] = self.depth_bid.get(old_price, 0) - old_qty
            if self.depth_bid[old_price] <= 0:
                self.depth_bid.pop(old_price, None)
        else:
            self.depth_ask[old_price] = self.depth_ask.get(old_price, 0) - old_qty
            if self.depth_ask[old_price] <= 0:
                self.depth_ask.pop(old_price, None)

        if side == SIDE_BID:
            self.depth_bid[new_price] = self.depth_bid.get(new_price, 0) + new_qty
        else:
            self.depth_ask[new_price] = self.depth_ask.get(new_price, 0) + new_qty

        if new_price != old_price:
            if side == SIDE_BID:
                self.acc_pull_bid[old_price] = self.acc_pull_bid.get(old_price, 0.0) + float(old_qty)
                if ts - old_ts_enter >= self.rest_ns:
                    self.acc_pull_rest_bid[old_price] = self.acc_pull_rest_bid.get(old_price, 0.0) + float(old_qty)
                self.acc_add_bid[new_price] = self.acc_add_bid.get(new_price, 0.0) + float(new_qty)
            else:
                self.acc_pull_ask[old_price] = self.acc_pull_ask.get(old_price, 0.0) + float(old_qty)
                if ts - old_ts_enter >= self.rest_ns:
                    self.acc_pull_rest_ask[old_price] = self.acc_pull_rest_ask.get(old_price, 0.0) + float(old_qty)
                self.acc_add_ask[new_price] = self.acc_add_ask.get(new_price, 0.0) + float(new_qty)
            order.ts_enter_price = ts
        else:
            if new_qty < old_qty:
                delta = old_qty - new_qty
                if side == SIDE_BID:
                    self.acc_pull_bid[old_price] = self.acc_pull_bid.get(old_price, 0.0) + float(delta)
                    if ts - old_ts_enter >= self.rest_ns:
                        self.acc_pull_rest_bid[old_price] = self.acc_pull_rest_bid.get(old_price, 0.0) + float(delta)
                else:
                    self.acc_pull_ask[old_price] = self.acc_pull_ask.get(old_price, 0.0) + float(delta)
                    if ts - old_ts_enter >= self.rest_ns:
                        self.acc_pull_rest_ask[old_price] = self.acc_pull_rest_ask.get(old_price, 0.0) + float(delta)
            elif new_qty > old_qty:
                delta = new_qty - old_qty
                if side == SIDE_BID:
                    self.acc_add_bid[old_price] = self.acc_add_bid.get(old_price, 0.0) + float(delta)
                else:
                    self.acc_add_ask[old_price] = self.acc_add_ask.get(old_price, 0.0) + float(delta)

        order.price_int = new_price
        order.qty = new_qty

    def _fill_order(self, order_id: int, fill_qty: int) -> None:
        order = self.orders.get(order_id)
        if order is None:
            return
        side = order.side
        price = order.price_int
        if side == SIDE_BID:
            self.depth_bid[price] = self.depth_bid.get(price, 0) - fill_qty
            if self.depth_bid[price] <= 0:
                self.depth_bid.pop(price, None)
            self.acc_fill_bid[price] = self.acc_fill_bid.get(price, 0.0) + float(fill_qty)
        else:
            self.depth_ask[price] = self.depth_ask.get(price, 0) - fill_qty
            if self.depth_ask[price] <= 0:
                self.depth_ask.pop(price, None)
            self.acc_fill_ask[price] = self.acc_fill_ask.get(price, 0.0) + float(fill_qty)

        remaining = order.qty - fill_qty
        if remaining > 0:
            order.qty = remaining
        else:
            self.orders.pop(order_id, None)


def compute_futures_surfaces_1s_from_batches(
    batches: Iterable[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engine = FuturesBookEngine()
    seen = False

    for batch in batches:
        if batch.empty:
            continue
        seen = True
        required = {"ts_event", "action", "side", "price", "size", "order_id", "sequence", "flags"}
        missing = required.difference(batch.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        for row in batch.itertuples(index=False):
            engine.apply_event(
                int(row.ts_event),
                str(row.action),
                str(row.side),
                int(row.price),
                int(row.size),
                int(row.order_id),
                int(row.flags),
            )

    if not seen:
        return (
            pd.DataFrame(columns=SNAP_COLUMNS),
            pd.DataFrame(columns=WALL_COLUMNS),
            pd.DataFrame(columns=RADAR_COLUMNS),
        )

    engine.flush_final()

    df_snap = _rows_to_df(engine.snap_rows, SNAP_COLUMNS)
    df_wall = _rows_to_df(engine.wall_rows, WALL_COLUMNS)
    df_radar = _rows_to_df(engine.radar_rows, RADAR_COLUMNS)

    return df_snap, df_wall, df_radar


def compute_futures_surfaces_1s(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return compute_futures_surfaces_1s_from_batches([df])


def _rows_to_df(rows: List[Dict[str, object]], columns: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _clone_orders(orders: Dict[int, OrderState]) -> Dict[int, OrderState]:
    return {oid: OrderState(o.side, o.price_int, o.qty, o.ts_enter_price) for oid, o in orders.items()}


def _collect_prices(depth: Dict[int, int], add: Dict[int, float], pull: Dict[int, float], fill: Dict[int, float], min_price: int, max_price: int) -> List[int]:
    prices = set()
    for price in depth:
        if min_price <= price <= max_price:
            prices.add(price)
    for price in add:
        if min_price <= price <= max_price:
            prices.add(price)
    for price in pull:
        if min_price <= price <= max_price:
            prices.add(price)
    for price in fill:
        if min_price <= price <= max_price:
            prices.add(price)
    return sorted(prices)


def _best_level(depth: Dict[int, int], is_bid: bool) -> Tuple[int, int]:
    if not depth:
        return 0, 0
    if is_bid:
        price = max(depth.keys())
    else:
        price = min(depth.keys())
    return price, depth.get(price, 0)


def _resting_depth(orders: Dict[int, OrderState], side: str, min_price: int, max_price: int, window_end: int, rest_ns: int) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for order in orders.values():
        if order.side != side:
            continue
        if order.price_int < min_price or order.price_int > max_price:
            continue
        if window_end - order.ts_enter_price < rest_ns:
            continue
        result[order.price_int] = result.get(order.price_int, 0.0) + float(order.qty)
    return result


def _depth_derivatives(price: int, depth_end: float, prev_depth: Dict[int, float], prev_d1: Dict[int, float], prev_d2: Dict[int, float]) -> Tuple[float, float, float]:
    prev_val = prev_depth.get(price)
    if prev_val is None:
        d1 = 0.0
        d2 = 0.0
        d3 = 0.0
    else:
        d1 = depth_end - prev_val
        d2 = d1 - prev_d1.get(price, 0.0)
        d3 = d2 - prev_d2.get(price, 0.0)
    prev_depth[price] = depth_end
    prev_d1[price] = d1
    prev_d2[price] = d2
    return d1, d2, d3


def _bucket_for(side: str, price_int: int, spot_ref: int) -> str:
    if side == SIDE_ASK and price_int >= spot_ref:
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
    if side == SIDE_BID and price_int <= spot_ref:
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


def snapshot_from_orders(orders: Dict[int, OrderState], spot_ref: int) -> Dict[str, float]:
    depth_bid: Dict[int, int] = {}
    depth_ask: Dict[int, int] = {}
    for order in orders.values():
        if order.side == SIDE_BID:
            depth_bid[order.price_int] = depth_bid.get(order.price_int, 0) + order.qty
        elif order.side == SIDE_ASK:
            depth_ask[order.price_int] = depth_ask.get(order.price_int, 0) + order.qty
    return _snapshot_from_depth(depth_bid, depth_ask, spot_ref)


def _snapshot_from_depth(depth_bid: Dict[int, int], depth_ask: Dict[int, int], spot_ref: int) -> Dict[str, float]:
    data = {
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

    for price_int, qty in depth_ask.items():
        bucket = _bucket_for(SIDE_ASK, price_int, spot_ref)
        if bucket == BUCKET_OUT:
            continue
        qty_f = float(qty)
        data["ask_depth_total"] += qty_f
        ask_com_num += float(price_int) * qty_f
        if bucket == ASK_ABOVE_AT:
            data["ask_depth_at"] += qty_f
        elif bucket == ASK_ABOVE_NEAR:
            data["ask_depth_near"] += qty_f
        elif bucket == ASK_ABOVE_FAR:
            data["ask_depth_far"] += qty_f
        if min_ask is None or price_int < min_ask:
            min_ask = price_int

    for price_int, qty in depth_bid.items():
        bucket = _bucket_for(SIDE_BID, price_int, spot_ref)
        if bucket == BUCKET_OUT:
            continue
        qty_f = float(qty)
        data["bid_depth_total"] += qty_f
        bid_com_num += float(price_int) * qty_f
        if bucket == BID_BELOW_AT:
            data["bid_depth_at"] += qty_f
        elif bucket == BID_BELOW_NEAR:
            data["bid_depth_near"] += qty_f
        elif bucket == BID_BELOW_FAR:
            data["bid_depth_far"] += qty_f
        if max_bid is None or price_int > max_bid:
            max_bid = price_int

    if data["ask_depth_total"] < TINY_TOL:
        data["d_ask_ticks"] = float(DELTA_TICKS)
    else:
        com_price = ask_com_num / data["ask_depth_total"]
        data["d_ask_ticks"] = max((com_price - spot_ref) / TICK_INT, 0.0)

    if data["bid_depth_total"] < TINY_TOL:
        data["d_bid_ticks"] = float(DELTA_TICKS)
    else:
        com_price = bid_com_num / data["bid_depth_total"]
        data["d_bid_ticks"] = max((spot_ref - com_price) / TICK_INT, 0.0)

    if min_ask is None:
        data["bbo_ask_ticks"] = float(DELTA_TICKS)
    else:
        data["bbo_ask_ticks"] = max((min_ask - spot_ref) / TICK_INT, 0.0)

    if max_bid is None:
        data["bbo_bid_ticks"] = float(DELTA_TICKS)
    else:
        data["bbo_bid_ticks"] = max((spot_ref - max_bid) / TICK_INT, 0.0)

    return data


def _radar_accumulators(
    orders_start: Dict[int, OrderState],
    events: List[Tuple[int, str, str, int, int, int]],
    spot_ref: int,
    rest_ns: int,
) -> Dict[str, float]:
    orders = _clone_orders(orders_start)
    accum = _new_radar_accumulators()

    for ts, action, side, price, size, order_id in events:
        if action == ACTION_CLEAR:
            orders.clear()
            accum = _new_radar_accumulators()
            continue
        if action == ACTION_TRADE:
            continue
        if action == ACTION_NONE:
            continue

        old = orders.get(order_id)
        if old:
            old_side = old.side
            old_price = old.price_int
            old_qty = old.qty
            old_bucket = _bucket_for(old_side, old_price, spot_ref)
            old_ts_enter = old.ts_enter_price
        else:
            old_side = ""
            old_price = 0
            old_qty = 0
            old_bucket = BUCKET_OUT
            old_ts_enter = 0

        old_in_ask = old_side == SIDE_ASK and old_bucket in ASK_INFLUENCE
        old_in_bid = old_side == SIDE_BID and old_bucket in BID_INFLUENCE

        new_side = old_side
        new_price = old_price
        new_qty = old_qty
        new_bucket = old_bucket
        new_ts_enter = old_ts_enter
        has_new = False

        if action == ACTION_ADD:
            new_side = side
            new_price = price
            new_qty = size
            new_bucket = _bucket_for(new_side, new_price, spot_ref)
            new_ts_enter = ts
            has_new = True
            orders[order_id] = OrderState(new_side, new_price, new_qty, new_ts_enter)
        elif action == ACTION_CANCEL:
            orders.pop(order_id, None)
            has_new = False
        elif action == ACTION_MODIFY:
            new_price = price
            new_qty = size
            new_bucket = _bucket_for(old_side, new_price, spot_ref)
            if new_price != old_price:
                new_ts_enter = ts
            has_new = True
            orders[order_id] = OrderState(old_side, new_price, new_qty, new_ts_enter)
        elif action == ACTION_FILL:
            fill_qty = size
            new_qty = old_qty - fill_qty
            if new_qty > 0:
                orders[order_id].qty = new_qty
                has_new = True
            else:
                orders.pop(order_id, None)
                has_new = False

        new_in_ask = has_new and new_side == SIDE_ASK and new_bucket in ASK_INFLUENCE
        new_in_bid = has_new and new_side == SIDE_BID and new_bucket in BID_INFLUENCE

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

        if delta_ask < 0 and action in {ACTION_CANCEL, ACTION_MODIFY}:
            age = ts - old_ts_enter
            if age >= rest_ns:
                pull = float(-delta_ask)
                accum["ask_pull_rest_qty"] += pull
                if old_bucket == ASK_ABOVE_AT:
                    accum["ask_pull_rest_qty_at"] += pull
                elif old_bucket == ASK_ABOVE_NEAR:
                    accum["ask_pull_rest_qty_near"] += pull

        if delta_bid < 0 and action in {ACTION_CANCEL, ACTION_MODIFY}:
            age = ts - old_ts_enter
            if age >= rest_ns:
                pull = float(-delta_bid)
                accum["bid_pull_rest_qty"] += pull
                if old_bucket == BID_BELOW_AT:
                    accum["bid_pull_rest_qty_at"] += pull
                elif old_bucket == BID_BELOW_NEAR:
                    accum["bid_pull_rest_qty_near"] += pull

        if action == ACTION_MODIFY:
            if old_side == SIDE_ASK and old_in_ask:
                age = ts - old_ts_enter
                if age >= rest_ns:
                    dist_old = old_price - spot_ref
                    dist_new = new_price - spot_ref
                    if new_price <= spot_ref:
                        accum["ask_reprice_toward_rest_qty"] += float(old_qty)
                    else:
                        if dist_new > dist_old:
                            accum["ask_reprice_away_rest_qty"] += float(old_qty)
                        elif dist_new < dist_old:
                            accum["ask_reprice_toward_rest_qty"] += float(old_qty)
            if old_side == SIDE_BID and old_in_bid:
                age = ts - old_ts_enter
                if age >= rest_ns:
                    dist_old = spot_ref - old_price
                    dist_new = spot_ref - new_price
                    if new_price >= spot_ref:
                        accum["bid_reprice_toward_rest_qty"] += float(old_qty)
                    else:
                        if dist_new > dist_old:
                            accum["bid_reprice_away_rest_qty"] += float(old_qty)
                        elif dist_new < dist_old:
                            accum["bid_reprice_toward_rest_qty"] += float(old_qty)

    return accum


def _new_radar_accumulators() -> Dict[str, float]:
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
    f1_ask_com_disp_log = math.log((end["d_ask_ticks"] + EPS_DIST_TICKS) / (start["d_ask_ticks"] + EPS_DIST_TICKS))
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

    f3_bid_com_disp_log = math.log((end["d_bid_ticks"] + EPS_DIST_TICKS) / (start["d_bid_ticks"] + EPS_DIST_TICKS))
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
    u1_ask_com_disp_log = base["f1_ask_com_disp_log"]
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
    u9_bid_slope_support_log = -base["f3_bid_slope_convex_log"]
    u9_bid_slope_inner_log = -base["f3_bid_slope_inner_log"]
    u10_bid_at_share_rise = base["f3_bid_at_share_delta"]
    u10_bid_near_share_rise = base["f3_bid_near_share_delta"]
    u11_bid_reprice_toward_share_rest = 1.0 - base["f3_bid_reprice_away_share_rest"]
    u12_bid_add_pull_log_rest = -base["f4_bid_pull_add_log_rest"]
    u13_bid_add_intensity_log = math.log1p(
        acc["bid_add_qty"] / (start["bid_depth_total"] + EPS_QTY)
    )
    u14_bid_far_pull_share_rest = 1.0 - base["f4_bid_at_pull_share_rest"] - base["f4_bid_near_pull_share_rest"]

    u15_up_expansion_log = u1_ask_com_disp_log + u8_bid_com_approach_log
    u16_up_flow_log = u5_ask_pull_add_log_rest + u12_bid_add_pull_log_rest
    u17_up_total_log = u15_up_expansion_log + u16_up_flow_log

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
        "u18_ask_bbo_dist_ticks": float(base["f8_ask_bbo_dist_ticks"]),
        "u19_bid_bbo_dist_ticks": float(base["f9_bid_bbo_dist_ticks"]),
    }
