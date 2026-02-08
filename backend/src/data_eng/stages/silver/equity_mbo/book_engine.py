from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd

PRICE_SCALE = 1e-9
RAW_TICK_SIZE = 0.01
RAW_TICK_INT = int(round(RAW_TICK_SIZE / PRICE_SCALE))
BUCKET_SIZE = 0.50
BUCKET_INT = int(round(BUCKET_SIZE / PRICE_SCALE))
WINDOW_NS = 1_000_000_000
REST_NS = 500_000_000
GRID_MAX_BUCKETS = 100  # +/- $50 range (100 * 0.50)

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

EPS_QTY = 1.0

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

DEPTH_FLOW_COLUMNS = [
    "window_start_ts_ns",
    "window_end_ts_ns",
    "price_int",
    "side",
    "spot_ref_price_int",
    "rel_ticks",
    "rel_ticks_side",
    "depth_qty_start",
    "depth_qty_end",
    "add_qty",
    "pull_qty",
    "depth_qty_rest",
    "pull_qty_rest",
    "fill_qty",
    "window_valid",
]


@dataclass
class OrderState:
    side: str
    price_int: int
    qty: int
    ts_enter_price: int


class EquityBookEngine:
    def __init__(
        self,
        bucket_int: int = BUCKET_INT,
        window_ns: int = WINDOW_NS,
        rest_ns: int = REST_NS,
        grid_max_buckets: int = GRID_MAX_BUCKETS,
        compute_depth_flow: bool = True,
    ) -> None:
        self.bucket_int = bucket_int
        self.window_ns = window_ns
        self.rest_ns = rest_ns
        self.grid_max_buckets = grid_max_buckets
        self.compute_depth_flow = compute_depth_flow

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
        self.window_spot_ref_price_int = 0
        self.window_best_bid_start = 0
        self.window_best_ask_start = 0

        self.acc_add_bid: Dict[int, float] = {}
        self.acc_add_ask: Dict[int, float] = {}
        self.acc_pull_bid: Dict[int, float] = {}
        self.acc_pull_ask: Dict[int, float] = {}
        self.acc_pull_rest_bid: Dict[int, float] = {}
        self.acc_pull_rest_ask: Dict[int, float] = {}
        self.acc_fill_bid: Dict[int, float] = {}
        self.acc_fill_ask: Dict[int, float] = {}

        self.prev_depth_bid: Dict[int, float] = {}
        self.prev_depth_ask: Dict[int, float] = {}

        self.snap_rows: List[Dict[str, object]] = []
        self.depth_flow_rows: List[Dict[str, object]] = []

    def process(self, events: Iterable[Tuple[int, str, str, int, int, int, int]]) -> None:
        for ts, action, side, price, size, order_id, flags in events:
            self.apply_event(ts, action, side, price, size, order_id, flags)
        self.flush_final()

    def apply_event(
        self,
        ts: int,
        action: str,
        side: str,
        price: int,
        size: int,
        order_id: int,
        flags: int,
    ) -> None:
        is_snapshot_msg = (flags & F_SNAPSHOT) != 0

        # Explicit snapshot logic disabled to avoid persistent clears.
        if not self.book_valid and not self.snapshot_in_progress:
            self.book_valid = True

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
            return

        if action == ACTION_ADD:
            self._add_order(ts, side, price, size, order_id)
        elif action == ACTION_CANCEL:
            self._cancel_order(ts, order_id)
        elif action == ACTION_MODIFY:
            self._modify_order(ts, order_id, price, size)
        elif action == ACTION_FILL:
            self._fill_order(order_id, size)

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
        raw_spot = _resolve_spot_ref(
            last_trade=self.last_trade_price_int,
            depth_bid=self.depth_bid,
            depth_ask=self.depth_ask,
        )
        self.window_spot_ref_price_int = _round_to_bucket(raw_spot, self.bucket_int)
        self.window_best_bid_start, _ = _best_level(self.window_depth_start_bid, is_bid=True)
        self.window_best_ask_start, _ = _best_level(self.window_depth_start_ask, is_bid=False)
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

        spot_ref = self.window_spot_ref_price_int

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
            if self.compute_depth_flow:
                self._emit_depth_flow_rows(spot_ref, window_valid)

        self._reset_accumulators()

    def _emit_depth_flow_rows(self, spot_ref: int, window_valid: bool) -> None:
        min_bucket = spot_ref - self.grid_max_buckets * self.bucket_int
        max_bucket = spot_ref + self.grid_max_buckets * self.bucket_int

        bid_anchor = self.window_best_bid_start
        if bid_anchor <= 0:
            bid_anchor, _ = _best_level(self.depth_bid, is_bid=True)
        if bid_anchor <= 0:
            bid_anchor = spot_ref
        bid_anchor = _bucket_price(bid_anchor, spot_ref, self.bucket_int)

        ask_anchor = self.window_best_ask_start
        if ask_anchor <= 0:
            ask_anchor, _ = _best_level(self.depth_ask, is_bid=False)
        if ask_anchor <= 0:
            ask_anchor = spot_ref
        ask_anchor = _bucket_price(ask_anchor, spot_ref, self.bucket_int)

        bid_depth = _bucket_aggregate(self.depth_bid, spot_ref, self.bucket_int, min_bucket, max_bucket)
        ask_depth = _bucket_aggregate(self.depth_ask, spot_ref, self.bucket_int, min_bucket, max_bucket)
        bid_add = _bucket_aggregate(self.acc_add_bid, spot_ref, self.bucket_int, min_bucket, max_bucket)
        ask_add = _bucket_aggregate(self.acc_add_ask, spot_ref, self.bucket_int, min_bucket, max_bucket)
        bid_pull = _bucket_aggregate(self.acc_pull_bid, spot_ref, self.bucket_int, min_bucket, max_bucket)
        ask_pull = _bucket_aggregate(self.acc_pull_ask, spot_ref, self.bucket_int, min_bucket, max_bucket)
        bid_pull_rest = _bucket_aggregate(self.acc_pull_rest_bid, spot_ref, self.bucket_int, min_bucket, max_bucket)
        ask_pull_rest = _bucket_aggregate(self.acc_pull_rest_ask, spot_ref, self.bucket_int, min_bucket, max_bucket)
        bid_fill = _bucket_aggregate(self.acc_fill_bid, spot_ref, self.bucket_int, min_bucket, max_bucket)
        ask_fill = _bucket_aggregate(self.acc_fill_ask, spot_ref, self.bucket_int, min_bucket, max_bucket)

        rest_bid = _resting_depth_bucketed(
            self.orders, SIDE_BID, spot_ref, self.bucket_int, min_bucket, max_bucket, self.window_end_ts, self.rest_ns
        )
        rest_ask = _resting_depth_bucketed(
            self.orders, SIDE_ASK, spot_ref, self.bucket_int, min_bucket, max_bucket, self.window_end_ts, self.rest_ns
        )

        bid_prices = _collect_buckets(bid_depth, bid_add, bid_pull, bid_fill, rest_bid)
        ask_prices = _collect_buckets(ask_depth, ask_add, ask_pull, ask_fill, rest_ask)

        for price in bid_prices:
            depth_end = float(bid_depth.get(price, 0.0))
            add_qty = bid_add.get(price, 0.0)
            pull_qty = bid_pull.get(price, 0.0)
            pull_rest = bid_pull_rest.get(price, 0.0)
            fill_qty = bid_fill.get(price, 0.0)
            depth_start = depth_end - add_qty + pull_qty + fill_qty
            if depth_start < 0:
                depth_start = 0.0
            depth_rest_raw = float(rest_bid.get(price, 0.0))
            depth_rest_clamped = min(depth_rest_raw, depth_end)
            rel_ticks = int((price - spot_ref) // self.bucket_int)
            rel_ticks_side = int((price - bid_anchor) // self.bucket_int)
            self.prev_depth_bid[price] = depth_end

            self.depth_flow_rows.append(
                {
                    "window_start_ts_ns": self.window_start_ts,
                    "window_end_ts_ns": self.window_end_ts,
                    "price_int": price,
                    "side": SIDE_BID,
                    "spot_ref_price_int": spot_ref,
                    "rel_ticks": rel_ticks,
                    "rel_ticks_side": rel_ticks_side,
                    "depth_qty_start": float(depth_start),
                    "depth_qty_end": float(depth_end),
                    "add_qty": float(add_qty),
                    "pull_qty": float(pull_qty),
                    "depth_qty_rest": depth_rest_clamped,
                    "pull_qty_rest": float(pull_rest),
                    "fill_qty": float(fill_qty),
                    "window_valid": window_valid,
                }
            )

        for price in ask_prices:
            depth_end = float(ask_depth.get(price, 0.0))
            add_qty = ask_add.get(price, 0.0)
            pull_qty = ask_pull.get(price, 0.0)
            pull_rest = ask_pull_rest.get(price, 0.0)
            fill_qty = ask_fill.get(price, 0.0)
            depth_start = depth_end - add_qty + pull_qty + fill_qty
            if depth_start < 0:
                depth_start = 0.0
            depth_rest_raw = float(rest_ask.get(price, 0.0))
            depth_rest_clamped = min(depth_rest_raw, depth_end)
            rel_ticks = int((price - spot_ref) // self.bucket_int)
            rel_ticks_side = int((price - ask_anchor) // self.bucket_int)
            self.prev_depth_ask[price] = depth_end

            self.depth_flow_rows.append(
                {
                    "window_start_ts_ns": self.window_start_ts,
                    "window_end_ts_ns": self.window_end_ts,
                    "price_int": price,
                    "side": SIDE_ASK,
                    "spot_ref_price_int": spot_ref,
                    "rel_ticks": rel_ticks,
                    "rel_ticks_side": rel_ticks_side,
                    "depth_qty_start": float(depth_start),
                    "depth_qty_end": float(depth_end),
                    "add_qty": float(add_qty),
                    "pull_qty": float(pull_qty),
                    "depth_qty_rest": depth_rest_clamped,
                    "pull_qty_rest": float(pull_rest),
                    "fill_qty": float(fill_qty),
                    "window_valid": window_valid,
                }
            )

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


def compute_equity_surfaces_1s_from_batches(
    batches: Iterable[pd.DataFrame],
    compute_depth_flow: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute book snapshots and bucketed depth/flow from equity MBO event batches."""
    engine = EquityBookEngine(compute_depth_flow=compute_depth_flow)
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
            pd.DataFrame(columns=DEPTH_FLOW_COLUMNS) if compute_depth_flow else pd.DataFrame(),
            pd.DataFrame(),
        )

    engine.flush_final()

    df_snap = _rows_to_df(engine.snap_rows, SNAP_COLUMNS)
    df_depth_flow = _rows_to_df(engine.depth_flow_rows, DEPTH_FLOW_COLUMNS) if compute_depth_flow else pd.DataFrame()

    return df_snap, df_depth_flow, pd.DataFrame()


def compute_equity_surfaces_1s(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return compute_equity_surfaces_1s_from_batches([df])


def _rows_to_df(rows: List[Dict[str, object]], columns: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _clone_orders(orders: Dict[int, OrderState]) -> Dict[int, OrderState]:
    return {oid: OrderState(o.side, o.price_int, o.qty, o.ts_enter_price) for oid, o in orders.items()}


def _collect_buckets(*dicts: Dict[int, float]) -> List[int]:
    prices = set()
    for d in dicts:
        prices.update(d.keys())
    return sorted(prices)


def _best_level(depth: Dict[int, int], is_bid: bool) -> Tuple[int, int]:
    if not depth:
        return 0, 0
    price = max(depth.keys()) if is_bid else min(depth.keys())
    return price, depth.get(price, 0)


def _resolve_spot_ref(
    last_trade: int,
    depth_bid: Dict[int, int],
    depth_ask: Dict[int, int],
) -> int:
    if last_trade > 0 and (last_trade in depth_bid or last_trade in depth_ask):
        return last_trade

    best_bid, _ = _best_level(depth_bid, is_bid=True)
    best_ask, _ = _best_level(depth_ask, is_bid=False)

    if best_bid > 0 and best_ask > 0:
        if last_trade > 0:
            if abs(last_trade - best_bid) <= abs(best_ask - last_trade):
                return best_bid
            return best_ask
        return best_bid

    if best_bid > 0:
        return best_bid
    if best_ask > 0:
        return best_ask
    return last_trade if last_trade > 0 else 0


def _round_to_bucket(price_int: int, bucket_int: int) -> int:
    if price_int <= 0:
        return 0
    return int(round(price_int / bucket_int)) * bucket_int


def _bucket_price(price_int: int, anchor_int: int, bucket_int: int) -> int:
    if anchor_int <= 0:
        return _round_to_bucket(price_int, bucket_int)
    offset = int(round((price_int - anchor_int) / bucket_int))
    return anchor_int + offset * bucket_int


def _bucket_aggregate(
    values: Dict[int, float] | Dict[int, int],
    anchor_int: int,
    bucket_int: int,
    min_bucket: int,
    max_bucket: int,
) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for price, qty in values.items():
        bucket = _bucket_price(int(price), anchor_int, bucket_int)
        if bucket < min_bucket or bucket > max_bucket:
            continue
        result[bucket] = result.get(bucket, 0.0) + float(qty)
    return result


def _resting_depth_bucketed(
    orders: Dict[int, OrderState],
    side: str,
    anchor_int: int,
    bucket_int: int,
    min_bucket: int,
    max_bucket: int,
    window_end: int,
    rest_ns: int,
) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for order in orders.values():
        if order.side != side:
            continue
        if window_end - order.ts_enter_price < rest_ns:
            continue
        bucket = _bucket_price(order.price_int, anchor_int, bucket_int)
        if bucket < min_bucket or bucket > max_bucket:
            continue
        result[bucket] = result.get(bucket, 0.0) + float(order.qty)
    return result
