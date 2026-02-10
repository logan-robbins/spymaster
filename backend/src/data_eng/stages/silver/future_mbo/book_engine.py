from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from ....config import PRICE_SCALE, ProductConfig

TICK_SIZE = 0.25
TICK_INT = int(round(TICK_SIZE / PRICE_SCALE))
WINDOW_NS = 1_000_000_000
REST_NS = 500_000_000
GRID_MAX_TICKS = 200  # +/- $50 range (200 * 0.25)
HUD_MAX_TICKS = GRID_MAX_TICKS  # Alias for backward compatibility

F_LAST = 128       # bit 7 (0x80): last record in event for instrument_id
F_SNAPSHOT = 32    # bit 5 (0x20): sourced from replay/snapshot server

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


class FuturesBookEngine:
    def __init__(
        self,
        tick_int: int = TICK_INT,
        window_ns: int = WINDOW_NS,
        rest_ns: int = REST_NS,
        grid_max_ticks: int = GRID_MAX_TICKS,
        compute_depth_flow: bool = True,
    ) -> None:
        self.tick_int = tick_int
        self.window_ns = window_ns
        self.rest_ns = rest_ns
        self.grid_max_ticks = grid_max_ticks
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

    def apply_event(self, ts: int, action: str, side: str, price: int, size: int, order_id: int, flags: int) -> None:
        # Databento MBO flags (u8 bitmask):
        #   F_LAST    = 128 (0x80, bit 7) — last record in event for instrument_id
        #   F_SNAPSHOT =  32 (0x20, bit 5) — sourced from replay/snapshot server
        #
        # Implicit snapshot detection (auto-clear on first F_SNAPSHOT) is disabled.
        # We rely on ACTION_CLEAR ('R') to enter snapshot mode via _clear_book().
        # For GLBX futures, F_SNAPSHOT=32 appears only on actual snapshot records
        # (rare, e.g. 00:00 UTC). Keeping ACTION_CLEAR as the sole trigger avoids
        # false positives.

        # Exit snapshot mode when F_LAST (128) is seen — end of snapshot sequence
        is_last_msg = (flags & F_LAST) != 0
        if self.snapshot_in_progress and is_last_msg:
            self.snapshot_in_progress = False
            self.book_valid = True

        # Non-snapshot recovery: if book invalid and not mid-snapshot, mark valid
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
        self.window_spot_ref_price_int = _resolve_spot_ref(
            last_trade=self.last_trade_price_int,
            depth_bid=self.depth_bid,
            depth_ask=self.depth_ask,
        )
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
        # Fallback: if spot_ref was 0 at window start, try to compute at window end
        if spot_ref == 0:
            spot_ref = _resolve_spot_ref(
                last_trade=self.last_trade_price_int,
                depth_bid=self.depth_bid,
                depth_ask=self.depth_ask,
            )

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
                return  # _emit_depth_flow_rows already resets accumulators

        self._reset_accumulators()

    def _emit_depth_flow_rows(self, spot_ref: int, window_valid: bool) -> None:
        min_price = spot_ref - self.grid_max_ticks * self.tick_int
        max_price = spot_ref + self.grid_max_ticks * self.tick_int

        bid_anchor = self.window_best_bid_start
        if bid_anchor <= 0:
            bid_anchor, _ = _best_level(self.depth_bid, is_bid=True)
        if bid_anchor <= 0:
            bid_anchor = spot_ref

        ask_anchor = self.window_best_ask_start
        if ask_anchor <= 0:
            ask_anchor, _ = _best_level(self.depth_ask, is_bid=False)
        if ask_anchor <= 0:
            ask_anchor = spot_ref

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
            rel_ticks_side = int(round((price - bid_anchor) / self.tick_int))
            # Simple State Track: Current Depth
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
                    "depth_qty_rest": float(rest_bid.get(price, 0.0)),
                    "pull_qty_rest": float(pull_rest),
                    "fill_qty": float(fill_qty),
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
            rel_ticks_side = int(round((price - ask_anchor) / self.tick_int))
            # Simple State Track: Current Depth
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
                    "depth_qty_rest": float(rest_ask.get(price, 0.0)),
                    "pull_qty_rest": float(pull_rest),
                    "fill_qty": float(fill_qty),
                    "window_valid": window_valid,
                }
            )

        self._reset_accumulators()

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
        self.book_valid = False
        self.window_has_snapshot = True
        # Enter snapshot mode only when F_SNAPSHOT (32) is set on the Clear record.
        # Non-snapshot clears (trading halts) don't start a snapshot sequence;
        # the next event will re-enable book_valid via the recovery check.
        self.snapshot_in_progress = bool(flags & F_SNAPSHOT)

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
        
        # Cap fill quantity at order's remaining quantity.
        # Databento MBO data can report fill_qty > order.qty for aggressor orders
        # where the fill event shows the total trade size, not the order's fill amount.
        actual_fill = min(fill_qty, order.qty)
        
        if side == SIDE_BID:
            self.depth_bid[price] = self.depth_bid.get(price, 0) - actual_fill
            if self.depth_bid[price] <= 0:
                self.depth_bid.pop(price, None)
            self.acc_fill_bid[price] = self.acc_fill_bid.get(price, 0.0) + float(actual_fill)
        else:
            self.depth_ask[price] = self.depth_ask.get(price, 0) - actual_fill
            if self.depth_ask[price] <= 0:
                self.depth_ask.pop(price, None)
            self.acc_fill_ask[price] = self.acc_fill_ask.get(price, 0.0) + float(actual_fill)

        remaining = order.qty - actual_fill
        if remaining > 0:
            order.qty = remaining
        else:
            self.orders.pop(order_id, None)


def compute_futures_surfaces_1s_from_batches(
    batches: Iterable[pd.DataFrame],
    compute_depth_flow: bool = True,
    product: ProductConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute book snapshots and depth/flow from MBO event batches."""
    kwargs: dict = {"compute_depth_flow": compute_depth_flow}
    if product is not None:
        kwargs["tick_int"] = product.tick_int
        kwargs["grid_max_ticks"] = product.grid_max_ticks
    engine = FuturesBookEngine(**kwargs)
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
