"""Absolute-tick event-driven vacuum/pressure engine — two-force model.

Architecture:
    1. Maintain internal order book (orders by order_id -> {price_int, size, side}).
    2. Pre-allocated NumPy arrays of N_TICKS absolute price levels.
    3. Each tick is fully independent — no cross-tick coupling.
    4. Anchor set once from first valid BBO; never changes.
    5. Incremental O(1) BBO tracking in book operations.
    6. Spot is a read-only property, used only at serve time for windowing.
    7. No grid shift, no spot-relative indexing during event processing.

Two-force model:
    Pressure (depth BUILDING — liquidity arriving/replenishing):
        pressure = c1*v_add + c2*max(v_rest_depth, 0) + c3*max(a_add, 0)

    Vacuum (depth DRAINING — liquidity removed/consumed):
        vacuum = c4*v_pull + c5*v_fill + c6*max(-v_rest_depth, 0)
               + c7*max(a_pull, 0)

Derivative chain math (continuous-time EMA):
    For each mechanics quantity at a tick, we maintain v, a, j:
        alpha = 1 - exp(-dt / tau)
        ema_new = alpha * x + (1 - alpha) * ema_old

Guarantees:
    G1: Each event advances engine state exactly once.
    G2: No value is null/NaN/Inf.
    G3: Untouched ticks persist prior values.
    G4: Any source adapter produces identical outputs for identical event stream.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRICE_SCALE: float = 1e-9
"""System-wide price scale: price_dollars = price_int * PRICE_SCALE."""

EPS_QTY: float = 1.0
"""Laplace smoothing in intensity denominators."""

# MBO action codes (matching databento convention)
ACTION_ADD: str = "A"
ACTION_CANCEL: str = "C"
ACTION_MODIFY: str = "M"
ACTION_CLEAR: str = "R"
ACTION_TRADE: str = "T"
ACTION_FILL: str = "F"

SIDE_BID: str = "B"
SIDE_ASK: str = "A"

F_SNAPSHOT: int = 32
F_LAST: int = 128

# Derivative chain time constants (seconds)
TAU_VELOCITY: float = 2.0
"""Time constant for velocity EMA (1st derivative)."""

TAU_ACCELERATION: float = 5.0
"""Time constant for acceleration EMA (2nd derivative)."""

TAU_JERK: float = 10.0
"""Time constant for jerk EMA (3rd derivative)."""

# Two-force model coefficients — Pressure (depth building)
C1_V_ADD: float = 1.0
C2_V_REST_POS: float = 0.5
C3_A_ADD: float = 0.3

# Two-force model coefficients — Vacuum (depth draining)
C4_V_PULL: float = 1.0
C5_V_FILL: float = 1.5
C6_V_REST_NEG: float = 0.5
C7_A_PULL: float = 0.3

# Rest depth exponential decay time constant (seconds)
TAU_REST_DECAY: float = 30.0


# ---------------------------------------------------------------------------
# Internal order state
# ---------------------------------------------------------------------------

@dataclass
class _OrderEntry:
    """Minimal order state for internal book tracking."""
    side: str
    price_int: int
    qty: int
    idx: int = -1


# ---------------------------------------------------------------------------
# Derivative chain helpers
# ---------------------------------------------------------------------------

def _ema_alpha(dt_s: float, tau: float) -> float:
    """Compute EMA blending factor for variable time intervals.

    Uses the continuous-time formula: alpha = 1 - exp(-dt / tau).
    """
    if dt_s <= 0.0:
        return 0.0
    ratio = dt_s / tau
    if ratio > 50.0:
        return 1.0
    return 1.0 - math.exp(-ratio)


def _update_derivative_chain(
    prev_value: float,
    new_value: float,
    dt_s: float,
    v_prev: float,
    a_prev: float,
    j_prev: float,
    tau_v: float = TAU_VELOCITY,
    tau_a: float = TAU_ACCELERATION,
    tau_j: float = TAU_JERK,
) -> Tuple[float, float, float]:
    """Update a three-level derivative chain from value changes.

    Use for snapshot quantities (e.g. rest_depth) where the rate of
    change of the signal itself is the correct input.

    Returns:
        (v_new, a_new, j_new). Guaranteed finite.
    """
    if dt_s <= 0.0:
        return v_prev, a_prev, j_prev

    rate = (new_value - prev_value) / dt_s

    alpha_v = _ema_alpha(dt_s, tau_v)
    v_new = alpha_v * rate + (1.0 - alpha_v) * v_prev

    dv_rate = (v_new - v_prev) / dt_s
    alpha_a = _ema_alpha(dt_s, tau_a)
    a_new = alpha_a * dv_rate + (1.0 - alpha_a) * a_prev

    da_rate = (a_new - a_prev) / dt_s
    alpha_j = _ema_alpha(dt_s, tau_j)
    j_new = alpha_j * da_rate + (1.0 - alpha_j) * j_prev

    if not (math.isfinite(v_new) and math.isfinite(a_new) and math.isfinite(j_new)):
        return v_prev, a_prev, j_prev

    return v_new, a_new, j_new


def _update_derivative_chain_from_delta(
    delta: float,
    dt_s: float,
    v_prev: float,
    a_prev: float,
    j_prev: float,
    tau_v: float = TAU_VELOCITY,
    tau_a: float = TAU_ACCELERATION,
    tau_j: float = TAU_JERK,
) -> Tuple[float, float, float]:
    """Update a three-level derivative chain from a raw event delta.

    Use for decayed accumulators (add_mass, pull_mass, fill_mass) to
    separate passive decay from the derivative signal.

    Returns:
        (v_new, a_new, j_new). Guaranteed finite.
    """
    if dt_s <= 0.0:
        return v_prev, a_prev, j_prev

    rate = delta / dt_s

    alpha_v = _ema_alpha(dt_s, tau_v)
    v_new = alpha_v * rate + (1.0 - alpha_v) * v_prev

    dv_rate = (v_new - v_prev) / dt_s
    alpha_a = _ema_alpha(dt_s, tau_a)
    a_new = alpha_a * dv_rate + (1.0 - alpha_a) * a_prev

    da_rate = (a_new - a_prev) / dt_s
    alpha_j = _ema_alpha(dt_s, tau_j)
    j_new = alpha_j * da_rate + (1.0 - alpha_j) * j_prev

    if not (math.isfinite(v_new) and math.isfinite(a_new) and math.isfinite(j_new)):
        return v_prev, a_prev, j_prev

    return v_new, a_new, j_new


# ---------------------------------------------------------------------------
# Main engine — absolute tick indexed
# ---------------------------------------------------------------------------

class AbsoluteTickEngine:
    """Absolute-tick event-driven vacuum/pressure engine.

    Pre-allocates N_TICKS price-level slots backed by NumPy arrays.
    Each tick is fully independent — events at price P only update the
    slot for P.  No grid shift, no spot coupling during event processing.

    Spot is computed on demand from BBO for serve-time window extraction.

    Args:
        n_ticks: Total pre-allocated tick slots (default 500).
        tick_int: Price integer units per tick (e.g. 250000000 for $0.25).
        bucket_size_dollars: Size of each bucket in dollars.

    Example:
        >>> engine = AbsoluteTickEngine(n_ticks=500, tick_int=250000000)
        >>> engine.update(
        ...     ts_ns=1707220800_000_000_000,
        ...     action="A", side="B",
        ...     price_int=21500_000_000_000,
        ...     size=1, order_id=12345, flags=0,
        ... )
        >>> engine.event_count
        1
    """

    def __init__(
        self,
        n_ticks: int = 500,
        tick_int: int = 250_000_000,
        bucket_size_dollars: float = 0.25,
        *,
        anchor_tick_idx: int | None = None,
        auto_anchor_from_bbo: bool = True,
        fail_on_out_of_range: bool = False,
    ) -> None:
        if n_ticks < 3:
            raise ValueError(f"n_ticks must be >= 3, got {n_ticks}")
        if tick_int <= 0:
            raise ValueError(f"tick_int must be > 0, got {tick_int}")
        if bucket_size_dollars <= 0.0:
            raise ValueError(
                f"bucket_size_dollars must be > 0, got {bucket_size_dollars}"
            )

        self.n_ticks: int = n_ticks
        self.tick_int: int = tick_int
        self.bucket_size_dollars: float = bucket_size_dollars
        self._auto_anchor_from_bbo: bool = bool(auto_anchor_from_bbo)
        self._fail_on_out_of_range: bool = bool(fail_on_out_of_range)

        # Anchor: absolute tick index of the center slot. Set once from
        # first valid BBO, never changes. -1 means not yet established.
        if anchor_tick_idx is None:
            self._anchor_tick_idx = -1
        else:
            if anchor_tick_idx < 0:
                raise ValueError(
                    f"anchor_tick_idx must be >= 0 when provided, got {anchor_tick_idx}"
                )
            self._anchor_tick_idx = int(anchor_tick_idx)

        # --- NumPy state arrays, shape (n_ticks,) ---
        # Mechanics
        self._add_mass = np.zeros(n_ticks, dtype=np.float64)
        self._pull_mass = np.zeros(n_ticks, dtype=np.float64)
        self._fill_mass = np.zeros(n_ticks, dtype=np.float64)
        self._rest_depth = np.zeros(n_ticks, dtype=np.float64)
        # Velocity (1st derivative)
        self._v_add = np.zeros(n_ticks, dtype=np.float64)
        self._v_pull = np.zeros(n_ticks, dtype=np.float64)
        self._v_fill = np.zeros(n_ticks, dtype=np.float64)
        self._v_rest_depth = np.zeros(n_ticks, dtype=np.float64)
        # Acceleration (2nd derivative)
        self._a_add = np.zeros(n_ticks, dtype=np.float64)
        self._a_pull = np.zeros(n_ticks, dtype=np.float64)
        self._a_fill = np.zeros(n_ticks, dtype=np.float64)
        self._a_rest_depth = np.zeros(n_ticks, dtype=np.float64)
        # Jerk (3rd derivative)
        self._j_add = np.zeros(n_ticks, dtype=np.float64)
        self._j_pull = np.zeros(n_ticks, dtype=np.float64)
        self._j_fill = np.zeros(n_ticks, dtype=np.float64)
        self._j_rest_depth = np.zeros(n_ticks, dtype=np.float64)
        # Force
        self._pressure_variant = np.zeros(n_ticks, dtype=np.float64)
        self._vacuum_variant = np.zeros(n_ticks, dtype=np.float64)
        # Metadata
        self._last_ts_ns = np.zeros(n_ticks, dtype=np.int64)
        self._last_event_id = np.zeros(n_ticks, dtype=np.int64)

        # --- Internal order book ---
        self._orders: Dict[int, _OrderEntry] = {}
        self._depth_bid_arr = np.zeros(n_ticks, dtype=np.int64)
        self._depth_ask_arr = np.zeros(n_ticks, dtype=np.int64)

        # --- Incremental BBO ---
        self._best_bid_idx: int = -1
        self._best_ask_idx: int = -1
        self._best_bid: int = 0
        self._best_ask: int = 0

        # --- Lifecycle ---
        self._event_counter: int = 0
        self._prev_ts_ns: int = 0
        self._book_valid: bool = False
        self._snapshot_in_progress: bool = False

    # ------------------------------------------------------------------
    # Absolute tick indexing
    # ------------------------------------------------------------------

    def _price_to_idx(self, price_int: int) -> Optional[int]:
        """Map an absolute price_int to an array index.

        Returns None if anchor is not set or price is out of range.
        """
        if self._anchor_tick_idx < 0:
            return None
        tick_abs = round(price_int / self.tick_int)
        idx = tick_abs - self._anchor_tick_idx + self.n_ticks // 2
        if 0 <= idx < self.n_ticks:
            return int(idx)
        return None

    def spot_to_idx(self, spot_price_int: int) -> Optional[int]:
        """Map a spot price_int to an array index (for serve-time windowing)."""
        return self._price_to_idx(spot_price_int)

    def _idx_to_price_int(self, idx: int) -> int:
        """Map a grid index back to absolute price_int."""
        if idx < 0 or self._anchor_tick_idx < 0:
            return 0
        tick_abs = self._anchor_tick_idx - self.n_ticks // 2 + idx
        return int(tick_abs * self.tick_int)

    def _repair_best_bid_idx(self, start_idx: int) -> None:
        """Repair best bid by scanning down from start_idx."""
        idx = min(max(start_idx, 0), self.n_ticks - 1)
        while idx >= 0 and self._depth_bid_arr[idx] <= 0:
            idx -= 1
        self._best_bid_idx = idx
        self._best_bid = self._idx_to_price_int(idx) if idx >= 0 else 0

    def _repair_best_ask_idx(self, start_idx: int) -> None:
        """Repair best ask by scanning up from start_idx."""
        idx = min(max(start_idx, 0), self.n_ticks - 1)
        while idx < self.n_ticks and self._depth_ask_arr[idx] <= 0:
            idx += 1
        if idx >= self.n_ticks:
            idx = -1
        self._best_ask_idx = idx
        self._best_ask = self._idx_to_price_int(idx) if idx >= 0 else 0

    def _recompute_best_from_depth_arrays(self) -> None:
        """Recompute best bid/ask from depth arrays after bulk rebuild."""
        bid_levels = np.flatnonzero(self._depth_bid_arr > 0)
        ask_levels = np.flatnonzero(self._depth_ask_arr > 0)

        self._best_bid_idx = int(bid_levels[-1]) if bid_levels.size > 0 else -1
        self._best_ask_idx = int(ask_levels[0]) if ask_levels.size > 0 else -1

        self._best_bid = (
            self._idx_to_price_int(self._best_bid_idx)
            if self._best_bid_idx >= 0
            else 0
        )
        self._best_ask = (
            self._idx_to_price_int(self._best_ask_idx)
            if self._best_ask_idx >= 0
            else 0
        )

    def _recompute_provisional_bbo_from_orders(self) -> None:
        """Recompute BBO from order map when anchor is not available."""
        best_bid = 0
        best_ask = 0
        for entry in self._orders.values():
            if entry.qty <= 0:
                continue
            if entry.side == SIDE_BID:
                if entry.price_int > best_bid:
                    best_bid = entry.price_int
            elif entry.side == SIDE_ASK:
                if best_ask == 0 or entry.price_int < best_ask:
                    best_ask = entry.price_int
        self._best_bid_idx = -1
        self._best_ask_idx = -1
        self._best_bid = best_bid
        self._best_ask = best_ask

    def _rebuild_depth_from_orders(self) -> None:
        """Rebuild depth arrays and BBO from current order map."""
        self._depth_bid_arr[:] = 0
        self._depth_ask_arr[:] = 0
        self._best_bid_idx = -1
        self._best_ask_idx = -1
        self._best_bid = 0
        self._best_ask = 0

        if self._anchor_tick_idx < 0:
            for entry in self._orders.values():
                entry.idx = -1
            self._recompute_provisional_bbo_from_orders()
            return

        for entry in self._orders.values():
            idx = self._price_to_idx(entry.price_int)
            if idx is None:
                entry.idx = -1
                if self._fail_on_out_of_range:
                    raise ValueError(
                        "Order price mapped outside configured absolute grid during rebuild "
                        f"(price_int={entry.price_int}, anchor_tick_idx={self._anchor_tick_idx}, "
                        f"n_ticks={self.n_ticks}, tick_int={self.tick_int})."
                    )
                continue
            entry.idx = idx
            if entry.side == SIDE_BID:
                self._depth_bid_arr[idx] += entry.qty
            else:
                self._depth_ask_arr[idx] += entry.qty

        self._recompute_best_from_depth_arrays()

    def _apply_depth_delta(self, side: str, idx: int, delta: int) -> None:
        """Apply signed qty delta to one depth array slot and maintain BBO."""
        if delta == 0 or idx < 0:
            return

        if side == SIDE_BID:
            arr = self._depth_bid_arr
            best_idx = self._best_bid_idx
            direction_bid = True
        else:
            arr = self._depth_ask_arr
            best_idx = self._best_ask_idx
            direction_bid = False

        cur = int(arr[idx])
        new_val = cur + int(delta)
        if new_val < 0:
            raise ValueError(
                f"Depth underflow at idx={idx} side={side}: cur={cur}, delta={delta}"
            )
        arr[idx] = new_val

        if direction_bid:
            if new_val > 0 and (best_idx < 0 or idx > best_idx):
                self._best_bid_idx = idx
                self._best_bid = self._idx_to_price_int(idx)
            elif cur > 0 and new_val == 0 and idx == best_idx:
                self._repair_best_bid_idx(idx)
        else:
            if new_val > 0 and (best_idx < 0 or idx < best_idx):
                self._best_ask_idx = idx
                self._best_ask = self._idx_to_price_int(idx)
            elif cur > 0 and new_val == 0 and idx == best_idx:
                self._repair_best_ask_idx(idx)

    def _try_set_anchor(self) -> None:
        """Set anchor from first valid BBO if not already set."""
        if self._anchor_tick_idx >= 0:
            return
        if not self._auto_anchor_from_bbo:
            return
        if self._best_bid > 0 and self._best_ask > 0:
            mid_tick = int(math.floor(
                (self._best_bid + self._best_ask) / (2.0 * self.tick_int) + 0.5
            ))
            self._anchor_tick_idx = mid_tick
            self._rebuild_depth_from_orders()
            logger.info(
                "Anchor set: tick_idx=%d (bid=%d, ask=%d, tick_int=%d)",
                mid_tick, self._best_bid, self._best_ask, self.tick_int,
            )

    def reanchor_to_bbo(self) -> bool:
        """Reset anchor to current BBO midpoint.

        Call after importing book state and BEFORE sync_rest_depth_from_book().
        The grid arrays must already be zeroed (which import_book_state does).

        Returns True if anchor was moved, False if BBO is invalid.
        """
        if self._best_bid <= 0 or self._best_ask <= 0:
            return False

        new_tick = int(math.floor(
            (self._best_bid + self._best_ask) / (2.0 * self.tick_int) + 0.5
        ))
        old_tick = self._anchor_tick_idx
        self._anchor_tick_idx = new_tick
        self._rebuild_depth_from_orders()

        if old_tick != new_tick:
            logger.info(
                "Anchor recentered: %d -> %d (bid=$%.2f, ask=$%.2f, mid=$%.2f)",
                old_tick,
                new_tick,
                self._best_bid * PRICE_SCALE,
                self._best_ask * PRICE_SCALE,
                (self._best_bid + self._best_ask) * 0.5 * PRICE_SCALE,
            )
        return True

    def set_anchor_tick_idx(self, anchor_tick_idx: int) -> None:
        """Set the absolute anchor tick index explicitly.

        This is used by core-grid pipelines that decouple force math from BBO.
        """
        if anchor_tick_idx < 0:
            raise ValueError(f"anchor_tick_idx must be >= 0, got {anchor_tick_idx}")
        if self._anchor_tick_idx >= 0 and self._anchor_tick_idx != anchor_tick_idx:
            raise ValueError(
                "anchor_tick_idx is already set and cannot be changed without "
                f"reinitializing the engine (current={self._anchor_tick_idx}, "
                f"requested={anchor_tick_idx})."
            )
        self._anchor_tick_idx = int(anchor_tick_idx)
        self._rebuild_depth_from_orders()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        ts_ns: int,
        action: str,
        side: str,
        price_int: int,
        size: int,
        order_id: int,
        flags: int,
    ) -> None:
        """Process one MBO event: book update + derivative chain + force.

        This is the single entry point for all source adapters.
        Every call advances engine state exactly once.
        """
        self._event_counter += 1
        event_id = self._event_counter
        is_snapshot = (flags & F_SNAPSHOT) != 0

        # --- Snapshot lifecycle ---
        was_snapshot_in_progress = self._snapshot_in_progress

        if action == ACTION_CLEAR and is_snapshot:
            self._snapshot_in_progress = True
        if self._snapshot_in_progress and (flags & F_LAST) != 0:
            self._snapshot_in_progress = False
            self._book_valid = True
        if not self._book_valid and not self._snapshot_in_progress:
            self._book_valid = True

        self._prev_ts_ns = ts_ns

        # --- Apply event to order book (with incremental BBO) ---
        touched_info: List[Tuple[int, float, float, float]] = []
        # Each entry: (idx, add_delta, pull_delta, fill_delta)

        if action == ACTION_CLEAR:
            self._clear_book()
        elif action == ACTION_TRADE:
            pass
        elif action == ACTION_ADD:
            self._add_order(side, price_int, size, order_id)
            new_entry = self._orders.get(order_id)
            if new_entry is not None:
                touched_info.append((new_entry.idx, float(size), 0.0, 0.0))
        elif action == ACTION_CANCEL:
            old_entry = self._orders.get(order_id)
            if old_entry is not None:
                cancel_idx = old_entry.idx
                cancel_size = old_entry.qty
                self._cancel_order(order_id)
                touched_info.append((cancel_idx, 0.0, float(cancel_size), 0.0))
            else:
                self._cancel_order(order_id)
        elif action == ACTION_MODIFY:
            old_entry = self._orders.get(order_id)
            if old_entry is not None:
                old_idx = old_entry.idx
                old_size = old_entry.qty
                self._modify_order(order_id, side, price_int, size)
                new_entry = self._orders.get(order_id)
                new_idx = new_entry.idx if new_entry is not None else -1
                if old_idx != new_idx:
                    touched_info.append((old_idx, 0.0, float(old_size), 0.0))
                    touched_info.append((new_idx, float(size), 0.0, 0.0))
                else:
                    size_diff = size - old_size
                    if size_diff > 0:
                        touched_info.append(
                            (new_idx, float(size_diff), 0.0, 0.0)
                        )
                    elif size_diff < 0:
                        touched_info.append(
                            (new_idx, 0.0, float(-size_diff), 0.0)
                        )
                    else:
                        touched_info.append((new_idx, 0.0, 0.0, 0.0))
            else:
                self._modify_order(order_id, side, price_int, size)
        elif action == ACTION_FILL:
            old_entry = self._orders.get(order_id)
            if old_entry is not None:
                fill_idx = old_entry.idx
                self._fill_order(order_id, size)
                touched_info.append((fill_idx, 0.0, 0.0, float(size)))
            else:
                self._fill_order(order_id, size)

        # --- Snapshot completion ---
        snapshot_completed = (
            was_snapshot_in_progress and not self._snapshot_in_progress
        )

        # --- Set anchor from first valid BBO ---
        anchor_was_unset = self._anchor_tick_idx < 0
        self._try_set_anchor()
        anchor_just_set = anchor_was_unset and self._anchor_tick_idx >= 0

        # --- Sync rest_depth after snapshot or anchor establishment ---
        if (snapshot_completed or anchor_just_set) and self._anchor_tick_idx >= 0:
            self.sync_rest_depth_from_book()

        # --- Update mechanics + derivatives at touched ticks ---
        for idx, add_delta, pull_delta, fill_delta in touched_info:
            if idx < 0 or idx >= self.n_ticks:
                continue

            # Per-tick dt
            last_ts = int(self._last_ts_ns[idx])
            dt_s = 0.0
            if last_ts > 0 and ts_ns > last_ts:
                dt_s = (ts_ns - last_ts) / 1e9

            # Previous rest_depth for derivative chain
            prev_rest = float(self._rest_depth[idx])

            # Update rest_depth from book (futures: 1 tick = 1 price level)
            self._rest_depth[idx] = float(
                self._depth_bid_arr[idx] + self._depth_ask_arr[idx]
            )

            # Decay and accumulate mass
            if dt_s > 0.0:
                decay = math.exp(-dt_s / TAU_REST_DECAY)
                self._add_mass[idx] = self._add_mass[idx] * decay + add_delta
                self._pull_mass[idx] = self._pull_mass[idx] * decay + pull_delta
                self._fill_mass[idx] = self._fill_mass[idx] * decay + fill_delta
            else:
                self._add_mass[idx] += add_delta
                self._pull_mass[idx] += pull_delta
                self._fill_mass[idx] += fill_delta

            # Derivative chains
            if dt_s > 0.0:
                v, a, j = _update_derivative_chain_from_delta(
                    add_delta, dt_s,
                    float(self._v_add[idx]),
                    float(self._a_add[idx]),
                    float(self._j_add[idx]),
                )
                self._v_add[idx] = v
                self._a_add[idx] = a
                self._j_add[idx] = j

                v, a, j = _update_derivative_chain_from_delta(
                    pull_delta, dt_s,
                    float(self._v_pull[idx]),
                    float(self._a_pull[idx]),
                    float(self._j_pull[idx]),
                )
                self._v_pull[idx] = v
                self._a_pull[idx] = a
                self._j_pull[idx] = j

                v, a, j = _update_derivative_chain_from_delta(
                    fill_delta, dt_s,
                    float(self._v_fill[idx]),
                    float(self._a_fill[idx]),
                    float(self._j_fill[idx]),
                )
                self._v_fill[idx] = v
                self._a_fill[idx] = a
                self._j_fill[idx] = j

                v, a, j = _update_derivative_chain(
                    prev_rest, float(self._rest_depth[idx]), dt_s,
                    float(self._v_rest_depth[idx]),
                    float(self._a_rest_depth[idx]),
                    float(self._j_rest_depth[idx]),
                )
                self._v_rest_depth[idx] = v
                self._a_rest_depth[idx] = a
                self._j_rest_depth[idx] = j

            # Pressure / vacuum
            self._pressure_variant[idx] = (
                C1_V_ADD * self._v_add[idx]
                + C2_V_REST_POS * max(float(self._v_rest_depth[idx]), 0.0)
                + C3_A_ADD * max(float(self._a_add[idx]), 0.0)
            )
            self._vacuum_variant[idx] = (
                C4_V_PULL * self._v_pull[idx]
                + C5_V_FILL * self._v_fill[idx]
                + C6_V_REST_NEG * max(-float(self._v_rest_depth[idx]), 0.0)
                + C7_A_PULL * max(float(self._a_pull[idx]), 0.0)
            )

            # Metadata
            self._last_event_id[idx] = event_id
            self._last_ts_ns[idx] = ts_ns

    # ------------------------------------------------------------------
    # Order book operations (with incremental BBO)
    # ------------------------------------------------------------------

    def _clear_book(self) -> None:
        """Clear all orders and depth."""
        self._orders.clear()
        self._depth_bid_arr[:] = 0
        self._depth_ask_arr[:] = 0
        self._best_bid_idx = -1
        self._best_ask_idx = -1
        self._best_bid = 0
        self._best_ask = 0
        self._book_valid = False

    def _add_order(
        self, side: str, price_int: int, size: int, order_id: int
    ) -> None:
        """Add a new order to the internal book with incremental BBO."""
        idx = self._price_to_idx(price_int)
        if idx is None:
            if self._anchor_tick_idx >= 0 and self._fail_on_out_of_range:
                raise ValueError(
                    "Add price mapped outside configured absolute grid "
                    f"(price_int={price_int}, anchor_tick_idx={self._anchor_tick_idx}, "
                    f"n_ticks={self.n_ticks}, tick_int={self.tick_int})."
                )
            idx = -1

        self._orders[order_id] = _OrderEntry(
            side=side, price_int=price_int, qty=size, idx=idx
        )
        if idx >= 0:
            self._apply_depth_delta(side, idx, size)
        elif self._anchor_tick_idx < 0:
            # Pre-anchor: maintain provisional BBO by raw price.
            if side == SIDE_BID:
                if price_int > self._best_bid:
                    self._best_bid = price_int
            else:
                if self._best_ask == 0 or price_int < self._best_ask:
                    self._best_ask = price_int

    def _cancel_order(self, order_id: int) -> None:
        """Cancel an order with incremental BBO update."""
        entry = self._orders.pop(order_id, None)
        if entry is None:
            return
        if entry.idx >= 0:
            self._apply_depth_delta(entry.side, entry.idx, -entry.qty)
        elif self._anchor_tick_idx < 0:
            self._recompute_provisional_bbo_from_orders()

    def _modify_order(
        self, order_id: int, side: str, price_int: int, size: int
    ) -> None:
        """Modify an existing order (cancel old, add new) with BBO update."""
        old = self._orders.pop(order_id, None)
        if old is not None:
            if old.idx >= 0:
                self._apply_depth_delta(old.side, old.idx, -old.qty)

        idx = self._price_to_idx(price_int)
        if idx is None:
            if self._anchor_tick_idx >= 0 and self._fail_on_out_of_range:
                raise ValueError(
                    "Modify price mapped outside configured absolute grid "
                    f"(price_int={price_int}, anchor_tick_idx={self._anchor_tick_idx}, "
                    f"n_ticks={self.n_ticks}, tick_int={self.tick_int})."
                )
            idx = -1
        self._orders[order_id] = _OrderEntry(
            side=side, price_int=price_int, qty=size, idx=idx
        )
        if idx >= 0:
            self._apply_depth_delta(side, idx, size)
        elif self._anchor_tick_idx < 0:
            self._recompute_provisional_bbo_from_orders()

    def _fill_order(self, order_id: int, fill_size: int) -> None:
        """Fill an order with incremental BBO update."""
        entry = self._orders.get(order_id)
        if entry is None:
            return

        effective_fill = min(fill_size, entry.qty)
        if entry.idx >= 0:
            self._apply_depth_delta(entry.side, entry.idx, -effective_fill)

        entry.qty -= effective_fill
        if entry.qty <= 0:
            self._orders.pop(order_id, None)
            if self._anchor_tick_idx < 0:
                self._recompute_provisional_bbo_from_orders()

    # ------------------------------------------------------------------
    # Bulk rest_depth sync (after snapshot)
    # ------------------------------------------------------------------

    def sync_rest_depth_from_book(self) -> None:
        """Synchronize all rest_depth values from current order book."""
        if self._anchor_tick_idx < 0:
            return

        if self._fail_on_out_of_range:
            for entry in self._orders.values():
                if entry.idx < 0:
                    raise ValueError(
                        "Order price mapped outside configured absolute grid "
                        f"(price_int={entry.price_int}, anchor_tick_idx={self._anchor_tick_idx}, "
                        f"n_ticks={self.n_ticks}, tick_int={self.tick_int})."
                    )

        self._rest_depth[:] = self._depth_bid_arr + self._depth_ask_arr

    # ------------------------------------------------------------------
    # Lightweight book-only processing (pre-warmup fast-forward)
    # ------------------------------------------------------------------

    def apply_book_event(
        self,
        ts_ns: int,
        action: str,
        side: str,
        price_int: int,
        size: int,
        order_id: int,
        flags: int,
    ) -> None:
        """Lightweight book-only event processing for pre-warmup fast-forward.

        Updates the internal order book and BBO *without* computing
        mechanics, derivatives, or force variants.  10-50x faster than
        update().

        After the last apply_book_event call, the caller should invoke
        sync_rest_depth_from_book() to initialize rest_depth.
        """
        self._event_counter += 1
        is_snapshot = (flags & F_SNAPSHOT) != 0

        # Snapshot lifecycle
        if action == ACTION_CLEAR and is_snapshot:
            self._snapshot_in_progress = True
        if self._snapshot_in_progress and (flags & F_LAST) != 0:
            self._snapshot_in_progress = False
            self._book_valid = True
        if not self._book_valid and not self._snapshot_in_progress:
            self._book_valid = True

        self._prev_ts_ns = ts_ns

        # Apply to order book (incremental BBO built into each op)
        if action == ACTION_CLEAR:
            self._clear_book()
        elif action == ACTION_ADD:
            self._add_order(side, price_int, size, order_id)
        elif action == ACTION_CANCEL:
            self._cancel_order(order_id)
        elif action == ACTION_MODIFY:
            self._modify_order(order_id, side, price_int, size)
        elif action == ACTION_FILL:
            self._fill_order(order_id, size)

        # Set anchor from first valid BBO
        self._try_set_anchor()

    # ------------------------------------------------------------------
    # Book state serialization (for checkpoint caching)
    # ------------------------------------------------------------------

    _BOOK_STATE_VERSION: int = 3

    def export_book_state(self) -> bytes:
        """Serialize book state to bytes for caching.

        Grid arrays are NOT included — they are all zeros during
        book-only mode and warmup will populate them.
        """
        import pickle

        state = {
            "_v": self._BOOK_STATE_VERSION,
            "orders": {
                oid: (e.side, e.price_int, e.qty)
                for oid, e in self._orders.items()
            },
            "best_bid": self._best_bid,
            "best_ask": self._best_ask,
            "anchor_tick_idx": self._anchor_tick_idx,
            "event_counter": self._event_counter,
            "prev_ts_ns": self._prev_ts_ns,
            "book_valid": self._book_valid,
            "snapshot_in_progress": self._snapshot_in_progress,
        }
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    def import_book_state(self, data: bytes) -> None:
        """Restore book state from cached bytes.

        Resets grid arrays to zero — warmup will populate them.
        Caller should invoke sync_rest_depth_from_book() after import.
        """
        import pickle

        state = pickle.loads(data)  # noqa: S301
        state_version = int(state.get("_v", 0))
        if state_version not in {2, self._BOOK_STATE_VERSION}:
            raise ValueError(
                f"Unsupported book state version {state.get('_v')}, "
                f"expected one of {[2, self._BOOK_STATE_VERSION]}. "
                "Delete the cache and retry."
            )

        self._orders = {
            oid: _OrderEntry(side=s, price_int=p, qty=q, idx=-1)
            for oid, (s, p, q) in state["orders"].items()
        }
        self._anchor_tick_idx = state["anchor_tick_idx"]
        self._event_counter = state["event_counter"]
        self._prev_ts_ns = state["prev_ts_ns"]
        self._book_valid = state["book_valid"]
        self._snapshot_in_progress = state["snapshot_in_progress"]
        self._depth_bid_arr[:] = 0
        self._depth_ask_arr[:] = 0
        self._best_bid_idx = -1
        self._best_ask_idx = -1
        self._best_bid = 0
        self._best_ask = 0

        # Rebuild array depth from order map under current anchor.
        if self._anchor_tick_idx >= 0:
            self._rebuild_depth_from_orders()
        else:
            self._recompute_provisional_bbo_from_orders()

        # Reset grid arrays — warmup will populate
        self._add_mass[:] = 0.0
        self._pull_mass[:] = 0.0
        self._fill_mass[:] = 0.0
        self._rest_depth[:] = 0.0
        self._v_add[:] = 0.0
        self._v_pull[:] = 0.0
        self._v_fill[:] = 0.0
        self._v_rest_depth[:] = 0.0
        self._a_add[:] = 0.0
        self._a_pull[:] = 0.0
        self._a_fill[:] = 0.0
        self._a_rest_depth[:] = 0.0
        self._j_add[:] = 0.0
        self._j_pull[:] = 0.0
        self._j_fill[:] = 0.0
        self._j_rest_depth[:] = 0.0
        self._pressure_variant[:] = 0.0
        self._vacuum_variant[:] = 0.0
        self._last_ts_ns[:] = 0
        self._last_event_id[:] = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def spot_ref_price_int(self) -> int:
        """Spot reference (mid of BBO) computed on demand. For serve-time use."""
        if self._best_bid > 0 and self._best_ask > 0:
            raw_mid = (self._best_bid + self._best_ask) / 2.0
            return int(math.floor(raw_mid / self.tick_int + 0.5)) * self.tick_int
        return 0

    @property
    def book_valid(self) -> bool:
        return self._book_valid

    @property
    def order_count(self) -> int:
        return len(self._orders)

    @property
    def event_count(self) -> int:
        return self._event_counter

    @property
    def best_bid_price_int(self) -> int:
        return self._best_bid

    @property
    def best_ask_price_int(self) -> int:
        return self._best_ask

    @property
    def mid_price(self) -> float:
        """Mid price in dollars from current BBO, or 0.0 when invalid."""
        if self._best_bid > 0 and self._best_ask > 0:
            return (self._best_bid + self._best_ask) * 0.5 * PRICE_SCALE
        return 0.0

    @property
    def anchor_tick_idx(self) -> int:
        """Absolute tick index of the array center. -1 if not yet set."""
        return self._anchor_tick_idx

    def book_metrics(self) -> Dict[str, Any]:
        """Return diagnostic metrics for monitoring book health.

        Intended for logging/monitoring — not for hot-path use.
        """
        bid_levels = int(np.count_nonzero(self._depth_bid_arr > 0))
        ask_levels = int(np.count_nonzero(self._depth_ask_arr > 0))
        total_bid_qty = int(self._depth_bid_arr.sum())
        total_ask_qty = int(self._depth_ask_arr.sum())

        return {
            "order_count": len(self._orders),
            "bid_levels": bid_levels,
            "ask_levels": ask_levels,
            "total_bid_qty": total_bid_qty,
            "total_ask_qty": total_ask_qty,
            "best_bid": self._best_bid,
            "best_ask": self._best_ask,
            "anchor_tick_idx": self._anchor_tick_idx,
            "event_count": self._event_counter,
            "book_valid": self._book_valid,
        }

    # ------------------------------------------------------------------
    # Grid snapshot (full N_TICKS arrays)
    # ------------------------------------------------------------------

    def grid_snapshot_arrays(self) -> Dict[str, np.ndarray]:
        """Return full grid state as numpy arrays.

        Returns views into internal arrays (zero-copy). Caller must not
        mutate them.  Length is always n_ticks.

        Returns:
            Dict with per-field 1D arrays of length n_ticks.
        """
        return {
            "add_mass": self._add_mass,
            "pull_mass": self._pull_mass,
            "fill_mass": self._fill_mass,
            "rest_depth": self._rest_depth,
            "v_add": self._v_add,
            "v_pull": self._v_pull,
            "v_fill": self._v_fill,
            "v_rest_depth": self._v_rest_depth,
            "a_add": self._a_add,
            "a_pull": self._a_pull,
            "a_fill": self._a_fill,
            "a_rest_depth": self._a_rest_depth,
            "j_add": self._j_add,
            "j_pull": self._j_pull,
            "j_fill": self._j_fill,
            "j_rest_depth": self._j_rest_depth,
            "pressure_variant": self._pressure_variant,
            "vacuum_variant": self._vacuum_variant,
            "last_event_id": self._last_event_id,
        }
