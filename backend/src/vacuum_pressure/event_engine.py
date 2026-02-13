"""Event-driven vacuum/pressure engine.

Canonical event-time force-field engine over relative price buckets around spot.
Every MBO order book event updates pressure-state math. Live and replay run
identical code paths (only the event source differs).

Architecture:
    1. Maintain internal order book (orders by order_id -> {price_int, size, side}).
    2. Track spot = mid(BBO) from internal book state.
    3. Dense grid of 2K+1 buckets centered on spot. Each bucket carries:
       - mechanics: add_mass, pull_mass, fill_mass, rest_depth
       - derivatives: v_*, a_*, j_* (dt-normalized via exponential decay)
       - force: pressure_variant, vacuum_variant, resistance_variant
       - metadata: last_event_id, cell_valid
    4. On each event: resolve bucket, update mechanics, decay-update derivatives,
       recompute force, emit dense grid.
    5. On spot shift: translate grid state by shift amount, fill edge buckets
       with finite defaults.

Derivative chain math (continuous-time EMA):
    For each mechanics quantity m_k at bucket k, we maintain:
        v_k (velocity)     = EMA of dm/dt
        a_k (acceleration) = EMA of dv/dt
        j_k (jerk)         = EMA of da/dt

    The EMA uses dt-normalized exponential decay:
        alpha = 1 - exp(-dt / tau)
        ema_new = alpha * x + (1 - alpha) * ema_old

    This properly handles variable inter-event times.
    Source: Jason S, Stack Overflow #1027808; standard single-pole IIR filter
    theory with continuous-time analog Y + tau * dY/dt = X.

Pressure variant formula (derivative-led):
    pressure_variant_k = c1*v_add_k + c2*v_fill_k - c3*v_pull_k
                       + c4*max(-a_rest_depth_k, 0)
                       + c5*j_flow_k

    where j_flow_k = j_add_k - j_pull_k + j_fill_k

    Raw levels appear only as conditioning terms (rest_depth for resistance),
    not primary drivers.

Guarantees:
    G1: For each event, engine state advances once and recomputes pressure variant.
    G2: Emitted grid contains all buckets k in [-K, +K] every time (2K+1 total).
    G3: No bucket value is null/NaN/Inf.
    G4: Untouched buckets persist prior values (after spot-frame remap).
    G5: Replay and live produce identical outputs for identical event stream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

# Pressure variant coefficients
C1_V_ADD: float = 1.0
"""Weight for velocity of add_mass."""

C2_V_FILL: float = 1.5
"""Weight for velocity of fill_mass."""

C3_V_PULL: float = 1.0
"""Weight for velocity of pull_mass."""

C4_A_REST: float = 0.5
"""Weight for acceleration of rest_depth (deceleration -> resistance decay)."""

C5_J_FLOW: float = 0.3
"""Weight for jerk of net flow (add - pull + fill)."""

# Rest depth exponential decay time constant (seconds)
TAU_REST_DECAY: float = 30.0
"""Time constant for passive rest_depth decay (absent direct events)."""


# ---------------------------------------------------------------------------
# Per-bucket state
# ---------------------------------------------------------------------------

@dataclass
class BucketState:
    """Full per-bucket state model.

    All fields initialized to finite defaults (zero). No field is ever
    null/NaN/Inf by construction.

    Mechanics:
        add_mass: cumulative add quantity (decayed)
        pull_mass: cumulative pull quantity (decayed)
        fill_mass: cumulative fill quantity (decayed)
        rest_depth: current resting depth at this bucket
        rest_decay: decayed resting depth (passive decay when no events)

    Derivatives (dt-normalized via exponential decay EMA):
        v_add, v_pull, v_fill, v_rest_depth: velocity (1st derivative)
        a_add, a_pull, a_fill, a_rest_depth: acceleration (2nd derivative)
        j_add, j_pull, j_fill, j_rest_depth: jerk (3rd derivative)

    Force:
        pressure_variant: derivative-led directional force
        vacuum_variant: pull-driven vacuum signal
        resistance_variant: rest-depth-derived resistance

    Metadata:
        last_event_id: id of last event that touched this bucket
        cell_valid: always True once initialized
        last_ts_ns: timestamp of last event at this bucket (for dt computation)
    """
    # Mechanics
    add_mass: float = 0.0
    pull_mass: float = 0.0
    fill_mass: float = 0.0
    rest_depth: float = 0.0

    # Velocity (1st derivative)
    v_add: float = 0.0
    v_pull: float = 0.0
    v_fill: float = 0.0
    v_rest_depth: float = 0.0

    # Acceleration (2nd derivative)
    a_add: float = 0.0
    a_pull: float = 0.0
    a_fill: float = 0.0
    a_rest_depth: float = 0.0

    # Jerk (3rd derivative)
    j_add: float = 0.0
    j_pull: float = 0.0
    j_fill: float = 0.0
    j_rest_depth: float = 0.0

    # Force
    pressure_variant: float = 0.0
    vacuum_variant: float = 0.0
    resistance_variant: float = 0.0

    # Metadata
    last_event_id: int = 0
    cell_valid: bool = True
    last_ts_ns: int = 0

    def to_dict(self, k: int) -> Dict[str, Any]:
        """Serialize to a flat dict with bucket index prefix.

        Args:
            k: Relative tick index of this bucket.

        Returns:
            Dict with all state fields, keyed by field name.
        """
        return {
            "k": k,
            "add_mass": self.add_mass,
            "pull_mass": self.pull_mass,
            "fill_mass": self.fill_mass,
            "rest_depth": self.rest_depth,
            "v_add": self.v_add,
            "v_pull": self.v_pull,
            "v_fill": self.v_fill,
            "v_rest_depth": self.v_rest_depth,
            "a_add": self.a_add,
            "a_pull": self.a_pull,
            "a_fill": self.a_fill,
            "a_rest_depth": self.a_rest_depth,
            "j_add": self.j_add,
            "j_pull": self.j_pull,
            "j_fill": self.j_fill,
            "j_rest_depth": self.j_rest_depth,
            "pressure_variant": self.pressure_variant,
            "vacuum_variant": self.vacuum_variant,
            "resistance_variant": self.resistance_variant,
            "last_event_id": self.last_event_id,
            "cell_valid": self.cell_valid,
        }


def _new_bucket() -> BucketState:
    """Create a fresh bucket with all-zero finite defaults."""
    return BucketState()


# ---------------------------------------------------------------------------
# Internal order state
# ---------------------------------------------------------------------------

@dataclass
class _OrderEntry:
    """Minimal order state for internal book tracking."""
    side: str
    price_int: int
    qty: int


# ---------------------------------------------------------------------------
# Derivative chain helpers
# ---------------------------------------------------------------------------

def _ema_alpha(dt_s: float, tau: float) -> float:
    """Compute EMA blending factor for variable time intervals.

    Uses the continuous-time formula:
        alpha = 1 - exp(-dt / tau)

    This properly handles irregular inter-event times. When dt is small
    relative to tau, alpha ~ dt/tau (standard EMA behavior). When dt is
    large, alpha -> 1 (new value dominates, old state forgotten).

    Args:
        dt_s: Time delta in seconds (must be >= 0).
        tau: Time constant in seconds (must be > 0).

    Returns:
        Alpha in [0, 1].
    """
    if dt_s <= 0.0:
        return 0.0
    ratio = dt_s / tau
    # Clamp to avoid exp underflow for very large dt
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
    """Update a three-level derivative chain with dt normalization.

    Computes velocity (1st derivative), acceleration (2nd), and jerk (3rd)
    of a mechanics quantity using exponential-decay EMA for each level.

    The instantaneous rate is (new_value - prev_value) / dt, then smoothed
    through the EMA chain.

    Args:
        prev_value: Previous mechanics value.
        new_value: Current mechanics value.
        dt_s: Time delta in seconds.
        v_prev: Previous velocity EMA.
        a_prev: Previous acceleration EMA.
        j_prev: Previous jerk EMA.
        tau_v: Time constant for velocity EMA.
        tau_a: Time constant for acceleration EMA.
        tau_j: Time constant for jerk EMA.

    Returns:
        (v_new, a_new, j_new): Updated derivative chain values.
    """
    if dt_s <= 0.0:
        return v_prev, a_prev, j_prev

    # Instantaneous rate of change
    rate = (new_value - prev_value) / dt_s

    # Velocity: EMA of rate
    alpha_v = _ema_alpha(dt_s, tau_v)
    v_new = alpha_v * rate + (1.0 - alpha_v) * v_prev

    # Acceleration: EMA of velocity change rate
    dv_rate = (v_new - v_prev) / dt_s
    alpha_a = _ema_alpha(dt_s, tau_a)
    a_new = alpha_a * dv_rate + (1.0 - alpha_a) * a_prev

    # Jerk: EMA of acceleration change rate
    da_rate = (a_new - a_prev) / dt_s
    alpha_j = _ema_alpha(dt_s, tau_j)
    j_new = alpha_j * da_rate + (1.0 - alpha_j) * j_prev

    return v_new, a_new, j_new


def _compute_pressure_variant(bucket: BucketState) -> float:
    """Compute derivative-led pressure variant for a bucket.

    Formula:
        pressure_variant = c1*v_add + c2*v_fill - c3*v_pull
                         + c4*max(-a_rest_depth, 0)
                         + c5*j_flow

    where j_flow = j_add - j_pull + j_fill

    Positive pressure_variant indicates upward force (buying pressure).
    Negative indicates downward force (selling pressure).

    The sign semantics depend on the bucket side:
    - Bid-side bucket (k < 0): positive add velocity = buying pressure = bullish
    - Ask-side bucket (k > 0): positive add velocity = selling pressure = bearish
    The caller (or downstream consumer) interprets the sign in context.

    Args:
        bucket: Current bucket state.

    Returns:
        Finite float pressure variant value.
    """
    j_flow = bucket.j_add - bucket.j_pull + bucket.j_fill

    pv = (
        C1_V_ADD * bucket.v_add
        + C2_V_FILL * bucket.v_fill
        - C3_V_PULL * bucket.v_pull
        + C4_A_REST * max(-bucket.a_rest_depth, 0.0)
        + C5_J_FLOW * j_flow
    )
    return pv


def _compute_vacuum_variant(bucket: BucketState) -> float:
    """Compute vacuum variant (pull-driven liquidity drainage).

    vacuum_variant = v_pull - v_add (positive = draining = vacuum)
    """
    return bucket.v_pull - bucket.v_add


def _compute_resistance_variant(bucket: BucketState) -> float:
    """Compute resistance variant from resting depth.

    Uses log-compressed rest_depth as the resistance signal.
    This is a conditioning term (raw level), not derivative-led,
    but the spec permits raw levels as conditioning terms.

    Returns:
        Non-negative resistance value.
    """
    return math.log1p(max(bucket.rest_depth, 0.0))


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class EventDrivenVPEngine:
    """Canonical event-driven vacuum/pressure engine.

    Processes MBO events one at a time. Maintains internal order book
    state, spot reference, and per-bucket derivative chains. Emits a
    dense grid snapshot after every event.

    Args:
        K: Grid half-width in ticks. Grid spans [-K, +K] (2K+1 buckets).
        tick_int: Price integer units per tick (e.g. 250000000 for $0.25
            tick with PRICE_SCALE=1e-9).
        bucket_size_dollars: Size of each bucket in dollars (typically
            equals tick_size for futures).

    Example:
        >>> engine = EventDrivenVPEngine(K=40, tick_int=250000000)
        >>> grid = engine.update(
        ...     ts_ns=1707220800_000_000_000,
        ...     action="A", side="B",
        ...     price_int=21500_000_000_000,
        ...     size=1, order_id=12345, flags=0,
        ... )
        >>> len(grid["buckets"])  # 2*40+1 = 81
        81
    """

    def __init__(
        self,
        K: int = 40,
        tick_int: int = 250_000_000,
        bucket_size_dollars: float = 0.25,
    ) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if tick_int <= 0:
            raise ValueError(f"tick_int must be > 0, got {tick_int}")
        if bucket_size_dollars <= 0.0:
            raise ValueError(
                f"bucket_size_dollars must be > 0, got {bucket_size_dollars}"
            )

        self.K: int = K
        self.tick_int: int = tick_int
        self.bucket_size_dollars: float = bucket_size_dollars

        # Grid: dict mapping k -> BucketState, always has 2K+1 entries
        self._grid: Dict[int, BucketState] = {}
        self._init_grid()

        # Internal order book: order_id -> _OrderEntry
        self._orders: Dict[int, _OrderEntry] = {}

        # Depth tracking: price_int -> total qty (bid side and ask side separate)
        self._depth_bid: Dict[int, int] = {}
        self._depth_ask: Dict[int, int] = {}

        # Spot reference (price_int of mid)
        self._spot_ref_price_int: int = 0

        # Event counter (monotonic, used as event_id for provenance)
        self._event_counter: int = 0

        # Previous event timestamp for global dt
        self._prev_ts_ns: int = 0

        # Book validity
        self._book_valid: bool = False
        self._snapshot_in_progress: bool = False

        # Best bid/ask cached for fast mid computation
        self._best_bid: int = 0
        self._best_ask: int = 0

    def _init_grid(self) -> None:
        """Initialize grid with 2K+1 buckets, all with finite zero defaults."""
        self._grid = {k: _new_bucket() for k in range(-self.K, self.K + 1)}

    def _bucket_count(self) -> int:
        """Expected number of buckets."""
        return 2 * self.K + 1

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
    ) -> Dict[str, Any]:
        """Process one MBO event and return dense grid snapshot.

        This is the single entry point for both replay and live feeds.
        Every call advances engine state exactly once.

        Args:
            ts_ns: Event timestamp in nanoseconds.
            action: MBO action code (A/C/M/R/T/F).
            side: Side code (A=ask, B=bid).
            price_int: Price in integer units (price_dollars = price_int * 1e-9).
            size: Order quantity.
            order_id: Unique order identifier.
            flags: Databento flags bitmask.

        Returns:
            Dict with keys:
                - ``ts_ns``: Event timestamp.
                - ``event_id``: Monotonic event counter.
                - ``spot_ref_price_int``: Current spot reference.
                - ``mid_price``: Mid price in dollars.
                - ``best_bid_price_int``: Best bid price_int.
                - ``best_ask_price_int``: Best ask price_int.
                - ``book_valid``: Whether order book is valid.
                - ``buckets``: List of 2K+1 bucket dicts, sorted by k.
                - ``touched_k``: Set of k values touched by this event.
        """
        self._event_counter += 1
        event_id = self._event_counter
        is_snapshot = (flags & F_SNAPSHOT) != 0

        # --- Handle snapshot lifecycle ---
        if action == ACTION_CLEAR and is_snapshot:
            self._snapshot_in_progress = True
        if self._snapshot_in_progress and (flags & F_LAST) != 0:
            self._snapshot_in_progress = False
            self._book_valid = True
        if not self._book_valid and not self._snapshot_in_progress:
            self._book_valid = True

        # --- Compute dt ---
        dt_s = 0.0
        if self._prev_ts_ns > 0 and ts_ns > self._prev_ts_ns:
            dt_s = (ts_ns - self._prev_ts_ns) / 1e9
        self._prev_ts_ns = ts_ns

        # --- Apply event to internal order book ---
        # Track (price_int, event_side) pairs for bucket updates
        touched_info: List[Tuple[int, str, float, float, float]] = []
        # Each entry: (price_int, order_side, add_delta, pull_delta, fill_delta)

        was_snapshot_in_progress = self._snapshot_in_progress

        if action == ACTION_CLEAR:
            self._clear_book()
        elif action == ACTION_TRADE:
            pass  # Trades don't modify order book state directly
        elif action == ACTION_ADD:
            self._add_order(side, price_int, size, order_id)
            touched_info.append(
                (price_int, side, float(size), 0.0, 0.0)
            )
        elif action == ACTION_CANCEL:
            old_entry = self._orders.get(order_id)
            if old_entry is not None:
                cancel_side = old_entry.side
                cancel_price = old_entry.price_int
                cancel_size = old_entry.qty
                self._cancel_order(order_id)
                touched_info.append(
                    (cancel_price, cancel_side, 0.0, float(cancel_size), 0.0)
                )
            else:
                self._cancel_order(order_id)
        elif action == ACTION_MODIFY:
            old_entry = self._orders.get(order_id)
            if old_entry is not None:
                old_price = old_entry.price_int
                old_side = old_entry.side
                old_size = old_entry.qty
                self._modify_order(order_id, side, price_int, size)
                # If price changed, treat as pull from old + add to new
                if old_price != price_int:
                    touched_info.append(
                        (old_price, old_side, 0.0, float(old_size), 0.0)
                    )
                    touched_info.append(
                        (price_int, side, float(size), 0.0, 0.0)
                    )
                else:
                    # Same price: size delta
                    size_diff = size - old_size
                    if size_diff > 0:
                        touched_info.append(
                            (price_int, side, float(size_diff), 0.0, 0.0)
                        )
                    elif size_diff < 0:
                        touched_info.append(
                            (price_int, side, 0.0, float(-size_diff), 0.0)
                        )
                    else:
                        # Size unchanged but event happened -- touch for provenance
                        touched_info.append(
                            (price_int, side, 0.0, 0.0, 0.0)
                        )
            else:
                self._modify_order(order_id, side, price_int, size)
        elif action == ACTION_FILL:
            old_entry = self._orders.get(order_id)
            if old_entry is not None:
                fill_side = old_entry.side
                fill_price = old_entry.price_int
                self._fill_order(order_id, size)
                touched_info.append(
                    (fill_price, fill_side, 0.0, 0.0, float(size))
                )
            else:
                self._fill_order(order_id, size)

        # --- Detect snapshot completion ---
        snapshot_just_completed = (
            was_snapshot_in_progress and not self._snapshot_in_progress
        )

        # --- Update BBO and spot ---
        self._update_bbo()
        new_spot = self._compute_spot()

        # --- Handle spot shift ---
        touched_k: set[int] = set()
        spot_just_established = (
            new_spot > 0 and self._spot_ref_price_int == 0
        )
        if new_spot > 0 and self._spot_ref_price_int > 0:
            shift_ticks = round(
                (new_spot - self._spot_ref_price_int) / self.tick_int
            )
            if shift_ticks != 0:
                self._shift_grid(int(shift_ticks))
        if new_spot > 0:
            self._spot_ref_price_int = new_spot

        # --- Sync rest_depth from full book after snapshot or first spot ---
        if (snapshot_just_completed or spot_just_established) and self._spot_ref_price_int > 0:
            self.sync_rest_depth_from_book()

        # --- Map touched prices to bucket indices and update mechanics ---
        for p, order_side, add_delta, pull_delta, fill_delta in touched_info:
            if self._spot_ref_price_int <= 0:
                continue
            k = self._price_to_k(p)
            if k is None:
                continue
            touched_k.add(k)
            bucket = self._grid[k]

            # Compute rest_depth at this price level (sum both sides)
            depth_at_price = self._total_depth_at_price(p)

            # Compute dt for this specific bucket
            bucket_dt_s = 0.0
            if bucket.last_ts_ns > 0 and ts_ns > bucket.last_ts_ns:
                bucket_dt_s = (ts_ns - bucket.last_ts_ns) / 1e9

            # Save previous mechanics for derivative computation
            prev_add = bucket.add_mass
            prev_pull = bucket.pull_mass
            prev_fill = bucket.fill_mass
            prev_rest = bucket.rest_depth

            # Update mechanics (cumulative with passive decay)
            if bucket_dt_s > 0.0:
                decay = math.exp(-bucket_dt_s / TAU_REST_DECAY)
                bucket.add_mass = bucket.add_mass * decay + add_delta
                bucket.pull_mass = bucket.pull_mass * decay + pull_delta
                bucket.fill_mass = bucket.fill_mass * decay + fill_delta
            else:
                bucket.add_mass += add_delta
                bucket.pull_mass += pull_delta
                bucket.fill_mass += fill_delta

            # rest_depth is a snapshot (semi-additive: last across time)
            bucket.rest_depth = float(depth_at_price)

            # Update derivative chains
            if bucket_dt_s > 0.0:
                bucket.v_add, bucket.a_add, bucket.j_add = (
                    _update_derivative_chain(
                        prev_add, bucket.add_mass, bucket_dt_s,
                        bucket.v_add, bucket.a_add, bucket.j_add,
                    )
                )
                bucket.v_pull, bucket.a_pull, bucket.j_pull = (
                    _update_derivative_chain(
                        prev_pull, bucket.pull_mass, bucket_dt_s,
                        bucket.v_pull, bucket.a_pull, bucket.j_pull,
                    )
                )
                bucket.v_fill, bucket.a_fill, bucket.j_fill = (
                    _update_derivative_chain(
                        prev_fill, bucket.fill_mass, bucket_dt_s,
                        bucket.v_fill, bucket.a_fill, bucket.j_fill,
                    )
                )
                bucket.v_rest_depth, bucket.a_rest_depth, bucket.j_rest_depth = (
                    _update_derivative_chain(
                        prev_rest, bucket.rest_depth, bucket_dt_s,
                        bucket.v_rest_depth, bucket.a_rest_depth,
                        bucket.j_rest_depth,
                    )
                )

            # Recompute force variables
            bucket.pressure_variant = _compute_pressure_variant(bucket)
            bucket.vacuum_variant = _compute_vacuum_variant(bucket)
            bucket.resistance_variant = _compute_resistance_variant(bucket)

            # Mark provenance
            bucket.last_event_id = event_id
            bucket.last_ts_ns = ts_ns

        # --- Build output ---
        mid_price = 0.0
        if self._best_bid > 0 and self._best_ask > 0:
            mid_price = (self._best_bid + self._best_ask) * 0.5 * PRICE_SCALE

        buckets_out: List[Dict[str, Any]] = []
        for k in range(-self.K, self.K + 1):
            b = self._grid[k]
            d = b.to_dict(k)
            buckets_out.append(d)

        return {
            "ts_ns": ts_ns,
            "event_id": event_id,
            "spot_ref_price_int": self._spot_ref_price_int,
            "mid_price": mid_price,
            "best_bid_price_int": self._best_bid,
            "best_ask_price_int": self._best_ask,
            "book_valid": self._book_valid,
            "buckets": buckets_out,
            "touched_k": touched_k,
        }

    # ------------------------------------------------------------------
    # Order book operations
    # ------------------------------------------------------------------

    def _clear_book(self) -> None:
        """Clear all orders and depth (snapshot/clear event)."""
        self._orders.clear()
        self._depth_bid.clear()
        self._depth_ask.clear()
        self._best_bid = 0
        self._best_ask = 0
        self._book_valid = False
        self._snapshot_in_progress = True

    def _add_order(
        self, side: str, price_int: int, size: int, order_id: int
    ) -> None:
        """Add a new order to the internal book."""
        self._orders[order_id] = _OrderEntry(
            side=side, price_int=price_int, qty=size
        )
        depth = self._depth_bid if side == SIDE_BID else self._depth_ask
        depth[price_int] = depth.get(price_int, 0) + size

    def _cancel_order(self, order_id: int) -> None:
        """Cancel (remove) an order from the internal book."""
        entry = self._orders.pop(order_id, None)
        if entry is None:
            return
        depth = (
            self._depth_bid if entry.side == SIDE_BID else self._depth_ask
        )
        cur = depth.get(entry.price_int, 0)
        new_qty = cur - entry.qty
        if new_qty <= 0:
            depth.pop(entry.price_int, None)
        else:
            depth[entry.price_int] = new_qty

    def _modify_order(
        self, order_id: int, side: str, price_int: int, size: int
    ) -> None:
        """Modify an existing order (cancel old, add new)."""
        old = self._orders.pop(order_id, None)
        if old is not None:
            depth = (
                self._depth_bid if old.side == SIDE_BID else self._depth_ask
            )
            cur = depth.get(old.price_int, 0)
            new_qty = cur - old.qty
            if new_qty <= 0:
                depth.pop(old.price_int, None)
            else:
                depth[old.price_int] = new_qty

        # Add the modified order
        self._orders[order_id] = _OrderEntry(
            side=side, price_int=price_int, qty=size
        )
        depth = self._depth_bid if side == SIDE_BID else self._depth_ask
        depth[price_int] = depth.get(price_int, 0) + size

    def _fill_order(self, order_id: int, fill_size: int) -> None:
        """Fill (partially or fully) an order."""
        entry = self._orders.get(order_id)
        if entry is None:
            return
        depth = (
            self._depth_bid if entry.side == SIDE_BID else self._depth_ask
        )
        # Reduce depth
        cur = depth.get(entry.price_int, 0)
        new_depth = cur - fill_size
        if new_depth <= 0:
            depth.pop(entry.price_int, None)
        else:
            depth[entry.price_int] = new_depth

        # Reduce order qty
        entry.qty -= fill_size
        if entry.qty <= 0:
            self._orders.pop(order_id, None)

    def _depth_at_price(self, price_int: int, side: str) -> int:
        """Get current depth at a specific price level for one side."""
        if side == SIDE_BID:
            return self._depth_bid.get(price_int, 0)
        return self._depth_ask.get(price_int, 0)

    def _total_depth_at_price(self, price_int: int) -> int:
        """Get total depth at a price level across both sides.

        In practice, a given price level will only have orders from one
        side (bids below spot, asks above). But during crossed markets
        or near the spread, summing both sides is the correct semantic.
        """
        return (
            self._depth_bid.get(price_int, 0)
            + self._depth_ask.get(price_int, 0)
        )

    # ------------------------------------------------------------------
    # BBO / Spot
    # ------------------------------------------------------------------

    def _update_bbo(self) -> None:
        """Update cached best bid/ask from depth maps."""
        if self._depth_bid:
            self._best_bid = max(self._depth_bid.keys())
        else:
            self._best_bid = 0

        if self._depth_ask:
            self._best_ask = min(self._depth_ask.keys())
        else:
            self._best_ask = 0

    def _compute_spot(self) -> int:
        """Compute spot reference as mid of BBO (in price_int).

        Returns 0 if either side is missing.
        """
        if self._best_bid > 0 and self._best_ask > 0:
            # Round to nearest tick
            raw_mid = (self._best_bid + self._best_ask) / 2.0
            return int(round(raw_mid / self.tick_int) * self.tick_int)
        return 0

    # ------------------------------------------------------------------
    # Grid management
    # ------------------------------------------------------------------

    def _price_to_k(self, price_int: int) -> Optional[int]:
        """Map an absolute price to a relative tick index k.

        Returns None if the price falls outside the grid.
        """
        if self._spot_ref_price_int <= 0:
            return None
        k = round((price_int - self._spot_ref_price_int) / self.tick_int)
        if -self.K <= k <= self.K:
            return int(k)
        return None

    def _shift_grid(self, shift_ticks: int) -> None:
        """Shift grid state by shift_ticks positions.

        When spot moves by N ticks, existing bucket state at index k
        moves to index k - N (because the reference frame shifted by +N).
        New edge buckets are initialized with finite defaults.

        Args:
            shift_ticks: Number of ticks spot moved (positive = spot moved up).
        """
        if shift_ticks == 0:
            return

        old_grid = self._grid
        new_grid: Dict[int, BucketState] = {}

        for new_k in range(-self.K, self.K + 1):
            # The bucket that was at old_k = new_k + shift now maps to new_k
            old_k = new_k + shift_ticks
            if -self.K <= old_k <= self.K and old_k in old_grid:
                new_grid[new_k] = old_grid[old_k]
            else:
                new_grid[new_k] = _new_bucket()

        self._grid = new_grid

    # ------------------------------------------------------------------
    # Bulk rest_depth sync (call sparingly, e.g. after snapshot)
    # ------------------------------------------------------------------

    def sync_rest_depth_from_book(self) -> None:
        """Synchronize all grid bucket rest_depth values from current order book.

        Useful after snapshot completion to initialize rest_depth from the
        full book state. For normal event processing, rest_depth is updated
        incrementally at touched buckets only.
        """
        if self._spot_ref_price_int <= 0:
            return

        # Zero out all rest_depth first
        for b in self._grid.values():
            b.rest_depth = 0.0

        # Bid side
        for price_int, qty in self._depth_bid.items():
            k = self._price_to_k(price_int)
            if k is not None:
                self._grid[k].rest_depth += float(qty)

        # Ask side
        for price_int, qty in self._depth_ask.items():
            k = self._price_to_k(price_int)
            if k is not None:
                self._grid[k].rest_depth += float(qty)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def spot_ref_price_int(self) -> int:
        """Current spot reference in price_int units."""
        return self._spot_ref_price_int

    @property
    def book_valid(self) -> bool:
        """Whether the internal order book is in a valid state."""
        return self._book_valid

    @property
    def order_count(self) -> int:
        """Number of live orders in the internal book."""
        return len(self._orders)

    @property
    def event_count(self) -> int:
        """Total events processed."""
        return self._event_counter

    def grid_snapshot_arrays(self) -> Dict[str, np.ndarray]:
        """Return grid state as numpy arrays for efficient downstream use.

        Returns:
            Dict with keys matching BucketState fields, each a 1D array
            of length 2K+1, indexed by k from -K to +K.
        """
        n = self._bucket_count()
        ks = np.arange(-self.K, self.K + 1, dtype=np.int32)

        arrays: Dict[str, np.ndarray] = {
            "k": ks,
            "add_mass": np.empty(n, dtype=np.float64),
            "pull_mass": np.empty(n, dtype=np.float64),
            "fill_mass": np.empty(n, dtype=np.float64),
            "rest_depth": np.empty(n, dtype=np.float64),
            "v_add": np.empty(n, dtype=np.float64),
            "v_pull": np.empty(n, dtype=np.float64),
            "v_fill": np.empty(n, dtype=np.float64),
            "v_rest_depth": np.empty(n, dtype=np.float64),
            "a_add": np.empty(n, dtype=np.float64),
            "a_pull": np.empty(n, dtype=np.float64),
            "a_fill": np.empty(n, dtype=np.float64),
            "a_rest_depth": np.empty(n, dtype=np.float64),
            "j_add": np.empty(n, dtype=np.float64),
            "j_pull": np.empty(n, dtype=np.float64),
            "j_fill": np.empty(n, dtype=np.float64),
            "j_rest_depth": np.empty(n, dtype=np.float64),
            "pressure_variant": np.empty(n, dtype=np.float64),
            "vacuum_variant": np.empty(n, dtype=np.float64),
            "resistance_variant": np.empty(n, dtype=np.float64),
            "last_event_id": np.empty(n, dtype=np.int64),
        }

        for i, k in enumerate(range(-self.K, self.K + 1)):
            b = self._grid[k]
            arrays["add_mass"][i] = b.add_mass
            arrays["pull_mass"][i] = b.pull_mass
            arrays["fill_mass"][i] = b.fill_mass
            arrays["rest_depth"][i] = b.rest_depth
            arrays["v_add"][i] = b.v_add
            arrays["v_pull"][i] = b.v_pull
            arrays["v_fill"][i] = b.v_fill
            arrays["v_rest_depth"][i] = b.v_rest_depth
            arrays["a_add"][i] = b.a_add
            arrays["a_pull"][i] = b.a_pull
            arrays["a_fill"][i] = b.a_fill
            arrays["a_rest_depth"][i] = b.a_rest_depth
            arrays["j_add"][i] = b.j_add
            arrays["j_pull"][i] = b.j_pull
            arrays["j_fill"][i] = b.j_fill
            arrays["j_rest_depth"][i] = b.j_rest_depth
            arrays["pressure_variant"][i] = b.pressure_variant
            arrays["vacuum_variant"][i] = b.vacuum_variant
            arrays["resistance_variant"][i] = b.resistance_variant
            arrays["last_event_id"][i] = b.last_event_id

        return arrays
