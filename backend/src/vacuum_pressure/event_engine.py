"""Event-driven vacuum/pressure engine — two-force model.

Canonical event-time force-field engine over relative price buckets around spot.
Every MBO order book event updates pressure-state math. Live and replay run
identical code paths (only the event source differs).

Architecture:
    1. Maintain internal order book (orders by order_id -> {price_int, size, side}).
    2. Track spot = mid(BBO) from internal book state.
    3. Dense grid of 2K+1 buckets centered on spot. Each bucket carries:
       - mechanics: add_mass, pull_mass, fill_mass, rest_depth
       - derivatives: v_*, a_*, j_* (dt-normalized via exponential decay)
       - force: pressure_variant (building), vacuum_variant (draining)
       - metadata: last_event_id, cell_valid
    4. On each event: resolve bucket, update mechanics, decay-update derivatives,
       recompute force, emit dense grid.
    5. On spot shift: translate grid state by shift amount, fill edge buckets
       with finite defaults.

Two-force model:
    Only three market states matter:
        1. Vacuum above + Pressure below → price goes up
        2. Pressure above + Vacuum below → price goes down
        3. Weak/balanced → chop

    "Pressure above spot" IS resistance. "Pressure below spot" IS support.
    A separate resistance variant (formerly log1p(rest_depth)) is redundant —
    it's a static level in a derivative-led system.

    Pressure (depth BUILDING — liquidity arriving/replenishing):
        pressure_variant_k = c1*v_add_k
                           + c2*max(v_rest_depth_k, 0)
                           + c3*max(a_add_k, 0)

    Vacuum (depth DRAINING — liquidity removed/consumed):
        vacuum_variant_k = c4*v_pull_k
                         + c5*v_fill_k
                         + c6*max(-v_rest_depth_k, 0)
                         + c7*max(a_pull_k, 0)

    Both forces are non-negative (magnitude of building/draining).
    Fills belong to VACUUM: they drain the passive side's depth.

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

Guarantees:
    G1: For each event, engine state advances once and recomputes force variants.
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

# Two-force model coefficients — Pressure (depth building)
C1_V_ADD: float = 1.0
"""Weight for v_add in pressure. Primary signal: rate of new order arrival."""

C2_V_REST_POS: float = 0.5
"""Weight for max(v_rest_depth, 0) in pressure. Conditioning: net depth
growth confirms building. Half-weight vs primary flow derivative."""

C3_A_ADD: float = 0.3
"""Weight for max(a_add, 0) in pressure. Detects intensifying adds.
Noisier than velocity → lower weight (~5-8% of total signal)."""

# Two-force model coefficients — Vacuum (depth draining)
C4_V_PULL: float = 1.0
"""Weight for v_pull in vacuum. Primary drain: cancellation rate."""

C5_V_FILL: float = 1.5
"""Weight for v_fill in vacuum. Fills are realized executions with zero
spoof risk. Higher weight rewards information quality and compensates
for lower frequency relative to pulls."""

C6_V_REST_NEG: float = 0.5
"""Weight for max(-v_rest_depth, 0) in vacuum. Conditioning: net depth
shrinkage confirms draining. Half-weight vs primary flow derivative."""

C7_A_PULL: float = 0.3
"""Weight for max(a_pull, 0) in vacuum. Detects intensifying pulls.
Same noise rationale as C3_A_ADD."""

# Rest depth exponential decay time constant (seconds)
TAU_REST_DECAY: float = 30.0
"""Time constant for passive rest_depth decay (absent direct events)."""

TAU_FRESHNESS: float = 10.0
"""Time constant for stale-bucket force/derivative discount in grid output.

Untouched buckets retain frozen derivative and force values (Guarantee G4).
When emitting the grid, we apply exp(-age / TAU_FRESHNESS) to derivative-led
fields so stale edge buckets don't display phantom pressure indefinitely.
"""


# ---------------------------------------------------------------------------
# Per-bucket state
# ---------------------------------------------------------------------------

@dataclass
class BucketState:
    """Full per-bucket state model (two-force variant).

    All fields initialized to finite defaults (zero). No field is ever
    null/NaN/Inf by construction.

    Mechanics:
        add_mass: cumulative add quantity (decayed)
        pull_mass: cumulative pull quantity (decayed)
        fill_mass: cumulative fill quantity (decayed)
        rest_depth: current resting depth at this bucket

    Derivatives (dt-normalized via exponential decay EMA):
        v_add, v_pull, v_fill, v_rest_depth: velocity (1st derivative)
        a_add, a_pull, a_fill, a_rest_depth: acceleration (2nd derivative)
        j_add, j_pull, j_fill, j_rest_depth: jerk (3rd derivative)

    Force (two-force model):
        pressure_variant: non-negative magnitude of depth building
            pressure = c1*v_add + c2*max(v_rest_depth, 0) + c3*max(a_add, 0)
        vacuum_variant: non-negative magnitude of depth draining
            vacuum = c4*v_pull + c5*v_fill + c6*max(-v_rest_depth, 0)
                   + c7*max(a_pull, 0)

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

    # Force (two-force model: both non-negative)
    pressure_variant: float = 0.0
    vacuum_variant: float = 0.0

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
    """Update a three-level derivative chain from value changes.

    Computes velocity (1st derivative), acceleration (2nd), and jerk (3rd)
    using exponential-decay EMA for each level.  The instantaneous rate is
    (new_value - prev_value) / dt, then smoothed through the EMA chain.

    Use this for **snapshot** quantities (e.g. rest_depth) where the rate of
    change of the signal itself is the correct input.

    For **decayed accumulators** (add_mass, pull_mass, fill_mass) use
    ``_update_derivative_chain_from_delta`` instead to avoid conflating
    passive decay with market activity.

    Returns:
        (v_new, a_new, j_new): Updated derivative chain values.
        Guaranteed finite (falls back to prev on non-finite result).
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

    # G3: guarantee finite outputs
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

    Unlike ``_update_derivative_chain``, this takes the **raw event delta**
    (quantity added/pulled/filled at this bucket) rather than the change in a
    decayed accumulator.  This separates passive exponential decay from the
    derivative signal, ensuring that velocity measures *market activity rate*
    rather than *rate of change of a decaying signal*.

    When delta=0 and dt_s>0, the EMA decays toward zero (no activity).
    When delta>0, the activity rate delta/dt_s is blended in.

    Args:
        delta: Raw event quantity (add_delta, pull_delta, or fill_delta).
        dt_s: Time delta in seconds (must be > 0 for any update).
        v_prev, a_prev, j_prev: Previous derivative chain values.
        tau_v, tau_a, tau_j: Time constants for each derivative level.

    Returns:
        (v_new, a_new, j_new): Updated derivative chain values.
        Guaranteed finite (falls back to prev on non-finite result).
    """
    if dt_s <= 0.0:
        return v_prev, a_prev, j_prev

    # Activity rate: raw delta per unit time
    rate = delta / dt_s

    alpha_v = _ema_alpha(dt_s, tau_v)
    v_new = alpha_v * rate + (1.0 - alpha_v) * v_prev

    dv_rate = (v_new - v_prev) / dt_s
    alpha_a = _ema_alpha(dt_s, tau_a)
    a_new = alpha_a * dv_rate + (1.0 - alpha_a) * a_prev

    da_rate = (a_new - a_prev) / dt_s
    alpha_j = _ema_alpha(dt_s, tau_j)
    j_new = alpha_j * da_rate + (1.0 - alpha_j) * j_prev

    # G3: guarantee finite outputs
    if not (math.isfinite(v_new) and math.isfinite(a_new) and math.isfinite(j_new)):
        return v_prev, a_prev, j_prev

    return v_new, a_new, j_new


def _compute_pressure(bucket: BucketState) -> float:
    """Compute pressure force (depth BUILDING — liquidity arriving).

    Pressure measures the rate at which depth is being built at a price
    level, capturing the intention of participants to defend/establish it.

    Formula:
        pressure = c1*v_add + c2*max(v_rest_depth, 0) + c3*max(a_add, 0)

    Components:
        v_add:               velocity of add_mass (rate of new orders arriving)
        max(v_rest_depth, 0): positive velocity of rest_depth (depth growing)
        max(a_add, 0):       acceleration of add activity (is adding intensifying?)

    Directional interpretation depends on bucket position relative to spot:
        k < 0 (below spot): pressure = support being built   (bullish)
        k > 0 (above spot): pressure = resistance being built (bearish)

    Returns:
        Non-negative float. Zero when no building activity.
    """
    return (
        C1_V_ADD * bucket.v_add
        + C2_V_REST_POS * max(bucket.v_rest_depth, 0.0)
        + C3_A_ADD * max(bucket.a_add, 0.0)
    )


def _compute_vacuum(bucket: BucketState) -> float:
    """Compute vacuum force (depth DRAINING — liquidity removed/consumed).

    Vacuum measures the rate at which depth is being drained from a price
    level, capturing passive withdrawal (market makers pulling) and
    aggressive consumption (fills eating through depth).

    Formula:
        vacuum = c4*v_pull + c5*v_fill + c6*max(-v_rest_depth, 0)
               + c7*max(a_pull, 0)

    Components:
        v_pull:                velocity of pull_mass (rate of cancellations)
        v_fill:                velocity of fill_mass (rate of depth consumed by trades)
        max(-v_rest_depth, 0): negative velocity of rest_depth (depth shrinking)
        max(a_pull, 0):        acceleration of pull activity (pulling intensifying?)

    Why fills belong to VACUUM:
        Fills drain the passive side's depth.  When aggressive buyers fill
        resting asks at k > 0:
            Old model: fills → pressure > 0 at k > 0 → frontend reads as bearish (WRONG)
            New model: fills → vacuum > 0 at k > 0 → "depth draining above" → bullish (CORRECT)
        This fixes the D4 fill-attribution problem.

    Directional interpretation depends on bucket position relative to spot:
        k < 0 (below spot): vacuum = support draining away  (bearish)
        k > 0 (above spot): vacuum = resistance being eaten  (bullish)

    Returns:
        Non-negative float. Zero when no draining activity.
    """
    return (
        C4_V_PULL * bucket.v_pull
        + C5_V_FILL * bucket.v_fill
        + C6_V_REST_NEG * max(-bucket.v_rest_depth, 0.0)
        + C7_A_PULL * max(bucket.a_pull, 0.0)
    )


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

        # Capture BEFORE lifecycle transitions (B1 fix: enables
        # snapshot_just_completed detection on the F_LAST event).
        was_snapshot_in_progress = self._snapshot_in_progress

        # --- Handle snapshot lifecycle ---
        if action == ACTION_CLEAR and is_snapshot:
            self._snapshot_in_progress = True
        if self._snapshot_in_progress and (flags & F_LAST) != 0:
            self._snapshot_in_progress = False
            self._book_valid = True
        if not self._book_valid and not self._snapshot_in_progress:
            self._book_valid = True

        # --- Compute dt ---
        if self._prev_ts_ns > 0 and ts_ns > self._prev_ts_ns:
            pass  # per-bucket dt is used below; global dt removed (was dead code)
        self._prev_ts_ns = ts_ns

        # --- Apply event to internal order book ---
        # Track (price_int, event_side) pairs for bucket updates
        touched_info: List[Tuple[int, str, float, float, float]] = []
        # Each entry: (price_int, order_side, add_delta, pull_delta, fill_delta)

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

            # B3 fix: aggregate rest_depth across ALL price levels that
            # map to this bucket (correct for equities where one bucket
            # spans multiple penny tick levels).
            bucket_rest_depth = self._aggregate_bucket_rest_depth(k)

            # Compute dt for this specific bucket
            bucket_dt_s = 0.0
            if bucket.last_ts_ns > 0 and ts_ns > bucket.last_ts_ns:
                bucket_dt_s = (ts_ns - bucket.last_ts_ns) / 1e9

            # Save previous rest_depth for value-change derivative
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

            # rest_depth: full bucket aggregate (B3 fix)
            bucket.rest_depth = bucket_rest_depth

            # D1 fix: Use delta-based derivative chain for decayed
            # accumulators (add/pull/fill).  This separates passive
            # exponential decay from the derivative signal so velocity
            # measures *market activity rate* not *rate of change of a
            # decaying signal*.
            #
            # rest_depth uses value-change derivative (no decay artifact
            # because it's a snapshot value, not a decayed accumulator).
            if bucket_dt_s > 0.0:
                bucket.v_add, bucket.a_add, bucket.j_add = (
                    _update_derivative_chain_from_delta(
                        add_delta, bucket_dt_s,
                        bucket.v_add, bucket.a_add, bucket.j_add,
                    )
                )
                bucket.v_pull, bucket.a_pull, bucket.j_pull = (
                    _update_derivative_chain_from_delta(
                        pull_delta, bucket_dt_s,
                        bucket.v_pull, bucket.a_pull, bucket.j_pull,
                    )
                )
                bucket.v_fill, bucket.a_fill, bucket.j_fill = (
                    _update_derivative_chain_from_delta(
                        fill_delta, bucket_dt_s,
                        bucket.v_fill, bucket.a_fill, bucket.j_fill,
                    )
                )
                # rest_depth: value-change chain (no decay artifact)
                bucket.v_rest_depth, bucket.a_rest_depth, bucket.j_rest_depth = (
                    _update_derivative_chain(
                        prev_rest, bucket.rest_depth, bucket_dt_s,
                        bucket.v_rest_depth, bucket.a_rest_depth,
                        bucket.j_rest_depth,
                    )
                )

            # Recompute two-force model
            bucket.pressure_variant = _compute_pressure(bucket)
            bucket.vacuum_variant = _compute_vacuum(bucket)

            # Mark provenance
            bucket.last_event_id = event_id
            bucket.last_ts_ns = ts_ns

        # --- Build output with D2 freshness discount ---
        mid_price = 0.0
        if self._best_bid > 0 and self._best_ask > 0:
            mid_price = (self._best_bid + self._best_ask) * 0.5 * PRICE_SCALE

        _DERIVATIVE_FIELDS = (
            "v_add", "v_pull", "v_fill", "v_rest_depth",
            "a_add", "a_pull", "a_fill", "a_rest_depth",
            "j_add", "j_pull", "j_fill", "j_rest_depth",
            "pressure_variant", "vacuum_variant",
        )
        _MASS_FIELDS = ("add_mass", "pull_mass", "fill_mass")

        buckets_out: List[Dict[str, Any]] = []
        for k in range(-self.K, self.K + 1):
            b = self._grid[k]
            d = b.to_dict(k)

            # D2: Freshness discount for untouched buckets.  Derivative-led
            # fields decay toward zero; mass fields continue natural decay.
            # rest_depth is NOT discounted (raw level / conditioning term).
            if b.last_ts_ns > 0 and ts_ns > b.last_ts_ns:
                age_s = (ts_ns - b.last_ts_ns) / 1e9
                if age_s > 0.5:  # Only discount if meaningfully stale
                    freshness = math.exp(-age_s / TAU_FRESHNESS)
                    mass_decay = math.exp(-age_s / TAU_REST_DECAY)
                    for f in _DERIVATIVE_FIELDS:
                        d[f] = d[f] * freshness
                    for f in _MASS_FIELDS:
                        d[f] = d[f] * mass_decay

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
        """Clear all orders and depth (snapshot/clear event).

        Does NOT set ``_snapshot_in_progress``; that is managed by the
        snapshot lifecycle state machine in ``update()`` /
        ``apply_book_event()``.  This prevents non-snapshot Clear events
        (e.g. trading halts) from trapping the engine in permanent
        snapshot-in-progress state.
        """
        self._orders.clear()
        self._depth_bid.clear()
        self._depth_ask.clear()
        self._best_bid = 0
        self._best_ask = 0
        self._book_valid = False

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
        """Fill (partially or fully) an order.

        Clamps effective fill to the order's remaining qty to prevent
        depth corruption across sibling orders at the same price level
        in the event of a feed anomaly (fill_size > remaining qty).
        """
        entry = self._orders.get(order_id)
        if entry is None:
            return

        # Defensive clamp: never reduce depth by more than order's remaining qty
        effective_fill = min(fill_size, entry.qty)

        depth = (
            self._depth_bid if entry.side == SIDE_BID else self._depth_ask
        )
        cur = depth.get(entry.price_int, 0)
        new_depth = cur - effective_fill
        if new_depth <= 0:
            depth.pop(entry.price_int, None)
        else:
            depth[entry.price_int] = new_depth

        # Reduce order qty
        entry.qty -= effective_fill
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

        Uses floor(x + 0.5) for consistent half-up rounding instead of
        Python's round() which uses banker's round-half-to-even.  This
        avoids tick-boundary asymmetry for 1-tick spreads where the mid
        falls exactly on a half-tick.

        Returns 0 if either side is missing.
        """
        if self._best_bid > 0 and self._best_ask > 0:
            raw_mid = (self._best_bid + self._best_ask) / 2.0
            # Consistent half-up rounding (no banker's asymmetry)
            return int(math.floor(raw_mid / self.tick_int + 0.5)) * self.tick_int
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

    def _aggregate_bucket_rest_depth(self, k: int) -> float:
        """Compute total rest depth for bucket k across all mapped price levels.

        For futures (1 price level per bucket) this returns the same value as
        ``_total_depth_at_price(p)``.  For equities (many penny levels per
        bucket, e.g. 50 for $0.50 bucket / $0.01 tick), this correctly
        aggregates depth from every price_int that maps to the same bucket.
        """
        if self._spot_ref_price_int <= 0:
            return 0.0
        total = 0.0
        for price_int, qty in self._depth_bid.items():
            if self._price_to_k(price_int) == k:
                total += qty
        for price_int, qty in self._depth_ask.items():
            if self._price_to_k(price_int) == k:
                total += qty
        return total

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

        Updates the internal order book, BBO, and spot reference *without*
        computing grid mechanics, derivatives, or force variants.  This is
        10-50x faster than ``update()`` and is used to fast-forward through
        hours of pre-warmup events where correct book state is needed but
        VP grid output is not.

        After the last ``apply_book_event`` call (transition to VP mode),
        the caller should invoke ``sync_rest_depth_from_book()`` to
        initialize grid rest_depth from the fully-built book.

        Args:
            Same as ``update()``.
        """
        self._event_counter += 1
        is_snapshot = (flags & F_SNAPSHOT) != 0

        # Snapshot lifecycle (identical to update())
        was_snapshot_in_progress = self._snapshot_in_progress

        if action == ACTION_CLEAR and is_snapshot:
            self._snapshot_in_progress = True
        if self._snapshot_in_progress and (flags & F_LAST) != 0:
            self._snapshot_in_progress = False
            self._book_valid = True
        if not self._book_valid and not self._snapshot_in_progress:
            self._book_valid = True

        self._prev_ts_ns = ts_ns

        # Apply to order book
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
        # ACTION_TRADE: no book impact

        # Update BBO and spot reference (no grid operations).
        # We track spot to keep _spot_ref_price_int current so the
        # first full update() sees zero or minimal grid shift.
        self._update_bbo()
        new_spot = self._compute_spot()
        if new_spot > 0:
            self._spot_ref_price_int = new_spot

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
            arrays["last_event_id"][i] = b.last_event_id

        return arrays
