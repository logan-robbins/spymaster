"""
Barrier Engine: ES MBP-10 + Trades barrier physics.

Implements §5.1.1 of PLAN.md: L2/MBP-10 + Trades inference for cancel-vs-fill.

Physics:
- Track depth changes across MBP-10 levels near the critical level
- Infer FILLED vs PULLED by comparing depth_lost to Vpassive (trades)
- ABSORPTION/WALL when depth consumed but replenished
- VACUUM when depth pulled without fills
- CONSUMED when filled > canceled

Price:
- Input level_price is in ES points (no conversion)
- Output defending_quote.price is in ES points

No equity L1 fallback - ES MBP-10 is the only data source.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, TYPE_CHECKING

from src.common.event_types import FuturesTrade, MBP10, BidAskLevel, Aggressor
from src.common.config import CONFIG

if TYPE_CHECKING:
    from .market_state import MarketState


class BarrierState(str, Enum):
    """Classification of liquidity dynamics at a level"""
    VACUUM = "VACUUM"              # Liquidity pulled without fills
    WALL = "WALL"                  # Strong replenishment
    ABSORPTION = "ABSORPTION"      # Liquidity consumed but replenished
    CONSUMED = "CONSUMED"          # Liquidity eaten faster than replenished
    WEAK = "WEAK"                  # Defending size below baseline
    NEUTRAL = "NEUTRAL"            # Normal state


class Direction(str, Enum):
    """Test direction at a level"""
    SUPPORT = "SUPPORT"    # Spot > L, approaching from above, break = DOWN
    RESISTANCE = "RESISTANCE"  # Spot < L, approaching from below, break = UP


@dataclass
class BarrierMetrics:
    """Output of barrier physics computation"""
    state: BarrierState
    delta_liq: float           # Net liquidity change
    replenishment_ratio: float # R = added / (canceled + filled + ε)
    added_size: float
    canceled_size: float
    filled_size: float
    defending_quote: dict      # {price, size} - best defending level
    confidence: float          # 0-1, based on sample size and stability
    churn: float               # gross_added + gross_removed (activity measure)
    depth_in_zone: int         # Total depth in the monitoring zone


# ES futures tick size: $0.25 per tick
ES_TICK_SIZE = 0.25


class BarrierEngine:
    """
    Computes liquidity state at a level using ES MBP-10 + trades.

    Per §5.1.1 of PLAN.md:
    - Track depth(p, side, t) from MBP-10 state
    - Compute Δdepth = depth(t1) - depth(t0)
    - Track Vpassive = passive fills on defending side from trades
    - Infer: filled = min(depth_lost, Vpassive), pulled = max(0, depth_lost - Vpassive)
    - ABSORPTION when Δdepth >= 0 but Vpassive is positive
    """

    def __init__(self, config=None):
        """
        Args:
            config: Config object (defaults to global CONFIG)
        """
        self.config = config or CONFIG

        # Window for computing flows
        self.window_seconds = self.config.W_b

        # Zone around level (in ES ticks) - tight around strike-aligned levels
        self.zone_es_ticks = self.config.BARRIER_ZONE_ES_TICKS

        # Thresholds (adjusted for ES contract sizes)
        self.r_vacuum = self.config.R_vac
        self.r_wall = self.config.R_wall
        self.delta_liq_thresh = self.config.F_thresh

    def compute_barrier_state(
        self,
        level_price: float,
        direction: Direction,
        market_state: 'MarketState'
    ) -> BarrierMetrics:
        """
        Compute barrier state for a level using ES MBP-10 + trades.

        Args:
            level_price: The critical level being tested (ES points, e.g., 6870.0)
            direction: SUPPORT or RESISTANCE
            market_state: MarketState instance with ES MBP-10 and trade buffers

        Returns:
            BarrierMetrics with state classification and flows
        """
        ts_now_ns = market_state.get_current_ts_ns()

        # Get current MBP-10 snapshot
        current_mbp = market_state.get_es_mbp10_snapshot()
        if current_mbp is None:
            return self._neutral_metrics()

        # Get historical MBP-10 snapshots for flow computation
        mbp_history = market_state.get_es_mbp10_in_window(ts_now_ns, self.window_seconds)
        if len(mbp_history) < 2:
            return self._neutral_metrics()

        # Get ES trades in window (for passive fill detection)
        es_trades = market_state.get_es_trades_in_window(ts_now_ns, self.window_seconds)

        # ES system: level_price is already in ES points (no conversion needed)
        # ES strikes are 5pt ATM on expiry, wider farther OTM
        es_level_price = level_price  # ES futures = ES options

        # Zone is ±N ES ticks around the level (tight around strike)
        zone_es = self.zone_es_ticks * ES_TICK_SIZE  # e.g., ±2 ticks = ±$0.50 ES
        zone_low = es_level_price - zone_es
        zone_high = es_level_price + zone_es

        # Determine defending side
        if direction == Direction.SUPPORT:
            # Level is support, defending side is BID
            side = 'bid'
        else:
            # Level is resistance, defending side is ASK
            side = 'ask'

        # Compute depth flows over the window (all in ES terms)
        added_size, canceled_size, filled_size, churn = self._compute_depth_flows(
            mbp_history=mbp_history,
            trades=es_trades,
            level_price=es_level_price,
            zone_low=zone_low,
            zone_high=zone_high,
            side=side
        )

        # Get current defending depth in zone
        depth_in_zone = self._get_zone_depth(current_mbp, zone_low, zone_high, side)
        defending_quote_es = self._get_best_defending_quote(current_mbp, side)

        # ES system: price is already in ES points (no conversion needed)
        defending_quote = {
            "price": defending_quote_es["price"] if defending_quote_es["price"] else 0.0,  # ES = ES
            "size": defending_quote_es["size"]
        }

        # Compute metrics
        epsilon = 1e-6
        delta_liq = added_size - canceled_size - filled_size
        replenishment_ratio = added_size / (canceled_size + filled_size + epsilon)

        # Classify state
        state = self._classify_state(
            delta_liq=delta_liq,
            replenishment_ratio=replenishment_ratio,
            canceled_size=canceled_size,
            filled_size=filled_size,
            depth_in_zone=depth_in_zone,
            churn=churn
        )

        # Compute confidence based on activity
        total_activity = added_size + canceled_size + filled_size
        # ES contracts are smaller counts than SPY shares, scale differently
        confidence = min(1.0, total_activity / 500.0)

        return BarrierMetrics(
            state=state,
            delta_liq=delta_liq,
            replenishment_ratio=replenishment_ratio,
            added_size=added_size,
            canceled_size=canceled_size,
            filled_size=filled_size,
            defending_quote=defending_quote,
            confidence=confidence,
            churn=churn,
            depth_in_zone=depth_in_zone
        )

    def _neutral_metrics(self) -> BarrierMetrics:
        """Return neutral metrics when no data available."""
        return BarrierMetrics(
            state=BarrierState.NEUTRAL,
            delta_liq=0.0,
            replenishment_ratio=0.0,
            added_size=0.0,
            canceled_size=0.0,
            filled_size=0.0,
            defending_quote={"price": 0.0, "size": 0},
            confidence=0.0,
            churn=0.0,
            depth_in_zone=0
        )

    def _compute_depth_flows(
        self,
        mbp_history: List[MBP10],
        trades: List[FuturesTrade],
        level_price: float,
        zone_low: float,
        zone_high: float,
        side: str
    ) -> tuple:
        """
        Compute added, canceled, filled size by analyzing MBP-10 deltas and trades.

        Per §5.1.1:
        - gross_removed = Σ max(0, -Δdepth_update) across updates
        - gross_added = Σ max(0, +Δdepth_update) across updates
        - For each depth decrease, check passive fills to split filled vs pulled

        Returns:
            (added_size, canceled_size, filled_size, churn)
        """
        if len(mbp_history) < 2:
            return 0.0, 0.0, 0.0, 0.0

        added_size = 0.0
        canceled_size = 0.0
        filled_size = 0.0
        gross_added = 0.0
        gross_removed = 0.0

        # Build trade lookup by time window
        # Group trades by timestamp ranges for concurrent fill detection
        trade_index = 0

        for i in range(1, len(mbp_history)):
            prev_mbp = mbp_history[i - 1]
            curr_mbp = mbp_history[i]

            # Compute zone depth for prev and curr
            prev_depth = self._get_zone_depth(prev_mbp, zone_low, zone_high, side)
            curr_depth = self._get_zone_depth(curr_mbp, zone_low, zone_high, side)

            delta_depth = curr_depth - prev_depth

            if delta_depth > 0:
                # Depth increased -> ADDED
                added_size += delta_depth
                gross_added += delta_depth
            elif delta_depth < 0:
                # Depth decreased -> either FILLED or CANCELED
                depth_lost = abs(delta_depth)
                gross_removed += depth_lost

                # Get passive volume in this time window
                v_passive = self._get_passive_volume(
                    trades=trades,
                    start_ts=prev_mbp.ts_event_ns,
                    end_ts=curr_mbp.ts_event_ns,
                    zone_low=zone_low,
                    zone_high=zone_high,
                    side=side
                )

                # Per §5.1.1:
                # inferred_filled = min(depth_lost, Vpassive + ε_fill)
                # inferred_pulled = max(0, depth_lost - Vpassive - ε_pull)
                epsilon_fill = 1  # small tolerance for timing misalignment
                inferred_filled = min(depth_lost, v_passive + epsilon_fill)
                inferred_pulled = max(0, depth_lost - v_passive)

                filled_size += inferred_filled
                canceled_size += inferred_pulled

        churn = gross_added + gross_removed
        return added_size, canceled_size, filled_size, churn

    def _get_zone_depth(
        self,
        mbp: MBP10,
        zone_low: float,
        zone_high: float,
        side: str
    ) -> int:
        """
        Get total depth in zone from MBP-10 snapshot.

        Per §5.1.1: compute over a zone (L ± N ticks) for stability.
        """
        total_depth = 0

        for level in mbp.levels:
            if side == 'bid':
                price = level.bid_px
                size = level.bid_sz
            else:
                price = level.ask_px
                size = level.ask_sz

            # Check if this price level is in our zone
            if zone_low <= price <= zone_high:
                total_depth += size

        return total_depth

    def _get_passive_volume(
        self,
        trades: List[FuturesTrade],
        start_ts: int,
        end_ts: int,
        zone_low: float,
        zone_high: float,
        side: str
    ) -> int:
        """
        Get volume of passive fills on the defending side.

        Per §5.1.1:
        - BID side: SELL-initiated trades (aggressor=SELL)
        - ASK side: BUY-initiated trades (aggressor=BUY)
        """
        volume = 0

        for trade in trades:
            # Check time window
            if not (start_ts <= trade.ts_event_ns <= end_ts):
                continue

            # Check price zone
            if not (zone_low <= trade.price <= zone_high):
                continue

            # Check aggressor matches defending side
            if side == 'bid' and trade.aggressor == Aggressor.SELL:
                # Sell aggressor hits bid -> passive bid fill
                volume += trade.size
            elif side == 'ask' and trade.aggressor == Aggressor.BUY:
                # Buy aggressor lifts ask -> passive ask fill
                volume += trade.size

        return volume

    def _get_best_defending_quote(self, mbp: MBP10, side: str) -> dict:
        """Get best defending quote from MBP-10."""
        if not mbp.levels:
            return {"price": 0.0, "size": 0}

        best = mbp.levels[0]
        if side == 'bid':
            return {"price": best.bid_px, "size": best.bid_sz}
        else:
            return {"price": best.ask_px, "size": best.ask_sz}

    def _classify_state(
        self,
        delta_liq: float,
        replenishment_ratio: float,
        canceled_size: float,
        filled_size: float,
        depth_in_zone: int,
        churn: float
    ) -> BarrierState:
        """
        Classify barrier state based on computed metrics.

        Per §5.1.1 and §5.1:
        - VACUUM: R < R_vac AND Δliq < -F_thresh (pulled without fills)
        - WALL/ABSORPTION: R > R_wall AND Δliq > +F_thresh (strong replenishment)
        - CONSUMED: Δliq << 0 AND filled > canceled
        - NEUTRAL: otherwise

        Additional per §5.1.1:
        - High churn with stable depth suggests defended/contested
        - If Vpassive >> depth_lost, interpret as hidden/iceberg
        """
        # VACUUM condition: depth pulled without fills
        if replenishment_ratio < self.r_vacuum and delta_liq < -self.delta_liq_thresh:
            return BarrierState.VACUUM

        # WALL condition: strong replenishment
        if replenishment_ratio > self.r_wall and delta_liq > self.delta_liq_thresh:
            return BarrierState.WALL

        # CONSUMED condition: being eaten (fills dominate cancels)
        if delta_liq < -self.delta_liq_thresh and filled_size > canceled_size:
            return BarrierState.CONSUMED

        # ABSORPTION: replenishment happening but not as strong as WALL
        if delta_liq > 0 and replenishment_ratio > 1.0:
            return BarrierState.ABSORPTION

        # WEAK: low depth in zone (could add baseline tracking)
        if depth_in_zone < 50:  # ES typically has smaller sizes than SPY shares
            return BarrierState.WEAK

        # High churn but stable depth suggests contested
        if abs(delta_liq) < self.delta_liq_thresh / 2 and churn > self.delta_liq_thresh:
            return BarrierState.ABSORPTION  # Contested/defended

        return BarrierState.NEUTRAL
