"""
Tape Engine: ES prints confirm direction

Computes aggressor imbalance, velocity, and sweep detection near a level.

Based on PLAN.md §5.2 - adapted for ES futures instead of SPY equities.

Price Conversion:
- Input level_price is in SPY terms (e.g., 687.0)
- Internally convert to ES (e.g., 6870.0) for trade queries
- Velocity is computed in ES terms but could be converted for display

Interfaces consumed:
- event_types (Agent A): Aggressor
- market_state (Agent C): MarketState for windowed ES trade queries
- config (Agent A): window sizes, thresholds
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque
import numpy as np

from src.common.event_types import Aggressor
from .market_state import MarketState, TimestampedESTrade
from src.common.config import CONFIG


@dataclass
class SweepDetection:
    """Sweep detection result"""
    detected: bool
    direction: str  # 'UP' or 'DOWN' or 'NONE'
    notional: float
    num_prints: int
    window_ms: float
    venues: Optional[List[int]] = None


@dataclass
class TapeMetrics:
    """Output of tape engine computation"""
    imbalance: float  # -1 to +1
    buy_vol: int
    sell_vol: int
    velocity: float  # $/sec
    sweep: SweepDetection
    confidence: float  # 0-1


class TapeEngine:
    """
    Computes tape momentum and aggression near a level using ES trades.

    For a given level L:
    - Compute buy/sell imbalance in price band around level
    - Compute velocity (price slope over time)
    - Detect sweeps (large clustered aggression)

    Interfaces consumed:
    - MarketState queries (Agent C) for windowed ES trades
    - Config thresholds (Agent A) for bands, sweep detection
    """

    def __init__(self, config=None):
        """
        Args:
            config: Config object (defaults to global CONFIG)
        """
        self.config = config or CONFIG

        # Extract config values
        self.window_seconds = self.config.W_t  # imbalance window
        self.velocity_window_seconds = self.config.W_v  # velocity window
        self.tape_band_dollars = self.config.TAPE_BAND  # price band for imbalance

        # Sweep detection
        self.sweep_min_notional = self.config.SWEEP_MIN_NOTIONAL
        self.sweep_max_gap_ms = self.config.SWEEP_MAX_GAP_MS
        self.sweep_min_venues = self.config.SWEEP_MIN_VENUES

    def compute_tape_state(
        self,
        level_price: float,
        market_state: MarketState
    ) -> TapeMetrics:
        """
        Compute tape metrics for a level using ES trades.

        Args:
            level_price: The critical level being tested (SPY price, e.g., 687.0)
            market_state: MarketState instance with ES trade buffers

        Returns:
            TapeMetrics with imbalance, velocity, and sweep detection
        """
        ts_now_ns = market_state.get_current_ts_ns()

        # Convert SPY level to ES equivalent for trade queries
        es_level_price = market_state.price_converter.spy_to_es(level_price)

        # Convert SPY tape band to ES equivalent
        es_tape_band = self.tape_band_dollars * market_state.price_converter.ratio

        # Get ES trades near level for imbalance (using ES prices)
        trades_near_level = market_state.get_es_trades_near_level(
            ts_now_ns=ts_now_ns,
            window_seconds=self.window_seconds,
            level_price=es_level_price,
            band_dollars=es_tape_band
        )

        # Compute imbalance
        buy_vol, sell_vol = self._compute_imbalance(trades_near_level)
        epsilon = 1e-6
        imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + epsilon)

        # Get all recent ES trades for velocity
        all_trades = market_state.get_es_trades_in_window(
            ts_now_ns=ts_now_ns,
            window_seconds=self.velocity_window_seconds
        )

        # Compute velocity (in ES $/sec - scales with ES prices)
        velocity = self._compute_velocity(all_trades)

        # Detect sweeps (using ES level price)
        sweep = self._detect_sweep(trades_near_level, es_level_price)

        # Compute confidence (scaled for ES contract sizes)
        total_vol = buy_vol + sell_vol
        confidence = min(1.0, total_vol / 500.0)  # ES contracts are smaller than SPY shares

        return TapeMetrics(
            imbalance=imbalance,
            buy_vol=buy_vol,
            sell_vol=sell_vol,
            velocity=velocity,
            sweep=sweep,
            confidence=confidence
        )

    def _compute_imbalance(
        self,
        trades: List[TimestampedESTrade]
    ) -> Tuple[int, int]:
        """
        Compute buy and sell volume from ES trades.

        Args:
            trades: List of timestamped ES trades

        Returns:
            (buy_vol, sell_vol) in contracts
        """
        buy_vol = 0
        sell_vol = 0

        for trade in trades:
            if trade.aggressor == Aggressor.BUY:
                buy_vol += trade.size
            elif trade.aggressor == Aggressor.SELL:
                sell_vol += trade.size
            # MID/UNKNOWN trades are ignored for imbalance

        return buy_vol, sell_vol

    def _compute_velocity(
        self,
        trades: List[TimestampedESTrade]
    ) -> float:
        """
        Compute price velocity ($/sec) using linear regression.

        Args:
            trades: List of timestamped ES trades

        Returns:
            Velocity in $/sec (positive = rising, negative = falling)
        """
        if len(trades) < 2:
            return 0.0

        # Extract times (in seconds) and prices
        times = np.array([t.ts_event_ns / 1e9 for t in trades])
        prices = np.array([t.price for t in trades])

        # Normalize time to start at 0
        times = times - times[0]

        # Linear regression: price = slope * time + intercept
        try:
            slope, _ = np.polyfit(times, prices, 1)
            return float(slope)
        except:
            return 0.0

    def _detect_sweep(
        self,
        trades: List[TimestampedESTrade],
        level_price: float
    ) -> SweepDetection:
        """
        Detect sweep activity (clustered aggressive prints).

        A sweep is characterized by:
        - Rapid sequence of trades (max gap < SWEEP_MAX_GAP_MS)
        - Large total notional (> SWEEP_MIN_NOTIONAL)
        - Consistent direction (all BUY or all SELL)
        - Optional: Multiple venues (if exchange data available)

        Args:
            trades: List of timestamped ES trades near level
            level_price: Level price for direction inference

        Returns:
            SweepDetection result
        """
        if len(trades) < 3:
            return SweepDetection(
                detected=False,
                direction='NONE',
                notional=0.0,
                num_prints=0,
                window_ms=0.0
            )

        # Find clusters of trades with small time gaps
        clusters = self._find_trade_clusters(trades)

        # Check each cluster for sweep characteristics
        best_sweep = None
        max_notional = 0.0

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            # Compute cluster metrics (ES contract = $50 per point)
            total_notional = sum(t.price * t.size * 50 for t in cluster)
            buy_count = sum(1 for t in cluster if t.aggressor == Aggressor.BUY)
            sell_count = sum(1 for t in cluster if t.aggressor == Aggressor.SELL)

            # Check for consistent direction
            is_buy_sweep = buy_count > 0 and sell_count == 0
            is_sell_sweep = sell_count > 0 and buy_count == 0

            if not (is_buy_sweep or is_sell_sweep):
                continue

            # Check notional threshold
            if total_notional < self.sweep_min_notional:
                continue

            # This is a valid sweep
            if total_notional > max_notional:
                max_notional = total_notional

                # Infer direction relative to level
                avg_price = sum(t.price for t in cluster) / len(cluster)
                if is_buy_sweep:
                    direction = 'UP' if avg_price >= level_price else 'UP'
                else:
                    direction = 'DOWN' if avg_price <= level_price else 'DOWN'

                window_ms = (cluster[-1].ts_event_ns - cluster[0].ts_event_ns) / 1e6

                best_sweep = SweepDetection(
                    detected=True,
                    direction=direction,
                    notional=total_notional,
                    num_prints=len(cluster),
                    window_ms=window_ms,
                    venues=None  # TODO: extract from trades if available
                )

        if best_sweep is not None:
            return best_sweep

        # No sweep detected
        return SweepDetection(
            detected=False,
            direction='NONE',
            notional=0.0,
            num_prints=0,
            window_ms=0.0
        )

    def _find_trade_clusters(
        self,
        trades: List[TimestampedESTrade]
    ) -> List[List[TimestampedESTrade]]:
        """
        Group trades into clusters based on time gaps.

        Args:
            trades: List of timestamped ES trades (assumed sorted by time)

        Returns:
            List of clusters (each cluster is a list of trades)
        """
        if not trades:
            return []

        clusters = []
        current_cluster = [trades[0]]

        for i in range(1, len(trades)):
            prev_trade = trades[i-1]
            curr_trade = trades[i]

            gap_ms = (curr_trade.ts_event_ns - prev_trade.ts_event_ns) / 1e6

            if gap_ms <= self.sweep_max_gap_ms:
                # Continue current cluster
                current_cluster.append(curr_trade)
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [curr_trade]

        # Add final cluster
        if current_cluster:
            clusters.append(current_cluster)

        return clusters
