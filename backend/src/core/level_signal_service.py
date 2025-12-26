"""
Level Signal Service: orchestrator for level physics engine.

Agent G deliverable per §12 of PLAN.md.

This service:
- Integrates all engines (Barrier, Tape, Fuel, Score, Smoothing)
- Generates level universe
- Computes runway for each level
- Produces complete WS payload per §6.4

This is the main entry point for computing level signals on each snap tick.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import time

from .market_state import MarketState
from .level_universe import LevelUniverse, Level, LevelKind
from .room_to_run import RoomToRun, get_break_direction, get_reject_direction
from .barrier_engine import BarrierEngine, Direction as BarrierDirection
from .tape_engine import TapeEngine
from .fuel_engine import FuelEngine
from .score_engine import ScoreEngine, Signal, Confidence
from .smoothing import SmootherSet
from .confluence_computer import ConfluenceComputer, get_confluence_level_name
from src.common.config import CONFIG


@dataclass
class LevelSignal:
    """
    Complete signal output for a single level.
    
    Matches §6.4 WS payload structure.
    """
    # Level identity
    id: str
    price: float
    kind: str
    direction: str  # SUPPORT or RESISTANCE
    distance: float  # Distance from spot to level
    
    # Scores
    break_score_raw: float
    break_score_smooth: Optional[float]
    signal: str  # BREAK, REJECT, CONTESTED, NEUTRAL
    confidence: str  # HIGH, MEDIUM, LOW

    # Approach context (backward-looking)
    approach_velocity: float
    approach_bars: int
    approach_distance: float
    prior_touches: int
    
    # Barrier metrics
    barrier: Dict[str, Any]
    
    # Tape metrics
    tape: Dict[str, Any]
    
    # Fuel metrics
    fuel: Dict[str, Any]
    
    # Runway
    runway: Dict[str, Any]
    
    # Confluence metrics
    confluence_count: int
    confluence_pressure: float
    confluence_alignment: int  # -1=OPPOSED, 0=NEUTRAL, 1=ALIGNED
    confluence_level: int  # 0-10 hierarchical scale
    confluence_level_name: str  # Human-readable name
    
    # Optional human-readable note
    note: Optional[str] = None


class LevelSignalService:
    """
    Main orchestrator for level physics computation.
    
    Usage:
        service = LevelSignalService(market_state)
        payload = service.compute_level_signals()
        # Returns payload ready for WS broadcast
    """
    
    def __init__(
        self,
        market_state: MarketState,
        user_hotzones: Optional[List[float]] = None,
        config=None,
        trading_date: Optional[str] = None
    ):
        """
        Initialize level signal service.

        Args:
            market_state: MarketState instance (shared)
            user_hotzones: Optional user-defined levels to monitor
            config: Config object (defaults to global CONFIG)
            trading_date: Trading date for 0DTE filter (YYYY-MM-DD format).
                         If None, uses current date. We ONLY do 0DTE.
        """
        self.market_state = market_state
        self.config = config or CONFIG

        # 0DTE filter: We ONLY process same-day expiration options
        if trading_date:
            self._trading_date = trading_date
        else:
            # Default to current date for live trading
            from datetime import datetime, timezone
            self._trading_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        # Initialize engines
        self.level_universe = LevelUniverse(user_hotzones=user_hotzones)
        self.room_to_run = RoomToRun()
        self.barrier_engine = BarrierEngine(config=self.config)
        self.tape_engine = TapeEngine(config=self.config)
        self.fuel_engine = FuelEngine(config=self.config)
        self.score_engine = ScoreEngine(config=self.config)
        
        # Smoothers: one set per level (keyed by level.id)
        self.smoothers: Dict[str, SmootherSet] = {}
        
        # Confluence computer
        self.confluence_computer = ConfluenceComputer()

        # Touch tracking for approach context (per level ID)
        self._touch_counts: Dict[str, int] = {}
        self._touch_active: Dict[str, bool] = {}
    
    def compute_level_signals(self) -> Dict[str, Any]:
        """
        Compute complete level signals for current market state.
        
        Returns:
            Dictionary payload ready for WS broadcast (per §6.4)
        """
        ts_now_ns = self.market_state.get_current_ts_ns()
        spot = self.market_state.get_spot()
        bid_ask = self.market_state.get_bid_ask()
        
        if spot is None or bid_ask is None:
            # No market data yet
            return self._empty_payload(ts_now_ns)
        
        bid, ask = bid_ask
        
        # Generate level universe
        levels = self.level_universe.get_levels(self.market_state)
        
        # Filter to levels within monitoring band
        active_levels = self._filter_active_levels(levels, spot)

        # Any level outside monitor band is definitely outside touch band too.
        # Reset "in-touch" state so future touches are counted correctly.
        active_ids = {level.id for level in active_levels}
        for level_id in list(self._touch_active.keys()):
            if level_id not in active_ids:
                self._touch_active[level_id] = False
        
        # Compute signals for each active level
        level_signals = []
        for level in active_levels:
            signal = self._compute_level_signal(level, spot, ts_now_ns)
            if signal is not None:
                level_signals.append(signal)
        
        # Sort by distance (nearest first)
        level_signals.sort(key=lambda s: s.distance)
        
        # Build payload
        payload = {
            "ts": ts_now_ns // 1_000_000,  # Convert to Unix ms for WS
            "spy": {
                "spot": spot,
                "bid": bid,
                "ask": ask
            },
            "levels": [self._signal_to_dict(sig) for sig in level_signals]
        }
        
        return payload
    
    def _empty_payload(self, ts_ns: int) -> Dict[str, Any]:
        """Return empty payload when no market data available."""
        return {
            "ts": ts_ns // 1_000_000,
            "spy": {"spot": None, "bid": None, "ask": None},
            "levels": []
        }
    
    def _filter_active_levels(self, levels: List[Level], spot: float) -> List[Level]:
        """
        Filter levels to those within monitoring band.
        
        Per §3.3 of PLAN.md: only compute full signals if
        abs(spot - L) <= MONITOR_BAND
        
        Args:
            levels: All levels from universe
            spot: Current spot price
            
        Returns:
            Filtered list of active levels
        """
        monitor_band = self.config.MONITOR_BAND
        return [
            level for level in levels
            if abs(spot - level.price) <= monitor_band
        ]
    
    def _compute_level_signal(
        self,
        level: Level,
        spot: float,
        ts_now_ns: int
    ) -> Optional[LevelSignal]:
        """
        Compute complete signal for a single level.
        
        Args:
            level: Level to analyze
            spot: Current spot price
            ts_now_ns: Current timestamp
            
        Returns:
            LevelSignal or None if computation fails
        """
        level_price = level.price
        distance = abs(spot - level_price)
        
        # Determine direction (SUPPORT or RESISTANCE)
        if spot > level_price:
            direction_str = "SUPPORT"
            barrier_direction = BarrierDirection.SUPPORT
            break_dir = "DOWN"
        else:
            direction_str = "RESISTANCE"
            barrier_direction = BarrierDirection.RESISTANCE
            break_dir = "UP"

        # ========== Backward-looking approach context ==========
        approach_velocity, approach_bars, approach_distance = self._compute_approach_context(break_dir)
        prior_touches = self._get_prior_touches(level.id, distance)
        
        # ========== Compute engine states ==========
        
        # Barrier engine
        barrier_metrics = self.barrier_engine.compute_barrier_state(
            level_price=level_price,
            direction=barrier_direction,
            market_state=self.market_state
        )
        
        # Tape engine
        tape_metrics = self.tape_engine.compute_tape_state(
            level_price=level_price,
            market_state=self.market_state
        )
        
        # Fuel engine - ALWAYS filter to 0DTE (same-day expiration only)
        fuel_metrics = self.fuel_engine.compute_fuel_state(
            level_price=level_price,
            market_state=self.market_state,
            exp_date_filter=self._trading_date
        )
        
        # Score engine
        composite_score = self.score_engine.compute_score(
            barrier_metrics=barrier_metrics,
            tape_metrics=tape_metrics,
            fuel_metrics=fuel_metrics,
            break_direction=break_dir,
            ts_ns=ts_now_ns,
            distance_to_level=distance
        )
        
        # ========== Smoothing ==========
        
        # Get or create smoother set for this level
        if level.id not in self.smoothers:
            self.smoothers[level.id] = SmootherSet(config=self.config)
        
        smoother = self.smoothers[level.id]
        
        # Update smoothers
        score_smooth = smoother.update_score(composite_score.raw_score, ts_now_ns)
        delta_liq_smooth = smoother.update_delta_liq(barrier_metrics.delta_liq, ts_now_ns)
        replenish_smooth = smoother.update_replenishment(barrier_metrics.replenishment_ratio, ts_now_ns)
        velocity_smooth = smoother.update_velocity(tape_metrics.velocity, ts_now_ns)
        dealer_gamma_smooth = smoother.update_dealer_gamma(fuel_metrics.net_dealer_gamma, ts_now_ns)
        
        # ========== Runway ==========
        
        # Get all levels for runway computation
        all_levels = self.level_universe.get_levels(self.market_state)
        
        # Determine runway direction based on signal
        if composite_score.signal == Signal.BREAK_IMMINENT:
            # Break direction
            from .room_to_run import Direction as RunwayDirection
            runway_direction = RunwayDirection.DOWN if break_dir == "DOWN" else RunwayDirection.UP
        elif composite_score.signal == Signal.REJECT:
            # Reject direction (opposite of break)
            from .room_to_run import Direction as RunwayDirection
            runway_direction = RunwayDirection.UP if break_dir == "DOWN" else RunwayDirection.DOWN
        else:
            # Default to break direction
            from .room_to_run import Direction as RunwayDirection
            runway_direction = RunwayDirection.DOWN if break_dir == "DOWN" else RunwayDirection.UP
        
        runway = self.room_to_run.compute_runway(
            current_level=level,
            direction=runway_direction,
            all_levels=all_levels,
            spot=spot
        )
        
        # ========== Confluence Computation ==========
        
        # Compute confluence metrics
        confluence_metrics = self.confluence_computer.compute(
            target_level=level,
            all_levels=all_levels,
            market_state=self.market_state,
            direction=direction_str
        )
        
        # Compute hierarchical confluence level (0-10)
        confluence_level = self.confluence_computer.compute_hierarchical_level(
            metrics=confluence_metrics,
            level_kind=level.kind
        )
        confluence_level_name = get_confluence_level_name(confluence_level)
        
        # ========== Build signal output ==========
        
        # Generate human-readable note
        note = self._generate_note(
            barrier_metrics.state.value,
            fuel_metrics.effect.value,
            tape_metrics.sweep.detected
        )
        
        level_signal = LevelSignal(
            id=level.id,
            price=level_price,
            kind=level.kind.value,
            direction=direction_str,
            distance=distance,
            break_score_raw=composite_score.raw_score,
            break_score_smooth=score_smooth,
            signal=composite_score.signal.value,
            confidence=composite_score.confidence.value,
            approach_velocity=approach_velocity,
            approach_bars=approach_bars,
            approach_distance=approach_distance,
            prior_touches=prior_touches,
            barrier={
                "state": barrier_metrics.state.value,
                "delta_liq": barrier_metrics.delta_liq,
                "delta_liq_smooth": delta_liq_smooth,
                "replenishment_ratio": barrier_metrics.replenishment_ratio,
                "replenishment_ratio_smooth": replenish_smooth,
                "added": barrier_metrics.added_size,
                "canceled": barrier_metrics.canceled_size,
                "filled": barrier_metrics.filled_size,
                "defending_quote": barrier_metrics.defending_quote,
                "churn": barrier_metrics.churn,
                "depth_in_zone": barrier_metrics.depth_in_zone
            },
            tape={
                "imbalance": tape_metrics.imbalance,
                "buy_vol": tape_metrics.buy_vol,
                "sell_vol": tape_metrics.sell_vol,
                "velocity": tape_metrics.velocity,
                "velocity_smooth": velocity_smooth,
                "sweep": {
                    "detected": tape_metrics.sweep.detected,
                    "direction": tape_metrics.sweep.direction,
                    "notional": tape_metrics.sweep.notional,
                    "num_prints": tape_metrics.sweep.num_prints,
                    "window_ms": tape_metrics.sweep.window_ms
                }
            },
            fuel={
                "effect": fuel_metrics.effect.value,
                "net_dealer_gamma": fuel_metrics.net_dealer_gamma,
                "net_dealer_gamma_smooth": dealer_gamma_smooth,
                "call_wall": fuel_metrics.call_wall.strike if fuel_metrics.call_wall else None,
                "put_wall": fuel_metrics.put_wall.strike if fuel_metrics.put_wall else None,
                "hvl": fuel_metrics.hvl
            },
            runway={
                "direction": runway.direction.value,
                "distance": runway.distance,
                "next_obstacle": {
                    "id": runway.next_obstacle.id,
                    "price": runway.next_obstacle.price
                } if runway.next_obstacle else None,
                "quality": runway.quality.value
            },
            confluence_count=confluence_metrics.confluence_count,
            confluence_pressure=confluence_metrics.confluence_pressure,
            confluence_alignment=confluence_metrics.confluence_alignment.value,
            confluence_level=confluence_level,
            confluence_level_name=confluence_level_name,
            note=note
        )
        
        return level_signal

    def _compute_approach_context(self, direction_updown: str) -> Tuple[float, int, float]:
        """
        Compute backward-looking approach context (v1) from recent 1-minute closes.

        direction_updown:
          - "UP": approaching resistance from below
          - "DOWN": approaching support from above
        """
        closes = self.market_state.get_recent_minute_closes(self.config.LOOKBACK_MINUTES)
        if len(closes) < 2:
            return 0.0, 0, 0.0

        price_change = closes[-1] - closes[0]
        time_minutes = len(closes)
        if time_minutes <= 0:
            return 0.0, 0, 0.0

        if direction_updown == "UP":
            approach_velocity = price_change / time_minutes
        else:
            # Moving down toward support is positive approach velocity
            approach_velocity = -price_change / time_minutes

        consecutive = 0
        for j in range(len(closes) - 1, 0, -1):
            bar_move = closes[j] - closes[j - 1]
            if direction_updown == "UP":
                if bar_move > 0:
                    consecutive += 1
                else:
                    break
            else:
                if bar_move < 0:
                    consecutive += 1
                else:
                    break

        approach_distance = abs(closes[-1] - closes[0])
        return approach_velocity, consecutive, approach_distance

    def _get_prior_touches(self, level_id: str, distance: float) -> int:
        """
        Count distinct "touch" events (entering TOUCH_BAND) per level.
        Returns the count prior to the current touch (matches vectorized pipeline semantics).
        """
        prior = self._touch_counts.get(level_id, 0)
        touching = distance <= self.config.TOUCH_BAND
        if touching and not self._touch_active.get(level_id, False):
            self._touch_counts[level_id] = prior + 1
            self._touch_active[level_id] = True
        elif not touching:
            self._touch_active[level_id] = False
        return prior
    
    def _generate_note(
        self,
        barrier_state: str,
        fuel_effect: str,
        sweep_detected: bool
    ) -> str:
        """
        Generate human-readable note for level signal.
        
        Args:
            barrier_state: Barrier state string
            fuel_effect: Fuel effect string
            sweep_detected: Whether sweep was detected
            
        Returns:
            Human-readable note
        """
        parts = []
        
        if barrier_state == "VACUUM":
            parts.append("Vacuum")
        elif barrier_state == "WALL":
            parts.append("Wall holding")
        elif barrier_state == "CONSUMED":
            parts.append("Being consumed")
        
        if fuel_effect == "AMPLIFY":
            parts.append("dealers chase")
        elif fuel_effect == "DAMPEN":
            parts.append("dealers fade")
        
        if sweep_detected:
            parts.append("sweep confirms")
        
        if parts:
            return "; ".join(parts)
        else:
            return "Monitoring"
    
    def _signal_to_dict(self, signal: LevelSignal) -> Dict[str, Any]:
        """Convert LevelSignal to dict for JSON serialization."""
        return asdict(signal)
    
    def reset(self):
        """Reset all engine and smoother state."""
        self.score_engine.reset()
        self.smoothers.clear()

    @property
    def trading_date(self) -> str:
        """Return the trading date used for 0DTE filtering."""
        return self._trading_date
