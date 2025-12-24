"""
Score Engine: composite break score and trigger state machine.

Agent G deliverable per §12 of PLAN.md.

This module combines Barrier, Tape, and Fuel states into:
- Component scores (S_L, S_H, S_T)
- Composite break score (0-100)
- Discrete signal triggers (BREAK, REJECT, CONTESTED, NEUTRAL)

Per §5.4 of PLAN.md:
- S = w_L * S_L + w_H * S_H + w_T * S_T
- Default weights: w_L=0.45, w_H=0.35, w_T=0.20
- Triggers require score sustained over time (hysteresis)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from .barrier_engine import BarrierMetrics, BarrierState
from .tape_engine import TapeMetrics
from .fuel_engine import FuelMetrics, FuelEffect
from src.common.config import CONFIG


class Signal(str, Enum):
    """Discrete signal classification."""
    BREAK_IMMINENT = "BREAK"  # Score > 80, sustained
    REJECT = "REJECT"  # Score < 20, touching level
    CONTESTED = "CONTESTED"  # Mid scores with high activity
    NEUTRAL = "NEUTRAL"  # Default state


class Confidence(str, Enum):
    """Signal confidence level."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class ComponentScores:
    """Individual component scores before composite."""
    liquidity_score: float  # S_L: 0-100
    hedge_score: float  # S_H: 0-100
    tape_score: float  # S_T: 0-100


@dataclass
class CompositeScore:
    """
    Complete scoring output for a level.
    
    Attributes:
        raw_score: Raw composite score (0-100)
        component_scores: Individual component scores
        signal: Discrete signal (BREAK/REJECT/CONTESTED/NEUTRAL)
        confidence: Signal confidence level
    """
    raw_score: float
    component_scores: ComponentScores
    signal: Signal
    confidence: Confidence


class TriggerStateMachine:
    """
    State machine for signal triggers with hysteresis.
    
    Per §5.4.3 of PLAN.md:
    - BREAK IMMINENT: score > 80 sustained for T_hold seconds
    - REJECT: score < 20 while price touches level
    - CONTESTED: mid scores with high activity
    
    This prevents flickering by requiring sustained conditions.
    """
    
    def __init__(self, hold_time_seconds: float = 3.0):
        """
        Initialize trigger state machine.
        
        Args:
            hold_time_seconds: Time score must be sustained for trigger
        """
        self.hold_time_seconds = hold_time_seconds
        self.hold_time_ns = int(hold_time_seconds * 1e9)
        
        # Track sustained score state
        self.current_signal = Signal.NEUTRAL
        self.high_score_since_ns: Optional[int] = None
        self.low_score_since_ns: Optional[int] = None
    
    def update(
        self,
        score: float,
        ts_ns: int,
        distance_to_level: float,
        barrier_state: BarrierState,
        tape_activity: float
    ) -> Signal:
        """
        Update state machine with current score and conditions.
        
        Args:
            score: Current break score (0-100)
            ts_ns: Current timestamp
            distance_to_level: Distance from spot to level (dollars)
            barrier_state: Current barrier state
            tape_activity: Tape activity metric (total volume)
            
        Returns:
            Current signal classification
        """
        # ========== BREAK IMMINENT trigger ==========
        if score > CONFIG.BREAK_SCORE_THRESHOLD:
            if self.high_score_since_ns is None:
                # Start tracking high score
                self.high_score_since_ns = ts_ns
            elif ts_ns - self.high_score_since_ns >= self.hold_time_ns:
                # Sustained high score for required duration
                self.current_signal = Signal.BREAK_IMMINENT
                # Reset low score tracking
                self.low_score_since_ns = None
        else:
            # Score dropped below threshold, reset tracking
            self.high_score_since_ns = None
        
        # ========== REJECT trigger ==========
        if score < CONFIG.REJECT_SCORE_THRESHOLD:
            # Check if price is touching the level (within TOUCH_BAND)
            if abs(distance_to_level) <= CONFIG.TOUCH_BAND:
                if self.low_score_since_ns is None:
                    self.low_score_since_ns = ts_ns
                elif ts_ns - self.low_score_since_ns >= self.hold_time_ns:
                    self.current_signal = Signal.REJECT
                    # Reset high score tracking
                    self.high_score_since_ns = None
            else:
                # Not touching level, reset tracking
                self.low_score_since_ns = None
        else:
            # Score rose above threshold, reset tracking
            self.low_score_since_ns = None
        
        # ========== CONTESTED trigger ==========
        # Mid scores (30-70) with high activity and CONSUMED state
        if 30 <= score <= 70:
            if barrier_state == BarrierState.CONSUMED and tape_activity > 50000:
                self.current_signal = Signal.CONTESTED
        
        # ========== NEUTRAL fallback ==========
        # If no sustained trigger conditions, return to neutral
        if (self.high_score_since_ns is None and 
            self.low_score_since_ns is None and
            self.current_signal not in [Signal.BREAK_IMMINENT, Signal.REJECT]):
            self.current_signal = Signal.NEUTRAL
        
        return self.current_signal
    
    def reset(self):
        """Reset state machine."""
        self.current_signal = Signal.NEUTRAL
        self.high_score_since_ns = None
        self.low_score_since_ns = None


class ScoreEngine:
    """
    Computes composite break score from Barrier, Tape, and Fuel states.
    
    Per §5.4 of PLAN.md:
    - Component scores: S_L (liquidity), S_H (hedge), S_T (tape)
    - Composite: S = w_L * S_L + w_H * S_H + w_T * S_T
    - Signal triggers with hysteresis
    
    Usage:
        engine = ScoreEngine()
        score = engine.compute_score(barrier_metrics, tape_metrics, fuel_metrics)
    """
    
    def __init__(self, config=None):
        """
        Initialize score engine.
        
        Args:
            config: Config object (defaults to global CONFIG)
        """
        self.config = config or CONFIG
        
        # Extract weights
        self.w_L = self.config.w_L
        self.w_H = self.config.w_H
        self.w_T = self.config.w_T
        
        # Trigger state machine
        self.trigger = TriggerStateMachine(
            hold_time_seconds=self.config.TRIGGER_HOLD_TIME
        )
    
    def compute_score(
        self,
        barrier_metrics: BarrierMetrics,
        tape_metrics: TapeMetrics,
        fuel_metrics: FuelMetrics,
        break_direction: str,  # 'UP' or 'DOWN'
        ts_ns: int,
        distance_to_level: float
    ) -> CompositeScore:
        """
        Compute composite break score.
        
        Args:
            barrier_metrics: Barrier engine output
            tape_metrics: Tape engine output
            fuel_metrics: Fuel engine output
            break_direction: Expected break direction ('UP' or 'DOWN')
            ts_ns: Current timestamp
            distance_to_level: Distance from spot to level
            
        Returns:
            CompositeScore with raw score, components, signal, and confidence
        """
        # Compute component scores
        S_L = self._compute_liquidity_score(barrier_metrics)
        S_H = self._compute_hedge_score(fuel_metrics, break_direction)
        S_T = self._compute_tape_score(tape_metrics, break_direction)
        
        # Composite score
        raw_score = self.w_L * S_L + self.w_H * S_H + self.w_T * S_T
        raw_score = max(0.0, min(100.0, raw_score))  # Clamp to [0, 100]
        
        # Trigger state machine
        tape_activity = tape_metrics.buy_vol + tape_metrics.sell_vol
        signal = self.trigger.update(
            score=raw_score,
            ts_ns=ts_ns,
            distance_to_level=distance_to_level,
            barrier_state=barrier_metrics.state,
            tape_activity=tape_activity
        )
        
        # Compute confidence based on component confidences
        confidence = self._compute_confidence(
            barrier_metrics,
            tape_metrics,
            fuel_metrics
        )
        
        return CompositeScore(
            raw_score=raw_score,
            component_scores=ComponentScores(
                liquidity_score=S_L,
                hedge_score=S_H,
                tape_score=S_T
            ),
            signal=signal,
            confidence=confidence
        )
    
    def _compute_liquidity_score(self, barrier: BarrierMetrics) -> float:
        """
        Compute liquidity component score (S_L).
        
        Per §5.4.1 of PLAN.md:
        - VACUUM: 100
        - WEAK: 75
        - NEUTRAL/CONSUMED: 50-60
        - WALL/ABSORPTION: 0
        
        Args:
            barrier: Barrier metrics
            
        Returns:
            Liquidity score (0-100)
        """
        state = barrier.state
        
        if state == BarrierState.VACUUM:
            return 100.0
        elif state == BarrierState.WEAK:
            return 75.0
        elif state == BarrierState.CONSUMED:
            # Consumed with negative delta_liq is more breakable
            if barrier.delta_liq < -self.config.F_thresh:
                return 60.0
            else:
                return 50.0
        elif state == BarrierState.NEUTRAL:
            return 50.0
        elif state in [BarrierState.WALL, BarrierState.ABSORPTION]:
            return 0.0
        else:
            return 50.0
    
    def _compute_hedge_score(self, fuel: FuelMetrics, break_direction: str) -> float:
        """
        Compute hedge component score (S_H).
        
        Per §5.4.1 of PLAN.md:
        - AMPLIFY in break direction: 100
        - DAMPEN: 0
        - NEUTRAL: 50
        
        Args:
            fuel: Fuel metrics
            break_direction: Expected break direction ('UP' or 'DOWN')
            
        Returns:
            Hedge score (0-100)
        """
        effect = fuel.effect
        
        if effect == FuelEffect.AMPLIFY:
            return 100.0
        elif effect == FuelEffect.DAMPEN:
            return 0.0
        else:  # NEUTRAL
            return 50.0
    
    def _compute_tape_score(self, tape: TapeMetrics, break_direction: str) -> float:
        """
        Compute tape component score (S_T).
        
        Per §5.4.1 of PLAN.md:
        - If sweep detected in break direction: 100
        - Else: scale 0-50 from velocity magnitude and imbalance consistency
        
        Args:
            tape: Tape metrics
            break_direction: Expected break direction ('UP' or 'DOWN')
            
        Returns:
            Tape score (0-100)
        """
        # Check for sweep in break direction
        if tape.sweep.detected:
            sweep_dir = tape.sweep.direction
            if sweep_dir == break_direction:
                return 100.0
        
        # No sweep, score based on velocity and imbalance
        # Velocity is in $/sec, normalize to 0-1 range
        # Typical velocity might be -0.5 to +0.5 $/sec
        velocity_norm = abs(tape.velocity) / 0.5  # Normalize to typical range
        velocity_norm = min(1.0, velocity_norm)
        
        # Check if velocity direction matches break direction
        velocity_aligned = False
        if break_direction == 'UP' and tape.velocity > 0:
            velocity_aligned = True
        elif break_direction == 'DOWN' and tape.velocity < 0:
            velocity_aligned = True
        
        # Imbalance consistency
        # Imbalance is -1 to +1
        imbalance_aligned = False
        if break_direction == 'UP' and tape.imbalance > 0:
            imbalance_aligned = True
        elif break_direction == 'DOWN' and tape.imbalance < 0:
            imbalance_aligned = True
        
        # Combine factors
        score = 0.0
        
        if velocity_aligned:
            score += 25.0 * velocity_norm
        
        if imbalance_aligned:
            score += 25.0 * abs(tape.imbalance)
        
        return min(50.0, score)  # Cap at 50 (sweep gets 100)
    
    def _compute_confidence(
        self,
        barrier: BarrierMetrics,
        tape: TapeMetrics,
        fuel: FuelMetrics
    ) -> Confidence:
        """
        Compute overall signal confidence.
        
        Based on component confidences:
        - HIGH: All components have high confidence
        - MEDIUM: Mixed confidences
        - LOW: Weak data quality
        
        Args:
            barrier: Barrier metrics
            tape: Tape metrics
            fuel: Fuel metrics
            
        Returns:
            Confidence level
        """
        # Average component confidences
        avg_confidence = (
            barrier.confidence + tape.confidence + fuel.confidence
        ) / 3.0
        
        if avg_confidence >= 0.7:
            return Confidence.HIGH
        elif avg_confidence >= 0.4:
            return Confidence.MEDIUM
        else:
            return Confidence.LOW
    
    def reset(self):
        """Reset score engine state."""
        self.trigger.reset()
