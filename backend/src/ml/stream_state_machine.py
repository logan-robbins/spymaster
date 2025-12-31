"""
TA-style State Machine for Stream Interpretation - STREAMS.md Section 10.

Rule-based interpretation layer that detects:
- Exhaustion/continuation/reversal patterns
- Flow-momentum divergence
- Barrier support/opposition
- Setup and dealer regime gating
- Position-aware exit signals

Usage:
    from src.ml.stream_state_machine import StreamStateMachine, detect_alerts
    
    alerts = detect_alerts(current_bar, history_df)
    exit_score = compute_exit_score(current_bar, position_sign=1.0)
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    """Stream alert classifications per STREAMS.md Section 10."""
    # Exhaustion patterns (Section 10.1)
    EXHAUSTION_UP = "EXHAUSTION_UP"  # Buying pressure fading
    EXHAUSTION_DOWN = "EXHAUSTION_DOWN"  # Selling pressure fading
    CONTINUATION_UP = "CONTINUATION_UP"  # Buying pressure building
    CONTINUATION_DOWN = "CONTINUATION_DOWN"  # Selling pressure building
    REVERSAL_RISK_UP = "REVERSAL_RISK_UP"  # Early reversal warning (from up)
    REVERSAL_RISK_DOWN = "REVERSAL_RISK_DOWN"  # Early reversal warning (from down)
    
    # Divergence patterns (Section 10.2)
    FLOW_DIVERGENCE = "FLOW_DIVERGENCE"  # Flow-momentum mismatch
    FLOW_CONFIRMATION = "FLOW_CONFIRMATION"  # Flow-momentum alignment
    
    # Barrier patterns (Section 10.3)
    BARRIER_BREAK_SUPPORT = "BARRIER_BREAK_SUPPORT"  # Barrier favors break
    BARRIER_OPPOSES_PRESSURE = "BARRIER_OPPOSES_PRESSURE"  # Barrier opposes pressure
    BARRIER_WEAKENING = "BARRIER_WEAKENING"  # Barrier losing strength
    
    # Quality gates (Sections 10.4-10.5)
    LOW_QUALITY_SETUP = "LOW_QUALITY_SETUP"  # sigma_s < -0.25
    HIGH_QUALITY_SETUP = "HIGH_QUALITY_SETUP"  # sigma_s > +0.25
    FUEL_REGIME = "FUEL_REGIME"  # sigma_d > +0.25 (amplification)
    PIN_REGIME = "PIN_REGIME"  # sigma_d < -0.25 (dampening)


class AlertSeverity(str, Enum):
    """Alert priority level."""
    CRITICAL = "CRITICAL"  # Immediate action suggested
    WARNING = "WARNING"  # Attention required
    INFO = "INFO"  # Informational only


@dataclass
class StreamAlert:
    """Individual stream alert with context."""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    stream_name: str  # Which stream triggered (pressure, flow, etc.)
    confidence: float  # 0.0 to 1.0
    metadata: Dict  # Additional context


# ============================================================================
# DETECTION FUNCTIONS (STREAMS.md Section 10)
# ============================================================================


def detect_exhaustion_continuation(
    bar: pd.Series,
    n_bars_lookback: int = 3
) -> List[StreamAlert]:
    """
    Detect exhaustion and continuation patterns using Pressure stream.
    
    Per STREAMS.md Section 10.1:
    - CONTINUATION_UP: P > +0.35 and P1 > +0.05
    - EXHAUSTION_UP: P > +0.35 and P1 < 0 for n_bars
    - REVERSAL_RISK_UP: P > +0.35 and P1 < 0 and P2 < 0 and |P3| > thresh
    
    Args:
        bar: Current bar with stream values and derivatives
        n_bars_lookback: How many bars slope must be negative (for exhaustion)
    
    Returns:
        List of detected alerts
    """
    alerts = []
    
    P = bar.get('sigma_p', 0.0)
    P1 = bar.get('sigma_p_slope', 0.0)
    P2 = bar.get('sigma_p_curvature', 0.0)
    P3 = bar.get('sigma_p_jerk', 0.0)
    
    # Thresholds per STREAMS.md
    P_THRESH = 0.35
    P1_CONT_THRESH = 0.05
    P3_THRESH = 0.02  # Jerk threshold for reversal risk
    
    # ========== UPWARD PATTERNS ==========
    if P > P_THRESH:
        if P1 > P1_CONT_THRESH:
            # Continuation: buying building
            alerts.append(StreamAlert(
                alert_type=AlertType.CONTINUATION_UP,
                severity=AlertSeverity.INFO,
                message=f"Buying pressure building (P={P:.3f}, slope={P1:.3f})",
                stream_name='pressure',
                confidence=min(P / 1.0, 1.0),
                metadata={'P': P, 'P1': P1}
            ))
        elif P1 < 0:
            # Exhaustion: buying slowing
            alerts.append(StreamAlert(
                alert_type=AlertType.EXHAUSTION_UP,
                severity=AlertSeverity.WARNING,
                message=f"Buying pressure fading (P={P:.3f}, slope={P1:.3f})",
                stream_name='pressure',
                confidence=min(abs(P1) * 5, 1.0),
                metadata={'P': P, 'P1': P1}
            ))
            
            # Check for reversal risk (early warning)
            if P2 < 0 and abs(P3) > P3_THRESH:
                alerts.append(StreamAlert(
                    alert_type=AlertType.REVERSAL_RISK_UP,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Reversal risk from up (P={P:.3f}, jerk={P3:.3f})",
                    stream_name='pressure',
                    confidence=min(abs(P3) * 20, 1.0),
                    metadata={'P': P, 'P1': P1, 'P2': P2, 'P3': P3}
                ))
    
    # ========== DOWNWARD PATTERNS ==========
    if P < -P_THRESH:
        if P1 < -P1_CONT_THRESH:
            # Continuation: selling building
            alerts.append(StreamAlert(
                alert_type=AlertType.CONTINUATION_DOWN,
                severity=AlertSeverity.INFO,
                message=f"Selling pressure building (P={P:.3f}, slope={P1:.3f})",
                stream_name='pressure',
                confidence=min(abs(P) / 1.0, 1.0),
                metadata={'P': P, 'P1': P1}
            ))
        elif P1 > 0:
            # Exhaustion: selling slowing
            alerts.append(StreamAlert(
                alert_type=AlertType.EXHAUSTION_DOWN,
                severity=AlertSeverity.WARNING,
                message=f"Selling pressure fading (P={P:.3f}, slope={P1:.3f})",
                stream_name='pressure',
                confidence=min(P1 * 5, 1.0),
                metadata={'P': P, 'P1': P1}
            ))
            
            # Check for reversal risk
            if P2 > 0 and abs(P3) > P3_THRESH:
                alerts.append(StreamAlert(
                    alert_type=AlertType.REVERSAL_RISK_DOWN,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Reversal risk from down (P={P:.3f}, jerk={P3:.3f})",
                    stream_name='pressure',
                    confidence=min(abs(P3) * 20, 1.0),
                    metadata={'P': P, 'P1': P1, 'P2': P2, 'P3': P3}
                ))
    
    return alerts


def detect_flow_divergence(bar: pd.Series) -> List[StreamAlert]:
    """
    Detect flow-momentum divergence (reversal/squeeze indicator).
    
    Per STREAMS.md Section 10.2:
    - FLOW_DIVERGENCE: sign(F) != sign(M) and both |·| > 0.30
    - FLOW_CONFIRMATION: sign(F) == sign(M) and min(|·|) > 0.35
    
    Args:
        bar: Current bar with stream values
    
    Returns:
        List of detected alerts
    """
    alerts = []
    
    F = bar.get('sigma_f', 0.0)  # Flow
    M = bar.get('sigma_m', 0.0)  # Momentum
    
    THRESH_DIV = 0.30
    THRESH_CONF = 0.35
    
    # ========== DIVERGENCE ==========
    if np.sign(F) != np.sign(M) and abs(F) > THRESH_DIV and abs(M) > THRESH_DIV:
        # Flow and momentum disagree - reversal/squeeze risk
        if M > 0 and F < 0:
            message = f"Price drifting up (M={M:.3f}) while sell aggression dominates (F={F:.3f}) - fragile/short-covering prone"
        else:
            message = f"Price drifting down (M={M:.3f}) while buy aggression dominates (F={F:.3f}) - absorption/squeeze risk"
        
        alerts.append(StreamAlert(
            alert_type=AlertType.FLOW_DIVERGENCE,
            severity=AlertSeverity.WARNING,
            message=message,
            stream_name='flow',
            confidence=min(abs(F - M) / 2.0, 1.0),
            metadata={'F': F, 'M': M}
        ))
    
    # ========== CONFIRMATION ==========
    if np.sign(F) == np.sign(M) and min(abs(F), abs(M)) > THRESH_CONF:
        # Flow and momentum aligned - strong directional conviction
        direction = "up" if F > 0 else "down"
        alerts.append(StreamAlert(
            alert_type=AlertType.FLOW_CONFIRMATION,
            severity=AlertSeverity.INFO,
            message=f"Flow-momentum aligned {direction} (F={F:.3f}, M={M:.3f})",
            stream_name='flow',
            confidence=min(min(abs(F), abs(M)) / 1.0, 1.0),
            metadata={'F': F, 'M': M}
        ))
    
    return alerts


def detect_barrier_phase(bar: pd.Series) -> List[StreamAlert]:
    """
    Detect barrier microstructure phase (break vs reject support).
    
    Per STREAMS.md Section 10.3:
    - BARRIER_BREAK_SUPPORT: sign(B) == sign(P) and |B| > 0.30
    - BARRIER_OPPOSES_PRESSURE: sign(B) != sign(P) and |B| > 0.30
    - BARRIER_WEAKENING: |B| > 0.30 and sign(slope_B) != sign(B)
    
    Args:
        bar: Current bar with stream values
    
    Returns:
        List of detected alerts
    """
    alerts = []
    
    B = bar.get('sigma_b', 0.0)  # Barrier
    P = bar.get('sigma_p', 0.0)  # Pressure
    slope_B = bar.get('sigma_b_slope', 0.0)
    
    THRESH = 0.30
    
    if abs(B) < THRESH:
        return alerts  # Barrier not significant
    
    # ========== BREAK SUPPORT ==========
    if np.sign(B) == np.sign(P):
        # Barrier and pressure aligned - continuation likely
        direction = "upward" if B > 0 else "downward"
        alerts.append(StreamAlert(
            alert_type=AlertType.BARRIER_BREAK_SUPPORT,
            severity=AlertSeverity.INFO,
            message=f"Barrier supports {direction} continuation (B={B:.3f}, P={P:.3f})",
            stream_name='barrier',
            confidence=min(abs(B) / 1.0, 1.0),
            metadata={'B': B, 'P': P}
        ))
    
    # ========== BARRIER OPPOSES PRESSURE ==========
    elif np.sign(B) != np.sign(P):
        # Barrier opposes pressure - reject likely
        alerts.append(StreamAlert(
            alert_type=AlertType.BARRIER_OPPOSES_PRESSURE,
            severity=AlertSeverity.WARNING,
            message=f"Barrier opposes pressure (B={B:.3f}, P={P:.3f}) - reject risk",
            stream_name='barrier',
            confidence=min(abs(B - P) / 2.0, 1.0),
            metadata={'B': B, 'P': P}
        ))
    
    # ========== BARRIER WEAKENING ==========
    if np.sign(slope_B) != np.sign(B):
        # Barrier losing strength in its direction
        alerts.append(StreamAlert(
            alert_type=AlertType.BARRIER_WEAKENING,
            severity=AlertSeverity.WARNING,
            message=f"Barrier weakening (B={B:.3f}, slope={slope_B:.3f})",
            stream_name='barrier',
            confidence=min(abs(slope_B) * 5, 1.0),
            metadata={'B': B, 'slope_B': slope_B}
        ))
    
    return alerts


def detect_quality_gates(bar: pd.Series) -> List[StreamAlert]:
    """
    Detect setup quality and dealer regime gates.
    
    Per STREAMS.md Sections 10.4-10.5:
    - LOW_QUALITY: sigma_s < -0.25
    - HIGH_QUALITY: sigma_s > +0.25
    - FUEL_REGIME: sigma_d > +0.25 (amplification)
    - PIN_REGIME: sigma_d < -0.25 (dampening)
    
    Args:
        bar: Current bar with stream values
    
    Returns:
        List of detected alerts
    """
    alerts = []
    
    S = bar.get('sigma_s', 0.0)  # Setup quality
    D = bar.get('sigma_d', 0.0)  # Dealer regime
    
    # ========== SETUP QUALITY ==========
    if S < -0.25:
        alerts.append(StreamAlert(
            alert_type=AlertType.LOW_QUALITY_SETUP,
            severity=AlertSeverity.WARNING,
            message=f"Degraded setup quality (S={S:.3f}) - suppress aggressive signals",
            stream_name='setup',
            confidence=min(abs(S + 0.25) * 2, 1.0),
            metadata={'sigma_s': S}
        ))
    elif S > +0.25:
        alerts.append(StreamAlert(
            alert_type=AlertType.HIGH_QUALITY_SETUP,
            severity=AlertSeverity.INFO,
            message=f"High-quality setup (S={S:.3f}) - trust signals",
            stream_name='setup',
            confidence=min((S - 0.25) * 2, 1.0),
            metadata={'sigma_s': S}
        ))
    
    # ========== DEALER REGIME ==========
    if D > +0.25:
        alerts.append(StreamAlert(
            alert_type=AlertType.FUEL_REGIME,
            severity=AlertSeverity.INFO,
            message=f"Fuel regime (D={D:.3f}) - moves amplified, tighten reversal thresholds",
            stream_name='dealer',
            confidence=min((D - 0.25) * 2, 1.0),
            metadata={'sigma_d': D}
        ))
    elif D < -0.25:
        alerts.append(StreamAlert(
            alert_type=AlertType.PIN_REGIME,
            severity=AlertSeverity.INFO,
            message=f"Pin regime (D={D:.3f}) - moves dampened, reduce conviction",
            stream_name='dealer',
            confidence=min(abs(D + 0.25) * 2, 1.0),
            metadata={'sigma_d': D}
        ))
    
    return alerts


def compute_exit_score(
    bar: pd.Series,
    position_sign: float
) -> Dict[str, float]:
    """
    Compute position-aware exit score.
    
    Per STREAMS.md Section 10.6:
    - pos_sign = +1 for LONG, -1 for SHORT
    - E_exit = tanh(0.45*A + 0.25*P1 + 0.15*F1 + 0.15*B1)
    - Zones: > +0.50 = HOLD/ADD, [-0.20, +0.50] = HOLD/TRAIL,
             [-0.50, -0.20] = REDUCE, < -0.50 = EXIT
    
    Args:
        bar: Current bar with stream values and derivatives
        position_sign: +1.0 for LONG, -1.0 for SHORT
    
    Returns:
        Dictionary with exit_score, zone, and recommendation
    """
    # Extract streams and derivatives
    A = bar.get('alignment_adj', 0.0)  # Dealer-adjusted alignment
    P1 = bar.get('sigma_p_slope', 0.0)
    F1 = bar.get('sigma_f_slope', 0.0)
    B1 = bar.get('sigma_b_slope', 0.0)
    P3 = bar.get('sigma_p_jerk', 0.0)
    
    # Position-aware components
    A_pos = position_sign * A
    P1_pos = position_sign * P1
    F1_pos = position_sign * F1
    B1_pos = position_sign * B1
    
    # Compute raw exit score per STREAMS.md Section 10.6
    E_raw = 0.45 * A_pos + 0.25 * P1_pos + 0.15 * F1_pos + 0.15 * B1_pos
    E_exit = float(np.tanh(E_raw))
    
    # Jerk booster (early warning)
    jerk_penalty = 0.0
    if position_sign * P3 < -0.02:  # Jerk opposing position
        jerk_penalty = 0.15
        E_exit -= jerk_penalty
    
    # Determine zone and recommendation
    if E_exit > +0.50:
        zone = "HOLD_ADD"
        recommendation = "Hold position / Consider adding"
    elif E_exit >= -0.20:
        zone = "HOLD_TRAIL"
        recommendation = "Hold position / Trail stop"
    elif E_exit >= -0.50:
        zone = "REDUCE"
        recommendation = "Reduce position size"
    else:
        zone = "EXIT"
        recommendation = "Exit position"
    
    return {
        'exit_score': E_exit,
        'zone': zone,
        'recommendation': recommendation,
        'jerk_penalty': jerk_penalty,
        'components': {
            'alignment': A_pos,
            'pressure_slope': P1_pos,
            'flow_slope': F1_pos,
            'barrier_slope': B1_pos
        }
    }


def detect_alerts(
    bar: pd.Series,
    history_df: Optional[pd.DataFrame] = None
) -> List[StreamAlert]:
    """
    Detect all stream alerts for current bar.
    
    Main entry point for stream state machine.
    
    Args:
        bar: Current bar with stream values and derivatives
        history_df: Optional history for multi-bar patterns
    
    Returns:
        List of all detected alerts
    """
    alerts = []
    
    # Exhaustion/continuation patterns (Section 10.1)
    alerts.extend(detect_exhaustion_continuation(bar))
    
    # Flow-momentum divergence (Section 10.2)
    alerts.extend(detect_flow_divergence(bar))
    
    # Barrier phase detection (Section 10.3)
    alerts.extend(detect_barrier_phase(bar))
    
    # Quality gates (Sections 10.4-10.5)
    alerts.extend(detect_quality_gates(bar))
    
    return alerts


class StreamStateMachine:
    """
    Stateful alert tracker with hysteresis.
    
    Similar to TriggerStateMachine in score_engine.py,
    this tracks sustained conditions to prevent alert flickering.
    """
    
    def __init__(self, hold_time_seconds: float = 5.0):
        """
        Initialize state machine.
        
        Args:
            hold_time_seconds: How long condition must persist
        """
        self.hold_time_seconds = hold_time_seconds
        self.hold_time_ns = int(hold_time_seconds * 1e9)
        
        # Track when each alert type first triggered
        self.alert_triggered_ns: Dict[AlertType, Optional[int]] = {
            alert_type: None for alert_type in AlertType
        }
        
        # Currently active sustained alerts
        self.active_alerts: Dict[AlertType, StreamAlert] = {}
    
    def update(
        self,
        bar: pd.Series,
        ts_ns: int
    ) -> List[StreamAlert]:
        """
        Update state machine and return sustained alerts.
        
        Args:
            bar: Current bar with stream values
            ts_ns: Current timestamp (nanoseconds)
        
        Returns:
            List of sustained alerts (held for > hold_time_seconds)
        """
        # Detect instantaneous alerts
        instant_alerts = detect_alerts(bar)
        
        # Update tracking for each alert type
        instant_alert_types = {alert.alert_type for alert in instant_alerts}
        
        for alert_type in AlertType:
            if alert_type in instant_alert_types:
                # Alert triggered
                if self.alert_triggered_ns[alert_type] is None:
                    # First time triggering
                    self.alert_triggered_ns[alert_type] = ts_ns
                elif ts_ns - self.alert_triggered_ns[alert_type] >= self.hold_time_ns:
                    # Sustained for required duration - activate
                    alert = next(a for a in instant_alerts if a.alert_type == alert_type)
                    self.active_alerts[alert_type] = alert
            else:
                # Alert not triggered - reset
                self.alert_triggered_ns[alert_type] = None
                if alert_type in self.active_alerts:
                    del self.active_alerts[alert_type]
        
        # Return currently active sustained alerts
        return list(self.active_alerts.values())
    
    def reset(self):
        """Reset state machine."""
        self.alert_triggered_ns = {alert_type: None for alert_type in AlertType}
        self.active_alerts = {}
    
    def get_active_alert_types(self) -> List[AlertType]:
        """Get list of currently active alert types."""
        return list(self.active_alerts.keys())

