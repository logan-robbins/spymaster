"""
Smoothing: EWMA and robust smoothing for stable signal outputs.

Agent G deliverable per §12 of PLAN.md.

This module provides:
- EWMA (Exponential Weighted Moving Average) for time-series smoothing
- Optional robust smoothing (median, Huber-like) for outlier rejection

Per §5.6 of PLAN.md, we smooth:
- BreakScore
- delta_liq (barrier)
- replenishment_ratio (barrier)
- velocity (tape)
- net_dealer_gamma (fuel)

All smoothers maintain state and are designed to be called on each snap tick.
"""

import math
from typing import Optional
from collections import deque


class EWMA:
    """
    Exponential Weighted Moving Average smoother.
    
    Formula:
        x_smooth(t) = α * x(t) + (1 - α) * x_smooth(t-1)
        
    Where α = 1 - exp(-Δt / τ) and τ is the half-life parameter.
    
    Usage:
        smoother = EWMA(tau=2.0)
        for value, timestamp in values:
            smoothed = smoother.update(value, timestamp)
    """
    
    def __init__(self, tau: float):
        """
        Initialize EWMA smoother.
        
        Args:
            tau: Half-life in seconds (time constant for exponential decay)
        """
        self.tau = tau
        self.value: Optional[float] = None
        self.last_ts_ns: Optional[int] = None
    
    def update(self, value: float, ts_ns: int) -> float:
        """
        Update smoother with new value.
        
        Args:
            value: New value to incorporate
            ts_ns: Timestamp in nanoseconds
            
        Returns:
            Smoothed value
        """
        if self.value is None:
            # First value, initialize
            self.value = value
            self.last_ts_ns = ts_ns
            return value
        
        # Compute time delta in seconds
        dt_seconds = (ts_ns - self.last_ts_ns) / 1e9
        
        # Compute alpha (adaptive based on actual time gap)
        alpha = 1.0 - math.exp(-dt_seconds / self.tau)
        alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]
        
        # Update smoothed value
        self.value = alpha * value + (1.0 - alpha) * self.value
        self.last_ts_ns = ts_ns
        
        return self.value
    
    def get(self) -> Optional[float]:
        """Get current smoothed value without updating."""
        return self.value
    
    def reset(self):
        """Reset smoother state."""
        self.value = None
        self.last_ts_ns = None


class RobustRollingMedian:
    """
    Rolling median filter for outlier rejection.
    
    Maintains a fixed-size window and returns the median value.
    Useful for filtering spike noise in quote sizes or scores.
    
    Usage:
        smoother = RobustRollingMedian(window_size=10)
        for value in values:
            smoothed = smoother.update(value)
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize rolling median filter.
        
        Args:
            window_size: Number of values to keep in window
        """
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
    
    def update(self, value: float) -> float:
        """
        Update filter with new value.
        
        Args:
            value: New value to incorporate
            
        Returns:
            Median of current window
        """
        self.buffer.append(value)
        
        if len(self.buffer) == 0:
            return value
        
        # Compute median
        sorted_buffer = sorted(self.buffer)
        n = len(sorted_buffer)
        
        if n % 2 == 0:
            # Even length, average middle two
            return (sorted_buffer[n // 2 - 1] + sorted_buffer[n // 2]) / 2.0
        else:
            # Odd length, return middle
            return sorted_buffer[n // 2]
    
    def get(self) -> Optional[float]:
        """Get current median without updating."""
        if len(self.buffer) == 0:
            return None
        
        sorted_buffer = sorted(self.buffer)
        n = len(sorted_buffer)
        
        if n % 2 == 0:
            return (sorted_buffer[n // 2 - 1] + sorted_buffer[n // 2]) / 2.0
        else:
            return sorted_buffer[n // 2]
    
    def reset(self):
        """Reset filter state."""
        self.buffer.clear()


class HybridSmoother:
    """
    Hybrid smoother combining EWMA with optional robust pre-filtering.
    
    Pipeline:
        raw_value -> [optional median filter] -> EWMA -> smooth_value
    
    Usage:
        smoother = HybridSmoother(tau=2.0, use_robust=True, robust_window=5)
        smoothed = smoother.update(value, timestamp)
    """
    
    def __init__(
        self, 
        tau: float, 
        use_robust: bool = False, 
        robust_window: int = 5
    ):
        """
        Initialize hybrid smoother.
        
        Args:
            tau: EWMA half-life in seconds
            use_robust: Whether to apply robust pre-filtering
            robust_window: Window size for robust filter
        """
        self.ewma = EWMA(tau=tau)
        self.use_robust = use_robust
        
        if use_robust:
            self.robust_filter = RobustRollingMedian(window_size=robust_window)
        else:
            self.robust_filter = None
    
    def update(self, value: float, ts_ns: int) -> float:
        """
        Update smoother with new value.
        
        Args:
            value: Raw value
            ts_ns: Timestamp in nanoseconds
            
        Returns:
            Smoothed value
        """
        if self.use_robust and self.robust_filter is not None:
            # Pre-filter with robust median
            filtered_value = self.robust_filter.update(value)
            # Then apply EWMA
            return self.ewma.update(filtered_value, ts_ns)
        else:
            # Direct EWMA
            return self.ewma.update(value, ts_ns)
    
    def get(self) -> Optional[float]:
        """Get current smoothed value without updating."""
        return self.ewma.get()
    
    def reset(self):
        """Reset all smoother state."""
        self.ewma.reset()
        if self.robust_filter is not None:
            self.robust_filter.reset()


class SmootherSet:
    """
    Collection of smoothers for all level signals.
    
    Per §5.6 of PLAN.md, we maintain smoothed versions of:
    - break_score
    - delta_liq
    - replenishment_ratio
    - velocity
    - net_dealer_gamma
    
    Usage:
        smoothers = SmootherSet()
        smoothers.update_score(raw_score, timestamp)
        smoothed_score = smoothers.get_score()
    """
    
    def __init__(self, config=None):
        """
        Initialize smoother set using config parameters.
        
        Args:
            config: Config object (defaults to global CONFIG)
        """
        from src.common.config import CONFIG
        config = config or CONFIG
        
        # Create smoothers with configured time constants
        self.score_smoother = EWMA(tau=config.tau_score)
        self.delta_liq_smoother = EWMA(tau=config.tau_delta_liq)
        self.replenish_smoother = EWMA(tau=config.tau_replenish)
        self.velocity_smoother = EWMA(tau=config.tau_velocity)
        self.dealer_gamma_smoother = EWMA(tau=config.tau_dealer_gamma)
    
    def update_score(self, score: float, ts_ns: int) -> float:
        """Update and return smoothed break score."""
        return self.score_smoother.update(score, ts_ns)
    
    def update_delta_liq(self, delta_liq: float, ts_ns: int) -> float:
        """Update and return smoothed delta_liq."""
        return self.delta_liq_smoother.update(delta_liq, ts_ns)
    
    def update_replenishment(self, ratio: float, ts_ns: int) -> float:
        """Update and return smoothed replenishment ratio."""
        return self.replenish_smoother.update(ratio, ts_ns)
    
    def update_velocity(self, velocity: float, ts_ns: int) -> float:
        """Update and return smoothed velocity."""
        return self.velocity_smoother.update(velocity, ts_ns)
    
    def update_dealer_gamma(self, gamma: float, ts_ns: int) -> float:
        """Update and return smoothed net dealer gamma."""
        return self.dealer_gamma_smoother.update(gamma, ts_ns)
    
    def get_score(self) -> Optional[float]:
        """Get current smoothed score without updating."""
        return self.score_smoother.get()
    
    def get_delta_liq(self) -> Optional[float]:
        """Get current smoothed delta_liq without updating."""
        return self.delta_liq_smoother.get()
    
    def get_replenishment(self) -> Optional[float]:
        """Get current smoothed replenishment ratio without updating."""
        return self.replenish_smoother.get()
    
    def get_velocity(self) -> Optional[float]:
        """Get current smoothed velocity without updating."""
        return self.velocity_smoother.get()
    
    def get_dealer_gamma(self) -> Optional[float]:
        """Get current smoothed dealer gamma without updating."""
        return self.dealer_gamma_smoother.get()
    
    def reset(self):
        """Reset all smoothers."""
        self.score_smoother.reset()
        self.delta_liq_smoother.reset()
        self.replenish_smoother.reset()
        self.velocity_smoother.reset()
        self.dealer_gamma_smoother.reset()
