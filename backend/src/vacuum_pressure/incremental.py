"""Incremental vacuum-pressure signal engine — Bernoulli lift model.

Stateful per-window signal computation implementing a fluid-dynamics-inspired
order flow model.  Computes "lift" (upward price force) and "drag" (downward
price force) from per-bucket order book flow data at three timescales:
fast (~5 s), medium (~15 s), and slow (~60 s).

Architecture:
    Stages 1-2 (per-bucket scores, window aggregation) are window-local
    with zero lookback — reuse existing pure functions from formulas.py.

    Stage 3 (spatial aggregation) computes pressure and resistance fields
    from flow_df using vectorized numpy with proximity weighting.

    Stage 4 (Bernoulli lift) combines pressure, vacuum, and resistance
    into a net directional force signal:
        lift_up  = P_below * V_above / (R_above + ε)
        lift_down = P_above * V_below / (R_below + ε)
        net_lift  = lift_up − lift_down

    Stages 5-7 (multi-timescale smoothing, confidence, regime, alerts)
    maintain rolling state via IncrementalEMA and DiffEMA accumulators.

State footprint:
    ~18 scalar floats (6 EMA states × 3 timescale engines)
    + 2 previous-value floats for alert detection
    = negligible memory.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .formulas import (
    EPS,
    NEAR_SPOT_DOLLARS,
    PROXIMITY_TAU_DOLLARS,
    GoldSignalConfig,
    aggregate_window_metrics,
    compute_per_bucket_scores,
    proximity_weight,
)

# Re-export for type checking
if False:  # TYPE_CHECKING
    from .config import VPRuntimeConfig


# ──────────────────────────────────────────────────────────────────────
# Alert flag constants
# ──────────────────────────────────────────────────────────────────────

ALERT_INFLECTION: int = 1
"""d1 crossed zero — velocity direction changed."""

ALERT_DECELERATION: int = 2
"""sign(d2) != sign(d1) — momentum fading."""

ALERT_REGIME_SHIFT: int = 4
"""d2 crossed zero while |d1| sustained — curvature reversal."""

REGIME_THRESHOLD: float = 0.5
"""Minimum |net_lift| magnitude for directional regime classification."""

EVENT_STATE_WATCH: str = "WATCH"
EVENT_STATE_ARMED: str = "ARMED"
EVENT_STATE_FIRE: str = "FIRE"
EVENT_STATE_COOLDOWN: str = "COOLDOWN"

EVENT_DIRECTION_UP: str = "UP"
EVENT_DIRECTION_DOWN: str = "DOWN"
EVENT_DIRECTION_NONE: str = "NONE"

REQUIRED_FLOW_COLUMNS: frozenset[str] = frozenset({
    "window_end_ts_ns",
    "rel_ticks",
    "side",
    "add_qty",
    "pull_qty",
    "fill_qty",
    "depth_qty_end",
    "depth_qty_rest",
    "pull_qty_rest",
    "spot_ref_price_int",
    "window_valid",
})
"""Required flow columns for deterministic per-window signal computation."""

NON_NEGATIVE_FLOW_COLUMNS: tuple[str, ...] = (
    "add_qty",
    "pull_qty",
    "fill_qty",
    "depth_qty_end",
    "depth_qty_rest",
    "pull_qty_rest",
)
"""Flow quantity columns that must be non-negative."""


@dataclass(frozen=True)
class EventStateConfig:
    """Thresholds and counters for deterministic directional eventing."""

    arm_net_lift_threshold: float = 0.75
    disarm_net_lift_threshold: float = 0.40
    fire_net_lift_threshold: float = 1.00
    min_abs_d1_arm: float = 0.04
    min_abs_d1_fire: float = 0.08
    max_countertrend_d2: float = 0.03
    min_cross_conf_arm: float = 0.25
    min_cross_conf_fire: float = 0.45
    min_projection_coh_arm: float = 0.20
    min_projection_coh_fire: float = 0.35
    arm_persistence_windows: int = 2
    fire_persistence_windows: int = 2
    fire_min_hold_windows: int = 1
    cooldown_windows: int = 5

    def validate(self) -> None:
        """Validate configuration and fail fast on invalid thresholds."""
        if not (
            self.arm_net_lift_threshold > 0.0
            and self.disarm_net_lift_threshold > 0.0
            and self.fire_net_lift_threshold > 0.0
        ):
            raise ValueError("Event net_lift thresholds must be > 0")
        if self.disarm_net_lift_threshold >= self.arm_net_lift_threshold:
            raise ValueError(
                "disarm_net_lift_threshold must be < arm_net_lift_threshold"
            )
        if self.arm_net_lift_threshold > self.fire_net_lift_threshold:
            raise ValueError(
                "arm_net_lift_threshold must be <= fire_net_lift_threshold"
            )
        if self.min_abs_d1_arm <= 0.0 or self.min_abs_d1_fire <= 0.0:
            raise ValueError("min_abs_d1_* thresholds must be > 0")
        if self.min_abs_d1_arm > self.min_abs_d1_fire:
            raise ValueError("min_abs_d1_arm must be <= min_abs_d1_fire")
        if self.max_countertrend_d2 < 0.0:
            raise ValueError("max_countertrend_d2 must be >= 0")
        for name, value in (
            ("min_cross_conf_arm", self.min_cross_conf_arm),
            ("min_cross_conf_fire", self.min_cross_conf_fire),
            ("min_projection_coh_arm", self.min_projection_coh_arm),
            ("min_projection_coh_fire", self.min_projection_coh_fire),
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.min_cross_conf_arm > self.min_cross_conf_fire:
            raise ValueError("min_cross_conf_arm must be <= min_cross_conf_fire")
        if self.min_projection_coh_arm > self.min_projection_coh_fire:
            raise ValueError(
                "min_projection_coh_arm must be <= min_projection_coh_fire"
            )
        for name, value in (
            ("arm_persistence_windows", self.arm_persistence_windows),
            ("fire_persistence_windows", self.fire_persistence_windows),
            ("fire_min_hold_windows", self.fire_min_hold_windows),
            ("cooldown_windows", self.cooldown_windows),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")


class DirectionalEventStateMachine:
    """Deterministic WATCH/ARMED/FIRE/COOLDOWN event state machine."""

    __slots__ = (
        "_cfg",
        "_state",
        "_direction",
        "_candidate_direction",
        "_candidate_windows",
        "_fire_windows",
        "_cooldown_remaining",
    )

    def __init__(self, config: EventStateConfig | None = None) -> None:
        self._cfg: EventStateConfig = config or EventStateConfig()
        self._cfg.validate()
        self._state: str = EVENT_STATE_WATCH
        self._direction: int = 0
        self._candidate_direction: int = 0
        self._candidate_windows: int = 0
        self._fire_windows: int = 0
        self._cooldown_remaining: int = 0

    @staticmethod
    def _sign(value: float) -> int:
        return int(np.sign(value))

    @staticmethod
    def _direction_label(direction: int) -> str:
        if direction > 0:
            return EVENT_DIRECTION_UP
        if direction < 0:
            return EVENT_DIRECTION_DOWN
        return EVENT_DIRECTION_NONE

    @staticmethod
    def _clip01(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    def _direction_vote(
        self,
        net_lift: float,
        d1_15s: float,
        projection_direction: int,
    ) -> int:
        """Vote a deterministic direction from medium trend + projection."""
        s_net = self._sign(net_lift)
        s_d1 = self._sign(d1_15s)
        if s_net == 0 or s_d1 == 0 or s_net != s_d1:
            return 0
        if projection_direction != 0 and projection_direction != s_net:
            return 0
        return s_net

    def _arm_gate(
        self,
        direction: int,
        net_lift: float,
        d1_15s: float,
        d2_15s: float,
        cross_confidence: float,
        projection_coherence: float,
    ) -> bool:
        if direction == 0:
            return False
        return (
            net_lift * direction >= self._cfg.arm_net_lift_threshold
            and d1_15s * direction >= self._cfg.min_abs_d1_arm
            and d2_15s * direction >= -self._cfg.max_countertrend_d2
            and cross_confidence >= self._cfg.min_cross_conf_arm
            and projection_coherence >= self._cfg.min_projection_coh_arm
        )

    def _fire_gate(
        self,
        direction: int,
        net_lift: float,
        d1_15s: float,
        d2_15s: float,
        cross_confidence: float,
        projection_coherence: float,
    ) -> bool:
        if direction == 0:
            return False
        return (
            net_lift * direction >= self._cfg.fire_net_lift_threshold
            and d1_15s * direction >= self._cfg.min_abs_d1_fire
            and d2_15s * direction >= -0.5 * self._cfg.max_countertrend_d2
            and cross_confidence >= self._cfg.min_cross_conf_fire
            and projection_coherence >= self._cfg.min_projection_coh_fire
        )

    def _hold_gate(
        self,
        direction: int,
        net_lift: float,
        d1_15s: float,
        cross_confidence: float,
        projection_coherence: float,
    ) -> bool:
        if direction == 0:
            return False
        return (
            net_lift * direction >= self._cfg.disarm_net_lift_threshold
            and d1_15s * direction >= 0.5 * self._cfg.min_abs_d1_arm
            and cross_confidence >= 0.8 * self._cfg.min_cross_conf_arm
            and projection_coherence >= 0.8 * self._cfg.min_projection_coh_arm
        )

    def _event_strength(
        self,
        direction: int,
        net_lift: float,
        d1_15s: float,
        d2_15s: float,
    ) -> float:
        if direction == 0:
            return 0.0
        net_term = min(1.0, abs(net_lift) / (self._cfg.fire_net_lift_threshold + EPS))
        d1_term = min(1.0, abs(d1_15s) / (self._cfg.min_abs_d1_fire + EPS))
        d2_term = self._clip01(
            (d2_15s * direction + self._cfg.max_countertrend_d2)
            / (2.0 * self._cfg.max_countertrend_d2 + EPS)
        )
        return self._clip01(0.50 * net_term + 0.30 * d1_term + 0.20 * d2_term)

    def update(
        self,
        net_lift: float,
        d1_15s: float,
        d2_15s: float,
        cross_confidence: float,
        projection_coherence: float,
        projection_direction: int,
    ) -> Dict[str, Any]:
        """Advance machine by one window and return event output fields."""
        projection_direction = int(np.sign(projection_direction))
        voted_direction = self._direction_vote(net_lift, d1_15s, projection_direction)

        if self._state == EVENT_STATE_COOLDOWN:
            self._cooldown_remaining = max(0, self._cooldown_remaining - 1)
            if self._cooldown_remaining == 0:
                self._state = EVENT_STATE_WATCH

        if self._state != EVENT_STATE_COOLDOWN:
            arm_ok = self._arm_gate(
                voted_direction,
                net_lift,
                d1_15s,
                d2_15s,
                cross_confidence,
                projection_coherence,
            )
            if arm_ok:
                if voted_direction == self._candidate_direction:
                    self._candidate_windows += 1
                else:
                    self._candidate_direction = voted_direction
                    self._candidate_windows = 1
            else:
                self._candidate_direction = 0
                self._candidate_windows = 0

            if self._state == EVENT_STATE_WATCH:
                if self._candidate_windows >= self._cfg.arm_persistence_windows:
                    self._state = EVENT_STATE_ARMED
                    self._direction = self._candidate_direction
                    self._fire_windows = 0

            elif self._state == EVENT_STATE_ARMED:
                hold_ok = self._hold_gate(
                    self._direction,
                    net_lift,
                    d1_15s,
                    cross_confidence,
                    projection_coherence,
                )
                if not hold_ok:
                    self._state = EVENT_STATE_WATCH
                    self._direction = 0
                    self._fire_windows = 0
                else:
                    fire_ok = self._fire_gate(
                        self._direction,
                        net_lift,
                        d1_15s,
                        d2_15s,
                        cross_confidence,
                        projection_coherence,
                    )
                    self._fire_windows = self._fire_windows + 1 if fire_ok else 0
                    if self._fire_windows >= self._cfg.fire_persistence_windows:
                        self._state = EVENT_STATE_FIRE
                        self._fire_windows = 0

            elif self._state == EVENT_STATE_FIRE:
                self._fire_windows += 1
                hold_ok = self._hold_gate(
                    self._direction,
                    net_lift,
                    d1_15s,
                    cross_confidence,
                    projection_coherence,
                )
                if (
                    self._fire_windows >= self._cfg.fire_min_hold_windows
                    and not hold_ok
                ):
                    self._state = EVENT_STATE_COOLDOWN
                    self._cooldown_remaining = self._cfg.cooldown_windows
                    self._direction = 0
                    self._candidate_direction = 0
                    self._candidate_windows = 0
                    self._fire_windows = 0

        active_direction = (
            self._direction if self._state in (EVENT_STATE_ARMED, EVENT_STATE_FIRE) else 0
        )
        strength = self._event_strength(active_direction, net_lift, d1_15s, d2_15s)
        coherence = self._clip01(0.5 * (cross_confidence + projection_coherence))
        confidence = (
            self._clip01(0.70 * coherence + 0.30 * strength)
            if active_direction != 0
            else 0.0
        )

        return {
            "event_state": self._state,
            "event_direction": self._direction_label(active_direction),
            "event_strength": strength,
            "event_confidence": confidence,
        }


# ──────────────────────────────────────────────────────────────────────
# Incremental math primitives
# ──────────────────────────────────────────────────────────────────────


class IncrementalEMA:
    """Exponential moving average with O(1) per-tick update.

    Implements: EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}
    where alpha = 2 / (span + 1), matching pandas ewm(span=N, adjust=False).
    """

    __slots__ = ("alpha", "value", "initialized")

    def __init__(self, span: int) -> None:
        self.alpha: float = 2.0 / (span + 1.0)
        self.value: float = 0.0
        self.initialized: bool = False

    def update(self, x: float) -> float:
        """Update with a new value and return the new EMA."""
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = 0.0
        self.initialized = False


class RollingStats:
    """Rolling window mean/std with O(1) amortized update.

    Uses a ring buffer with running sum and sum-of-squares for
    numerically stable incremental statistics.
    """

    __slots__ = ("window", "buffer", "sum_x", "sum_x2", "count")

    def __init__(self, window: int) -> None:
        self.window: int = window
        self.buffer: deque[float] = deque(maxlen=window)
        self.sum_x: float = 0.0
        self.sum_x2: float = 0.0
        self.count: int = 0

    def update(self, x: float) -> tuple[float, float]:
        """Add a value, return (mean, std)."""
        # Evict oldest if buffer is full
        if self.count >= self.window:
            old = self.buffer[0]
            self.sum_x -= old
            self.sum_x2 -= old * old
            self.count -= 1

        self.buffer.append(x)
        self.sum_x += x
        self.sum_x2 += x * x
        self.count += 1

        mean = self.sum_x / self.count
        variance = (self.sum_x2 / self.count) - (mean * mean)
        if variance < 0:
            variance = 0.0  # numerical guard
        std = math.sqrt(variance)
        return mean, std

    def reset(self) -> None:
        self.buffer.clear()
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.count = 0


class DiffEMA:
    """First-difference followed by EMA smoothing.

    Implements: diff_t = x_t - x_{t-1}, then EMA(diff, span).
    This replaces pd.Series.diff().ewm(span=N, adjust=False).mean().
    """

    __slots__ = ("prev", "ema", "has_prev")

    def __init__(self, span: int) -> None:
        self.prev: float = 0.0
        self.ema: IncrementalEMA = IncrementalEMA(span)
        self.has_prev: bool = False

    def update(self, x: float) -> float:
        """Update with new value, return smoothed derivative."""
        if not self.has_prev:
            self.has_prev = True
            self.prev = x
            # First diff is NaN in pandas; we use 0.0 to match fillna(0)
            return self.ema.update(0.0)
        diff = x - self.prev
        self.prev = x
        return self.ema.update(diff)

    def reset(self) -> None:
        self.prev = 0.0
        self.ema.reset()
        self.has_prev = False


# ──────────────────────────────────────────────────────────────────────
# Multi-timescale smoothing engine
# ──────────────────────────────────────────────────────────────────────


class TimescaleEngine:
    r"""Per-timescale smoothing + derivatives + Taylor projection.

    Applies a pre-smoothing EMA, then computes 1st and 2nd derivatives
    via DiffEMA, and produces a 2nd-order Taylor forward projection:

    .. math::
        \hat{x}(t+k) = \bar{x}(t) + \dot{x}(t) \, k
                      + \ddot{x}(t) \, \frac{k^2}{2}

    Args:
        pre_smooth_span: EMA span for pre-smoothing the raw signal.
        d1_span: EMA span for 1st derivative (velocity) smoothing.
        d2_span: EMA span for 2nd derivative (acceleration) smoothing.
        proj_horizon_s: Forward projection horizon in seconds.
    """

    __slots__ = ("pre_smooth", "d1", "d2", "proj_horizon")

    def __init__(
        self,
        pre_smooth_span: int,
        d1_span: int,
        d2_span: int,
        proj_horizon_s: float,
    ) -> None:
        self.pre_smooth: IncrementalEMA = IncrementalEMA(pre_smooth_span)
        self.d1: DiffEMA = DiffEMA(d1_span)
        self.d2: DiffEMA = DiffEMA(d2_span)
        self.proj_horizon: float = proj_horizon_s

    def update(self, value: float) -> Dict[str, float]:
        """Update with a new raw value, return smoothed state dict.

        Returns:
            Dict with keys: ``smooth``, ``d1``, ``d2``, ``projection``.
        """
        smooth = self.pre_smooth.update(value)
        v1 = self.d1.update(smooth)
        v2 = self.d2.update(v1)
        k = self.proj_horizon
        proj = smooth + v1 * k + v2 * (k * k * 0.5)
        return {"smooth": smooth, "d1": v1, "d2": v2, "projection": proj}


# ──────────────────────────────────────────────────────────────────────
# Incremental Signal Engine
# ──────────────────────────────────────────────────────────────────────


class IncrementalSignalEngine:
    r"""Stateful per-window Bernoulli lift signal computation.

    Accepts one window's worth of ``(snap_dict, flow_rows_df)`` and returns
    the full signals dict, maintaining rolling state across windows.

    The fluid dynamics model computes:

    1. **Pressure fields** — bid/ask activity near spot:

       .. math::
           P_{\text{below}} = \sum_{\substack{s=B \\ |k| \le N}}
               (\text{fill} + \text{add})_{k,s} \, w(k)

    2. **Resistance fields** — resting depth walls:

       .. math::
           R_{\text{above}} = \sum_{\substack{s=A \\ k > 0}}
               \text{depth\_rest}_{k,s} \, w(k)

    3. **Bernoulli lift** — pressure through vacuum past resistance:

       .. math::
           L_{\uparrow} = \frac{P_{\text{below}} \cdot V_{\text{above}}}
                               {R_{\text{above}} + \varepsilon}

    4. **Multi-timescale** smoothing (fast/medium/slow) with Taylor
       projections at each scale.
    5. **Cross-timescale confidence** and regime classification.
    """

    def __init__(
        self,
        bucket_size_dollars: float,
        gold_config: GoldSignalConfig | None = None,
    ) -> None:
        if not np.isfinite(bucket_size_dollars) or bucket_size_dollars <= 0.0:
            raise ValueError(
                f"bucket_size_dollars must be finite and > 0, got {bucket_size_dollars}"
            )
        self.bucket_size_dollars: float = bucket_size_dollars
        self.gold_cfg: GoldSignalConfig = gold_config or GoldSignalConfig()
        self.gold_cfg.validate()

        self.window_count: int = 0

        # Proximity weighting parameters (match formulas.py conventions)
        self._tau_ticks: float = PROXIMITY_TAU_DOLLARS / bucket_size_dollars
        self._near_ticks: int = int(round(NEAR_SPOT_DOLLARS / bucket_size_dollars))
        if not np.isfinite(self._tau_ticks) or self._tau_ticks <= 0.0:
            raise ValueError(f"Invalid proximity tau (ticks): {self._tau_ticks}")
        if self._near_ticks < 1:
            raise ValueError(f"Invalid near-spot tick radius: {self._near_ticks}")

        # Multi-timescale engines: fast (~5 s), medium (~15 s), slow (~60 s)
        self._ts_fast: TimescaleEngine = TimescaleEngine(
            pre_smooth_span=3, d1_span=3, d2_span=5, proj_horizon_s=2.0,
        )
        self._ts_medium: TimescaleEngine = TimescaleEngine(
            pre_smooth_span=8, d1_span=8, d2_span=15, proj_horizon_s=10.0,
        )
        self._ts_slow: TimescaleEngine = TimescaleEngine(
            pre_smooth_span=30, d1_span=20, d2_span=40, proj_horizon_s=30.0,
        )

        # Alert state: previous medium-timescale derivatives
        self._prev_d1_medium: Optional[float] = None
        self._prev_d2_medium: Optional[float] = None
        self._event_machine: DirectionalEventStateMachine = (
            DirectionalEventStateMachine(EventStateConfig())
        )

    @staticmethod
    def _validate_flow_df(flow_df: pd.DataFrame) -> None:
        """Fail fast for missing schema columns and degenerate quantities."""
        missing = REQUIRED_FLOW_COLUMNS.difference(flow_df.columns)
        if missing:
            missing_sorted = ", ".join(sorted(missing))
            raise KeyError(
                f"flow_df missing required columns: {missing_sorted}"
            )

        numeric_cols = (
            "rel_ticks",
            "add_qty",
            "pull_qty",
            "fill_qty",
            "depth_qty_end",
            "depth_qty_rest",
            "pull_qty_rest",
        )
        for col in numeric_cols:
            col_values = flow_df[col].values
            if not np.isfinite(col_values).all():
                raise ValueError(f"Non-finite values in flow_df['{col}']")

        for col in NON_NEGATIVE_FLOW_COLUMNS:
            if (flow_df[col].values < 0.0).any():
                raise ValueError(
                    f"Negative quantity in flow_df['{col}']; expected non-negative values"
                )

    @staticmethod
    def _compute_feasibility_gate(
        lift_up: float,
        lift_down: float,
    ) -> tuple[float, float, float]:
        r"""Compute bounded directional feasibility from Bernoulli lift.

        The gate is deterministic and anti-noise by construction:

        .. math::
            u = \log(1 + \max(L_{\uparrow}, 0)), \quad
            d = \log(1 + \max(L_{\downarrow}, 0))

        .. math::
            a = \tanh(u + d)

        .. math::
            f_{\uparrow} = a \cdot \frac{u}{u + d + \varepsilon}, \quad
            f_{\downarrow} = a \cdot \frac{d}{u + d + \varepsilon}

        .. math::
            b = \tanh(u - d)

        Returns:
            ``(feasibility_up, feasibility_down, directional_bias)`` where
            feasibility values are in ``[0, 1]`` and directional_bias in
            ``[-1, 1]``.
        """
        u_raw = max(lift_up, 0.0)
        d_raw = max(lift_down, 0.0)
        u = float(np.log1p(u_raw))
        d = float(np.log1p(d_raw))

        if not np.isfinite(u) or not np.isfinite(d):
            raise ValueError(
                f"Invalid feasibility inputs after log1p transform: up={u}, down={d}"
            )

        score_sum = u + d
        if score_sum <= EPS:
            return 0.0, 0.0, 0.0

        activation = float(np.tanh(score_sum))
        feasibility_up = activation * (u / (score_sum + EPS))
        feasibility_down = activation * (d / (score_sum + EPS))
        directional_bias = float(np.tanh(u - d))

        # Hard numeric guards for wire compatibility.
        feasibility_up = min(1.0, max(0.0, feasibility_up))
        feasibility_down = min(1.0, max(0.0, feasibility_down))
        directional_bias = min(1.0, max(-1.0, directional_bias))
        return feasibility_up, feasibility_down, directional_bias

    def process_window(
        self,
        snap_dict: Dict[str, Any],
        flow_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        r"""Process one window and return the full signals dict.

        Implements the Bernoulli-inspired fluid dynamics model:

        1. Spatial aggregation — pressure + resistance fields from flow_df.
        2. Bernoulli lift:
           :math:`L = P \cdot V / (R + \varepsilon)`.
        3. Multi-timescale derivative chains (fast / medium / slow).
        4. Cross-timescale confidence + regime classification.
        5. Slope-change alerts on medium timescale.

        Args:
            snap_dict: Single snap row as a dict with keys matching
                ``book_snapshot_1s`` schema (``window_end_ts_ns``,
                ``mid_price``, etc.).
            flow_df: DataFrame of ``depth_and_flow_1s`` rows for this
                window.  Required columns: ``rel_ticks``, ``side``,
                ``fill_qty``, ``add_qty``, ``depth_qty_rest``,
                ``window_valid``.

        Returns:
            Dict with all signal columns for the wire protocol.
        """
        self.window_count += 1
        bucket = self.bucket_size_dollars

        window_end_ts_ns = snap_dict.get("window_end_ts_ns", 0)

        # ── Stage 1+2: Per-bucket scores + window aggregation ────────
        if flow_df.empty:
            event_ctx = self._event_machine.update(
                net_lift=0.0,
                d1_15s=0.0,
                d2_15s=0.0,
                cross_confidence=0.0,
                projection_coherence=0.0,
                projection_direction=0,
            )
            return self._empty_signals(window_end_ts_ns, snap_dict, event_ctx)
        self._validate_flow_df(flow_df)
        if not flow_df["window_valid"].astype(bool).any():
            event_ctx = self._event_machine.update(
                net_lift=0.0,
                d1_15s=0.0,
                d2_15s=0.0,
                cross_confidence=0.0,
                projection_coherence=0.0,
                projection_direction=0,
            )
            return self._empty_signals(window_end_ts_ns, snap_dict, event_ctx)

        # compute_per_bucket_scores adds derived columns to flow_df
        compute_per_bucket_scores(flow_df, bucket)

        # aggregate_window_metrics groups by window — we have exactly 1
        metrics_df = aggregate_window_metrics(flow_df, bucket)
        if metrics_df.empty:
            event_ctx = self._event_machine.update(
                net_lift=0.0,
                d1_15s=0.0,
                d2_15s=0.0,
                cross_confidence=0.0,
                projection_coherence=0.0,
                projection_direction=0,
            )
            return self._empty_signals(window_end_ts_ns, snap_dict, event_ctx)

        # Extract the single row as a dict
        m = metrics_df.iloc[0].to_dict()

        # ── Stage 3: Spatial aggregation (pressure + resistance) ─────
        # Vectorised computation using same proximity weighting as vacuum
        valid_mask = flow_df["window_valid"].values.astype(bool)
        rt = flow_df["rel_ticks"].values
        side = flow_df["side"].values
        w = proximity_weight(rt, self._tau_ticks)

        fill = flow_df["fill_qty"].values
        add = flow_df["add_qty"].values
        depth_rest = flow_df["depth_qty_rest"].values

        is_ask = side == "A"
        is_bid = side == "B"
        above = rt > 0
        below = rt < 0
        near = np.abs(rt) <= self._near_ticks

        # Pressure: active force near spot (fills + adds, proximity-weighted)
        # P_below = bid-side activity pushing UP toward spot
        # P_above = ask-side activity pushing DOWN toward spot
        activity = (fill + add) * w
        P_below = float(np.sum(
            np.where(is_bid & near & valid_mask, activity, 0.0)
        ))
        P_above = float(np.sum(
            np.where(is_ask & near & valid_mask, activity, 0.0)
        ))

        # Resistance: resting depth walls (proximity-weighted)
        # R_above = ask-side resting depth above spot (wall above)
        # R_below = bid-side resting depth below spot (wall below)
        rest_weighted = depth_rest * w
        R_above = float(np.sum(
            np.where(is_ask & above & valid_mask, rest_weighted, 0.0)
        ))
        R_below = float(np.sum(
            np.where(is_bid & below & valid_mask, rest_weighted, 0.0)
        ))

        # ── Stage 4: Bernoulli lift ──────────────────────────────────
        # lift_up is large when: pressure from below AND vacuum above
        #   AND low resistance above.  All three must be present.
        V_above: float = m["vacuum_above"]
        V_below: float = m["vacuum_below"]

        den_up = R_above + EPS
        den_down = R_below + EPS
        if not np.isfinite(den_up) or den_up <= 0.0:
            raise ValueError(f"Invalid upward denominator: {den_up}")
        if not np.isfinite(den_down) or den_down <= 0.0:
            raise ValueError(f"Invalid downward denominator: {den_down}")

        lift_up: float = P_below * V_above / den_up
        lift_down: float = P_above * V_below / den_down
        net_lift: float = lift_up - lift_down
        if not np.isfinite(lift_up) or not np.isfinite(lift_down) or not np.isfinite(net_lift):
            raise ValueError(
                f"Non-finite lift values: lift_up={lift_up}, lift_down={lift_down}, net_lift={net_lift}"
            )

        feasibility_up, feasibility_down, directional_bias = self._compute_feasibility_gate(
            lift_up,
            lift_down,
        )

        # ── Stage 5: Multi-timescale derivative chains ───────────────
        ts_fast: Dict[str, float] = self._ts_fast.update(net_lift)
        ts_medium: Dict[str, float] = self._ts_medium.update(net_lift)
        ts_slow: Dict[str, float] = self._ts_slow.update(net_lift)

        # ── Stage 6: Cross-timescale confidence ──────────────────────
        sign_fast = np.sign(ts_fast["smooth"])
        sign_medium = np.sign(ts_medium["smooth"])
        sign_slow = np.sign(ts_slow["smooth"])

        all_agree: bool = (
            sign_fast == sign_medium == sign_slow
        ) and sign_fast != 0

        mag_fast = abs(ts_fast["smooth"])
        mag_medium = abs(ts_medium["smooth"])
        mag_slow = abs(ts_slow["smooth"])

        if all_agree:
            # Confidence = ratio of weakest to strongest timescale
            confidence: float = (
                min(mag_fast, mag_medium, mag_slow)
                / (max(mag_fast, mag_medium, mag_slow) + EPS)
            )
        else:
            confidence = 0.0

        proj_sign_fast = np.sign(ts_fast["projection"])
        proj_sign_medium = np.sign(ts_medium["projection"])
        proj_sign_slow = np.sign(ts_slow["projection"])
        projections_agree = (
            proj_sign_fast == proj_sign_medium == proj_sign_slow
        ) and proj_sign_fast != 0
        if projections_agree:
            proj_mag_fast = abs(ts_fast["projection"])
            proj_mag_medium = abs(ts_medium["projection"])
            proj_mag_slow = abs(ts_slow["projection"])
            projection_coherence: float = (
                min(proj_mag_fast, proj_mag_medium, proj_mag_slow)
                / (max(proj_mag_fast, proj_mag_medium, proj_mag_slow) + EPS)
            )
            projection_direction: int = int(proj_sign_medium)
        else:
            projection_coherence = 0.0
            projection_direction = 0

        # ── Stage 7: Regime classification ───────────────────────────
        if not all_agree:
            regime = "CHOP"
        elif net_lift > REGIME_THRESHOLD:
            regime = "LIFT"
        elif net_lift < -REGIME_THRESHOLD:
            regime = "DRAG"
        else:
            regime = "NEUTRAL"

        # ── Stage 8: Slope change alerts (medium timescale) ──────────
        alert_flags: int = 0

        if self._prev_d1_medium is not None:
            # Inflection: d1 crossed zero (velocity reversal)
            if (
                np.sign(ts_medium["d1"]) != np.sign(self._prev_d1_medium)
                and self._prev_d1_medium != 0.0
            ):
                alert_flags |= ALERT_INFLECTION

            # Deceleration: d2 opposes d1 while d1 is meaningful
            if (
                np.sign(ts_medium["d2"]) != np.sign(ts_medium["d1"])
                and abs(ts_medium["d1"]) > 0.1
            ):
                alert_flags |= ALERT_DECELERATION

            # Regime shift: d2 crossed zero while |d1| sustained
            if self._prev_d2_medium is not None:
                if (
                    np.sign(ts_medium["d2"]) != np.sign(self._prev_d2_medium)
                    and abs(ts_medium["d1"]) > 0.5
                ):
                    alert_flags |= ALERT_REGIME_SHIFT

        self._prev_d1_medium = ts_medium["d1"]
        self._prev_d2_medium = ts_medium["d2"]
        event_ctx = self._event_machine.update(
            net_lift=net_lift,
            d1_15s=ts_medium["d1"],
            d2_15s=ts_medium["d2"],
            cross_confidence=confidence,
            projection_coherence=projection_coherence,
            projection_direction=projection_direction,
        )

        # ── Assemble output ──────────────────────────────────────────
        return {
            "window_end_ts_ns": window_end_ts_ns,
            # Existing metrics (backward compat)
            "vacuum_above": m["vacuum_above"],
            "vacuum_below": m["vacuum_below"],
            "resting_drain_ask": m["resting_drain_ask"],
            "resting_drain_bid": m["resting_drain_bid"],
            "flow_imbalance": m["flow_imbalance"],
            "fill_imbalance": m["fill_imbalance"],
            "depth_imbalance": m["depth_imbalance"],
            "rest_depth_imbalance": m["rest_depth_imbalance"],
            "bid_migration_com": m.get("bid_migration_com", 0.0),
            "ask_migration_com": m.get("ask_migration_com", 0.0),
            # NEW: Pressure and resistance fields
            "pressure_above": P_above,
            "pressure_below": P_below,
            "resistance_above": R_above,
            "resistance_below": R_below,
            # NEW: Lift model
            "lift_up": lift_up,
            "lift_down": lift_down,
            "net_lift": net_lift,
            "feasibility_up": feasibility_up,
            "feasibility_down": feasibility_down,
            "directional_bias": directional_bias,
            # NEW: Multi-timescale
            "lift_5s": ts_fast["smooth"],
            "d1_5s": ts_fast["d1"],
            "d2_5s": ts_fast["d2"],
            "proj_5s": ts_fast["projection"],
            "lift_15s": ts_medium["smooth"],
            "d1_15s": ts_medium["d1"],
            "d2_15s": ts_medium["d2"],
            "proj_15s": ts_medium["projection"],
            "lift_60s": ts_slow["smooth"],
            "d1_60s": ts_slow["d1"],
            "d2_60s": ts_slow["d2"],
            "proj_60s": ts_slow["projection"],
            # NEW: Confidence and alerts
            "cross_confidence": confidence,
            "projection_coherence": projection_coherence,
            "alert_flags": alert_flags,
            "regime": regime,
            "event_state": event_ctx["event_state"],
            "event_direction": event_ctx["event_direction"],
            "event_strength": event_ctx["event_strength"],
            "event_confidence": event_ctx["event_confidence"],
            # Backward compat: map from new model
            "composite": net_lift,
            "composite_smooth": ts_medium["smooth"],
            "confidence": confidence,
            "d1_composite": ts_medium["d1"],
            "d2_composite": ts_medium["d2"],
            "d3_composite": 0.0,
            "d1_smooth": ts_medium["d1"],
            "d2_smooth": ts_medium["d2"],
            "d3_smooth": 0.0,
            "wtd_slope": ts_medium["d1"],
            "wtd_projection": ts_medium["projection"],
            "wtd_projection_500ms": ts_fast["projection"],
            "wtd_deriv_conf": confidence,
            "z_composite": 0.0,
            "z_composite_raw": 0.0,
            "z_composite_smooth": 0.0,
            "strength": min(1.0, abs(net_lift) / 100.0),
            "strength_smooth": min(1.0, abs(ts_medium["smooth"]) / 100.0),
            # Snap reference
            "mid_price": snap_dict.get("mid_price", 0.0),
            "best_bid_price_int": snap_dict.get("best_bid_price_int", 0),
            "best_ask_price_int": snap_dict.get("best_ask_price_int", 0),
            "book_valid": snap_dict.get("book_valid", False),
        }

    def _empty_signals(
        self,
        window_end_ts_ns: int,
        snap_dict: Dict[str, Any],
        event_ctx: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Return a zero-filled signals dict for invalid/empty windows."""
        event_ctx = event_ctx or {
            "event_state": EVENT_STATE_WATCH,
            "event_direction": EVENT_DIRECTION_NONE,
            "event_strength": 0.0,
            "event_confidence": 0.0,
        }
        return {
            "window_end_ts_ns": window_end_ts_ns,
            # Existing metrics
            "vacuum_above": 0.0,
            "vacuum_below": 0.0,
            "resting_drain_ask": 0.0,
            "resting_drain_bid": 0.0,
            "flow_imbalance": 0.0,
            "fill_imbalance": 0.0,
            "depth_imbalance": 0.0,
            "rest_depth_imbalance": 0.0,
            "bid_migration_com": 0.0,
            "ask_migration_com": 0.0,
            # Pressure and resistance
            "pressure_above": 0.0,
            "pressure_below": 0.0,
            "resistance_above": 0.0,
            "resistance_below": 0.0,
            # Lift model
            "lift_up": 0.0,
            "lift_down": 0.0,
            "net_lift": 0.0,
            "feasibility_up": 0.0,
            "feasibility_down": 0.0,
            "directional_bias": 0.0,
            # Multi-timescale
            "lift_5s": 0.0,
            "d1_5s": 0.0,
            "d2_5s": 0.0,
            "proj_5s": 0.0,
            "lift_15s": 0.0,
            "d1_15s": 0.0,
            "d2_15s": 0.0,
            "proj_15s": 0.0,
            "lift_60s": 0.0,
            "d1_60s": 0.0,
            "d2_60s": 0.0,
            "proj_60s": 0.0,
            # Confidence and alerts
            "cross_confidence": 0.0,
            "projection_coherence": 0.0,
            "alert_flags": 0,
            "regime": "NEUTRAL",
            "event_state": event_ctx["event_state"],
            "event_direction": event_ctx["event_direction"],
            "event_strength": event_ctx["event_strength"],
            "event_confidence": event_ctx["event_confidence"],
            # Backward compat
            "composite": 0.0,
            "composite_smooth": 0.0,
            "confidence": 0.0,
            "d1_composite": 0.0,
            "d2_composite": 0.0,
            "d3_composite": 0.0,
            "d1_smooth": 0.0,
            "d2_smooth": 0.0,
            "d3_smooth": 0.0,
            "wtd_slope": 0.0,
            "wtd_projection": 0.0,
            "wtd_projection_500ms": 0.0,
            "wtd_deriv_conf": 0.0,
            "z_composite": 0.0,
            "z_composite_raw": 0.0,
            "z_composite_smooth": 0.0,
            "strength": 0.0,
            "strength_smooth": 0.0,
            # Snap reference
            "mid_price": snap_dict.get("mid_price", 0.0),
            "best_bid_price_int": snap_dict.get("best_bid_price_int", 0),
            "best_ask_price_int": snap_dict.get("best_ask_price_int", 0),
            "book_valid": snap_dict.get("book_valid", False),
        }


class _DerivativeChain:
    """Three-level derivative chain: d1, d2, d3 with EMA smoothing.

    Replicates compute_derivatives() from formulas.py but incrementally.
    """

    __slots__ = ("d1", "d2", "d3")

    def __init__(self, span_d1: int, span_d2: int, span_d3: int) -> None:
        self.d1 = DiffEMA(span_d1)
        self.d2 = DiffEMA(span_d2)
        self.d3 = DiffEMA(span_d3)

    def update(self, value: float) -> tuple[float, float, float]:
        """Update with a new value, return (d1, d2, d3)."""
        v1 = self.d1.update(value)
        v2 = self.d2.update(v1)
        v3 = self.d3.update(v2)
        return v1, v2, v3
