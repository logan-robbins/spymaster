"""Spatial Pressure Gradient (SPG) signal.

Computes directional trading signals from the spatial first derivative
(central difference) of pressure and vacuum fields around the spot price.

Signal logic:
    1. Spatial gradient dP/dk and dV/dk via np.gradient along the k axis.
    2. Mean gradient in configurable ask/bid bands around spot.
    3. Directional signals:
       wall_signal = grad_P_below - grad_P_above (positive = wall above = bearish)
       pull_signal = grad_V_above - grad_V_below (positive = pull above = bullish)
       net = -wall_signal + pull_signal
    4. Dual EMA smoothing (fast + slow spans).
    5. Spatial curvature correction (d2P/dk2 near spot, EMA-smoothed).
    6. Final = blend_smooth * smoothed + blend_curv * curvature_signal.

Thresholds are adaptive, derived from post-warmup signal percentiles.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from experiment_harness.signals.base import SignalResult, StatisticalSignal
from experiment_harness.signals.features import ema_1d
from experiment_harness.signals import register_signal


class SPGSignal(StatisticalSignal):
    """Spatial Pressure Gradient signal.

    Detects directional bias from the spatial derivative structure of the
    pressure and vacuum fields. Walls (positive pressure gradient) resist
    price movement; vacuums (negative vacuum gradient) attract it.

    Args:
        fast_span: EMA span for the fast smoothing component.
        slow_span: EMA span for the slow smoothing component.
        curv_cols: Column indices for spatial curvature evaluation near spot.
        curv_span: EMA span for curvature smoothing.
        blend_smooth: Weight for the dual-EMA smoothed net signal.
        blend_curv: Weight for the curvature correction signal.
        ema_fast_weight: Blend weight for the fast EMA component.
        ema_slow_weight: Blend weight for the slow EMA component.
        ask_band: Tuple of (start, end) column indices for the ask side.
        bid_band: Tuple of (start, end) column indices for the bid side.
        cooldown_bins: Minimum bins between signal firings.
        warmup_bins: Bins to skip before computing adaptive thresholds.
    """

    def __init__(
        self,
        fast_span: int = 5,
        slow_span: int = 20,
        curv_cols: tuple[int, ...] | list[int] = (48, 49, 50, 51, 52),
        curv_span: int = 10,
        blend_smooth: float = 0.7,
        blend_curv: float = 0.3,
        ema_fast_weight: float = 0.6,
        ema_slow_weight: float = 0.4,
        ask_band: tuple[int, int] = (51, 67),
        bid_band: tuple[int, int] = (34, 50),
        cooldown_bins: int = 20,
        warmup_bins: int = 300,
    ) -> None:
        self.fast_span: int = fast_span
        self.slow_span: int = slow_span
        self.curv_cols: list[int] = list(curv_cols)
        self.curv_span: int = curv_span
        self.blend_smooth: float = blend_smooth
        self.blend_curv: float = blend_curv
        self.ema_fast_weight: float = ema_fast_weight
        self.ema_slow_weight: float = ema_slow_weight
        self.ask_lo: int = ask_band[0]
        self.ask_hi: int = ask_band[1]
        self.bid_lo: int = bid_band[0]
        self.bid_hi: int = bid_band[1]
        self.cooldown_bins: int = cooldown_bins
        self.warmup_bins: int = warmup_bins

    @property
    def name(self) -> str:
        """Canonical signal name."""
        return "spg"

    @property
    def required_columns(self) -> list[str]:
        """Grid columns required by this signal."""
        return ["pressure_variant", "vacuum_variant"]

    def default_thresholds(self) -> list[float]:
        """Default threshold grid.

        SPG uses adaptive thresholds computed from signal percentiles
        inside compute(). This returns a fallback grid for cases where
        the adaptive calibration fails (e.g. near-zero signal).
        """
        return [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

    def compute(self, dataset: dict[str, Any]) -> SignalResult:
        """Compute the SPG signal from pressure and vacuum grids.

        Args:
            dataset: Dict with keys ``pressure_variant``, ``vacuum_variant``
                as (n_bins, 101) arrays, plus ``n_bins`` (int).

        Returns:
            SignalResult with the composite signal and metadata containing
            adaptive thresholds and signal percentile statistics.
        """
        pressure: np.ndarray = dataset["pressure_variant"]
        vacuum: np.ndarray = dataset["vacuum_variant"]
        n_bins: int = dataset["n_bins"]

        # Step 1: Spatial first derivative along k axis
        dp_dk: np.ndarray = np.gradient(pressure, axis=1)
        dv_dk: np.ndarray = np.gradient(vacuum, axis=1)

        # Step 2: Mean gradient in bid/ask bands
        grad_p_above: np.ndarray = np.mean(dp_dk[:, self.ask_lo:self.ask_hi], axis=1)
        grad_p_below: np.ndarray = np.mean(dp_dk[:, self.bid_lo:self.bid_hi], axis=1)
        grad_v_above: np.ndarray = np.mean(dv_dk[:, self.ask_lo:self.ask_hi], axis=1)
        grad_v_below: np.ndarray = np.mean(dv_dk[:, self.bid_lo:self.bid_hi], axis=1)

        # Step 3: Directional signals
        wall_signal: np.ndarray = grad_p_below - grad_p_above
        pull_signal: np.ndarray = grad_v_above - grad_v_below
        net: np.ndarray = -wall_signal + pull_signal

        # Step 4: Dual EMA smoothing
        alpha_fast: float = 2.0 / (self.fast_span + 1)
        alpha_slow: float = 2.0 / (self.slow_span + 1)

        ema_fast: np.ndarray = ema_1d(net, alpha_fast)
        ema_slow: np.ndarray = ema_1d(net, alpha_slow)
        smoothed: np.ndarray = (
            self.ema_fast_weight * ema_fast
            + self.ema_slow_weight * ema_slow
        )

        # Step 5: Spatial curvature near spot (second derivative)
        curv_raw: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        for c in self.curv_cols:
            curv_raw += pressure[:, c + 1] + pressure[:, c - 1] - 2.0 * pressure[:, c]
        curv_raw /= len(self.curv_cols)

        # Negate: positive curvature = wall = bearish
        curv_signal_raw: np.ndarray = -curv_raw
        alpha_curv: float = 2.0 / (self.curv_span + 1)
        curv_signal: np.ndarray = ema_1d(curv_signal_raw, alpha_curv)

        # Step 6: Final composite signal
        final: np.ndarray = (
            self.blend_smooth * smoothed
            + self.blend_curv * curv_signal
        )

        # Adaptive thresholds from signal percentiles
        post_warmup: np.ndarray = final[self.warmup_bins:]
        pcts: np.ndarray = np.percentile(post_warmup, [5, 25, 50, 75, 95])
        abs_p75: float = max(abs(float(pcts[1])), abs(float(pcts[3])))
        abs_p95: float = max(abs(float(pcts[0])), abs(float(pcts[4])))
        iqr: float = float(pcts[3]) - float(pcts[1])

        adaptive_thresholds: list[float] = sorted(set([
            round(0.5 * iqr, 6),
            round(1.0 * iqr, 6),
            round(abs_p75, 6),
            round(0.5 * (abs_p75 + abs_p95), 6),
            round(abs_p95, 6),
        ]))
        adaptive_thresholds = [t for t in adaptive_thresholds if t > 1e-8]

        return SignalResult(
            signal=final,
            metadata={
                "adaptive_thresholds": adaptive_thresholds,
                "signal_pcts": {
                    "p5": float(pcts[0]),
                    "p25": float(pcts[1]),
                    "p50": float(pcts[2]),
                    "p75": float(pcts[3]),
                    "p95": float(pcts[4]),
                },
                "net_signal_mean": float(np.mean(net)),
                "net_signal_std": float(np.std(net)),
            },
        )


register_signal("spg", SPGSignal)
