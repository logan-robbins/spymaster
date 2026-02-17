"""Asymmetric Derivative Slope (ADS) signal.

Computes directional asymmetry across three spatial bands (inner/mid/outer)
using v_add and v_pull grid columns, then takes rolling OLS slopes of the
combined asymmetry, robust z-scores them, and blends into a single signal.

Signal construction:
    1. Per-band asymmetry (bandwidth-weighted):
       add_asym(band) = mean(v_add[bid_cols]) - mean(v_add[ask_cols])
       pull_asym(band) = mean(v_pull[ask_cols]) - mean(v_pull[bid_cols])
    2. Combined = weighted sum of (add_asym + pull_asym) per band,
       weights = 1/sqrt(band_width), normalized.
    3. Rolling OLS slope over configurable windows (default [10, 25, 50]).
    4. Robust z-score each slope (default 200-bin window).
    5. Blend: tanh-compressed z-scores with configurable weights.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from experiment_harness.signals.base import SignalResult, StatisticalSignal
from experiment_harness.signals.features import DEFAULT_BAND_DEFS
from experiment_harness.eval_engine import rolling_ols_slope, robust_zscore
from experiment_harness.signals import register_signal


class ADSSignal(StatisticalSignal):
    """Asymmetric Derivative Slope signal.

    Detects directional bias from the rate of change of spatial order-flow
    asymmetry. Bandwidth-weighted band aggregation gives inner ticks more
    influence. Multi-timescale slope blending captures both fast reactions
    and sustained trends.

    Args:
        slope_windows: Rolling OLS windows for slope computation.
        zscore_window: Lookback window for robust z-score normalization.
        blend_weights: Weights for blending each slope timescale.
            Must have same length as slope_windows.
        blend_scale: Denominator for tanh compression of z-scores.
        cooldown_bins: Minimum bins between signal firings.
    """

    def __init__(
        self,
        slope_windows: tuple[int, ...] | list[int] = (10, 25, 50),
        zscore_window: int = 200,
        blend_weights: tuple[float, ...] | list[float] = (0.40, 0.35, 0.25),
        blend_scale: float = 3.0,
        cooldown_bins: int = 30,
    ) -> None:
        self.slope_windows: list[int] = list(slope_windows)
        self.zscore_window: int = zscore_window
        self.blend_weights: list[float] = list(blend_weights)
        self.blend_scale: float = blend_scale
        self.cooldown_bins: int = cooldown_bins

        if len(self.slope_windows) != len(self.blend_weights):
            raise ValueError(
                f"slope_windows ({len(self.slope_windows)}) and blend_weights "
                f"({len(self.blend_weights)}) must have the same length"
            )

    @property
    def name(self) -> str:
        """Canonical signal name."""
        return "ads"

    @property
    def required_columns(self) -> list[str]:
        """Grid columns required by this signal."""
        return ["v_add", "v_pull"]

    def default_thresholds(self) -> list[float]:
        """Default threshold grid for sweep evaluation."""
        return [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

    def compute(self, dataset: dict[str, Any]) -> SignalResult:
        """Compute the ADS signal from v_add and v_pull grids.

        Args:
            dataset: Dict with keys ``v_add``, ``v_pull`` as (n_bins, 101)
                arrays, plus ``n_bins`` (int).

        Returns:
            SignalResult with signal in roughly [-1, +1] and metadata
            containing the standard deviation of the combined asymmetry.
        """
        v_add: np.ndarray = dataset["v_add"]
        v_pull: np.ndarray = dataset["v_pull"]
        n_bins: int = dataset["n_bins"]

        # Step 1: Bandwidth-weighted combined asymmetry across bands
        combined: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        total_weight: float = 0.0

        for band in DEFAULT_BAND_DEFS:
            bid_cols: list[int] = band["bid_cols"]  # type: ignore[assignment]
            ask_cols: list[int] = band["ask_cols"]  # type: ignore[assignment]
            width: int = band["width"]  # type: ignore[assignment]

            w: float = 1.0 / np.sqrt(width)
            total_weight += w

            # add_asym: positive = more adding on bid side = bullish
            add_asym: np.ndarray = (
                v_add[:, bid_cols].mean(axis=1)
                - v_add[:, ask_cols].mean(axis=1)
            )

            # pull_asym: positive = more pulling on ask side = bullish
            pull_asym: np.ndarray = (
                v_pull[:, ask_cols].mean(axis=1)
                - v_pull[:, bid_cols].mean(axis=1)
            )

            combined += w * (add_asym + pull_asym)

        combined /= total_weight

        # Step 2: Rolling OLS slopes -> z-score -> tanh -> blend
        signal: np.ndarray = np.zeros(n_bins, dtype=np.float64)

        for i, win in enumerate(self.slope_windows):
            slope: np.ndarray = rolling_ols_slope(combined, win)
            z: np.ndarray = robust_zscore(slope, self.zscore_window)
            signal += self.blend_weights[i] * np.tanh(z / self.blend_scale)

        return SignalResult(
            signal=signal,
            metadata={"combined_asym_std": float(np.std(combined))},
        )


register_signal("ads", ADSSignal)
