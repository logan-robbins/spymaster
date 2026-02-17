"""Intensity Imbalance Rate-of-Change (IIRC) signal.

Captures order-flow toxicity momentum: the rate of change of the
add-to-pull intensity ratio on each side of the book.

Signal construction:
    1. Sum v_add, v_pull, v_fill by side for the configured tick band.
    2. Intensity ratio per side with Laplace smoothing:
       ratio = add_rate / (pull_rate + fill_rate + eps)
    3. Log imbalance:
       imbalance = log(ratio_bid + 1) - log(ratio_ask + 1)
       Positive = bid adding more relative to pulling (bullish).
    4. Rate of change via rolling OLS slope at fast/slow windows.
    5. Combined signal: fast_weight * d_fast + slow_weight * d_slow.
    6. Noise floor filter: zero where |imbalance| < noise_floor.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.experiment_harness.signals.base import SignalResult, StatisticalSignal
from src.experiment_harness.eval_engine import rolling_ols_slope
from src.experiment_harness.signals import register_signal


class IIRCSignal(StatisticalSignal):
    """Intensity Imbalance Rate-of-Change signal.

    Detects momentum in order-flow toxicity by tracking how fast the
    ratio of adding-to-pulling intensity changes on each side of the
    book. A rising bid-side ratio (relative to ask) signals bullish
    accumulation.

    Args:
        bid_slice: Tuple of (start, end) column indices for the bid side.
        ask_slice: Tuple of (start, end) column indices for the ask side.
        eps: Laplace smoothing in the intensity ratio denominator.
        fast_window: Rolling OLS window for fast slope.
        slow_window: Rolling OLS window for slow slope.
        fast_weight: Blend weight for the fast slope component.
        slow_weight: Blend weight for the slow slope component.
        noise_floor: Minimum |imbalance| required for signal activation.
        cooldown_bins: Minimum bins between signal firings.
    """

    def __init__(
        self,
        bid_slice: tuple[int, int] = (34, 50),
        ask_slice: tuple[int, int] = (51, 67),
        eps: float = 1.0,
        fast_window: int = 10,
        slow_window: int = 30,
        fast_weight: float = 0.6,
        slow_weight: float = 0.4,
        noise_floor: float = 0.1,
        cooldown_bins: int = 20,
    ) -> None:
        self.bid_slice: slice = slice(bid_slice[0], bid_slice[1])
        self.ask_slice: slice = slice(ask_slice[0], ask_slice[1])
        self.eps: float = eps
        self.fast_window: int = fast_window
        self.slow_window: int = slow_window
        self.fast_weight: float = fast_weight
        self.slow_weight: float = slow_weight
        self.noise_floor: float = noise_floor
        self.cooldown_bins: int = cooldown_bins

    @property
    def name(self) -> str:
        """Canonical signal name."""
        return "iirc"

    @property
    def required_columns(self) -> list[str]:
        """Grid columns required by this signal."""
        return ["v_add", "v_pull", "v_fill"]

    def default_thresholds(self) -> list[float]:
        """Default threshold grid for sweep evaluation."""
        return [0.001, 0.005, 0.01, 0.02, 0.05]

    def compute(self, dataset: dict[str, Any]) -> SignalResult:
        """Compute the IIRC signal from v_add, v_pull, and v_fill grids.

        Args:
            dataset: Dict with keys ``v_add``, ``v_pull``, ``v_fill`` as
                (n_bins, 101) arrays, plus ``n_bins`` (int).

        Returns:
            SignalResult with the rate-of-change signal and metadata
            containing imbalance statistics.
        """
        v_add: np.ndarray = dataset["v_add"]
        v_pull: np.ndarray = dataset["v_pull"]
        v_fill: np.ndarray = dataset["v_fill"]

        # Step 1: Sum velocities across each side's tick band
        add_rate_bid: np.ndarray = v_add[:, self.bid_slice].sum(axis=1)
        pull_rate_bid: np.ndarray = v_pull[:, self.bid_slice].sum(axis=1)
        fill_rate_bid: np.ndarray = v_fill[:, self.bid_slice].sum(axis=1)

        add_rate_ask: np.ndarray = v_add[:, self.ask_slice].sum(axis=1)
        pull_rate_ask: np.ndarray = v_pull[:, self.ask_slice].sum(axis=1)
        fill_rate_ask: np.ndarray = v_fill[:, self.ask_slice].sum(axis=1)

        # Step 2: Intensity ratios with Laplace smoothing
        ratio_bid: np.ndarray = add_rate_bid / (
            pull_rate_bid + fill_rate_bid + self.eps
        )
        ratio_ask: np.ndarray = add_rate_ask / (
            pull_rate_ask + fill_rate_ask + self.eps
        )

        # Step 3: Log imbalance
        imbalance: np.ndarray = (
            np.log(ratio_bid + 1.0) - np.log(ratio_ask + 1.0)
        )

        # Step 4: Rolling OLS slopes at fast/slow windows
        d_fast: np.ndarray = rolling_ols_slope(imbalance, self.fast_window)
        d_slow: np.ndarray = rolling_ols_slope(imbalance, self.slow_window)

        # Step 5: Weighted combination, NaN -> 0
        raw_signal: np.ndarray = (
            self.fast_weight * d_fast + self.slow_weight * d_slow
        )
        raw_signal = np.nan_to_num(raw_signal, nan=0.0)

        # Step 6: Noise floor filter
        noise_mask: np.ndarray = np.abs(imbalance) >= self.noise_floor
        signal: np.ndarray = raw_signal * noise_mask.astype(np.float64)

        # Metadata: nonzero signal statistics
        nonzero: np.ndarray = signal[signal != 0.0]
        nonzero_pcts: dict[str, float] = {}
        if len(nonzero) > 0:
            pcts: np.ndarray = np.percentile(nonzero, [5, 25, 50, 75, 95])
            nonzero_pcts = {
                "p5": float(pcts[0]),
                "p25": float(pcts[1]),
                "p50": float(pcts[2]),
                "p75": float(pcts[3]),
                "p95": float(pcts[4]),
            }

        adaptive_thresholds: list[float] = list(self.default_thresholds())
        if len(nonzero) > 0:
            abs_p95: float = float(np.percentile(np.abs(nonzero), 95))
            if abs_p95 > 0.1:
                adaptive_thresholds.extend([0.08, 0.10])
            if abs_p95 > 0.5:
                adaptive_thresholds.extend([0.20, 0.50])
        adaptive_thresholds = sorted(set(adaptive_thresholds))

        return SignalResult(
            signal=signal,
            metadata={
                "n_nonzero": int(len(nonzero)),
                "n_total": int(len(signal)),
                "nonzero_pcts": nonzero_pcts,
                "imbalance_mean": float(np.mean(imbalance)),
                "imbalance_std": float(np.std(imbalance)),
                "adaptive_thresholds": adaptive_thresholds,
            },
        )


register_signal("iirc", IIRCSignal)
