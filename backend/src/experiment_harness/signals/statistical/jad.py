"""Jerk-Acceleration Divergence (JAD) signal.

Tests the core thesis: when jerk (d3) of add/pull diverges between bid
and ask sides, it signals regime inflection before acceleration changes sign.

Signal construction:
    1. Distance-weighted spatial aggregation:
       w(k) = 1/|k| for bid cols 26..49 and ask cols 51..74.
       Weighted mean of j_add, j_pull, a_add, a_pull per side.
    2. Divergence signals (bullish-positive):
       jerk_add_div  = J_add_bid  - J_add_ask
       jerk_pull_div = J_pull_ask - J_pull_bid
       accel_add_div = A_add_bid  - A_add_ask
       accel_pull_div = A_pull_ask - A_pull_bid
    3. Combined: jerk_signal = 0.5 * jerk_add_div + 0.5 * jerk_pull_div
                 accel_signal = 0.5 * accel_add_div + 0.5 * accel_pull_div
    4. Agreement/disagreement weighting:
       Same sign: confirm_weights (default 0.4 jerk + 0.6 accel)
       Diff sign: disagree_weights (default 0.8 jerk + 0.2 accel)
    5. Robust z-score + tanh compression.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.experiment_harness.signals.base import SignalResult, StatisticalSignal
from src.experiment_harness.eval_engine import robust_zscore
from src.experiment_harness.signals import register_signal


def _build_weights(cols: np.ndarray, spot_col: int = 50) -> np.ndarray:
    """Build normalized 1/|k| distance weights for given column indices.

    Args:
        cols: Array of column indices.
        spot_col: The column index representing spot (k=0).

    Returns:
        Normalized weight array of same length as cols.
    """
    k_abs: np.ndarray = np.abs(cols - spot_col).astype(np.float64)
    weights: np.ndarray = 1.0 / k_abs
    return weights / weights.sum()


class JADSignal(StatisticalSignal):
    """Jerk-Acceleration Divergence signal.

    Detects regime inflection points by measuring divergence between
    jerk (3rd derivative) and acceleration (2nd derivative) of order-flow
    activity across bid and ask sides. When jerk and acceleration disagree
    on direction, jerk is weighted more heavily as an early reversal
    indicator.

    Args:
        bid_cols: Column indices for the bid side (k < 0).
        ask_cols: Column indices for the ask side (k > 0).
        zscore_window: Lookback window for robust z-score normalization.
        tanh_scale: Denominator for tanh compression of z-scores.
        confirm_weights: (jerk_w, accel_w) weights when jerk and accel agree.
        disagree_weights: (jerk_w, accel_w) weights when they disagree.
        cooldown_bins: Minimum bins between signal firings.
    """

    def __init__(
        self,
        bid_cols: np.ndarray | list[int] | None = None,
        ask_cols: np.ndarray | list[int] | None = None,
        zscore_window: int = 300,
        tanh_scale: float = 3.0,
        confirm_weights: tuple[float, float] = (0.4, 0.6),
        disagree_weights: tuple[float, float] = (0.8, 0.2),
        cooldown_bins: int = 25,
    ) -> None:
        self._bid_cols: np.ndarray = (
            np.asarray(bid_cols, dtype=np.int32)
            if bid_cols is not None
            else np.arange(26, 50, dtype=np.int32)
        )
        self._ask_cols: np.ndarray = (
            np.asarray(ask_cols, dtype=np.int32)
            if ask_cols is not None
            else np.arange(51, 75, dtype=np.int32)
        )
        self.zscore_window: int = zscore_window
        self.tanh_scale: float = tanh_scale
        self.confirm_weights: tuple[float, float] = confirm_weights
        self.disagree_weights: tuple[float, float] = disagree_weights
        self.cooldown_bins: int = cooldown_bins

        # Pre-compute normalized inverse-distance weights
        self._bid_weights: np.ndarray = _build_weights(self._bid_cols)
        self._ask_weights: np.ndarray = _build_weights(self._ask_cols)

    @property
    def name(self) -> str:
        """Canonical signal name."""
        return "jad"

    @property
    def required_columns(self) -> list[str]:
        """Grid columns required by this signal."""
        return ["j_add", "j_pull", "a_add", "a_pull"]

    def default_thresholds(self) -> list[float]:
        """Default threshold grid for sweep evaluation."""
        return [0.05, 0.10, 0.15, 0.20, 0.30]

    def _weighted_mean(
        self,
        grid: np.ndarray,
        cols: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Compute distance-weighted mean across selected columns.

        Args:
            grid: (n_bins, 101) array.
            cols: Column indices to aggregate.
            weights: Normalized weights, same length as cols.

        Returns:
            (n_bins,) weighted mean per time bin.
        """
        return grid[:, cols] @ weights

    def compute(self, dataset: dict[str, Any]) -> SignalResult:
        """Compute the JAD signal from jerk and acceleration grids.

        Args:
            dataset: Dict with keys ``j_add``, ``j_pull``, ``a_add``,
                ``a_pull`` as (n_bins, 101) arrays, plus ``n_bins`` (int).

        Returns:
            SignalResult with signal in roughly [-1, +1] and metadata
            containing divergence statistics.
        """
        j_add: np.ndarray = dataset["j_add"]
        j_pull: np.ndarray = dataset["j_pull"]
        a_add: np.ndarray = dataset["a_add"]
        a_pull: np.ndarray = dataset["a_pull"]

        # Step 1: Distance-weighted spatial aggregation
        j_add_bid: np.ndarray = self._weighted_mean(
            j_add, self._bid_cols, self._bid_weights
        )
        j_add_ask: np.ndarray = self._weighted_mean(
            j_add, self._ask_cols, self._ask_weights
        )
        j_pull_bid: np.ndarray = self._weighted_mean(
            j_pull, self._bid_cols, self._bid_weights
        )
        j_pull_ask: np.ndarray = self._weighted_mean(
            j_pull, self._ask_cols, self._ask_weights
        )

        a_add_bid: np.ndarray = self._weighted_mean(
            a_add, self._bid_cols, self._bid_weights
        )
        a_add_ask: np.ndarray = self._weighted_mean(
            a_add, self._ask_cols, self._ask_weights
        )
        a_pull_bid: np.ndarray = self._weighted_mean(
            a_pull, self._bid_cols, self._bid_weights
        )
        a_pull_ask: np.ndarray = self._weighted_mean(
            a_pull, self._ask_cols, self._ask_weights
        )

        # Step 2: Divergence signals (bullish-positive)
        jerk_add_div: np.ndarray = j_add_bid - j_add_ask
        jerk_pull_div: np.ndarray = j_pull_ask - j_pull_bid
        accel_add_div: np.ndarray = a_add_bid - a_add_ask
        accel_pull_div: np.ndarray = a_pull_ask - a_pull_bid

        # Step 3: Combined jerk and accel signals
        jerk_signal: np.ndarray = 0.5 * jerk_add_div + 0.5 * jerk_pull_div
        accel_signal: np.ndarray = 0.5 * accel_add_div + 0.5 * accel_pull_div

        # Step 4: Agreement/disagreement weighting (vectorized)
        same_sign: np.ndarray = np.sign(jerk_signal) == np.sign(accel_signal)
        raw_signal: np.ndarray = np.where(
            same_sign,
            self.confirm_weights[0] * jerk_signal
            + self.confirm_weights[1] * accel_signal,
            self.disagree_weights[0] * jerk_signal
            + self.disagree_weights[1] * accel_signal,
        )

        # Step 5: Robust z-score + tanh compression
        z: np.ndarray = robust_zscore(raw_signal, self.zscore_window)
        signal: np.ndarray = np.tanh(z / self.tanh_scale)

        # Metadata
        n_agree: int = int(np.sum(same_sign))
        n_total: int = len(same_sign)

        return SignalResult(
            signal=signal,
            metadata={
                "jerk_signal_std": float(np.std(jerk_signal)),
                "accel_signal_std": float(np.std(accel_signal)),
                "agreement_rate": n_agree / n_total if n_total > 0 else 0.0,
                "raw_signal_std": float(np.std(raw_signal)),
            },
        )


register_signal("jad", JADSignal)
