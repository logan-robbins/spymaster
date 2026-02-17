"""Entropy Regime Detector (ERD) signal.

Detects entropy spikes in the spectrum state field as precursors to
regime transitions. Shannon entropy of the 3-state distribution across
ticks measures disorder; asymmetry between above/below spot provides
directional bias.

Two signal variants:
    A) signal = score_direction * max(0, z_H - spike_floor)
    B) signal = entropy_asym * max(0, z_H - spike_floor)

Variant A is returned as the primary signal. Variant B is included in
metadata for optional secondary evaluation by the experiment runner.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.experiment_harness.signals.base import SignalResult, StatisticalSignal
from src.experiment_harness.eval_engine import robust_zscore
from src.experiment_harness.signals import register_signal

# Maximum Shannon entropy for 3 states = log2(3) = 1.585 bits
_LOG2_3: float = float(np.log2(3.0))


def _compute_entropy_arrays(
    state_code: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute full, above-spot, and below-spot Shannon entropy per bin.

    Uses vectorized state counting for efficiency. Entropy is computed
    for the 3-state distribution {-1 (vacuum), 0 (neutral), 1 (pressure)}
    across the spatial tick axis.

    Args:
        state_code: (n_bins, 101) int8 array of spectrum state codes.
            Col 50 = spot (k=0). Cols 0..49 = below spot. Cols 51..100 = above.

    Returns:
        Tuple of (H_full, H_above, H_below), each shape (n_bins,).
        Entropy values in bits, range [0, log2(3)].
    """
    n_bins: int = state_code.shape[0]
    h_full: np.ndarray = np.zeros(n_bins, dtype=np.float64)
    h_above: np.ndarray = np.zeros(n_bins, dtype=np.float64)
    h_below: np.ndarray = np.zeros(n_bins, dtype=np.float64)

    # Accumulate entropy contributions from each of the 3 states
    for state_val in (1, -1, 0):
        mask_full: np.ndarray = state_code == state_val
        mask_above: np.ndarray = state_code[:, 51:101] == state_val
        mask_below: np.ndarray = state_code[:, 0:50] == state_val

        count_full: np.ndarray = mask_full.sum(axis=1).astype(np.float64)
        count_above: np.ndarray = mask_above.sum(axis=1).astype(np.float64)
        count_below: np.ndarray = mask_below.sum(axis=1).astype(np.float64)

        # Full spectrum: n=101 ticks
        p_full: np.ndarray = count_full / 101.0
        valid_full: np.ndarray = p_full > 0
        h_full[valid_full] -= (
            p_full[valid_full] * np.log2(p_full[valid_full] + 1e-12)
        )

        # Above spot: n=50 ticks (cols 51..100)
        p_above: np.ndarray = count_above / 50.0
        valid_above: np.ndarray = p_above > 0
        h_above[valid_above] -= (
            p_above[valid_above] * np.log2(p_above[valid_above] + 1e-12)
        )

        # Below spot: n=50 ticks (cols 0..49)
        p_below: np.ndarray = count_below / 50.0
        valid_below: np.ndarray = p_below > 0
        h_below[valid_below] -= (
            p_below[valid_below] * np.log2(p_below[valid_below] + 1e-12)
        )

    return h_full, h_above, h_below


class ERDSignal(StatisticalSignal):
    """Entropy Regime Detector signal.

    Uses Shannon entropy of the 3-state spectrum (pressure/vacuum/neutral)
    to detect high-disorder regimes. A spike gate activates the signal
    only when entropy is anomalously high (z-score above the floor).
    Directional bias comes from the score field asymmetry between
    above-spot and below-spot ticks.

    Args:
        zscore_window: Lookback window for robust z-score of H_full.
        cooldown_bins: Minimum bins between signal firings.
        spike_floor: Minimum z_H value to activate the spike gate.
            Signal is zero when z_H <= spike_floor.
        variant: Signal variant to emit.
            "a" -> score_direction * spike_gate
            "b" -> entropy_asym * spike_gate
    """

    def __init__(
        self,
        zscore_window: int = 100,
        cooldown_bins: int = 40,
        spike_floor: float = 0.5,
        variant: str = "a",
    ) -> None:
        if variant not in {"a", "b"}:
            raise ValueError(f"variant must be 'a' or 'b', got {variant!r}")
        self.zscore_window: int = zscore_window
        self.cooldown_bins: int = cooldown_bins
        self.spike_floor: float = spike_floor
        self.variant: str = variant

    @property
    def name(self) -> str:
        """Canonical signal name."""
        return "erd"

    @property
    def required_columns(self) -> list[str]:
        """Grid columns required by this signal."""
        return ["spectrum_state_code", "spectrum_score"]

    def default_thresholds(self) -> list[float]:
        """Default threshold grid for sweep evaluation."""
        return [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0]

    def compute(self, dataset: dict[str, Any]) -> SignalResult:
        """Compute the ERD signal from spectrum state and score grids.

        Returns variant A (score_direction * spike_gate) as the primary
        signal. Variant B (entropy_asym * spike_gate) is included in
        metadata for secondary evaluation.

        Args:
            dataset: Dict with keys ``spectrum_state_code`` and
                ``spectrum_score`` as (n_bins, 101) arrays, plus
                ``n_bins`` (int).

        Returns:
            SignalResult with the variant A signal and metadata containing
            variant B signal, entropy statistics, and diagnostic info.
        """
        state_code: np.ndarray = dataset["spectrum_state_code"]
        score: np.ndarray = dataset["spectrum_score"]
        n_bins: int = dataset["n_bins"]

        # Step 1: Cast state_code to int8 (harness returns float64)
        state_int: np.ndarray = state_code.astype(np.int8)

        # Step 2: Shannon entropy per bin (full, above, below)
        h_full, h_above, h_below = _compute_entropy_arrays(state_int)

        # Entropy asymmetry: positive = more disorder above spot
        entropy_asym: np.ndarray = h_above - h_below

        # Step 3: Rolling robust z-score of H_full
        z_h: np.ndarray = robust_zscore(h_full, window=self.zscore_window)

        # Step 4: Score direction from spatial score field
        mean_score_above: np.ndarray = score[:, 51:101].mean(axis=1)
        mean_score_below: np.ndarray = score[:, 0:50].mean(axis=1)
        score_direction: np.ndarray = mean_score_below - mean_score_above

        # Step 5: Spike gate -- signal only active during entropy spikes
        spike_gate: np.ndarray = np.maximum(0.0, z_h - self.spike_floor)

        # Variant A: score direction modulated by entropy spike
        signal_a: np.ndarray = score_direction * spike_gate

        # Variant B: entropy asymmetry modulated by entropy spike
        signal_b: np.ndarray = entropy_asym * spike_gate

        selected_signal = signal_a if self.variant == "a" else signal_b

        return SignalResult(
            signal=selected_signal,
            metadata={
                "signal_b": signal_b.tolist(),
                "variant": self.variant,
                "h_full_mean": float(h_full.mean()),
                "h_full_std": float(h_full.std()),
                "h_full_max": float(h_full.max()),
                "entropy_asym_mean": float(entropy_asym.mean()),
                "entropy_asym_std": float(entropy_asym.std()),
                "z_h_mean": float(z_h.mean()),
                "z_h_std": float(z_h.std()),
                "z_h_max": float(z_h.max()),
                "score_direction_mean": float(score_direction.mean()),
                "score_direction_std": float(score_direction.std()),
                "nonzero_a": int(np.count_nonzero(signal_a)),
                "nonzero_b": int(np.count_nonzero(signal_b)),
                "max_entropy_bits": _LOG2_3,
            },
        )


register_signal("erd", ERDSignal)
