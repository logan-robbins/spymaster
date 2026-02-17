"""Move-Size Signal Decomposition (MSD) harness signal.

Canonical harness implementation of the MSD experiment's spatial-vacuum
component. The legacy MSD script writes ``results.json`` from the
distance-weighted spatial vacuum asymmetry (variant_c), so this signal
exposes that as the default variant.

Signal variants:
    - "weighted": distance-weighted vacuum asymmetry (legacy variant_c)
    - "sum": raw side-sum vacuum asymmetry (legacy variant_a)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.experiment_harness.signals.base import SignalResult, StatisticalSignal
from src.experiment_harness.signals import register_signal


class MSDSignal(StatisticalSignal):
    """Spatial vacuum asymmetry signal from the legacy MSD experiment.

    Args:
        variant: Which spatial-vacuum variant to emit.
            "weighted" -> distance-weighted (legacy variant_c, default)
            "sum" -> simple side-sum asymmetry (legacy variant_a)
        cooldown_bins: Minimum bins between signal firings.
        warmup_bins: Bins to skip before adaptive threshold calibration.
    """

    def __init__(
        self,
        variant: str = "weighted",
        cooldown_bins: int = 30,
        warmup_bins: int = 300,
    ) -> None:
        if variant not in {"weighted", "sum"}:
            raise ValueError(
                f"variant must be 'weighted' or 'sum', got {variant!r}"
            )
        self.variant: str = variant
        self.cooldown_bins: int = cooldown_bins
        self.warmup_bins: int = warmup_bins

    @property
    def name(self) -> str:
        """Canonical signal name."""
        return "msd"

    @property
    def required_columns(self) -> list[str]:
        """Grid columns required by this signal."""
        return ["vacuum_variant", "pressure_variant"]

    def default_thresholds(self) -> list[float]:
        """Fallback threshold grid used when adaptive calibration fails."""
        return [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

    def compute(self, dataset: dict[str, Any]) -> SignalResult:
        """Compute MSD spatial vacuum asymmetry.

        Args:
            dataset: Dict with keys ``vacuum_variant`` and
                ``pressure_variant`` as (n_bins, 101) arrays.

        Returns:
            SignalResult with selected variant signal and adaptive
            thresholds derived from the post-warmup absolute distribution.
        """
        vacuum: np.ndarray = dataset["vacuum_variant"]
        pressure: np.ndarray = dataset["pressure_variant"]

        vac_below: np.ndarray = vacuum[:, 0:50]
        vac_above: np.ndarray = vacuum[:, 51:101]

        signal_sum: np.ndarray = vac_above.sum(axis=1) - vac_below.sum(axis=1)

        # 1/|k| distance weighting matches the legacy variant_c implementation.
        weights_below: np.ndarray = 1.0 / np.arange(50, 0, -1, dtype=np.float64)
        weights_above: np.ndarray = 1.0 / np.arange(1, 51, dtype=np.float64)
        weighted_below: np.ndarray = (vac_below * weights_below).sum(axis=1)
        weighted_above: np.ndarray = (vac_above * weights_above).sum(axis=1)
        signal_weighted: np.ndarray = weighted_above - weighted_below

        signal: np.ndarray = (
            signal_weighted if self.variant == "weighted" else signal_sum
        )

        active: np.ndarray = signal[self.warmup_bins :]
        if len(active) == 0:
            adaptive_thresholds = self.default_thresholds()
        else:
            pcts: np.ndarray = np.nanpercentile(
                np.abs(active), [50, 70, 80, 90, 95, 99]
            )
            adaptive_thresholds = sorted(
                set(round(float(v), 6) for v in pcts if float(v) > 0.0)
            )
            if not adaptive_thresholds:
                adaptive_thresholds = self.default_thresholds()

        return SignalResult(
            signal=signal,
            metadata={
                "variant": self.variant,
                "adaptive_thresholds": adaptive_thresholds,
                "signal_sum_std": float(np.std(signal_sum)),
                "signal_weighted_std": float(np.std(signal_weighted)),
                "pressure_balance_mean": float(
                    np.mean(
                        pressure[:, 51:101].sum(axis=1)
                        - pressure[:, 0:50].sum(axis=1)
                    )
                ),
            },
        )


register_signal("msd", MSDSignal)

