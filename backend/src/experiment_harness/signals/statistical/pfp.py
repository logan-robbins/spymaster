"""Pressure Front Propagation (PFP) signal.

Detects when inner-tick order activity leads outer-tick activity, indicating
aggressive directional intent propagating outward from the BBO.

Signal construction:
    1. Zones:
       - Inner bid: k=-3..-1 (cols 47-49), Inner ask: k=+1..+3 (cols 51-53)
       - Outer bid: k=-12..-5 (cols 38-45), Outer ask: k=+5..+12 (cols 55-62)
    2. Activity intensity per zone:
       I_zone[t] = mean(v_add[t, zone_cols] + v_fill[t, zone_cols])
    3. Lead-lag via EMA of lagged cross-products:
       lead_metric[t] = ema(inner[t] * outer[t-L]) / (ema(inner[t] * outer[t]) + eps)
       Ratio > 1 => inner leads outer (directional pressure front).
    4. Add channel: signal = lead_bid - lead_ask (positive = bullish).
    5. Pull channel: same zones on v_pull, signal = pull_lead_ask - pull_lead_bid.
    6. Final = add_weight * add_signal + pull_weight * pull_signal.

Thresholds are adaptive, derived from post-warmup signal percentiles.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from experiment_harness.signals.base import SignalResult, StatisticalSignal
from experiment_harness.signals.features import ema_1d
from experiment_harness.signals import register_signal


def _compute_zone_intensity(
    v_add: np.ndarray,
    v_fill: np.ndarray,
    zone_cols: list[int],
) -> np.ndarray:
    """Compute building-activity intensity for a zone.

    I_zone[t] = mean(v_add[t, zone_cols] + v_fill[t, zone_cols])

    Args:
        v_add: (n_bins, 101) add velocity grid.
        v_fill: (n_bins, 101) fill velocity grid.
        zone_cols: Column indices defining the zone.

    Returns:
        (n_bins,) intensity array.
    """
    return (v_add[:, zone_cols] + v_fill[:, zone_cols]).mean(axis=1)


def _compute_lead_metric(
    inner: np.ndarray,
    outer: np.ndarray,
    lag: int,
    alpha: float,
    eps: float,
) -> np.ndarray:
    """Compute EMA-based lead-lag metric between inner and outer zones.

    lead_metric[t] = ema(inner[t] * outer[t - lag]) /
                     (ema(inner[t] * outer[t]) + eps)

    When ratio > 1, inner activity ``lag`` bins ago correlates more strongly
    with current outer activity than contemporaneous inner does.

    Args:
        inner: (n_bins,) inner zone intensity.
        outer: (n_bins,) outer zone intensity.
        lag: Number of bins to lag the outer series.
        alpha: EMA smoothing factor.
        eps: Small constant to prevent division by zero.

    Returns:
        (n_bins,) lead-lag ratio.
    """
    n: int = len(inner)

    # Unlagged cross-product
    prod_unlagged: np.ndarray = inner * outer

    # Lagged cross-product: inner[t] * outer[t - lag]
    outer_lagged: np.ndarray = np.zeros(n, dtype=np.float64)
    outer_lagged[lag:] = outer[:n - lag]
    prod_lagged: np.ndarray = inner * outer_lagged

    ema_lagged: np.ndarray = ema_1d(prod_lagged, alpha)
    ema_unlagged: np.ndarray = ema_1d(prod_unlagged, alpha)

    return ema_lagged / (ema_unlagged + eps)


class PFPSignal(StatisticalSignal):
    """Pressure Front Propagation signal.

    Detects directional pressure fronts by measuring whether inner-tick
    (near-BBO) activity leads outer-tick activity via a lagged
    cross-product ratio.

    Args:
        inner_bid_cols: Column indices for the inner bid zone.
        inner_ask_cols: Column indices for the inner ask zone.
        outer_bid_cols: Column indices for the outer bid zone.
        outer_ask_cols: Column indices for the outer ask zone.
        lag_bins: Number of bins to lag outer against inner.
        ema_alpha: Smoothing factor for the lead-lag EMA.
        eps: Small constant for division safety.
        add_weight: Blend weight for the add/fill channel.
        pull_weight: Blend weight for the pull channel.
        cooldown_bins: Minimum bins between signal firings.
        warmup_bins: Bins to skip before computing adaptive thresholds.
    """

    def __init__(
        self,
        inner_bid_cols: tuple[int, ...] | list[int] = (47, 48, 49),
        inner_ask_cols: tuple[int, ...] | list[int] = (51, 52, 53),
        outer_bid_cols: tuple[int, ...] | list[int] | None = None,
        outer_ask_cols: tuple[int, ...] | list[int] | None = None,
        lag_bins: int = 5,
        ema_alpha: float = 0.1,
        eps: float = 1e-12,
        add_weight: float = 0.6,
        pull_weight: float = 0.4,
        cooldown_bins: int = 30,
        warmup_bins: int = 300,
    ) -> None:
        self.inner_bid_cols: list[int] = list(inner_bid_cols)
        self.inner_ask_cols: list[int] = list(inner_ask_cols)
        self.outer_bid_cols: list[int] = (
            list(outer_bid_cols)
            if outer_bid_cols is not None
            else list(range(38, 46))
        )
        self.outer_ask_cols: list[int] = (
            list(outer_ask_cols)
            if outer_ask_cols is not None
            else list(range(55, 63))
        )
        self.lag_bins: int = lag_bins
        self.ema_alpha: float = ema_alpha
        self.eps: float = eps
        self.add_weight: float = add_weight
        self.pull_weight: float = pull_weight
        self.cooldown_bins: int = cooldown_bins
        self.warmup_bins: int = warmup_bins

    @property
    def name(self) -> str:
        """Canonical signal name."""
        return "pfp"

    @property
    def required_columns(self) -> list[str]:
        """Grid columns required by this signal."""
        return ["v_add", "v_pull", "v_fill"]

    def default_thresholds(self) -> list[float]:
        """Default threshold grid.

        PFP uses adaptive thresholds computed from signal percentiles.
        This provides a fallback grid.
        """
        return [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

    def compute(self, dataset: dict[str, Any]) -> SignalResult:
        """Compute the PFP signal from v_add, v_pull, and v_fill grids.

        Args:
            dataset: Dict with keys ``v_add``, ``v_pull``, ``v_fill`` as
                (n_bins, 101) arrays, plus ``n_bins`` (int).

        Returns:
            SignalResult with the composite signal and metadata containing
            adaptive thresholds derived from signal percentiles.
        """
        v_add: np.ndarray = dataset["v_add"]
        v_pull: np.ndarray = dataset["v_pull"]
        v_fill: np.ndarray = dataset["v_fill"]

        # --- Add/fill (building) channel ---
        i_inner_bid: np.ndarray = _compute_zone_intensity(
            v_add, v_fill, self.inner_bid_cols
        )
        i_inner_ask: np.ndarray = _compute_zone_intensity(
            v_add, v_fill, self.inner_ask_cols
        )
        i_outer_bid: np.ndarray = _compute_zone_intensity(
            v_add, v_fill, self.outer_bid_cols
        )
        i_outer_ask: np.ndarray = _compute_zone_intensity(
            v_add, v_fill, self.outer_ask_cols
        )

        lead_bid: np.ndarray = _compute_lead_metric(
            i_inner_bid, i_outer_bid, self.lag_bins, self.ema_alpha, self.eps
        )
        lead_ask: np.ndarray = _compute_lead_metric(
            i_inner_ask, i_outer_ask, self.lag_bins, self.ema_alpha, self.eps
        )

        # Positive = bid-side inner leads outer more => bullish
        add_signal: np.ndarray = lead_bid - lead_ask

        # --- Pull (cancellation) channel ---
        p_inner_bid: np.ndarray = v_pull[:, self.inner_bid_cols].mean(axis=1)
        p_inner_ask: np.ndarray = v_pull[:, self.inner_ask_cols].mean(axis=1)
        p_outer_bid: np.ndarray = v_pull[:, self.outer_bid_cols].mean(axis=1)
        p_outer_ask: np.ndarray = v_pull[:, self.outer_ask_cols].mean(axis=1)

        pull_lead_bid: np.ndarray = _compute_lead_metric(
            p_inner_bid, p_outer_bid, self.lag_bins, self.ema_alpha, self.eps
        )
        pull_lead_ask: np.ndarray = _compute_lead_metric(
            p_inner_ask, p_outer_ask, self.lag_bins, self.ema_alpha, self.eps
        )

        # Pull on ask side leading => bullish
        pull_signal: np.ndarray = pull_lead_ask - pull_lead_bid

        # --- Blend channels ---
        final: np.ndarray = (
            self.add_weight * add_signal
            + self.pull_weight * pull_signal
        )

        # Adaptive thresholds from percentiles
        active_signal: np.ndarray = final[self.warmup_bins:]
        abs_signal: np.ndarray = np.abs(active_signal)
        abs_pcts: np.ndarray = np.nanpercentile(
            abs_signal, [50, 70, 80, 90, 95, 99]
        )
        adaptive_thresholds: list[float] = sorted(
            set(round(float(v), 6) for v in abs_pcts if v > 0)
        )
        if not adaptive_thresholds:
            adaptive_thresholds = self.default_thresholds()

        pcts: np.ndarray = np.nanpercentile(active_signal, [5, 25, 50, 75, 95])

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
            },
        )


register_signal("pfp", PFPSignal)
