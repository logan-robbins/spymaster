"""ADS+PFP+SVac composite signal for harness evaluation.

This module is the canonical implementation for ADS/PFP/SVac math used by
offline evaluation after removal of frontend-side duplicate signal models.

Design goals:
    1. Preserve directional math semantics used by prior campaigns.
    2. Keep parameters tunable from harness YAML sweeps.
    3. Preserve time-based semantics by converting ms -> bins via cell_width_ms.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.experiment_harness.eval_engine import rolling_ols_slope, robust_zscore
from src.experiment_harness.signals import register_signal
from src.experiment_harness.signals.base import SignalResult, StatisticalSignal
from src.experiment_harness.signals.features import ema_1d

# Grid layout is fixed at 101 ticks: k in [-50, +50], col 50 is spot.
N_TICKS = 101

# ADS k-band mappings expressed as column indices.
ADS_BANDS: tuple[tuple[np.ndarray, np.ndarray, int], ...] = (
    (np.arange(47, 50), np.arange(51, 54), 3),   # inner
    (np.arange(39, 47), np.arange(54, 62), 8),   # mid
    (np.arange(27, 39), np.arange(62, 74), 12),  # outer
)

# Frontend PFP zones (k values mapped to absolute columns).
PFP_INNER_BID_COLS = np.array([47, 48, 49], dtype=np.int32)
PFP_INNER_ASK_COLS = np.array([51, 52, 53], dtype=np.int32)
PFP_OUTER_BID_COLS = np.array([38, 39, 40, 41, 42, 43, 44, 45], dtype=np.int32)
PFP_OUTER_ASK_COLS = np.array([55, 56, 57, 58, 59, 60, 61, 62], dtype=np.int32)

# SVac distance-weighting from k-space: weight = 1/|k|.
SVAC_BELOW_WEIGHTS = 1.0 / np.arange(50, 0, -1, dtype=np.float64)
SVAC_ABOVE_WEIGHTS = 1.0 / np.arange(1, 51, dtype=np.float64)


def _ms_to_bins(duration_ms: int, cell_width_ms: int, minimum: int) -> int:
    """Convert duration in ms to a minimum-clamped number of bins."""
    return max(minimum, int(math.ceil(float(duration_ms) / float(cell_width_ms))))


def _validate_grid_shape(arr: np.ndarray, name: str) -> None:
    """Fail fast on unexpected grid shapes."""
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    if arr.shape[1] != N_TICKS:
        raise ValueError(
            f"{name} must have {N_TICKS} columns (k=-50..+50), got shape={arr.shape}"
        )


def _zone_mean(grid: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Mean activity for a set of columns across all bins."""
    return grid[:, cols].mean(axis=1)


class ADSPFPSVacSignal(StatisticalSignal):
    """Frontend ADS+PFP+SVac composite for offline harness sweeps."""

    def __init__(
        self,
        cell_width_ms: int = 100,
        ads_slope_windows_ms: tuple[int, ...] | list[int] = (1000, 2500, 5000),
        ads_zscore_window_ms: int = 20000,
        ads_zscore_min_periods_ratio: float = 0.15,
        ads_blend_weights: tuple[float, ...] | list[float] = (0.40, 0.35, 0.25),
        ads_blend_scale: float = 3.0,
        ads_warmup_ms: int = 20000,
        pfp_lag_ms: int = 500,
        pfp_ema_alpha: float = 0.1,
        pfp_eps: float = 1e-12,
        pfp_add_weight: float = 0.6,
        pfp_pull_weight: float = 0.4,
        svac_norm_alpha: float = 0.05,
        svac_norm_floor: float = 1e-6,
        composite_weight_ads: float = 0.40,
        composite_weight_pfp: float = 0.30,
        composite_weight_svac: float = 0.30,
        cooldown_bins: int = 30,
    ) -> None:
        if cell_width_ms <= 0:
            raise ValueError(f"cell_width_ms must be > 0, got {cell_width_ms}")

        slope_ms = [int(x) for x in ads_slope_windows_ms]
        if not slope_ms:
            raise ValueError("ads_slope_windows_ms must contain at least one value")
        if any(x <= 0 for x in slope_ms):
            raise ValueError(
                f"ads_slope_windows_ms values must be > 0, got {slope_ms}"
            )

        blend = [float(x) for x in ads_blend_weights]
        if len(slope_ms) != len(blend):
            raise ValueError(
                "ads_slope_windows_ms and ads_blend_weights must have the same length"
            )
        if any(x < 0.0 for x in blend):
            raise ValueError(
                f"ads_blend_weights values must be >= 0, got {blend}"
            )
        if sum(blend) <= 0.0:
            raise ValueError("ads_blend_weights must have positive sum")

        if ads_zscore_window_ms <= 0:
            raise ValueError(
                f"ads_zscore_window_ms must be > 0, got {ads_zscore_window_ms}"
            )
        if not (0.0 < ads_zscore_min_periods_ratio <= 1.0):
            raise ValueError(
                "ads_zscore_min_periods_ratio must be in (0, 1], "
                f"got {ads_zscore_min_periods_ratio}"
            )
        if ads_blend_scale <= 0.0:
            raise ValueError(f"ads_blend_scale must be > 0, got {ads_blend_scale}")
        if ads_warmup_ms <= 0:
            raise ValueError(f"ads_warmup_ms must be > 0, got {ads_warmup_ms}")

        if pfp_lag_ms <= 0:
            raise ValueError(f"pfp_lag_ms must be > 0, got {pfp_lag_ms}")
        if not (0.0 < pfp_ema_alpha <= 1.0):
            raise ValueError(
                f"pfp_ema_alpha must be in (0, 1], got {pfp_ema_alpha}"
            )
        if pfp_eps <= 0.0:
            raise ValueError(f"pfp_eps must be > 0, got {pfp_eps}")
        if pfp_add_weight < 0.0 or pfp_pull_weight < 0.0:
            raise ValueError(
                "pfp_add_weight and pfp_pull_weight must be >= 0, got "
                f"{pfp_add_weight}, {pfp_pull_weight}"
            )
        if pfp_add_weight + pfp_pull_weight <= 0.0:
            raise ValueError("pfp_add_weight + pfp_pull_weight must be > 0")

        if not (0.0 < svac_norm_alpha <= 1.0):
            raise ValueError(
                f"svac_norm_alpha must be in (0, 1], got {svac_norm_alpha}"
            )
        if svac_norm_floor <= 0.0:
            raise ValueError(f"svac_norm_floor must be > 0, got {svac_norm_floor}")

        if (
            composite_weight_ads < 0.0
            or composite_weight_pfp < 0.0
            or composite_weight_svac < 0.0
        ):
            raise ValueError(
                "composite weights must be >= 0, got "
                f"{composite_weight_ads}, {composite_weight_pfp}, {composite_weight_svac}"
            )
        if composite_weight_ads + composite_weight_pfp + composite_weight_svac <= 0.0:
            raise ValueError("at least one composite weight must be > 0")

        self.cell_width_ms = int(cell_width_ms)
        self.ads_slope_windows_ms = slope_ms
        self.ads_zscore_window_ms = int(ads_zscore_window_ms)
        self.ads_zscore_min_periods_ratio = float(ads_zscore_min_periods_ratio)
        self.ads_blend_weights = blend
        self.ads_blend_scale = float(ads_blend_scale)
        self.ads_warmup_ms = int(ads_warmup_ms)
        self.pfp_lag_ms = int(pfp_lag_ms)
        self.pfp_ema_alpha = float(pfp_ema_alpha)
        self.pfp_eps = float(pfp_eps)
        self.pfp_add_weight = float(pfp_add_weight)
        self.pfp_pull_weight = float(pfp_pull_weight)
        self.svac_norm_alpha = float(svac_norm_alpha)
        self.svac_norm_floor = float(svac_norm_floor)
        self.composite_weight_ads = float(composite_weight_ads)
        self.composite_weight_pfp = float(composite_weight_pfp)
        self.composite_weight_svac = float(composite_weight_svac)
        self.cooldown_bins = int(cooldown_bins)

    @property
    def name(self) -> str:
        return "ads_pfp_svac"

    @property
    def required_columns(self) -> list[str]:
        return ["v_add", "v_fill", "v_pull", "vacuum_variant"]

    def default_thresholds(self) -> list[float]:
        return [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

    def compute(self, dataset: dict[str, Any]) -> SignalResult:
        v_add: np.ndarray = dataset["v_add"]
        v_fill: np.ndarray = dataset["v_fill"]
        v_pull: np.ndarray = dataset["v_pull"]
        vacuum: np.ndarray = dataset["vacuum_variant"]
        n_bins: int = int(dataset["n_bins"])

        _validate_grid_shape(v_add, "v_add")
        _validate_grid_shape(v_fill, "v_fill")
        _validate_grid_shape(v_pull, "v_pull")
        _validate_grid_shape(vacuum, "vacuum_variant")
        if n_bins <= 0:
            raise ValueError(f"n_bins must be > 0, got {n_bins}")

        # ------------------------------------------------------------------
        # ADS (frontend-equivalent math with ms->bins semantics)
        # ------------------------------------------------------------------
        raw_w = np.asarray(
            [1.0 / math.sqrt(width) for _, _, width in ADS_BANDS],
            dtype=np.float64,
        )
        band_weights = raw_w / raw_w.sum()

        ads_combined = np.zeros(n_bins, dtype=np.float64)
        for band_idx, (bid_cols, ask_cols, _) in enumerate(ADS_BANDS):
            add_asym = (
                v_add[:, bid_cols].mean(axis=1)
                - v_add[:, ask_cols].mean(axis=1)
            )
            pull_asym = (
                v_pull[:, ask_cols].mean(axis=1)
                - v_pull[:, bid_cols].mean(axis=1)
            )
            ads_combined += band_weights[band_idx] * (add_asym + pull_asym)

        ads_slope_windows_bins = [
            _ms_to_bins(ms, self.cell_width_ms, minimum=2)
            for ms in self.ads_slope_windows_ms
        ]
        ads_zscore_window_bins = _ms_to_bins(
            self.ads_zscore_window_ms,
            self.cell_width_ms,
            minimum=2,
        )
        ads_zscore_min_periods = max(
            2,
            min(
                ads_zscore_window_bins,
                int(math.floor(
                    ads_zscore_window_bins * self.ads_zscore_min_periods_ratio
                )),
            ),
        )
        ads_warmup_bins = _ms_to_bins(self.ads_warmup_ms, self.cell_width_ms, minimum=1)

        ads_signal = np.zeros(n_bins, dtype=np.float64)
        for i, window_bins in enumerate(ads_slope_windows_bins):
            slope = rolling_ols_slope(ads_combined, window_bins)
            z = robust_zscore(
                slope,
                window=ads_zscore_window_bins,
                min_periods=ads_zscore_min_periods,
            )
            ads_signal += self.ads_blend_weights[i] * np.tanh(z / self.ads_blend_scale)

        # Frontend warm condition checks after update(), so first warm index is warmup_bins-1.
        ads_warm_mask = np.arange(n_bins) >= (ads_warmup_bins - 1)
        ads_signal = np.where(ads_warm_mask, ads_signal, 0.0)

        # ------------------------------------------------------------------
        # PFP (frontend-equivalent math with ms->bins lag)
        # ------------------------------------------------------------------
        pfp_lag_bins = _ms_to_bins(self.pfp_lag_ms, self.cell_width_ms, minimum=1)

        i_inner_bid = _zone_mean(v_add + v_fill, PFP_INNER_BID_COLS)
        i_inner_ask = _zone_mean(v_add + v_fill, PFP_INNER_ASK_COLS)
        i_outer_bid = _zone_mean(v_add + v_fill, PFP_OUTER_BID_COLS)
        i_outer_ask = _zone_mean(v_add + v_fill, PFP_OUTER_ASK_COLS)

        p_inner_bid = _zone_mean(v_pull, PFP_INNER_BID_COLS)
        p_inner_ask = _zone_mean(v_pull, PFP_INNER_ASK_COLS)
        p_outer_bid = _zone_mean(v_pull, PFP_OUTER_BID_COLS)
        p_outer_ask = _zone_mean(v_pull, PFP_OUTER_ASK_COLS)

        outer_bid_add_lagged = np.zeros(n_bins, dtype=np.float64)
        outer_ask_add_lagged = np.zeros(n_bins, dtype=np.float64)
        outer_bid_pull_lagged = np.zeros(n_bins, dtype=np.float64)
        outer_ask_pull_lagged = np.zeros(n_bins, dtype=np.float64)
        outer_bid_add_lagged[pfp_lag_bins:] = i_outer_bid[:-pfp_lag_bins]
        outer_ask_add_lagged[pfp_lag_bins:] = i_outer_ask[:-pfp_lag_bins]
        outer_bid_pull_lagged[pfp_lag_bins:] = p_outer_bid[:-pfp_lag_bins]
        outer_ask_pull_lagged[pfp_lag_bins:] = p_outer_ask[:-pfp_lag_bins]

        lead_bid_add = ema_1d(
            i_inner_bid * outer_bid_add_lagged,
            self.pfp_ema_alpha,
        ) / (ema_1d(i_inner_bid * i_outer_bid, self.pfp_ema_alpha) + self.pfp_eps)
        lead_ask_add = ema_1d(
            i_inner_ask * outer_ask_add_lagged,
            self.pfp_ema_alpha,
        ) / (ema_1d(i_inner_ask * i_outer_ask, self.pfp_ema_alpha) + self.pfp_eps)
        lead_bid_pull = ema_1d(
            p_inner_bid * outer_bid_pull_lagged,
            self.pfp_ema_alpha,
        ) / (ema_1d(p_inner_bid * p_outer_bid, self.pfp_ema_alpha) + self.pfp_eps)
        lead_ask_pull = ema_1d(
            p_inner_ask * outer_ask_pull_lagged,
            self.pfp_ema_alpha,
        ) / (ema_1d(p_inner_ask * p_outer_ask, self.pfp_ema_alpha) + self.pfp_eps)

        pfp_signal = (
            self.pfp_add_weight * (lead_bid_add - lead_ask_add)
            + self.pfp_pull_weight * (lead_ask_pull - lead_bid_pull)
        )
        pfp_warm_mask = np.arange(n_bins) >= (pfp_lag_bins - 1)
        pfp_signal = np.where(pfp_warm_mask, pfp_signal, 0.0)

        # ------------------------------------------------------------------
        # SVac (frontend-equivalent weighted asymmetry + running EMA magnitude)
        # ------------------------------------------------------------------
        vac_below = vacuum[:, 0:50]
        vac_above = vacuum[:, 51:101]
        weighted_below = (vac_below * SVAC_BELOW_WEIGHTS).sum(axis=1)
        weighted_above = (vac_above * SVAC_ABOVE_WEIGHTS).sum(axis=1)
        svac_raw = weighted_above - weighted_below

        abs_svac = np.abs(svac_raw)
        svac_running_mag = np.empty(n_bins, dtype=np.float64)
        svac_running_mag[0] = abs_svac[0]
        for i in range(1, n_bins):
            svac_running_mag[i] = (
                self.svac_norm_alpha * abs_svac[i]
                + (1.0 - self.svac_norm_alpha) * svac_running_mag[i - 1]
            )
        svac_signal = svac_raw / np.maximum(svac_running_mag, self.svac_norm_floor)
        # Frontend warm flag becomes true after first update; all emitted bins are warm here.
        svac_warm_mask = np.ones(n_bins, dtype=bool)

        # ------------------------------------------------------------------
        # Composite dynamic-warm blend with renormalization
        # ------------------------------------------------------------------
        weight_ads = np.where(ads_warm_mask, self.composite_weight_ads, 0.0)
        weight_pfp = np.where(pfp_warm_mask, self.composite_weight_pfp, 0.0)
        weight_svac = np.where(svac_warm_mask, self.composite_weight_svac, 0.0)

        total_weight = weight_ads + weight_pfp + weight_svac
        weighted_sum = (
            weight_ads * ads_signal
            + weight_pfp * pfp_signal
            + weight_svac * svac_signal
        )
        composite = np.zeros(n_bins, dtype=np.float64)
        nonzero = total_weight > 0.0
        composite[nonzero] = weighted_sum[nonzero] / total_weight[nonzero]

        full_warm_start = max(ads_warmup_bins, pfp_lag_bins) - 1
        active = composite[full_warm_start:]
        if len(active) == 0:
            adaptive_thresholds = self.default_thresholds()
        else:
            abs_pcts = np.nanpercentile(np.abs(active), [50, 70, 80, 90, 95, 99])
            adaptive_thresholds = sorted(
                set(round(float(v), 6) for v in abs_pcts if float(v) > 0.0)
            )
            if not adaptive_thresholds:
                adaptive_thresholds = self.default_thresholds()

        warm_fraction = (
            ads_warm_mask.astype(np.float64)
            + pfp_warm_mask.astype(np.float64)
            + svac_warm_mask.astype(np.float64)
        ) / 3.0

        return SignalResult(
            signal=composite,
            metadata={
                "adaptive_thresholds": adaptive_thresholds,
                "ads_warmup_bins": int(ads_warmup_bins),
                "pfp_lag_bins": int(pfp_lag_bins),
                "ads_slope_windows_bins": ads_slope_windows_bins,
                "ads_zscore_window_bins": int(ads_zscore_window_bins),
                "ads_zscore_min_periods": int(ads_zscore_min_periods),
                "component_std": {
                    "ads": float(np.std(ads_signal)),
                    "pfp": float(np.std(pfp_signal)),
                    "svac": float(np.std(svac_signal)),
                    "composite": float(np.std(composite)),
                },
                "warmup_fraction_mean": float(np.mean(warm_fraction)),
            },
        )


register_signal("ads_pfp_svac", ADSPFPSVacSignal)
