"""Permutation-derivative signal over state5/micro9 bucket labels.

Builds a directional score from per-bin state-intensity derivatives rather than
absolute resting liquidity levels.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.experiment_harness.signals import register_signal
from src.experiment_harness.signals.base import SignalResult, StatisticalSignal
from src.vp_shared.derivative_core import (
    STATE5_CODES,
    compute_state5_intensities,
    derivative_base_from_intensities,
    normalized_spatial_weights,
    validate_derivative_parameter_set,
)
from src.vp_shared.zscore import (
    robust_or_global_z_series,
    sanitize_unit_interval_array,
    weighted_tanh_blend,
)

N_TICKS = 101


def _validate_grid_shape(arr: np.ndarray, name: str) -> None:
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    if arr.shape[1] != N_TICKS:
        raise ValueError(
            f"{name} must have {N_TICKS} columns (k=-50..+50), got shape={arr.shape}"
        )


def _time_derivative(x: np.ndarray, dt_s: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    if x.size >= 2:
        out[1:] = (x[1:] - x[:-1]) / dt_s
    return out


class DerivativeSignal(StatisticalSignal):
    """Directional signal from microstructure permutation-state derivatives."""

    def __init__(
        self,
        cell_width_ms: int = 100,
        center_exclusion_radius: int = 0,
        spatial_decay_power: float = 0.0,
        zscore_window_bins: int = 300,
        zscore_min_periods: int = 75,
        tanh_scale: float = 3.0,
        d1_weight: float = 1.0,
        d2_weight: float = 0.0,
        d3_weight: float = 0.0,
        bull_pressure_weight: float = 1.0,
        bull_vacuum_weight: float = 1.0,
        bear_pressure_weight: float = 1.0,
        bear_vacuum_weight: float = 1.0,
        mixed_weight: float = 0.0,
    ) -> None:
        if cell_width_ms <= 0:
            raise ValueError(f"cell_width_ms must be > 0, got {cell_width_ms}")
        validate_derivative_parameter_set(
            center_exclusion_radius=center_exclusion_radius,
            spatial_decay_power=spatial_decay_power,
            zscore_window_bins=zscore_window_bins,
            zscore_min_periods=zscore_min_periods,
            tanh_scale=tanh_scale,
            d1_weight=d1_weight,
            d2_weight=d2_weight,
            d3_weight=d3_weight,
            bull_pressure_weight=bull_pressure_weight,
            bull_vacuum_weight=bull_vacuum_weight,
            bear_pressure_weight=bear_pressure_weight,
            bear_vacuum_weight=bear_vacuum_weight,
            mixed_weight=mixed_weight,
        )

        self.cell_width_ms = int(cell_width_ms)
        self.center_exclusion_radius = int(center_exclusion_radius)
        self.spatial_decay_power = float(spatial_decay_power)
        self.zscore_window_bins = int(zscore_window_bins)
        self.zscore_min_periods = int(zscore_min_periods)
        self.tanh_scale = float(tanh_scale)
        self.d1_weight = float(d1_weight)
        self.d2_weight = float(d2_weight)
        self.d3_weight = float(d3_weight)
        self.bull_pressure_weight = float(bull_pressure_weight)
        self.bull_vacuum_weight = float(bull_vacuum_weight)
        self.bear_pressure_weight = float(bear_pressure_weight)
        self.bear_vacuum_weight = float(bear_vacuum_weight)
        self.mixed_weight = float(mixed_weight)

    @property
    def name(self) -> str:
        return "derivative"

    @property
    def required_columns(self) -> list[str]:
        return ["state5_code", "microstate_id"]

    def default_thresholds(self) -> list[float]:
        return [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

    def compute(self, dataset: dict[str, Any]) -> SignalResult:
        state5 = dataset["state5_code"]
        micro9 = dataset["microstate_id"]
        n_bins = int(dataset["n_bins"])
        k_values = np.asarray(dataset["k_values"], dtype=np.int32)

        _validate_grid_shape(state5, "state5_code")
        _validate_grid_shape(micro9, "microstate_id")
        if n_bins <= 0:
            raise ValueError(f"n_bins must be > 0, got {n_bins}")

        weights = normalized_spatial_weights(
            k_values,
            center_exclusion_radius=self.center_exclusion_radius,
            spatial_decay_power=self.spatial_decay_power,
            expected_len=N_TICKS,
        )

        i_bear_vac, i_bear_press, i_mixed, i_bull_press, i_bull_vac = (
            compute_state5_intensities(state5, weights)
        )
        bull, bear, base = derivative_base_from_intensities(
            i_bear_vac=i_bear_vac,
            i_bear_press=i_bear_press,
            i_mixed=i_mixed,
            i_bull_press=i_bull_press,
            i_bull_vac=i_bull_vac,
            bull_pressure_weight=self.bull_pressure_weight,
            bull_vacuum_weight=self.bull_vacuum_weight,
            bear_pressure_weight=self.bear_pressure_weight,
            bear_vacuum_weight=self.bear_vacuum_weight,
            mixed_weight=self.mixed_weight,
        )

        dt_s = float(self.cell_width_ms) / 1000.0
        d1 = _time_derivative(base, dt_s)
        d2 = _time_derivative(d1, dt_s)
        d3 = _time_derivative(d2, dt_s)

        z1 = robust_or_global_z_series(
            d1,
            window=self.zscore_window_bins,
            min_periods=self.zscore_min_periods,
        )
        z2 = robust_or_global_z_series(
            d2,
            window=self.zscore_window_bins,
            min_periods=self.zscore_min_periods,
        )
        z3 = robust_or_global_z_series(
            d3,
            window=self.zscore_window_bins,
            min_periods=self.zscore_min_periods,
        )

        score = np.asarray(
            weighted_tanh_blend(
                z1,
                z2,
                z3,
                d1_weight=self.d1_weight,
                d2_weight=self.d2_weight,
                d3_weight=self.d3_weight,
                tanh_scale=self.tanh_scale,
            ),
            dtype=np.float64,
        )
        norm = abs(self.d1_weight) + abs(self.d2_weight) + abs(self.d3_weight)
        sanitize_unit_interval_array(score)

        center_col = N_TICKS // 2
        micro_ids = np.clip(np.rint(micro9[:, center_col]).astype(np.int32), 0, 8)
        micro_dist = {str(i): int((micro_ids == i).sum()) for i in range(9)}

        state_stack = np.column_stack(
            [i_bear_vac, i_bear_press, i_mixed, i_bull_press, i_bull_vac]
        )
        dominant_idx = np.argmax(state_stack, axis=1)
        dominant_codes = np.asarray([-2, -1, 0, 1, 2], dtype=np.int8)[dominant_idx]
        state_dist = {str(code): int((dominant_codes == code).sum()) for code in STATE5_CODES}

        idx_by_code = {code: idx for idx, code in enumerate(STATE5_CODES)}
        transition = np.zeros((5, 5), dtype=np.int64)
        for i in range(1, n_bins):
            src = idx_by_code[int(dominant_codes[i - 1])]
            dst = idx_by_code[int(dominant_codes[i])]
            transition[src, dst] += 1

        metadata: dict[str, Any] = {
            "state5_distribution": state_dist,
            "micro9_distribution": micro_dist,
            "state5_transition_matrix": transition.tolist(),
            "state5_labels": [str(x) for x in STATE5_CODES],
            "derivative_weight_sum": norm,
        }
        return SignalResult(signal=score, metadata=metadata)


register_signal("derivative", DerivativeSignal)
