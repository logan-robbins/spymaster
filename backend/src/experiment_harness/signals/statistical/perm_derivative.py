"""Permutation-derivative signal over state5/micro9 bucket labels.

Builds a directional score from per-bin state-intensity derivatives rather than
absolute resting liquidity levels.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.experiment_harness.eval_engine import robust_zscore
from src.experiment_harness.signals import register_signal
from src.experiment_harness.signals.base import SignalResult, StatisticalSignal

N_TICKS = 101
STATE5_CODES = (-2, -1, 0, 1, 2)


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


def _robust_or_global_z(
    x: np.ndarray,
    *,
    window: int,
    min_periods: int,
) -> np.ndarray:
    z = robust_zscore(x, window=window, min_periods=min_periods)
    if float(np.max(np.abs(z))) > 0.0:
        return z
    scale = float(np.std(x))
    if scale <= 1e-12:
        return z
    return x / scale


def _normalized_spatial_weights(
    k_values: np.ndarray,
    *,
    center_exclusion_radius: int,
    spatial_decay_power: float,
) -> np.ndarray:
    if k_values.shape != (N_TICKS,):
        raise ValueError(f"k_values must have shape ({N_TICKS},), got {k_values.shape}")

    w = np.ones(N_TICKS, dtype=np.float64)
    if center_exclusion_radius > 0:
        w[np.abs(k_values) <= center_exclusion_radius] = 0.0

    if spatial_decay_power > 0.0:
        dist = np.abs(k_values).astype(np.float64)
        dist[dist < 1.0] = 1.0
        w *= 1.0 / np.power(dist, spatial_decay_power)

    total = float(w.sum())
    if total <= 0.0:
        raise ValueError(
            "Spatial weights collapsed to zero. Reduce center_exclusion_radius "
            "or spatial_decay_power."
        )
    return w / total


class PermDerivativeSignal(StatisticalSignal):
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
        enable_weighted_blend: bool = False,
    ) -> None:
        if cell_width_ms <= 0:
            raise ValueError(f"cell_width_ms must be > 0, got {cell_width_ms}")
        if center_exclusion_radius < 0:
            raise ValueError(
                f"center_exclusion_radius must be >= 0, got {center_exclusion_radius}"
            )
        if spatial_decay_power < 0.0:
            raise ValueError(
                f"spatial_decay_power must be >= 0, got {spatial_decay_power}"
            )
        if zscore_window_bins < 2:
            raise ValueError(f"zscore_window_bins must be >= 2, got {zscore_window_bins}")
        if zscore_min_periods < 2:
            raise ValueError(f"zscore_min_periods must be >= 2, got {zscore_min_periods}")
        if zscore_min_periods > zscore_window_bins:
            raise ValueError("zscore_min_periods cannot exceed zscore_window_bins")
        if tanh_scale <= 0.0:
            raise ValueError(f"tanh_scale must be > 0, got {tanh_scale}")

        for name, value in (
            ("d1_weight", d1_weight),
            ("d2_weight", d2_weight),
            ("d3_weight", d3_weight),
            ("bull_pressure_weight", bull_pressure_weight),
            ("bull_vacuum_weight", bull_vacuum_weight),
            ("bear_pressure_weight", bear_pressure_weight),
            ("bear_vacuum_weight", bear_vacuum_weight),
            ("mixed_weight", mixed_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        if abs(d1_weight) + abs(d2_weight) + abs(d3_weight) <= 0.0:
            raise ValueError("At least one derivative weight must be > 0")

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
        self.enable_weighted_blend = bool(enable_weighted_blend)

    @property
    def name(self) -> str:
        return "perm_derivative"

    @property
    def required_columns(self) -> list[str]:
        return ["perm_state5_code", "perm_microstate_id"]

    def default_thresholds(self) -> list[float]:
        return [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

    def compute(self, dataset: dict[str, Any]) -> SignalResult:
        state5 = dataset["perm_state5_code"]
        micro9 = dataset["perm_microstate_id"]
        n_bins = int(dataset["n_bins"])
        k_values = np.asarray(dataset["k_values"], dtype=np.int32)

        _validate_grid_shape(state5, "perm_state5_code")
        _validate_grid_shape(micro9, "perm_microstate_id")
        if n_bins <= 0:
            raise ValueError(f"n_bins must be > 0, got {n_bins}")

        weights = _normalized_spatial_weights(
            k_values,
            center_exclusion_radius=self.center_exclusion_radius,
            spatial_decay_power=self.spatial_decay_power,
        )

        s = np.rint(state5).astype(np.int8)
        i_bear_vac = (s == -2).astype(np.float64) @ weights
        i_bear_press = (s == -1).astype(np.float64) @ weights
        i_mixed = (s == 0).astype(np.float64) @ weights
        i_bull_press = (s == 1).astype(np.float64) @ weights
        i_bull_vac = (s == 2).astype(np.float64) @ weights

        bull = self.bull_pressure_weight * i_bull_press + self.bull_vacuum_weight * i_bull_vac
        bear = self.bear_pressure_weight * i_bear_press + self.bear_vacuum_weight * i_bear_vac
        net = bull - bear

        if self.enable_weighted_blend:
            mixed_damp = 1.0 - self.mixed_weight * i_mixed
            mixed_damp = np.clip(mixed_damp, 0.0, 1.0)
            base = net * mixed_damp
        else:
            base = net

        dt_s = float(self.cell_width_ms) / 1000.0
        d1 = _time_derivative(base, dt_s)
        d2 = _time_derivative(d1, dt_s)
        d3 = _time_derivative(d2, dt_s)

        z1 = _robust_or_global_z(
            d1,
            window=self.zscore_window_bins,
            min_periods=self.zscore_min_periods,
        )
        z2 = _robust_or_global_z(
            d2,
            window=self.zscore_window_bins,
            min_periods=self.zscore_min_periods,
        )
        z3 = _robust_or_global_z(
            d3,
            window=self.zscore_window_bins,
            min_periods=self.zscore_min_periods,
        )

        score = (
            self.d1_weight * np.tanh(z1 / self.tanh_scale)
            + self.d2_weight * np.tanh(z2 / self.tanh_scale)
            + self.d3_weight * np.tanh(z3 / self.tanh_scale)
        )
        norm = abs(self.d1_weight) + abs(self.d2_weight) + abs(self.d3_weight)
        if norm > 0.0:
            score = score / norm
        np.nan_to_num(score, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(score, -1.0, 1.0, out=score)

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
            "perm_state5_distribution": state_dist,
            "perm_micro9_distribution": micro_dist,
            "perm_state5_transition_matrix": transition.tolist(),
            "perm_state5_labels": [str(x) for x in STATE5_CODES],
            "derivative_weight_sum": norm,
        }
        return SignalResult(signal=score, metadata=metadata)


register_signal("perm_derivative", PermDerivativeSignal)
