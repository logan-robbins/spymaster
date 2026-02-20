"""Incremental runtime models for live vacuum-pressure streaming.

This module currently implements the derivative scorer used by the
evaluation harness (`derivative`) in an online O(n_ticks) per-bin form.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


_STATE5_CODES: tuple[int, ...] = (-2, -1, 0, 1, 2)


@dataclass(frozen=True)
class DerivativeRuntimeParams:
    """Config for the incremental derivative runtime scorer."""

    center_exclusion_radius: int = 0
    spatial_decay_power: float = 0.0
    zscore_window_bins: int = 240
    zscore_min_periods: int = 60
    tanh_scale: float = 3.0
    d1_weight: float = 1.0
    d2_weight: float = 0.0
    d3_weight: float = 0.0
    bull_pressure_weight: float = 1.0
    bull_vacuum_weight: float = 1.0
    bear_pressure_weight: float = 1.0
    bear_vacuum_weight: float = 1.0
    mixed_weight: float = 0.0
    enable_weighted_blend: bool = False

    def validate(self) -> None:
        """Fail fast on invalid runtime model parameters."""
        if self.center_exclusion_radius < 0:
            raise ValueError(
                "center_exclusion_radius must be >= 0, "
                f"got {self.center_exclusion_radius}"
            )
        if self.spatial_decay_power < 0.0:
            raise ValueError(
                "spatial_decay_power must be >= 0, "
                f"got {self.spatial_decay_power}"
            )
        if self.zscore_window_bins < 2:
            raise ValueError(
                "zscore_window_bins must be >= 2, "
                f"got {self.zscore_window_bins}"
            )
        if self.zscore_min_periods < 2:
            raise ValueError(
                "zscore_min_periods must be >= 2, "
                f"got {self.zscore_min_periods}"
            )
        if self.zscore_min_periods > self.zscore_window_bins:
            raise ValueError(
                "zscore_min_periods cannot exceed zscore_window_bins"
            )
        if self.tanh_scale <= 0.0:
            raise ValueError(
                f"tanh_scale must be > 0, got {self.tanh_scale}"
            )
        for name, value in (
            ("d1_weight", self.d1_weight),
            ("d2_weight", self.d2_weight),
            ("d3_weight", self.d3_weight),
            ("bull_pressure_weight", self.bull_pressure_weight),
            ("bull_vacuum_weight", self.bull_vacuum_weight),
            ("bear_pressure_weight", self.bear_pressure_weight),
            ("bear_vacuum_weight", self.bear_vacuum_weight),
            ("mixed_weight", self.mixed_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if abs(self.d1_weight) + abs(self.d2_weight) + abs(self.d3_weight) <= 0.0:
            raise ValueError("At least one derivative weight must be > 0")


@dataclass(frozen=True)
class DerivativeRuntimeOutput:
    """Single-bin runtime model output emitted with stream updates."""

    name: str
    score: float
    ready: bool
    sample_count: int
    base: float
    d1: float
    d2: float
    d3: float
    z1: float
    z2: float
    z3: float
    bull_intensity: float
    bear_intensity: float
    mixed_intensity: float
    dominant_state5_code: int


def _normalized_spatial_weights(
    k_values: np.ndarray,
    *,
    center_exclusion_radius: int,
    spatial_decay_power: float,
) -> np.ndarray:
    if k_values.ndim != 1:
        raise ValueError(f"k_values must be 1D, got shape={k_values.shape}")

    w = np.ones(k_values.shape[0], dtype=np.float64)
    if center_exclusion_radius > 0:
        w[np.abs(k_values) <= center_exclusion_radius] = 0.0

    if spatial_decay_power > 0.0:
        dist = np.abs(k_values).astype(np.float64)
        dist[dist < 1.0] = 1.0
        w *= 1.0 / np.power(dist, spatial_decay_power)

    total = float(w.sum())
    if total <= 0.0:
        raise ValueError(
            "Spatial weights collapsed to zero. "
            "Reduce center_exclusion_radius or spatial_decay_power."
        )
    return w / total


def _robust_or_global_z_from_history(
    values: deque[float],
    *,
    min_periods: int,
) -> float:
    if len(values) < min_periods:
        return 0.0

    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if scale > 1e-12:
        return float((arr[-1] - med) / scale)

    std = float(np.std(arr))
    if std <= 1e-12:
        return 0.0
    return float(arr[-1] / std)


class DerivativeRuntime:
    """Incremental online variant of the harness `derivative` signal."""

    def __init__(
        self,
        *,
        k_values: Iterable[int],
        cell_width_ms: int,
        params: DerivativeRuntimeParams,
    ) -> None:
        if cell_width_ms <= 0:
            raise ValueError(f"cell_width_ms must be > 0, got {cell_width_ms}")
        params.validate()

        k_arr = np.asarray(tuple(k_values), dtype=np.int32)
        if k_arr.ndim != 1 or k_arr.size == 0:
            raise ValueError(f"k_values must be a non-empty 1D sequence, got {k_arr.shape}")

        self._params = params
        self._dt_s = float(cell_width_ms) / 1000.0
        self._weights = _normalized_spatial_weights(
            k_arr,
            center_exclusion_radius=params.center_exclusion_radius,
            spatial_decay_power=params.spatial_decay_power,
        )

        self._prev_base = 0.0
        self._prev_d1 = 0.0
        self._prev_d2 = 0.0
        self._sample_count = 0

        maxlen = params.zscore_window_bins
        self._d1_hist: deque[float] = deque(maxlen=maxlen)
        self._d2_hist: deque[float] = deque(maxlen=maxlen)
        self._d3_hist: deque[float] = deque(maxlen=maxlen)

    @property
    def params(self) -> DerivativeRuntimeParams:
        return self._params

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def _ready(self) -> bool:
        min_periods = self._params.zscore_min_periods
        ready = True
        if self._params.d1_weight > 0.0:
            ready = ready and (len(self._d1_hist) >= min_periods)
        if self._params.d2_weight > 0.0:
            ready = ready and (len(self._d2_hist) >= min_periods)
        if self._params.d3_weight > 0.0:
            ready = ready and (len(self._d3_hist) >= min_periods)
        return ready

    def update(self, state5_code: np.ndarray) -> DerivativeRuntimeOutput:
        """Advance runtime signal by one bin of state5 labels."""
        s = np.asarray(state5_code, dtype=np.int8)
        if s.ndim != 1:
            raise ValueError(f"state5_code must be 1D, got shape={s.shape}")
        if s.shape[0] != self._weights.shape[0]:
            raise ValueError(
                "state5_code length must match runtime k-axis length "
                f"({self._weights.shape[0]}), got {s.shape[0]}"
            )

        i_bear_vac = float(((s == -2).astype(np.float64)) @ self._weights)
        i_bear_press = float(((s == -1).astype(np.float64)) @ self._weights)
        i_mixed = float(((s == 0).astype(np.float64)) @ self._weights)
        i_bull_press = float(((s == 1).astype(np.float64)) @ self._weights)
        i_bull_vac = float(((s == 2).astype(np.float64)) @ self._weights)

        bull = (
            self._params.bull_pressure_weight * i_bull_press
            + self._params.bull_vacuum_weight * i_bull_vac
        )
        bear = (
            self._params.bear_pressure_weight * i_bear_press
            + self._params.bear_vacuum_weight * i_bear_vac
        )
        net = bull - bear

        if self._params.enable_weighted_blend:
            mixed_damp = 1.0 - self._params.mixed_weight * i_mixed
            mixed_damp = float(np.clip(mixed_damp, 0.0, 1.0))
            base = net * mixed_damp
        else:
            base = net

        if self._sample_count == 0:
            d1 = 0.0
            d2 = 0.0
            d3 = 0.0
        else:
            d1 = (base - self._prev_base) / self._dt_s
            d2 = (d1 - self._prev_d1) / self._dt_s
            d3 = (d2 - self._prev_d2) / self._dt_s

        self._d1_hist.append(float(d1))
        self._d2_hist.append(float(d2))
        self._d3_hist.append(float(d3))

        z1 = _robust_or_global_z_from_history(
            self._d1_hist,
            min_periods=self._params.zscore_min_periods,
        )
        z2 = _robust_or_global_z_from_history(
            self._d2_hist,
            min_periods=self._params.zscore_min_periods,
        )
        z3 = _robust_or_global_z_from_history(
            self._d3_hist,
            min_periods=self._params.zscore_min_periods,
        )

        score = (
            self._params.d1_weight * math.tanh(z1 / self._params.tanh_scale)
            + self._params.d2_weight * math.tanh(z2 / self._params.tanh_scale)
            + self._params.d3_weight * math.tanh(z3 / self._params.tanh_scale)
        )
        norm = (
            abs(self._params.d1_weight)
            + abs(self._params.d2_weight)
            + abs(self._params.d3_weight)
        )
        if norm > 0.0:
            score = score / norm
        score = float(np.clip(np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0))

        intensities = np.asarray(
            [i_bear_vac, i_bear_press, i_mixed, i_bull_press, i_bull_vac],
            dtype=np.float64,
        )
        dominant_state5_code = int(_STATE5_CODES[int(np.argmax(intensities))])

        self._prev_base = base
        self._prev_d1 = d1
        self._prev_d2 = d2
        self._sample_count += 1

        return DerivativeRuntimeOutput(
            name="derivative",
            score=score,
            ready=self._ready(),
            sample_count=self._sample_count,
            base=float(base),
            d1=float(d1),
            d2=float(d2),
            d3=float(d3),
            z1=float(z1),
            z2=float(z2),
            z3=float(z3),
            bull_intensity=float(bull),
            bear_intensity=float(bear),
            mixed_intensity=float(i_mixed),
            dominant_state5_code=dominant_state5_code,
        )
