"""Incremental runtime models for live vacuum-pressure streaming.

This module currently implements the derivative scorer used by the
evaluation harness (`derivative`) in an online O(n_ticks) per-bin form.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..vp_shared.derivative_core import (
    STATE5_CODES,
    compute_state5_intensities,
    derivative_base_from_intensities,
    normalized_spatial_weights,
    validate_derivative_parameter_set,
)
from ..vp_shared.zscore import (
    robust_or_global_z_latest,
    sanitize_unit_interval_scalar,
    weighted_tanh_blend,
)


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

    def validate(self) -> None:
        """Fail fast on invalid runtime model parameters."""
        validate_derivative_parameter_set(
            center_exclusion_radius=self.center_exclusion_radius,
            spatial_decay_power=self.spatial_decay_power,
            zscore_window_bins=self.zscore_window_bins,
            zscore_min_periods=self.zscore_min_periods,
            tanh_scale=self.tanh_scale,
            d1_weight=self.d1_weight,
            d2_weight=self.d2_weight,
            d3_weight=self.d3_weight,
            bull_pressure_weight=self.bull_pressure_weight,
            bull_vacuum_weight=self.bull_vacuum_weight,
            bear_pressure_weight=self.bear_pressure_weight,
            bear_vacuum_weight=self.bear_vacuum_weight,
            mixed_weight=self.mixed_weight,
        )


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
        self._weights = normalized_spatial_weights(
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

        i_bear_vac, i_bear_press, i_mixed, i_bull_press, i_bull_vac = (
            compute_state5_intensities(s, self._weights)
        )
        bull, bear, base = derivative_base_from_intensities(
            i_bear_vac=float(i_bear_vac),
            i_bear_press=float(i_bear_press),
            i_mixed=float(i_mixed),
            i_bull_press=float(i_bull_press),
            i_bull_vac=float(i_bull_vac),
            bull_pressure_weight=self._params.bull_pressure_weight,
            bull_vacuum_weight=self._params.bull_vacuum_weight,
            bear_pressure_weight=self._params.bear_pressure_weight,
            bear_vacuum_weight=self._params.bear_vacuum_weight,
            mixed_weight=self._params.mixed_weight,
        )

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

        z1 = robust_or_global_z_latest(
            self._d1_hist,
            min_periods=self._params.zscore_min_periods,
        )
        z2 = robust_or_global_z_latest(
            self._d2_hist,
            min_periods=self._params.zscore_min_periods,
        )
        z3 = robust_or_global_z_latest(
            self._d3_hist,
            min_periods=self._params.zscore_min_periods,
        )

        score_raw = weighted_tanh_blend(
            z1,
            z2,
            z3,
            d1_weight=self._params.d1_weight,
            d2_weight=self._params.d2_weight,
            d3_weight=self._params.d3_weight,
            tanh_scale=self._params.tanh_scale,
        )
        score = sanitize_unit_interval_scalar(float(score_raw))

        intensities = np.asarray(
            [
                float(i_bear_vac),
                float(i_bear_press),
                float(i_mixed),
                float(i_bull_press),
                float(i_bull_vac),
            ],
            dtype=np.float64,
        )
        dominant_state5_code = int(STATE5_CODES[int(np.argmax(intensities))])

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
