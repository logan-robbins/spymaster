from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from .zscore import validate_zscore_tanh_params

STATE5_CODES: tuple[int, ...] = (-2, -1, 0, 1, 2)


class State5Intensities(NamedTuple):
    """Weighted intensity fractions for each of the five derivative states."""

    bear_vac: Any
    bear_press: Any
    mixed: Any
    bull_press: Any
    bull_vac: Any


def validate_derivative_parameter_set(
    *,
    center_exclusion_radius: int,
    spatial_decay_power: float,
    zscore_window_bins: int,
    zscore_min_periods: int,
    tanh_scale: float,
    d1_weight: float,
    d2_weight: float,
    d3_weight: float,
    bull_pressure_weight: float,
    bull_vacuum_weight: float,
    bear_pressure_weight: float,
    bear_vacuum_weight: float,
    mixed_weight: float,
) -> None:
    if center_exclusion_radius < 0:
        raise ValueError(
            f"center_exclusion_radius must be >= 0, got {center_exclusion_radius}"
        )
    if spatial_decay_power < 0.0:
        raise ValueError(
            f"spatial_decay_power must be >= 0, got {spatial_decay_power}"
        )

    validate_zscore_tanh_params(
        zscore_window_bins=zscore_window_bins,
        zscore_min_periods=zscore_min_periods,
        tanh_scale=tanh_scale,
    )

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


def normalized_spatial_weights(
    k_values: np.ndarray,
    *,
    center_exclusion_radius: int,
    spatial_decay_power: float,
    expected_len: int | None = None,
) -> np.ndarray:
    if k_values.ndim != 1:
        raise ValueError(f"k_values must be 1D, got shape={k_values.shape}")
    if expected_len is not None and k_values.shape[0] != expected_len:
        raise ValueError(
            f"k_values must have shape ({expected_len},), got {k_values.shape}"
        )

    weights = np.ones(k_values.shape[0], dtype=np.float64)
    if center_exclusion_radius > 0:
        weights[np.abs(k_values) <= center_exclusion_radius] = 0.0

    if spatial_decay_power > 0.0:
        dist = np.abs(k_values).astype(np.float64)
        dist[dist < 1.0] = 1.0
        weights *= 1.0 / np.power(dist, spatial_decay_power)

    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(
            "Spatial weights collapsed to zero. "
            "Reduce center_exclusion_radius or spatial_decay_power."
        )
    return weights / total


def compute_state5_intensities(
    state5_code: np.ndarray,
    weights: np.ndarray,
) -> State5Intensities:
    """Compute weighted intensity fractions for each derivative state.

    Args:
        state5_code: 1-D array of state codes in {-2, -1, 0, 1, 2}.
        weights: 1-D spatial weight vector (same length as *state5_code*).

    Returns:
        A :class:`State5Intensities` named tuple with fields
        ``bear_vac``, ``bear_press``, ``mixed``, ``bull_press``, ``bull_vac``.
    """
    s = np.rint(state5_code).astype(np.int8)
    i_bear_vac = (s == -2).astype(np.float64) @ weights
    i_bear_press = (s == -1).astype(np.float64) @ weights
    i_mixed = (s == 0).astype(np.float64) @ weights
    i_bull_press = (s == 1).astype(np.float64) @ weights
    i_bull_vac = (s == 2).astype(np.float64) @ weights
    return State5Intensities(i_bear_vac, i_bear_press, i_mixed, i_bull_press, i_bull_vac)


def derivative_base_from_intensities(
    *,
    i_bear_vac: np.ndarray | float,
    i_bear_press: np.ndarray | float,
    i_mixed: np.ndarray | float,
    i_bull_press: np.ndarray | float,
    i_bull_vac: np.ndarray | float,
    bull_pressure_weight: float,
    bull_vacuum_weight: float,
    bear_pressure_weight: float,
    bear_vacuum_weight: float,
    mixed_weight: float,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    bull = bull_pressure_weight * i_bull_press + bull_vacuum_weight * i_bull_vac
    bear = bear_pressure_weight * i_bear_press + bear_vacuum_weight * i_bear_vac
    net = bull - bear
    mixed_damp = 1.0 - mixed_weight * i_mixed
    mixed_damp = np.clip(mixed_damp, 0.0, 1.0)
    base = net * mixed_damp

    return bull, bear, base
