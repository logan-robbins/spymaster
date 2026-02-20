from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

MAD_TO_SIGMA: float = 1.4826


def validate_zscore_tanh_params(
    *,
    zscore_window_bins: int,
    zscore_min_periods: int,
    tanh_scale: float,
    threshold_neutral: float | None = None,
) -> None:
    if zscore_window_bins < 2:
        raise ValueError(f"zscore_window_bins must be >= 2, got {zscore_window_bins}")
    if zscore_min_periods < 2:
        raise ValueError(f"zscore_min_periods must be >= 2, got {zscore_min_periods}")
    if zscore_min_periods > zscore_window_bins:
        raise ValueError("zscore_min_periods cannot exceed zscore_window_bins")
    if tanh_scale <= 0.0:
        raise ValueError(f"tanh_scale must be > 0, got {tanh_scale}")
    if threshold_neutral is not None and not (0.0 < threshold_neutral < 1.0):
        raise ValueError(
            f"threshold_neutral must be in (0, 1), got {threshold_neutral}"
        )


def validate_positive_weight_vector(
    values: Sequence[float],
    *,
    expected_size: int,
    field_name: str,
) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if weights.ndim != 1 or weights.size != expected_size:
        raise ValueError(
            f"{field_name} must contain exactly {expected_size} values, "
            f"got shape {weights.shape}"
        )
    if np.any(weights <= 0.0):
        raise ValueError(f"{field_name} values must be > 0")
    return weights / float(weights.sum())


def robust_zscore_rolling_1d(
    arr: np.ndarray,
    *,
    window: int,
    min_periods: int,
    scale_eps: float = 1e-12,
) -> np.ndarray:
    """Rolling robust z-score on a 1D series using median/MAD."""
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        start = max(0, i - window + 1)
        segment = arr[start : i + 1]
        if len(segment) < min_periods:
            result[i] = 0.0
            continue

        med = float(np.median(segment))
        mad = float(np.median(np.abs(segment - med)))
        scale = MAD_TO_SIGMA * mad
        if scale <= scale_eps:
            result[i] = 0.0
        else:
            result[i] = float((arr[i] - med) / scale)

    return result


def robust_or_global_z_series(
    arr: np.ndarray,
    *,
    window: int,
    min_periods: int,
    scale_eps: float = 1e-12,
) -> np.ndarray:
    z = robust_zscore_rolling_1d(
        arr,
        window=window,
        min_periods=min_periods,
        scale_eps=scale_eps,
    )
    if float(np.max(np.abs(z))) > 0.0:
        return z

    std = float(np.std(arr))
    if std <= scale_eps:
        return z
    return arr / std


def robust_or_global_z_latest(
    history: deque[float] | np.ndarray,
    *,
    min_periods: int,
    scale_eps: float = 1e-12,
) -> float:
    if len(history) < min_periods:
        return 0.0

    arr = np.asarray(history, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = MAD_TO_SIGMA * mad
    if scale > scale_eps:
        return float((arr[-1] - med) / scale)

    std = float(np.std(arr))
    if std <= scale_eps:
        return 0.0
    return float(arr[-1] / std)


def robust_z_current_vectorized(
    history: np.ndarray,
    current: np.ndarray,
    *,
    scale_eps: float,
    out: np.ndarray,
    work: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized robust z-score for current values against history matrix."""
    med = np.median(history, axis=0)
    if work is None:
        dev = np.abs(history - med)
    else:
        np.subtract(history, med, out=work)
        np.abs(work, out=work)
        dev = work
    mad = np.median(dev, axis=0)
    scale = MAD_TO_SIGMA * mad

    valid = scale > scale_eps
    out.fill(0.0)
    out[valid] = (current[valid] - med[valid]) / scale[valid]
    return out


def weighted_tanh_blend(
    z1: np.ndarray | float,
    z2: np.ndarray | float,
    z3: np.ndarray | float,
    *,
    d1_weight: float,
    d2_weight: float,
    d3_weight: float,
    tanh_scale: float,
) -> np.ndarray | float:
    score = (
        d1_weight * np.tanh(np.asarray(z1) / tanh_scale)
        + d2_weight * np.tanh(np.asarray(z2) / tanh_scale)
        + d3_weight * np.tanh(np.asarray(z3) / tanh_scale)
    )
    norm = abs(d1_weight) + abs(d2_weight) + abs(d3_weight)
    if norm > 0.0:
        score = score / norm
    return score


def sanitize_unit_interval_array(arr: np.ndarray) -> np.ndarray:
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(arr, -1.0, 1.0, out=arr)
    return arr


def sanitize_unit_interval_scalar(value: float) -> float:
    sanitized = float(np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0))
    return float(np.clip(sanitized, -1.0, 1.0))
