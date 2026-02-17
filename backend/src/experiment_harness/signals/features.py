"""Shared feature-building utilities for experiment harness signals.

Extracts and deduplicates common feature computations that were
previously inlined across multiple ML agent scripts:

    band_asymmetry   -- svm_sp, gbm_mf, knn_cl, lsvm_der
    rolling_mean_std -- svm_sp, gbm_mf, xgb_snap
    distance_weighted_sum -- lsvm_der
    ema_1d           -- msd (PFP sub-signal)

All functions operate on raw numpy arrays with explicit shapes
documented. No pandas dependency.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Default band definitions
# ---------------------------------------------------------------------------
# Used by ADS, SVM_SP, KNN_CL, LSVM_DER, GBM_MF for spatial asymmetry.
# Column 50 = k=0 (spot). Bid side < 50, ask side > 50.
DEFAULT_BAND_DEFS: list[dict[str, list[int] | str | int]] = [
    {
        "name": "inner",
        "bid_cols": list(range(47, 50)),    # k=-3, -2, -1
        "ask_cols": list(range(51, 54)),    # k=+1, +2, +3
        "width": 3,
    },
    {
        "name": "mid",
        "bid_cols": list(range(39, 47)),    # k=-11 .. -4
        "ask_cols": list(range(54, 62)),    # k=+4 .. +11
        "width": 8,
    },
    {
        "name": "outer",
        "bid_cols": list(range(27, 39)),    # k=-23 .. -12
        "ask_cols": list(range(62, 74)),    # k=+12 .. +23
        "width": 12,
    },
]


def band_asymmetry(
    grid: np.ndarray,
    col_name: str,
    band_defs: list[dict[str, list[int] | str | int]] | None = None,
) -> list[np.ndarray]:
    """Compute bid-ask asymmetry per band for a single grid column.

    Sign convention depends on whether the column represents "add"
    (order placement) or "pull" (order cancellation) activity:

        "add" columns (v_add, a_add, j_add):
            asym = mean(bid_side) - mean(ask_side)
            Positive = more adding on bid = bullish pressure.

        "pull" columns (v_pull, a_pull, j_pull):
            asym = mean(ask_side) - mean(bid_side)
            Positive = more pulling on ask = bullish pressure.

    Both conventions yield positive = bullish when combined.

    Args:
        grid: (n_bins, 101) array for one spatial column.
        col_name: Column name string. Must contain "add" or "pull"
            to determine sign convention. If neither is present,
            defaults to "add" convention.
        band_defs: List of band definitions, each a dict with keys
            "bid_cols" and "ask_cols" (lists of column indices).
            Defaults to DEFAULT_BAND_DEFS (inner/mid/outer).

    Returns:
        List of (n_bins,) asymmetry arrays, one per band, in the
        order of band_defs.
    """
    if band_defs is None:
        band_defs = DEFAULT_BAND_DEFS

    is_pull: bool = "pull" in col_name
    asymmetries: list[np.ndarray] = []

    for band in band_defs:
        bid_cols: list[int] = band["bid_cols"]  # type: ignore[assignment]
        ask_cols: list[int] = band["ask_cols"]  # type: ignore[assignment]

        bid_mean: np.ndarray = grid[:, bid_cols].mean(axis=1)
        ask_mean: np.ndarray = grid[:, ask_cols].mean(axis=1)

        if is_pull:
            asym = ask_mean - bid_mean
        else:
            asym = bid_mean - ask_mean

        asymmetries.append(asym)

    return asymmetries


def rolling_mean_std(
    arr: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Efficient rolling mean and standard deviation using cumulative sums.

    Uses the identity:
        var = E[X^2] - E[X]^2
    computed via prefix sums for O(1) per-element after O(n) setup.

    The first (window - 1) elements are left as zeros since the rolling
    window has not yet filled.

    Args:
        arr: (n,) input array of float64 values.
        window: Rolling window size in elements. Must be >= 1.

    Returns:
        Tuple of (rmean, rstd), each of shape (n,).
        rmean[i] = mean of arr[i-window+1 : i+1] for i >= window-1.
        rstd[i]  = std  of arr[i-window+1 : i+1] for i >= window-1.
        Both are zero for i < window-1.
    """
    n: int = len(arr)
    rmean = np.zeros(n, dtype=np.float64)
    rstd = np.zeros(n, dtype=np.float64)

    if n == 0 or window < 1:
        return rmean, rstd

    cs: np.ndarray = np.cumsum(arr)
    cs2: np.ndarray = np.cumsum(arr ** 2)

    for i in range(window - 1, n):
        s: float = cs[i] - (cs[i - window] if i >= window else 0.0)
        s2: float = cs2[i] - (cs2[i - window] if i >= window else 0.0)
        m: float = s / window
        rmean[i] = m
        variance: float = s2 / window - m * m
        rstd[i] = np.sqrt(max(variance, 0.0))

    return rmean, rstd


def distance_weighted_sum(
    grid: np.ndarray,
    bid_cols: np.ndarray,
    ask_cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1/|k| distance-weighted means for bid and ask sides.

    Weights each tick inversely proportional to its distance from spot
    (column 50), so near-spot activity dominates. Weights are normalized
    to sum to 1.0 on each side independently.

    Used by LSVM_DER for full-width divergence features.

    Args:
        grid: (n_bins, 101) array for one spatial column.
        bid_cols: 1D array of column indices for the bid side.
            Expected to be < 50. Distance = 50 - col_index.
        ask_cols: 1D array of column indices for the ask side.
            Expected to be > 50. Distance = col_index - 50.

    Returns:
        Tuple of (bid_weighted, ask_weighted), each (n_bins,).
        bid_weighted[i] = sum(grid[i, bid_cols] * bid_weights).
        ask_weighted[i] = sum(grid[i, ask_cols] * ask_weights).

    Raises:
        ValueError: If any bid_col >= 50 or ask_col <= 50 (wrong side).
    """
    bid_distances: np.ndarray = (50 - bid_cols).astype(np.float64)
    ask_distances: np.ndarray = (ask_cols - 50).astype(np.float64)

    if np.any(bid_distances <= 0):
        raise ValueError(
            f"bid_cols must all be < 50 (bid side). Got: {bid_cols}"
        )
    if np.any(ask_distances <= 0):
        raise ValueError(
            f"ask_cols must all be > 50 (ask side). Got: {ask_cols}"
        )

    bid_weights: np.ndarray = 1.0 / bid_distances
    bid_weights /= bid_weights.sum()

    ask_weights: np.ndarray = 1.0 / ask_distances
    ask_weights /= ask_weights.sum()

    bid_wm: np.ndarray = grid[:, bid_cols] @ bid_weights
    ask_wm: np.ndarray = grid[:, ask_cols] @ ask_weights

    return bid_wm, ask_wm


def ema_1d(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average over a 1D array.

    Recurrence:
        EMA[0] = arr[0]
        EMA[t] = alpha * arr[t] + (1 - alpha) * EMA[t-1]

    Used by PFP signal for smoothing lead-lag metrics.

    Args:
        arr: (n,) input array.
        alpha: Smoothing factor in (0, 1]. Higher alpha means more
            weight on the current value (less smoothing).

    Returns:
        (n,) float64 array of EMA values.

    Raises:
        ValueError: If alpha is not in (0, 1] or arr is empty.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    n: int = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    out = np.empty(n, dtype=np.float64)
    out[0] = arr[0]
    beta: float = 1.0 - alpha

    for i in range(1, n):
        out[i] = alpha * arr[i] + beta * out[i - 1]

    return out
