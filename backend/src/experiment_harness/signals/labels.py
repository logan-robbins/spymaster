"""Shared label computation for ML walk-forward signals.

Extracts and deduplicates the TP/SL labeling logic that was previously
copy-pasted across all 6 ML agent scripts (svm_sp, gbm_mf, knn_cl,
lsvm_der, xgb_snap, pca_ad). This is the canonical implementation.

Label semantics:
    For each bin i, scan forward up to max_hold_bins looking for the
    first price threshold crossing:
        +1 : price rose by tp_ticks * tick_size  (long TP / bullish)
        -1 : price fell by tp_ticks * tick_size  (short TP / bearish)

    Stop-loss crossings (sl_ticks) also resolve to the direction of
    the move that triggered them:
        -1 : price fell by sl_ticks * tick_size  (long SL hit)
        +1 : price rose by sl_ticks * tick_size  (short SL hit)

    The first crossing of any of these four boundaries determines the
    label. If none is hit within max_hold_bins, label = 0 (timeout).

This matches the dominant pattern used by knn_cl, lsvm_der, xgb_snap,
and pca_ad in the original experiment scripts.
"""
from __future__ import annotations

import numpy as np


def compute_labels(
    mid_price: np.ndarray,
    n_bins: int,
    tp_ticks: int = 8,
    sl_ticks: int = 4,
    tick_size: float = 0.25,
    max_hold_bins: int = 1200,
) -> np.ndarray:
    """Compute directional labels for each bin via first-exit TP/SL logic.

    For each bin i, scan forward up to max_hold_bins. The first of these
    four boundaries to be crossed determines the label:
        price >= entry + tp_ticks * tick_size  -->  +1 (long TP)
        price <= entry - tp_ticks * tick_size  -->  -1 (short TP)
        price <= entry - sl_ticks * tick_size  -->  -1 (long SL)
        price >= entry + sl_ticks * tick_size  -->  +1 (short SL)

    Bins where no boundary is crossed within max_hold_bins are labeled 0.
    Bins with non-positive entry price are also labeled 0.

    Args:
        mid_price: (n_bins,) array of mid prices at each bin.
        n_bins: Number of bins (length of mid_price to use).
        tp_ticks: Take-profit distance in ticks. Default 8 ($2.00
            for MNQ with $0.25 tick size).
        sl_ticks: Stop-loss distance in ticks. Default 4 ($1.00).
        tick_size: Dollar value per tick. Default $0.25.
        max_hold_bins: Maximum forward lookahead in bins before
            timeout. Default 1200 (120 seconds at 100ms bins).

    Returns:
        (n_bins,) int8 array with values in {-1, 0, +1}.
    """
    labels = np.zeros(n_bins, dtype=np.int8)
    tp_d: float = tp_ticks * tick_size
    sl_d: float = sl_ticks * tick_size

    for i in range(n_bins):
        entry: float = mid_price[i]
        if entry <= 0.0:
            continue
        end: int = min(i + max_hold_bins, n_bins)
        for j in range(i + 1, end):
            p: float = mid_price[j]
            if p <= 0.0:
                continue
            diff: float = p - entry
            if diff >= tp_d:
                labels[i] = 1
                break
            elif diff <= -tp_d:
                labels[i] = -1
                break
            elif diff <= -sl_d:
                labels[i] = -1
                break
            elif diff >= sl_d:
                labels[i] = 1
                break

    return labels
