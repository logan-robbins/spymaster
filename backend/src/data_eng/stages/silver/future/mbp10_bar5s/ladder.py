from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import EPSILON, POINT


def compute_ladder_features(bid_px: NDArray[np.float64], ask_px: NDArray[np.float64]) -> dict[str, float]:
    ask_gaps = []
    for i in range(9):
        if ask_px[i] > EPSILON and ask_px[i + 1] > EPSILON:
            gap = (ask_px[i + 1] - ask_px[i]) / POINT
            ask_gaps.append(gap)
    
    bid_gaps = []
    for i in range(9):
        if bid_px[i] > EPSILON and bid_px[i + 1] > EPSILON:
            gap = (bid_px[i] - bid_px[i + 1]) / POINT
            bid_gaps.append(gap)
    
    if len(ask_gaps) > 0:
        ask_gap_max = float(np.max(ask_gaps))
        ask_gap_mean = float(np.mean(ask_gaps))
    else:
        ask_gap_max = float("nan")
        ask_gap_mean = float("nan")
    
    if len(bid_gaps) > 0:
        bid_gap_max = float(np.max(bid_gaps))
        bid_gap_mean = float(np.mean(bid_gaps))
    else:
        bid_gap_max = float("nan")
        bid_gap_mean = float("nan")
    
    return {
        "ask_gap_max_pts": ask_gap_max,
        "ask_gap_mean_pts": ask_gap_mean,
        "bid_gap_max_pts": bid_gap_max,
        "bid_gap_mean_pts": bid_gap_mean,
    }

