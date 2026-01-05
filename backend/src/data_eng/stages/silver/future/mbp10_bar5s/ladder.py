from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import EPSILON, POINT


def compute_ladder_features(bid_px: NDArray[np.float64], ask_px: NDArray[np.float64]) -> dict[str, float]:
    valid_ask_pairs = (ask_px[:-1] > EPSILON) & (ask_px[1:] > EPSILON)
    ask_gaps = np.diff(ask_px) / POINT
    ask_gaps_valid = ask_gaps[valid_ask_pairs]
    
    valid_bid_pairs = (bid_px[:-1] > EPSILON) & (bid_px[1:] > EPSILON)
    bid_gaps = -np.diff(bid_px) / POINT
    bid_gaps_valid = bid_gaps[valid_bid_pairs]
    
    if len(ask_gaps_valid) > 0:
        ask_gap_max = float(np.max(ask_gaps_valid))
        ask_gap_mean = float(np.mean(ask_gaps_valid))
    else:
        ask_gap_max = float("nan")
        ask_gap_mean = float("nan")
    
    if len(bid_gaps_valid) > 0:
        bid_gap_max = float(np.max(bid_gaps_valid))
        bid_gap_mean = float(np.mean(bid_gaps_valid))
    else:
        bid_gap_max = float("nan")
        bid_gap_mean = float("nan")
    
    return {
        "ask_gap_max_pts": ask_gap_max,
        "ask_gap_mean_pts": ask_gap_mean,
        "bid_gap_max_pts": bid_gap_max,
        "bid_gap_mean_pts": bid_gap_mean,
    }
