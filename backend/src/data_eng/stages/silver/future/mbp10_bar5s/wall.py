from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import EPSILON, POINT, WALL_Z_THRESHOLD


def compute_wall_features(sizes: NDArray[np.float64], prices: NDArray[np.float64], p_ref: float, is_bid: bool) -> dict[str, float]:
    q = np.log1p(sizes)
    
    mu = q.mean()
    sigma = q.std(ddof=0)
    
    z_scores = (q - mu) / max(sigma, EPSILON)
    
    max_z = float(z_scores.max())
    max_z_idx = int(z_scores.argmax())
    
    strong_mask = z_scores >= WALL_Z_THRESHOLD
    
    if strong_mask.any():
        nearest_strong_idx = int(np.argmax(strong_mask))
        px = prices[nearest_strong_idx]
        if is_bid:
            nearest_strong_dist_pts = (p_ref - px) / POINT
        else:
            nearest_strong_dist_pts = (px - p_ref) / POINT
    else:
        nearest_strong_idx = -1
        nearest_strong_dist_pts = float("nan")
    
    return {
        "maxz": max_z,
        "maxz_levelidx": float(max_z_idx),
        "nearest_strong_dist_pts": nearest_strong_dist_pts,
        "nearest_strong_levelidx": float(nearest_strong_idx),
    }
