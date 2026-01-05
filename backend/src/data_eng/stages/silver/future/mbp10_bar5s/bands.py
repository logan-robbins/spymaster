from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .book_state import BookState
from .constants import BANDS, EPSILON, POINT


def assign_band(distance_pts: float) -> str | None:
    if distance_pts <= 1:
        return "p0_1"
    elif distance_pts <= 2:
        return "p1_2"
    elif distance_pts <= 3:
        return "p2_3"
    elif distance_pts <= 5:
        return "p3_5"
    elif distance_pts <= 10:
        return "p5_10"
    else:
        return None


def assign_band_vectorized(distances: NDArray[np.float64]) -> NDArray:
    bands = np.empty(len(distances), dtype=object)
    bands[:] = None
    
    bands[distances <= 1] = "p0_1"
    mask_p1_2 = (distances > 1) & (distances <= 2)
    bands[mask_p1_2] = "p1_2"
    mask_p2_3 = (distances > 2) & (distances <= 3)
    bands[mask_p2_3] = "p2_3"
    mask_p3_5 = (distances > 3) & (distances <= 5)
    bands[mask_p3_5] = "p3_5"
    mask_p5_10 = (distances > 5) & (distances <= 10)
    bands[mask_p5_10] = "p5_10"
    
    return bands


def compute_banded_quantities(book: BookState, p_ref: float) -> tuple[dict[str, float], dict[str, float]]:
    below_qty = {band: 0.0 for band in BANDS}
    above_qty = {band: 0.0 for band in BANDS}

    valid_bid_mask = book.bid_px > EPSILON
    valid_ask_mask = book.ask_px > EPSILON

    if valid_bid_mask.any():
        d_bid = (p_ref - book.bid_px[valid_bid_mask]) / POINT
        bid_sizes = book.bid_sz[valid_bid_mask]
        bid_bands = assign_band_vectorized(d_bid)
        
        for band in BANDS:
            mask = bid_bands == band
            if mask.any():
                below_qty[band] = float(bid_sizes[mask].sum())

    if valid_ask_mask.any():
        d_ask = (book.ask_px[valid_ask_mask] - p_ref) / POINT
        ask_sizes = book.ask_sz[valid_ask_mask]
        ask_bands = assign_band_vectorized(d_ask)
        
        for band in BANDS:
            mask = ask_bands == band
            if mask.any():
                above_qty[band] = float(ask_sizes[mask].sum())

    return below_qty, above_qty


def compute_cdi(below_qty: dict[str, float], above_qty: dict[str, float]) -> dict[str, float]:
    cdi = {}
    for band in BANDS:
        below = below_qty[band]
        above = above_qty[band]
        denom = below + above + EPSILON
        cdi[band] = (below - above) / denom
    return cdi


def compute_banded_fractions(below_qty: dict[str, float], above_qty: dict[str, float], bid_depth: float, ask_depth: float) -> tuple[dict[str, float], dict[str, float]]:
    below_frac = {}
    above_frac = {}
    
    bid_denom = bid_depth + EPSILON
    ask_denom = ask_depth + EPSILON
    
    for band in BANDS:
        below_frac[band] = below_qty[band] / bid_denom
        above_frac[band] = above_qty[band] / ask_denom
    
    return below_frac, above_frac
