from __future__ import annotations

import numpy as np

from .book_state import BookState
from .constants import EPSILON, POINT


def assign_band(distance_pts: float) -> str | None:
    if distance_pts <= 0:
        return "p0_1"
    elif distance_pts <= 1:
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


def compute_banded_quantities(book: BookState, p_ref: float) -> tuple[dict[str, float], dict[str, float]]:
    below_qty = {"p0_1": 0.0, "p1_2": 0.0, "p2_3": 0.0, "p3_5": 0.0, "p5_10": 0.0}
    above_qty = {"p0_1": 0.0, "p1_2": 0.0, "p2_3": 0.0, "p3_5": 0.0, "p5_10": 0.0}

    for i in range(10):
        bid_px = book.bid_px[i]
        ask_px = book.ask_px[i]
        bid_sz = book.bid_sz[i]
        ask_sz = book.ask_sz[i]

        if bid_px > EPSILON:
            d_bid = (p_ref - bid_px) / POINT
            band_bid = assign_band(d_bid)
            if band_bid is not None:
                below_qty[band_bid] += bid_sz

        if ask_px > EPSILON:
            d_ask = (ask_px - p_ref) / POINT
            band_ask = assign_band(d_ask)
            if band_ask is not None:
                above_qty[band_ask] += ask_sz

    return below_qty, above_qty


def compute_cdi(below_qty: dict[str, float], above_qty: dict[str, float]) -> dict[str, float]:
    cdi = {}
    for band in ["p0_1", "p1_2", "p2_3", "p3_5", "p5_10"]:
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
    
    for band in ["p0_1", "p1_2", "p2_3", "p3_5", "p5_10"]:
        below_frac[band] = below_qty[band] / bid_denom
        above_frac[band] = above_qty[band] / ask_denom
    
    return below_frac, above_frac

