from __future__ import annotations

from .bands import assign_band
from .book_state import BookState
from .constants import EPSILON, POINT, SIDE_ASK, SIDE_BID


def compute_delta_q(event_price: float, event_side: str, pre_state: BookState, post_state: BookState) -> tuple[float, float]:
    q_prev = 0.0
    q_new = 0.0
    
    if event_side == SIDE_BID:
        for i in range(10):
            if abs(pre_state.bid_px[i] - event_price) < EPSILON:
                q_prev = pre_state.bid_sz[i]
                break
        for i in range(10):
            if abs(post_state.bid_px[i] - event_price) < EPSILON:
                q_new = post_state.bid_sz[i]
                break
    else:
        for i in range(10):
            if abs(pre_state.ask_px[i] - event_price) < EPSILON:
                q_prev = pre_state.ask_sz[i]
                break
        for i in range(10):
            if abs(post_state.ask_px[i] - event_price) < EPSILON:
                q_new = post_state.ask_sz[i]
                break
    
    delta_q = q_new - q_prev
    add_vol = max(delta_q, 0.0)
    rem_vol = max(-delta_q, 0.0)
    
    return add_vol, rem_vol


def compute_flow_band(event_price: float, event_side: str, p_ref: float) -> str | None:
    if event_side == SIDE_BID:
        d = (p_ref - event_price) / POINT
    else:
        d = (event_price - p_ref) / POINT
    
    return assign_band(d)

