from __future__ import annotations

import numpy as np
from numba import jit, prange
from numpy.typing import NDArray

POINT = 1.0
BAR_DURATION_NS = 5_000_000_000
EPSILON = 1e-9
WALL_Z_THRESHOLD = 2.0

N_BANDS = 5
N_LEVELS = 10

ACTION_MODIFY = 0
ACTION_CLEAR = 1
ACTION_ADD = 2
ACTION_CANCEL = 3
ACTION_TRADE = 4

SIDE_BID = 0
SIDE_ASK = 1


@jit(nopython=True, cache=True)
def assign_band_idx(distance_pts: float) -> int:
    if distance_pts <= 1:
        return 0
    elif distance_pts <= 2:
        return 1
    elif distance_pts <= 3:
        return 2
    elif distance_pts <= 5:
        return 3
    elif distance_pts <= 10:
        return 4
    else:
        return -1


@jit(nopython=True, cache=True)
def compute_microprice(
    bid_px_0: float, ask_px_0: float, bid_sz_0: float, ask_sz_0: float
) -> float:
    total_sz = bid_sz_0 + ask_sz_0
    if total_sz < EPSILON:
        return (ask_px_0 + bid_px_0) / 2.0
    return (ask_px_0 * bid_sz_0 + bid_px_0 * ask_sz_0) / total_sz


@jit(nopython=True, cache=True)
def compute_spread_pts(bid_px_0: float, ask_px_0: float) -> float:
    return (ask_px_0 - bid_px_0) / POINT


@jit(nopython=True, cache=True)
def compute_obi0(bid_sz_0: float, ask_sz_0: float) -> float:
    denom = bid_sz_0 + ask_sz_0 + EPSILON
    return (bid_sz_0 - ask_sz_0) / denom


@jit(nopython=True, cache=True)
def compute_obi10(bid_sz: NDArray[np.float64], ask_sz: NDArray[np.float64]) -> float:
    bid_depth = bid_sz.sum()
    ask_depth = ask_sz.sum()
    denom = bid_depth + ask_depth + EPSILON
    return (bid_depth - ask_depth) / denom


@jit(nopython=True, cache=True)
def compute_total_depth(
    bid_sz: NDArray[np.float64], ask_sz: NDArray[np.float64]
) -> tuple:
    return bid_sz.sum(), ask_sz.sum()


@jit(nopython=True, cache=True)
def compute_banded_quantities(
    bid_px: NDArray[np.float64],
    ask_px: NDArray[np.float64],
    bid_sz: NDArray[np.float64],
    ask_sz: NDArray[np.float64],
    p_ref: float,
) -> tuple:
    below_qty = np.zeros(N_BANDS, dtype=np.float64)
    above_qty = np.zeros(N_BANDS, dtype=np.float64)

    for i in range(N_LEVELS):
        if bid_px[i] > EPSILON:
            d_bid = (p_ref - bid_px[i]) / POINT
            band_idx = assign_band_idx(d_bid)
            if band_idx >= 0:
                below_qty[band_idx] += bid_sz[i]

        if ask_px[i] > EPSILON:
            d_ask = (ask_px[i] - p_ref) / POINT
            band_idx = assign_band_idx(d_ask)
            if band_idx >= 0:
                above_qty[band_idx] += ask_sz[i]

    return below_qty, above_qty


@jit(nopython=True, cache=True)
def compute_cdi(
    below_qty: NDArray[np.float64], above_qty: NDArray[np.float64]
) -> NDArray[np.float64]:
    cdi = np.zeros(N_BANDS, dtype=np.float64)
    for i in range(N_BANDS):
        denom = below_qty[i] + above_qty[i] + EPSILON
        cdi[i] = (below_qty[i] - above_qty[i]) / denom
    return cdi


@jit(nopython=True, cache=True)
def compute_banded_fractions(
    below_qty: NDArray[np.float64],
    above_qty: NDArray[np.float64],
    bid_depth: float,
    ask_depth: float,
) -> tuple:
    below_frac = np.zeros(N_BANDS, dtype=np.float64)
    above_frac = np.zeros(N_BANDS, dtype=np.float64)

    bid_denom = bid_depth + EPSILON
    ask_denom = ask_depth + EPSILON

    for i in range(N_BANDS):
        below_frac[i] = below_qty[i] / bid_denom
        above_frac[i] = above_qty[i] / ask_denom

    return below_frac, above_frac


@jit(nopython=True, cache=True)
def compute_delta_q(
    event_price: float,
    event_side: int,
    pre_bid_px: NDArray[np.float64],
    pre_ask_px: NDArray[np.float64],
    pre_bid_sz: NDArray[np.float64],
    pre_ask_sz: NDArray[np.float64],
    post_bid_px: NDArray[np.float64],
    post_ask_px: NDArray[np.float64],
    post_bid_sz: NDArray[np.float64],
    post_ask_sz: NDArray[np.float64],
) -> tuple:
    q_prev = 0.0
    q_new = 0.0

    if event_side == SIDE_BID:
        for i in range(N_LEVELS):
            if np.abs(pre_bid_px[i] - event_price) < EPSILON:
                q_prev += pre_bid_sz[i]
            if np.abs(post_bid_px[i] - event_price) < EPSILON:
                q_new += post_bid_sz[i]
    else:
        for i in range(N_LEVELS):
            if np.abs(pre_ask_px[i] - event_price) < EPSILON:
                q_prev += pre_ask_sz[i]
            if np.abs(post_ask_px[i] - event_price) < EPSILON:
                q_new += post_ask_sz[i]

    delta_q = q_new - q_prev
    add_vol = max(delta_q, 0.0)
    rem_vol = max(-delta_q, 0.0)

    return add_vol, rem_vol


@jit(nopython=True, cache=True)
def compute_flow_band_idx(event_price: float, event_side: int, p_ref: float) -> int:
    if event_side == SIDE_BID:
        d = (p_ref - event_price) / POINT
    else:
        d = (event_price - p_ref) / POINT
    return assign_band_idx(d)


@jit(nopython=True, cache=True)
def compute_ladder_features(
    bid_px: NDArray[np.float64], ask_px: NDArray[np.float64]
) -> tuple:
    ask_gap_max = np.nan
    ask_gap_mean = np.nan
    bid_gap_max = np.nan
    bid_gap_mean = np.nan

    ask_gaps_sum = 0.0
    ask_gaps_cnt = 0
    for i in range(N_LEVELS - 1):
        if ask_px[i] > EPSILON and ask_px[i + 1] > EPSILON:
            gap = (ask_px[i + 1] - ask_px[i]) / POINT
            ask_gaps_sum += gap
            ask_gaps_cnt += 1
            if np.isnan(ask_gap_max) or gap > ask_gap_max:
                ask_gap_max = gap

    if ask_gaps_cnt > 0:
        ask_gap_mean = ask_gaps_sum / ask_gaps_cnt

    bid_gaps_sum = 0.0
    bid_gaps_cnt = 0
    for i in range(N_LEVELS - 1):
        if bid_px[i] > EPSILON and bid_px[i + 1] > EPSILON:
            gap = (bid_px[i] - bid_px[i + 1]) / POINT
            bid_gaps_sum += gap
            bid_gaps_cnt += 1
            if np.isnan(bid_gap_max) or gap > bid_gap_max:
                bid_gap_max = gap

    if bid_gaps_cnt > 0:
        bid_gap_mean = bid_gaps_sum / bid_gaps_cnt

    return ask_gap_max, ask_gap_mean, bid_gap_max, bid_gap_mean


@jit(nopython=True, cache=True)
def compute_shape_fractions(sizes: NDArray[np.float64]) -> NDArray[np.float64]:
    total = sizes.sum() + EPSILON
    return sizes / total


@jit(nopython=True, cache=True)
def compute_wall_features(
    sizes: NDArray[np.float64], prices: NDArray[np.float64], p_ref: float, is_bid: bool
) -> tuple:
    q = np.zeros(N_LEVELS, dtype=np.float64)
    for i in range(N_LEVELS):
        q[i] = np.log1p(sizes[i])

    mu = q.mean()
    sigma = q.std()

    z_scores = (q - mu) / max(sigma, EPSILON)

    max_z = z_scores[0]
    max_z_idx = 0
    for i in range(1, N_LEVELS):
        if z_scores[i] > max_z:
            max_z = z_scores[i]
            max_z_idx = i

    nearest_strong_idx = -1
    nearest_strong_dist_pts = np.nan
    for i in range(N_LEVELS):
        if z_scores[i] >= WALL_Z_THRESHOLD:
            nearest_strong_idx = i
            px = prices[i]
            if is_bid:
                nearest_strong_dist_pts = (p_ref - px) / POINT
            else:
                nearest_strong_dist_pts = (px - p_ref) / POINT
            break

    return max_z, float(max_z_idx), nearest_strong_dist_pts, float(nearest_strong_idx)


@jit(nopython=True, cache=True, parallel=False)
def process_all_ticks(
    ts_event: NDArray[np.int64],
    action: NDArray[np.int32],
    side: NDArray[np.int32],
    price: NDArray[np.float64],
    size: NDArray[np.float64],
    bid_px: NDArray[np.float64],
    ask_px: NDArray[np.float64],
    bid_sz: NDArray[np.float64],
    ask_sz: NDArray[np.float64],
    bid_ct: NDArray[np.float64],
    ask_ct: NDArray[np.float64],
    bar_starts: NDArray[np.int64],
    unique_bar_starts: NDArray[np.int64],
) -> tuple:
    n_ticks = len(ts_event)
    n_bars = len(unique_bar_starts)

    bar_meta_msg_cnt = np.zeros(n_bars, dtype=np.float64)
    bar_meta_clear_cnt = np.zeros(n_bars, dtype=np.float64)
    bar_meta_add_cnt = np.zeros(n_bars, dtype=np.float64)
    bar_meta_cancel_cnt = np.zeros(n_bars, dtype=np.float64)
    bar_meta_modify_cnt = np.zeros(n_bars, dtype=np.float64)
    bar_meta_trade_cnt = np.zeros(n_bars, dtype=np.float64)

    bar_trade_cnt = np.zeros(n_bars, dtype=np.float64)
    bar_trade_vol = np.zeros(n_bars, dtype=np.float64)
    bar_trade_aggbuy_vol = np.zeros(n_bars, dtype=np.float64)
    bar_trade_aggsell_vol = np.zeros(n_bars, dtype=np.float64)

    bar_flow_add_vol = np.zeros((n_bars, 2, N_BANDS), dtype=np.float64)
    bar_flow_rem_vol = np.zeros((n_bars, 2, N_BANDS), dtype=np.float64)
    bar_flow_net_vol = np.zeros((n_bars, 2, N_BANDS), dtype=np.float64)
    bar_flow_cnt_add = np.zeros((n_bars, 2, N_BANDS), dtype=np.float64)
    bar_flow_cnt_cancel = np.zeros((n_bars, 2, N_BANDS), dtype=np.float64)
    bar_flow_cnt_modify = np.zeros((n_bars, 2, N_BANDS), dtype=np.float64)

    bar_twa_spread_pts = np.zeros(n_bars, dtype=np.float64)
    bar_twa_obi0 = np.zeros(n_bars, dtype=np.float64)
    bar_twa_obi10 = np.zeros(n_bars, dtype=np.float64)
    bar_twa_cdi = np.zeros((n_bars, N_BANDS), dtype=np.float64)
    bar_twa_bid10_qty = np.zeros(n_bars, dtype=np.float64)
    bar_twa_ask10_qty = np.zeros(n_bars, dtype=np.float64)
    bar_twa_below_qty = np.zeros((n_bars, N_BANDS), dtype=np.float64)
    bar_twa_above_qty = np.zeros((n_bars, N_BANDS), dtype=np.float64)
    bar_twa_below_frac = np.zeros((n_bars, N_BANDS), dtype=np.float64)
    bar_twa_above_frac = np.zeros((n_bars, N_BANDS), dtype=np.float64)

    bar_eob_idx = np.zeros(n_bars, dtype=np.int64)

    bar_map = np.zeros(n_ticks, dtype=np.int64)
    for i in range(n_bars):
        bar_map[bar_starts == unique_bar_starts[i]] = i

    current_bar_idx = 0
    t_last = unique_bar_starts[0]

    pre_bid_px = bid_px[0].copy()
    pre_ask_px = ask_px[0].copy()
    pre_bid_sz = bid_sz[0].copy()
    pre_ask_sz = ask_sz[0].copy()

    for tick_idx in range(n_ticks):
        bar_idx = bar_map[tick_idx]

        if bar_idx != current_bar_idx:
            bar_eob_idx[current_bar_idx] = tick_idx - 1

            bar_twa_spread_pts[current_bar_idx] /= BAR_DURATION_NS
            bar_twa_obi0[current_bar_idx] /= BAR_DURATION_NS
            bar_twa_obi10[current_bar_idx] /= BAR_DURATION_NS
            bar_twa_bid10_qty[current_bar_idx] /= BAR_DURATION_NS
            bar_twa_ask10_qty[current_bar_idx] /= BAR_DURATION_NS
            for b in range(N_BANDS):
                bar_twa_cdi[current_bar_idx, b] /= BAR_DURATION_NS
                bar_twa_below_qty[current_bar_idx, b] /= BAR_DURATION_NS
                bar_twa_above_qty[current_bar_idx, b] /= BAR_DURATION_NS
                bar_twa_below_frac[current_bar_idx, b] /= BAR_DURATION_NS
                bar_twa_above_frac[current_bar_idx, b] /= BAR_DURATION_NS

            current_bar_idx = bar_idx
            t_last = unique_bar_starts[bar_idx]

        ts = ts_event[tick_idx]
        dt = ts - t_last

        if dt > 0:
            p_ref = compute_microprice(pre_bid_px[0], pre_ask_px[0], pre_bid_sz[0], pre_ask_sz[0])
            spread_pts = compute_spread_pts(pre_bid_px[0], pre_ask_px[0])
            obi0 = compute_obi0(pre_bid_sz[0], pre_ask_sz[0])
            obi10 = compute_obi10(pre_bid_sz, pre_ask_sz)
            bid10_qty, ask10_qty = compute_total_depth(pre_bid_sz, pre_ask_sz)
            below_qty, above_qty = compute_banded_quantities(pre_bid_px, pre_ask_px, pre_bid_sz, pre_ask_sz, p_ref)
            cdi = compute_cdi(below_qty, above_qty)
            below_frac, above_frac = compute_banded_fractions(below_qty, above_qty, bid10_qty, ask10_qty)

            bar_twa_spread_pts[bar_idx] += spread_pts * dt
            bar_twa_obi0[bar_idx] += obi0 * dt
            bar_twa_obi10[bar_idx] += obi10 * dt
            bar_twa_bid10_qty[bar_idx] += bid10_qty * dt
            bar_twa_ask10_qty[bar_idx] += ask10_qty * dt
            for b in range(N_BANDS):
                bar_twa_cdi[bar_idx, b] += cdi[b] * dt
                bar_twa_below_qty[bar_idx, b] += below_qty[b] * dt
                bar_twa_above_qty[bar_idx, b] += above_qty[b] * dt
                bar_twa_below_frac[bar_idx, b] += below_frac[b] * dt
                bar_twa_above_frac[bar_idx, b] += above_frac[b] * dt

            t_last = ts

        bar_meta_msg_cnt[bar_idx] += 1

        act = action[tick_idx]
        sd = side[tick_idx]
        px = price[tick_idx]
        sz = size[tick_idx]

        post_bid_px = bid_px[tick_idx]
        post_ask_px = ask_px[tick_idx]
        post_bid_sz = bid_sz[tick_idx]
        post_ask_sz = ask_sz[tick_idx]

        if act == ACTION_CLEAR:
            bar_meta_clear_cnt[bar_idx] += 1
        elif act == ACTION_ADD:
            bar_meta_add_cnt[bar_idx] += 1
            if px > EPSILON:
                add_vol, rem_vol = compute_delta_q(
                    px, sd, pre_bid_px, pre_ask_px, pre_bid_sz, pre_ask_sz,
                    post_bid_px, post_ask_px, post_bid_sz, post_ask_sz
                )
                p_ref = compute_microprice(pre_bid_px[0], pre_ask_px[0], pre_bid_sz[0], pre_ask_sz[0])
                band_idx = compute_flow_band_idx(px, sd, p_ref)
                if band_idx >= 0:
                    bar_flow_add_vol[bar_idx, sd, band_idx] += add_vol
                    bar_flow_rem_vol[bar_idx, sd, band_idx] += rem_vol
                    bar_flow_net_vol[bar_idx, sd, band_idx] += (add_vol - rem_vol)
                    bar_flow_cnt_add[bar_idx, sd, band_idx] += 1
        elif act == ACTION_CANCEL:
            bar_meta_cancel_cnt[bar_idx] += 1
            if px > EPSILON:
                add_vol, rem_vol = compute_delta_q(
                    px, sd, pre_bid_px, pre_ask_px, pre_bid_sz, pre_ask_sz,
                    post_bid_px, post_ask_px, post_bid_sz, post_ask_sz
                )
                p_ref = compute_microprice(pre_bid_px[0], pre_ask_px[0], pre_bid_sz[0], pre_ask_sz[0])
                band_idx = compute_flow_band_idx(px, sd, p_ref)
                if band_idx >= 0:
                    bar_flow_add_vol[bar_idx, sd, band_idx] += add_vol
                    bar_flow_rem_vol[bar_idx, sd, band_idx] += rem_vol
                    bar_flow_net_vol[bar_idx, sd, band_idx] += (add_vol - rem_vol)
                    bar_flow_cnt_cancel[bar_idx, sd, band_idx] += 1
        elif act == ACTION_MODIFY:
            bar_meta_modify_cnt[bar_idx] += 1
            if px > EPSILON:
                add_vol, rem_vol = compute_delta_q(
                    px, sd, pre_bid_px, pre_ask_px, pre_bid_sz, pre_ask_sz,
                    post_bid_px, post_ask_px, post_bid_sz, post_ask_sz
                )
                p_ref = compute_microprice(pre_bid_px[0], pre_ask_px[0], pre_bid_sz[0], pre_ask_sz[0])
                band_idx = compute_flow_band_idx(px, sd, p_ref)
                if band_idx >= 0:
                    bar_flow_add_vol[bar_idx, sd, band_idx] += add_vol
                    bar_flow_rem_vol[bar_idx, sd, band_idx] += rem_vol
                    bar_flow_net_vol[bar_idx, sd, band_idx] += (add_vol - rem_vol)
                    bar_flow_cnt_modify[bar_idx, sd, band_idx] += 1
        elif act == ACTION_TRADE:
            bar_meta_trade_cnt[bar_idx] += 1
            bar_trade_cnt[bar_idx] += 1
            bar_trade_vol[bar_idx] += sz
            if sd == SIDE_ASK:
                bar_trade_aggbuy_vol[bar_idx] += sz
            elif sd == SIDE_BID:
                bar_trade_aggsell_vol[bar_idx] += sz

        pre_bid_px[:] = post_bid_px
        pre_ask_px[:] = post_ask_px
        pre_bid_sz[:] = post_bid_sz
        pre_ask_sz[:] = post_ask_sz

    bar_eob_idx[current_bar_idx] = n_ticks - 1
    bar_twa_spread_pts[current_bar_idx] /= BAR_DURATION_NS
    bar_twa_obi0[current_bar_idx] /= BAR_DURATION_NS
    bar_twa_obi10[current_bar_idx] /= BAR_DURATION_NS
    bar_twa_bid10_qty[current_bar_idx] /= BAR_DURATION_NS
    bar_twa_ask10_qty[current_bar_idx] /= BAR_DURATION_NS
    for b in range(N_BANDS):
        bar_twa_cdi[current_bar_idx, b] /= BAR_DURATION_NS
        bar_twa_below_qty[current_bar_idx, b] /= BAR_DURATION_NS
        bar_twa_above_qty[current_bar_idx, b] /= BAR_DURATION_NS
        bar_twa_below_frac[current_bar_idx, b] /= BAR_DURATION_NS
        bar_twa_above_frac[current_bar_idx, b] /= BAR_DURATION_NS

    return (
        bar_meta_msg_cnt, bar_meta_clear_cnt, bar_meta_add_cnt, bar_meta_cancel_cnt,
        bar_meta_modify_cnt, bar_meta_trade_cnt,
        bar_trade_cnt, bar_trade_vol, bar_trade_aggbuy_vol, bar_trade_aggsell_vol,
        bar_flow_add_vol, bar_flow_rem_vol, bar_flow_net_vol,
        bar_flow_cnt_add, bar_flow_cnt_cancel, bar_flow_cnt_modify,
        bar_twa_spread_pts, bar_twa_obi0, bar_twa_obi10, bar_twa_cdi,
        bar_twa_bid10_qty, bar_twa_ask10_qty,
        bar_twa_below_qty, bar_twa_above_qty, bar_twa_below_frac, bar_twa_above_frac,
        bar_eob_idx,
    )
