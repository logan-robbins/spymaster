from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numba import jit

EPSILON = 1e-9
POINT = 1.0
WALL_Z_THRESHOLD = 2.0
N_BANDS = 5
N_LEVELS = 10

BANDS = ["p0_1", "p1_2", "p2_3", "p3_5", "p5_10"]


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
def compute_lvl_banded_quantities(
    bid_px: NDArray[np.float64],
    ask_px: NDArray[np.float64],
    bid_sz: NDArray[np.float64],
    ask_sz: NDArray[np.float64],
    level_price: float,
) -> tuple:
    below_qty = np.zeros(N_BANDS, dtype=np.float64)
    above_qty = np.zeros(N_BANDS, dtype=np.float64)

    for i in range(N_LEVELS):
        if bid_px[i] > EPSILON:
            d_bid = (level_price - bid_px[i]) / POINT
            if d_bid > 0:
                band_idx = assign_band_idx(d_bid)
                if band_idx >= 0:
                    below_qty[band_idx] += bid_sz[i]
            elif d_bid <= 0:
                d_above = -d_bid
                band_idx = assign_band_idx(d_above)
                if band_idx >= 0:
                    above_qty[band_idx] += bid_sz[i]

        if ask_px[i] > EPSILON:
            d_ask = (ask_px[i] - level_price) / POINT
            if d_ask > 0:
                band_idx = assign_band_idx(d_ask)
                if band_idx >= 0:
                    above_qty[band_idx] += ask_sz[i]
            elif d_ask <= 0:
                d_below = -d_ask
                band_idx = assign_band_idx(d_below)
                if band_idx >= 0:
                    below_qty[band_idx] += ask_sz[i]

    return below_qty, above_qty


@jit(nopython=True, cache=True)
def compute_lvl_cdi(
    below_qty: NDArray[np.float64], above_qty: NDArray[np.float64]
) -> NDArray[np.float64]:
    cdi = np.zeros(N_BANDS, dtype=np.float64)
    for i in range(N_BANDS):
        denom = below_qty[i] + above_qty[i] + EPSILON
        cdi[i] = (below_qty[i] - above_qty[i]) / denom
    return cdi


@jit(nopython=True, cache=True)
def compute_lvl_banded_fractions(
    below_qty: NDArray[np.float64],
    above_qty: NDArray[np.float64],
    total_depth: float,
) -> tuple:
    below_frac = np.zeros(N_BANDS, dtype=np.float64)
    above_frac = np.zeros(N_BANDS, dtype=np.float64)

    denom = total_depth + EPSILON

    for i in range(N_BANDS):
        below_frac[i] = below_qty[i] / denom
        above_frac[i] = above_qty[i] / denom

    return below_frac, above_frac


@jit(nopython=True, cache=True)
def compute_lvl_wall_features(
    sizes: NDArray[np.float64],
    prices: NDArray[np.float64],
    level_price: float,
    is_bid: bool
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
                nearest_strong_dist_pts = (level_price - px) / POINT
            else:
                nearest_strong_dist_pts = (px - level_price) / POINT
            break

    return max_z, float(max_z_idx), nearest_strong_dist_pts, float(nearest_strong_idx)


@jit(nopython=True, cache=True)
def compute_lvl_flow_band_idx(event_price: float, event_side: int, level_price: float) -> int:
    if event_side == 0:
        d = (level_price - event_price) / POINT
    else:
        d = (event_price - level_price) / POINT

    if d < 0:
        d = -d

    return assign_band_idx(d)


@jit(nopython=True, cache=True)
def compute_lvl_flow_direction(event_price: float, event_side: int, level_price: float) -> int:
    if event_side == 0:
        if event_price < level_price:
            return 0
        else:
            return 1
    else:
        if event_price > level_price:
            return 1
        else:
            return 0


def compute_level_relative_depth_features(
    bid_px: NDArray[np.float64],
    ask_px: NDArray[np.float64],
    bid_sz: NDArray[np.float64],
    ask_sz: NDArray[np.float64],
    level_price: float,
) -> dict[str, float]:
    below_qty, above_qty = compute_lvl_banded_quantities(
        bid_px, ask_px, bid_sz, ask_sz, level_price
    )

    cdi = compute_lvl_cdi(below_qty, above_qty)

    total_depth = bid_sz.sum() + ask_sz.sum()
    below_frac, above_frac = compute_lvl_banded_fractions(below_qty, above_qty, total_depth)

    features = {}

    for band_idx, band in enumerate(BANDS):
        features[f"bar5s_lvldepth_below_{band}_qty_eob"] = float(below_qty[band_idx])
        features[f"bar5s_lvldepth_above_{band}_qty_eob"] = float(above_qty[band_idx])
        features[f"bar5s_lvldepth_below_{band}_frac_eob"] = float(below_frac[band_idx])
        features[f"bar5s_lvldepth_above_{band}_frac_eob"] = float(above_frac[band_idx])
        features[f"bar5s_lvldepth_cdi_{band}_eob"] = float(cdi[band_idx])

    features["bar5s_lvldepth_below_total_qty_eob"] = float(below_qty.sum())
    features["bar5s_lvldepth_above_total_qty_eob"] = float(above_qty.sum())

    total_below = below_qty.sum()
    total_above = above_qty.sum()
    features["bar5s_lvldepth_imbal_eob"] = (total_below - total_above) / (total_below + total_above + EPSILON)

    return features


def compute_level_relative_wall_features(
    bid_px: NDArray[np.float64],
    ask_px: NDArray[np.float64],
    bid_sz: NDArray[np.float64],
    ask_sz: NDArray[np.float64],
    level_price: float,
) -> dict[str, float]:
    wall_bid = compute_lvl_wall_features(bid_sz, bid_px, level_price, True)
    wall_ask = compute_lvl_wall_features(ask_sz, ask_px, level_price, False)

    return {
        "bar5s_lvlwall_bid_maxz_eob": wall_bid[0],
        "bar5s_lvlwall_ask_maxz_eob": wall_ask[0],
        "bar5s_lvlwall_bid_maxz_levelidx_eob": wall_bid[1],
        "bar5s_lvlwall_ask_maxz_levelidx_eob": wall_ask[1],
        "bar5s_lvlwall_bid_nearest_strong_dist_pts_eob": wall_bid[2],
        "bar5s_lvlwall_ask_nearest_strong_dist_pts_eob": wall_ask[2],
        "bar5s_lvlwall_bid_nearest_strong_levelidx_eob": wall_bid[3],
        "bar5s_lvlwall_ask_nearest_strong_levelidx_eob": wall_ask[3],
    }


ACTION_MODIFY = 0
ACTION_CLEAR = 1
ACTION_ADD = 2
ACTION_CANCEL = 3
ACTION_TRADE = 4

SIDE_BID = 0
SIDE_ASK = 1


@jit(nopython=True, cache=True)
def compute_lvl_delta_q(
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
def process_lvl_flow_ticks(
    ts_event: NDArray[np.int64],
    action: NDArray[np.int32],
    side: NDArray[np.int32],
    price: NDArray[np.float64],
    size: NDArray[np.float64],
    bid_px: NDArray[np.float64],
    ask_px: NDArray[np.float64],
    bid_sz: NDArray[np.float64],
    ask_sz: NDArray[np.float64],
    bar_starts: NDArray[np.int64],
    unique_bar_starts: NDArray[np.int64],
    level_price: float,
) -> tuple:
    n_ticks = len(ts_event)
    n_bars = len(unique_bar_starts)

    bar_flow_add_vol = np.zeros((n_bars, 2, N_BANDS), dtype=np.float64)
    bar_flow_rem_vol = np.zeros((n_bars, 2, N_BANDS), dtype=np.float64)
    bar_flow_net_vol = np.zeros((n_bars, 2, N_BANDS), dtype=np.float64)
    bar_flow_cnt_add = np.zeros((n_bars, 2, N_BANDS), dtype=np.float64)
    bar_flow_cnt_cancel = np.zeros((n_bars, 2, N_BANDS), dtype=np.float64)
    bar_flow_cnt_modify = np.zeros((n_bars, 2, N_BANDS), dtype=np.float64)

    bar_twa_below_qty = np.zeros((n_bars, N_BANDS), dtype=np.float64)
    bar_twa_above_qty = np.zeros((n_bars, N_BANDS), dtype=np.float64)

    bar_map = np.zeros(n_ticks, dtype=np.int64)
    for i in range(n_bars):
        bar_map[bar_starts == unique_bar_starts[i]] = i

    pre_bid_px = bid_px[0].copy()
    pre_ask_px = ask_px[0].copy()
    pre_bid_sz = bid_sz[0].copy()
    pre_ask_sz = ask_sz[0].copy()

    BAR_DURATION_NS = 5_000_000_000
    current_bar_idx = 0
    t_last = unique_bar_starts[0]

    for tick_idx in range(n_ticks):
        bar_idx = bar_map[tick_idx]

        if bar_idx != current_bar_idx:
            for b in range(N_BANDS):
                bar_twa_below_qty[current_bar_idx, b] /= BAR_DURATION_NS
                bar_twa_above_qty[current_bar_idx, b] /= BAR_DURATION_NS

            current_bar_idx = bar_idx
            t_last = unique_bar_starts[bar_idx]

        ts = ts_event[tick_idx]
        dt = ts - t_last

        if dt > 0:
            below_qty, above_qty = compute_lvl_banded_quantities(
                pre_bid_px, pre_ask_px, pre_bid_sz, pre_ask_sz, level_price
            )
            for b in range(N_BANDS):
                bar_twa_below_qty[bar_idx, b] += below_qty[b] * dt
                bar_twa_above_qty[bar_idx, b] += above_qty[b] * dt
            t_last = ts

        act = action[tick_idx]
        sd = side[tick_idx]
        px = price[tick_idx]

        post_bid_px = bid_px[tick_idx]
        post_ask_px = ask_px[tick_idx]
        post_bid_sz = bid_sz[tick_idx]
        post_ask_sz = ask_sz[tick_idx]

        if act == ACTION_ADD:
            if px > EPSILON:
                add_vol, rem_vol = compute_lvl_delta_q(
                    px, sd, pre_bid_px, pre_ask_px, pre_bid_sz, pre_ask_sz,
                    post_bid_px, post_ask_px, post_bid_sz, post_ask_sz
                )
                band_idx = compute_lvl_flow_band_idx(px, sd, level_price)
                if band_idx >= 0:
                    bar_flow_add_vol[bar_idx, sd, band_idx] += add_vol
                    bar_flow_rem_vol[bar_idx, sd, band_idx] += rem_vol
                    bar_flow_net_vol[bar_idx, sd, band_idx] += (add_vol - rem_vol)
                    bar_flow_cnt_add[bar_idx, sd, band_idx] += 1
        elif act == ACTION_CANCEL:
            if px > EPSILON:
                add_vol, rem_vol = compute_lvl_delta_q(
                    px, sd, pre_bid_px, pre_ask_px, pre_bid_sz, pre_ask_sz,
                    post_bid_px, post_ask_px, post_bid_sz, post_ask_sz
                )
                band_idx = compute_lvl_flow_band_idx(px, sd, level_price)
                if band_idx >= 0:
                    bar_flow_add_vol[bar_idx, sd, band_idx] += add_vol
                    bar_flow_rem_vol[bar_idx, sd, band_idx] += rem_vol
                    bar_flow_net_vol[bar_idx, sd, band_idx] += (add_vol - rem_vol)
                    bar_flow_cnt_cancel[bar_idx, sd, band_idx] += 1
        elif act == ACTION_MODIFY:
            if px > EPSILON:
                add_vol, rem_vol = compute_lvl_delta_q(
                    px, sd, pre_bid_px, pre_ask_px, pre_bid_sz, pre_ask_sz,
                    post_bid_px, post_ask_px, post_bid_sz, post_ask_sz
                )
                band_idx = compute_lvl_flow_band_idx(px, sd, level_price)
                if band_idx >= 0:
                    bar_flow_add_vol[bar_idx, sd, band_idx] += add_vol
                    bar_flow_rem_vol[bar_idx, sd, band_idx] += rem_vol
                    bar_flow_net_vol[bar_idx, sd, band_idx] += (add_vol - rem_vol)
                    bar_flow_cnt_modify[bar_idx, sd, band_idx] += 1

        pre_bid_px[:] = post_bid_px
        pre_ask_px[:] = post_ask_px
        pre_bid_sz[:] = post_bid_sz
        pre_ask_sz[:] = post_ask_sz

    for b in range(N_BANDS):
        bar_twa_below_qty[current_bar_idx, b] /= BAR_DURATION_NS
        bar_twa_above_qty[current_bar_idx, b] /= BAR_DURATION_NS

    return (
        bar_flow_add_vol, bar_flow_rem_vol, bar_flow_net_vol,
        bar_flow_cnt_add, bar_flow_cnt_cancel, bar_flow_cnt_modify,
        bar_twa_below_qty, bar_twa_above_qty,
    )


def extract_level_relative_flow_features(
    bar_idx: int,
    bar_flow_add_vol: NDArray[np.float64],
    bar_flow_rem_vol: NDArray[np.float64],
    bar_flow_net_vol: NDArray[np.float64],
    bar_flow_cnt_add: NDArray[np.float64],
    bar_flow_cnt_cancel: NDArray[np.float64],
    bar_flow_cnt_modify: NDArray[np.float64],
    bar_twa_below_qty: NDArray[np.float64],
    bar_twa_above_qty: NDArray[np.float64],
) -> dict[str, float]:
    features = {}

    sides = ["bid", "ask"]

    for side_idx, side_str in enumerate(sides):
        for band_idx, band in enumerate(BANDS):
            features[f"bar5s_lvlflow_add_vol_{side_str}_{band}_sum"] = bar_flow_add_vol[bar_idx, side_idx, band_idx]
            features[f"bar5s_lvlflow_rem_vol_{side_str}_{band}_sum"] = bar_flow_rem_vol[bar_idx, side_idx, band_idx]
            features[f"bar5s_lvlflow_net_vol_{side_str}_{band}_sum"] = bar_flow_net_vol[bar_idx, side_idx, band_idx]
            features[f"bar5s_lvlflow_cnt_add_{side_str}_{band}_sum"] = bar_flow_cnt_add[bar_idx, side_idx, band_idx]
            features[f"bar5s_lvlflow_cnt_cancel_{side_str}_{band}_sum"] = bar_flow_cnt_cancel[bar_idx, side_idx, band_idx]
            features[f"bar5s_lvlflow_cnt_modify_{side_str}_{band}_sum"] = bar_flow_cnt_modify[bar_idx, side_idx, band_idx]

            net_vol = bar_flow_net_vol[bar_idx, side_idx, band_idx]
            if side_str == "bid":
                twa_qty = bar_twa_below_qty[bar_idx, band_idx]
            else:
                twa_qty = bar_twa_above_qty[bar_idx, band_idx]
            norm = net_vol / max(twa_qty, 1.0)
            features[f"bar5s_lvlflow_net_volnorm_{side_str}_{band}_sum"] = norm

    return features


def compute_all_level_relative_features(
    bid_px: NDArray[np.float64],
    ask_px: NDArray[np.float64],
    bid_sz: NDArray[np.float64],
    ask_sz: NDArray[np.float64],
    level_price: float,
) -> dict[str, float]:
    features = {}

    depth_features = compute_level_relative_depth_features(
        bid_px, ask_px, bid_sz, ask_sz, level_price
    )
    features.update(depth_features)

    wall_features = compute_level_relative_wall_features(
        bid_px, ask_px, bid_sz, ask_sz, level_price
    )
    features.update(wall_features)

    return features


LEVEL_RELATIVE_DEPTH_FEATURES = [
    f"bar5s_lvldepth_{direction}_{band}_{metric}_eob"
    for band in BANDS
    for direction in ["below", "above"]
    for metric in ["qty", "frac"]
] + [
    f"bar5s_lvldepth_cdi_{band}_eob"
    for band in BANDS
] + [
    "bar5s_lvldepth_below_total_qty_eob",
    "bar5s_lvldepth_above_total_qty_eob",
    "bar5s_lvldepth_imbal_eob",
]

LEVEL_RELATIVE_WALL_FEATURES = [
    "bar5s_lvlwall_bid_maxz_eob",
    "bar5s_lvlwall_ask_maxz_eob",
    "bar5s_lvlwall_bid_maxz_levelidx_eob",
    "bar5s_lvlwall_ask_maxz_levelidx_eob",
    "bar5s_lvlwall_bid_nearest_strong_dist_pts_eob",
    "bar5s_lvlwall_ask_nearest_strong_dist_pts_eob",
    "bar5s_lvlwall_bid_nearest_strong_levelidx_eob",
    "bar5s_lvlwall_ask_nearest_strong_levelidx_eob",
]

LEVEL_RELATIVE_FLOW_FEATURES = [
    f"bar5s_lvlflow_{metric}_{side}_{band}_sum"
    for band in BANDS
    for side in ["bid", "ask"]
    for metric in ["add_vol", "rem_vol", "net_vol", "cnt_add", "cnt_cancel", "cnt_modify", "net_volnorm"]
]

ALL_LEVEL_RELATIVE_FEATURES = (
    LEVEL_RELATIVE_DEPTH_FEATURES +
    LEVEL_RELATIVE_WALL_FEATURES +
    LEVEL_RELATIVE_FLOW_FEATURES
)
