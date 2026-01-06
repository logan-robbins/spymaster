from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .constants import (
    ACTION_ADD,
    ACTION_CANCEL,
    ACTION_CLEAR,
    ACTION_MODIFY,
    ACTION_TRADE,
    BANDS,
    BAR_DURATION_NS,
    EPSILON,
    SIDE_ASK,
    SIDE_BID,
)
from .numba_core import (
    process_all_ticks,
    compute_ladder_features,
    compute_shape_fractions,
    compute_wall_features,
    compute_microprice,
    compute_banded_quantities,
    compute_cdi,
    compute_banded_fractions,
    N_BANDS,
)
from .numba_core import (
    ACTION_ADD as NUMBA_ACTION_ADD,
    ACTION_CANCEL as NUMBA_ACTION_CANCEL,
    ACTION_CLEAR as NUMBA_ACTION_CLEAR,
    ACTION_MODIFY as NUMBA_ACTION_MODIFY,
    ACTION_TRADE as NUMBA_ACTION_TRADE,
    SIDE_BID as NUMBA_SIDE_BID,
    SIDE_ASK as NUMBA_SIDE_ASK,
)


ACTION_MAP = {
    ACTION_MODIFY: NUMBA_ACTION_MODIFY,
    ACTION_CLEAR: NUMBA_ACTION_CLEAR,
    ACTION_ADD: NUMBA_ACTION_ADD,
    ACTION_CANCEL: NUMBA_ACTION_CANCEL,
    ACTION_TRADE: NUMBA_ACTION_TRADE,
}

SIDE_MAP = {
    SIDE_BID: NUMBA_SIDE_BID,
    SIDE_ASK: NUMBA_SIDE_ASK,
}


class TickArrays:
    __slots__ = (
        "ts_event", "action", "side", "price", "size", "sequence",
        "bid_px", "ask_px", "bid_sz", "ask_sz", "bid_ct", "ask_ct",
    )

    def __init__(self, df: pd.DataFrame) -> None:
        n = len(df)
        self.ts_event: NDArray[np.int64] = df["ts_event"].values.astype(np.int64)

        action_str = df["action"].values
        self.action: NDArray[np.int32] = np.array([ACTION_MAP.get(a, -1) for a in action_str], dtype=np.int32)

        side_str = df["side"].values
        self.side: NDArray[np.int32] = np.array([SIDE_MAP.get(s, -1) for s in side_str], dtype=np.int32)

        self.price: NDArray[np.float64] = df["price"].values.astype(np.float64)
        self.size: NDArray[np.float64] = np.maximum(df["size"].values.astype(np.float64), 0.0)
        self.sequence: NDArray[np.int64] = df["sequence"].values.astype(np.int64) if "sequence" in df.columns else np.arange(n, dtype=np.int64)

        self.bid_px = np.zeros((n, 10), dtype=np.float64)
        self.ask_px = np.zeros((n, 10), dtype=np.float64)
        self.bid_sz = np.zeros((n, 10), dtype=np.float64)
        self.ask_sz = np.zeros((n, 10), dtype=np.float64)
        self.bid_ct = np.zeros((n, 10), dtype=np.float64)
        self.ask_ct = np.zeros((n, 10), dtype=np.float64)

        for i in range(10):
            idx = f"{i:02d}"
            self.bid_px[:, i] = df[f"bid_px_{idx}"].values.astype(np.float64)
            self.ask_px[:, i] = df[f"ask_px_{idx}"].values.astype(np.float64)
            self.bid_sz[:, i] = np.maximum(df[f"bid_sz_{idx}"].values.astype(np.float64), 0.0)
            self.ask_sz[:, i] = np.maximum(df[f"ask_sz_{idx}"].values.astype(np.float64), 0.0)
            self.bid_ct[:, i] = np.maximum(df[f"bid_ct_{idx}"].values.astype(np.float64), 0.0)
            self.ask_ct[:, i] = np.maximum(df[f"ask_ct_{idx}"].values.astype(np.float64), 0.0)


def fill_empty_bar(bar_ts: int, symbol: str, prev_bar: dict) -> dict:
    empty_bar = {
        "bar_ts": bar_ts,
        "symbol": symbol,
    }

    for key, value in prev_bar.items():
        if key in ("bar_ts", "symbol"):
            continue

        if key.endswith("_sum"):
            empty_bar[key] = 0.0
        elif key.endswith("_eob"):
            empty_bar[key] = value
        elif key.endswith("_twa"):
            eob_key = key.replace("_twa", "_eob")
            empty_bar[key] = prev_bar.get(eob_key, 0.0)
        else:
            empty_bar[key] = value

    return empty_bar


def compute_bar5s_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    if "ts_event" not in df.columns:
        raise ValueError("Missing ts_event column")

    df = df.copy()
    if df["ts_event"].dtype != "int64":
        df["ts_event"] = pd.to_datetime(df["ts_event"]).astype("int64")

    if "sequence" not in df.columns:
        df["sequence"] = range(len(df))

    df = df.sort_values(["ts_event", "sequence"]).reset_index(drop=True)

    ticks = TickArrays(df)
    n = len(df)

    bar_starts = (ticks.ts_event // BAR_DURATION_NS) * BAR_DURATION_NS
    unique_bar_starts = np.unique(bar_starts)
    n_bars = len(unique_bar_starts)

    if n_bars == 0:
        return pd.DataFrame()

    (
        bar_meta_msg_cnt, bar_meta_clear_cnt, bar_meta_add_cnt, bar_meta_cancel_cnt,
        bar_meta_modify_cnt, bar_meta_trade_cnt,
        bar_trade_cnt, bar_trade_vol, bar_trade_aggbuy_vol, bar_trade_aggsell_vol,
        bar_flow_add_vol, bar_flow_rem_vol, bar_flow_net_vol,
        bar_flow_cnt_add, bar_flow_cnt_cancel, bar_flow_cnt_modify,
        bar_twa_spread_pts, bar_twa_obi0, bar_twa_obi10, bar_twa_cdi,
        bar_twa_bid10_qty, bar_twa_ask10_qty,
        bar_twa_below_qty, bar_twa_above_qty, bar_twa_below_frac, bar_twa_above_frac,
        bar_eob_idx,
    ) = process_all_ticks(
        ticks.ts_event,
        ticks.action,
        ticks.side,
        ticks.price,
        ticks.size,
        ticks.bid_px,
        ticks.ask_px,
        ticks.bid_sz,
        ticks.ask_sz,
        ticks.bid_ct,
        ticks.ask_ct,
        bar_starts,
        unique_bar_starts,
    )

    bars = []
    for bar_idx in range(n_bars):
        bar_ts = unique_bar_starts[bar_idx]
        eob_idx = int(bar_eob_idx[bar_idx])

        eob_bid_px = ticks.bid_px[eob_idx]
        eob_ask_px = ticks.ask_px[eob_idx]
        eob_bid_sz = ticks.bid_sz[eob_idx]
        eob_ask_sz = ticks.ask_sz[eob_idx]
        eob_bid_ct = ticks.bid_ct[eob_idx]
        eob_ask_ct = ticks.ask_ct[eob_idx]

        p_ref_eob = compute_microprice(eob_bid_px[0], eob_ask_px[0], eob_bid_sz[0], eob_ask_sz[0])
        eob_spread_pts = (eob_ask_px[0] - eob_bid_px[0]) / 1.0
        eob_obi0 = (eob_bid_sz[0] - eob_ask_sz[0]) / (eob_bid_sz[0] + eob_ask_sz[0] + EPSILON)
        eob_obi10 = (eob_bid_sz.sum() - eob_ask_sz.sum()) / (eob_bid_sz.sum() + eob_ask_sz.sum() + EPSILON)
        eob_bid10_qty = float(eob_bid_sz.sum())
        eob_ask10_qty = float(eob_ask_sz.sum())

        eob_below_qty, eob_above_qty = compute_banded_quantities(
            eob_bid_px, eob_ask_px, eob_bid_sz, eob_ask_sz, p_ref_eob
        )
        eob_cdi = compute_cdi(eob_below_qty, eob_above_qty)
        eob_below_frac, eob_above_frac = compute_banded_fractions(
            eob_below_qty, eob_above_qty, eob_bid10_qty, eob_ask10_qty
        )

        ask_gap_max, ask_gap_mean, bid_gap_max, bid_gap_mean = compute_ladder_features(
            eob_bid_px, eob_ask_px
        )

        bid_sz_frac = compute_shape_fractions(eob_bid_sz)
        ask_sz_frac = compute_shape_fractions(eob_ask_sz)
        bid_ct_frac = compute_shape_fractions(eob_bid_ct)
        ask_ct_frac = compute_shape_fractions(eob_ask_ct)

        wall_bid = compute_wall_features(eob_bid_sz, eob_bid_px, p_ref_eob, True)
        wall_ask = compute_wall_features(eob_ask_sz, eob_ask_px, p_ref_eob, False)

        bar_dict = {
            "bar_ts": bar_ts,
            "symbol": symbol,
            "bar5s_microprice_eob": p_ref_eob,
            "bar5s_midprice_eob": (eob_bid_px[0] + eob_ask_px[0]) / 2.0,
            "bar5s_meta_msg_cnt_sum": bar_meta_msg_cnt[bar_idx],
            "bar5s_meta_clear_cnt_sum": bar_meta_clear_cnt[bar_idx],
            "bar5s_meta_add_cnt_sum": bar_meta_add_cnt[bar_idx],
            "bar5s_meta_cancel_cnt_sum": bar_meta_cancel_cnt[bar_idx],
            "bar5s_meta_modify_cnt_sum": bar_meta_modify_cnt[bar_idx],
            "bar5s_meta_trade_cnt_sum": bar_meta_trade_cnt[bar_idx],
            "bar5s_state_spread_pts_twa": bar_twa_spread_pts[bar_idx],
            "bar5s_state_spread_pts_eob": eob_spread_pts,
            "bar5s_state_obi0_twa": bar_twa_obi0[bar_idx],
            "bar5s_state_obi0_eob": eob_obi0,
            "bar5s_state_obi10_twa": bar_twa_obi10[bar_idx],
            "bar5s_state_obi10_eob": eob_obi10,
        }

        for band_idx, band in enumerate(BANDS):
            bar_dict[f"bar5s_state_cdi_{band}_twa"] = bar_twa_cdi[bar_idx, band_idx]
            bar_dict[f"bar5s_state_cdi_{band}_eob"] = eob_cdi[band_idx]

        bar_dict["bar5s_depth_bid10_qty_twa"] = bar_twa_bid10_qty[bar_idx]
        bar_dict["bar5s_depth_bid10_qty_eob"] = eob_bid10_qty
        bar_dict["bar5s_depth_ask10_qty_twa"] = bar_twa_ask10_qty[bar_idx]
        bar_dict["bar5s_depth_ask10_qty_eob"] = eob_ask10_qty

        for band_idx, band in enumerate(BANDS):
            bar_dict[f"bar5s_depth_below_{band}_qty_twa"] = bar_twa_below_qty[bar_idx, band_idx]
            bar_dict[f"bar5s_depth_below_{band}_qty_eob"] = eob_below_qty[band_idx]
            bar_dict[f"bar5s_depth_above_{band}_qty_twa"] = bar_twa_above_qty[bar_idx, band_idx]
            bar_dict[f"bar5s_depth_above_{band}_qty_eob"] = eob_above_qty[band_idx]

        for band_idx, band in enumerate(BANDS):
            bar_dict[f"bar5s_depth_below_{band}_frac_twa"] = bar_twa_below_frac[bar_idx, band_idx]
            bar_dict[f"bar5s_depth_below_{band}_frac_eob"] = eob_below_frac[band_idx]
            bar_dict[f"bar5s_depth_above_{band}_frac_twa"] = bar_twa_above_frac[bar_idx, band_idx]
            bar_dict[f"bar5s_depth_above_{band}_frac_eob"] = eob_above_frac[band_idx]

        bar_dict["bar5s_ladder_ask_gap_max_pts_eob"] = ask_gap_max
        bar_dict["bar5s_ladder_ask_gap_mean_pts_eob"] = ask_gap_mean
        bar_dict["bar5s_ladder_bid_gap_max_pts_eob"] = bid_gap_max
        bar_dict["bar5s_ladder_bid_gap_mean_pts_eob"] = bid_gap_mean

        for i in range(10):
            bar_dict[f"bar5s_shape_bid_px_l{i:02d}_eob"] = float(eob_bid_px[i])
        
        for i in range(10):
            bar_dict[f"bar5s_shape_ask_px_l{i:02d}_eob"] = float(eob_ask_px[i])
        
        for i in range(10):
            bar_dict[f"bar5s_shape_bid_sz_l{i:02d}_eob"] = float(eob_bid_sz[i])
        
        for i in range(10):
            bar_dict[f"bar5s_shape_ask_sz_l{i:02d}_eob"] = float(eob_ask_sz[i])

        for i in range(10):
            bar_dict[f"bar5s_shape_bid_ct_l{i:02d}_eob"] = float(eob_bid_ct[i])
        
        for i in range(10):
            bar_dict[f"bar5s_shape_ask_ct_l{i:02d}_eob"] = float(eob_ask_ct[i])

        for i in range(10):
            bar_dict[f"bar5s_shape_bid_sz_frac_l{i:02d}_eob"] = float(bid_sz_frac[i])
            bar_dict[f"bar5s_shape_ask_sz_frac_l{i:02d}_eob"] = float(ask_sz_frac[i])

        for i in range(10):
            bar_dict[f"bar5s_shape_bid_ct_frac_l{i:02d}_eob"] = float(bid_ct_frac[i])
            bar_dict[f"bar5s_shape_ask_ct_frac_l{i:02d}_eob"] = float(ask_ct_frac[i])

        for side_idx, side_str in enumerate(["bid", "ask"]):
            for band_idx, band in enumerate(BANDS):
                bar_dict[f"bar5s_flow_add_vol_{side_str}_{band}_sum"] = bar_flow_add_vol[bar_idx, side_idx, band_idx]
                bar_dict[f"bar5s_flow_rem_vol_{side_str}_{band}_sum"] = bar_flow_rem_vol[bar_idx, side_idx, band_idx]
                bar_dict[f"bar5s_flow_net_vol_{side_str}_{band}_sum"] = bar_flow_net_vol[bar_idx, side_idx, band_idx]

        for side_idx, side_str in enumerate(["bid", "ask"]):
            for band_idx, band in enumerate(BANDS):
                bar_dict[f"bar5s_flow_cnt_add_{side_str}_{band}_sum"] = bar_flow_cnt_add[bar_idx, side_idx, band_idx]
                bar_dict[f"bar5s_flow_cnt_cancel_{side_str}_{band}_sum"] = bar_flow_cnt_cancel[bar_idx, side_idx, band_idx]
                bar_dict[f"bar5s_flow_cnt_modify_{side_str}_{band}_sum"] = bar_flow_cnt_modify[bar_idx, side_idx, band_idx]

        for side_idx, side_str in enumerate(["bid", "ask"]):
            for band_idx, band in enumerate(BANDS):
                net_vol = bar_flow_net_vol[bar_idx, side_idx, band_idx]
                if side_str == "bid":
                    twa_qty = bar_twa_below_qty[bar_idx, band_idx]
                else:
                    twa_qty = bar_twa_above_qty[bar_idx, band_idx]
                norm = net_vol / max(twa_qty, 1.0)
                bar_dict[f"bar5s_flow_net_volnorm_{side_str}_{band}_sum"] = norm

        bar_dict["bar5s_trade_cnt_sum"] = bar_trade_cnt[bar_idx]
        bar_dict["bar5s_trade_vol_sum"] = bar_trade_vol[bar_idx]
        bar_dict["bar5s_trade_aggbuy_vol_sum"] = bar_trade_aggbuy_vol[bar_idx]
        bar_dict["bar5s_trade_aggsell_vol_sum"] = bar_trade_aggsell_vol[bar_idx]
        bar_dict["bar5s_trade_signed_vol_sum"] = bar_trade_aggbuy_vol[bar_idx] - bar_trade_aggsell_vol[bar_idx]

        bar_dict["bar5s_wall_bid_maxz_eob"] = wall_bid[0]
        bar_dict["bar5s_wall_ask_maxz_eob"] = wall_ask[0]
        bar_dict["bar5s_wall_bid_maxz_levelidx_eob"] = wall_bid[1]
        bar_dict["bar5s_wall_ask_maxz_levelidx_eob"] = wall_ask[1]
        bar_dict["bar5s_wall_bid_nearest_strong_dist_pts_eob"] = wall_bid[2]
        bar_dict["bar5s_wall_ask_nearest_strong_dist_pts_eob"] = wall_ask[2]
        bar_dict["bar5s_wall_bid_nearest_strong_levelidx_eob"] = wall_bid[3]
        bar_dict["bar5s_wall_ask_nearest_strong_levelidx_eob"] = wall_ask[3]

        bars.append(bar_dict)

    if len(bars) == 0:
        return pd.DataFrame()

    bars_dict = {bar["bar_ts"]: bar for bar in bars}

    first_bar_ts = min(bars_dict.keys())
    last_bar_ts = max(bars_dict.keys())

    filled_bars = []
    current_ts = first_bar_ts

    while current_ts <= last_bar_ts:
        if current_ts in bars_dict:
            filled_bars.append(bars_dict[current_ts])
        else:
            prev_ts = current_ts - BAR_DURATION_NS
            if prev_ts in bars_dict:
                prev_bar = bars_dict[prev_ts]
            else:
                for i in range(len(filled_bars) - 1, -1, -1):
                    if filled_bars[i]["bar_ts"] < current_ts:
                        prev_bar = filled_bars[i]
                        break

            empty_bar = fill_empty_bar(current_ts, symbol, prev_bar)
            filled_bars.append(empty_bar)
            bars_dict[current_ts] = empty_bar

        current_ts += BAR_DURATION_NS

    return pd.DataFrame(filled_bars)
