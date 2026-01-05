from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .bands import compute_banded_fractions, compute_banded_quantities, compute_cdi
from .bar_accumulator import BarAccumulator
from .book_state import BookState
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
from .flow import compute_delta_q_vectorized, compute_flow_band
from .ladder import compute_ladder_features
from .shape import compute_shape_fractions
from .wall import compute_wall_features


class TickArrays:
    __slots__ = (
        "ts_event", "action", "side", "price", "size", "sequence",
        "bid_px", "ask_px", "bid_sz", "ask_sz", "bid_ct", "ask_ct",
    )
    
    def __init__(self, df: pd.DataFrame) -> None:
        n = len(df)
        self.ts_event: NDArray[np.int64] = df["ts_event"].values.astype(np.int64)
        self.action: NDArray = df["action"].values
        self.side: NDArray = df["side"].values
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
    
    bars: list[dict] = []
    
    current_bar: BarAccumulator | None = None
    pre_state = BookState()
    post_state = BookState()
    
    post_state.load_from_arrays(ticks, 0)
    pre_state.copy_from(post_state)
    
    first_bar_start = int(bar_starts[0])
    current_bar = BarAccumulator(first_bar_start)
    
    for idx in range(n):
        ts_event = int(ticks.ts_event[idx])
        bar_start = int(bar_starts[idx])
        
        if current_bar is not None and bar_start > current_bar.bar_start_ns:
            finalize_bar(current_bar, pre_state, symbol, bars)
            current_bar = BarAccumulator(bar_start)
        
        post_state.load_from_arrays(ticks, idx)
        
        if current_bar is not None:
            process_event(ticks, idx, ts_event, pre_state, post_state, current_bar)
        
        pre_state.copy_from(post_state)
    
    if current_bar is not None:
        finalize_bar(current_bar, pre_state, symbol, bars)
    
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


def process_event(ticks: TickArrays, idx: int, ts_event: int, pre_state: BookState, post_state: BookState, bar: BarAccumulator) -> None:
    dt = ts_event - bar.t_last
    
    if dt > 0:
        p_ref = pre_state.compute_microprice()
        spread_pts = pre_state.compute_spread_pts()
        obi0 = pre_state.compute_obi0()
        obi10 = pre_state.compute_obi10()
        
        bid10_qty, ask10_qty = pre_state.compute_total_depth()
        
        below_qty, above_qty = compute_banded_quantities(pre_state, p_ref)
        cdi = compute_cdi(below_qty, above_qty)
        below_frac, above_frac = compute_banded_fractions(below_qty, above_qty, bid10_qty, ask10_qty)
        
        bar.accumulate_twa_state(dt, spread_pts, obi0, obi10, cdi, bid10_qty, ask10_qty, 
                                below_qty, above_qty, below_frac, above_frac)
        
        bar.t_last = ts_event
    
    bar.meta_msg_cnt += 1
    
    action = ticks.action[idx]
    side = ticks.side[idx]
    
    if action == ACTION_CLEAR:
        bar.meta_clear_cnt += 1
    elif action == ACTION_ADD:
        bar.meta_add_cnt += 1
        process_flow_event(ticks, idx, pre_state, post_state, bar, "add")
    elif action == ACTION_CANCEL:
        bar.meta_cancel_cnt += 1
        process_flow_event(ticks, idx, pre_state, post_state, bar, "cancel")
    elif action == ACTION_MODIFY:
        bar.meta_modify_cnt += 1
        process_flow_event(ticks, idx, pre_state, post_state, bar, "modify")
    elif action == ACTION_TRADE:
        bar.meta_trade_cnt += 1
        process_trade_event(ticks, idx, bar)


def process_flow_event(ticks: TickArrays, idx: int, pre_state: BookState, post_state: BookState, bar: BarAccumulator, action_type: str) -> None:
    event_price = ticks.price[idx]
    event_side = ticks.side[idx]
    
    if event_price < EPSILON:
        return
    
    add_vol, rem_vol = compute_delta_q_vectorized(event_price, event_side, pre_state, post_state)
    
    p_ref = pre_state.compute_microprice()
    band = compute_flow_band(event_price, event_side, p_ref)
    
    if band is None:
        return
    
    side_str = "bid" if event_side == SIDE_BID else "ask"
    key = f"{side_str}_{band}"
    
    bar.flow_add_vol[key] += add_vol
    bar.flow_rem_vol[key] += rem_vol
    bar.flow_net_vol[key] += (add_vol - rem_vol)
    
    if action_type == "add":
        bar.flow_cnt_add[key] += 1
    elif action_type == "cancel":
        bar.flow_cnt_cancel[key] += 1
    elif action_type == "modify":
        bar.flow_cnt_modify[key] += 1


def process_trade_event(ticks: TickArrays, idx: int, bar: BarAccumulator) -> None:
    size = ticks.size[idx]
    side = ticks.side[idx]
    
    bar.trade_cnt += 1
    bar.trade_vol += size
    
    if side == SIDE_ASK:
        bar.trade_aggbuy_vol += size
    elif side == SIDE_BID:
        bar.trade_aggsell_vol += size


def finalize_bar(bar: BarAccumulator, final_state: BookState, symbol: str, bars: list[dict]) -> None:
    dt_end = bar.bar_end_ns - bar.t_last
    
    if dt_end > 0:
        p_ref = final_state.compute_microprice()
        spread_pts = final_state.compute_spread_pts()
        obi0 = final_state.compute_obi0()
        obi10 = final_state.compute_obi10()
        
        bid10_qty, ask10_qty = final_state.compute_total_depth()
        
        below_qty, above_qty = compute_banded_quantities(final_state, p_ref)
        cdi = compute_cdi(below_qty, above_qty)
        below_frac, above_frac = compute_banded_fractions(below_qty, above_qty, bid10_qty, ask10_qty)
        
        bar.accumulate_twa_state(dt_end, spread_pts, obi0, obi10, cdi, bid10_qty, ask10_qty,
                                below_qty, above_qty, below_frac, above_frac)
    
    bar.finalize_twa(bar.bar_end_ns)
    
    p_ref_eob = final_state.compute_microprice()
    bar.eob_spread_pts = final_state.compute_spread_pts()
    bar.eob_obi0 = final_state.compute_obi0()
    bar.eob_obi10 = final_state.compute_obi10()
    
    bid10_qty_eob, ask10_qty_eob = final_state.compute_total_depth()
    bar.eob_bid10_qty = bid10_qty_eob
    bar.eob_ask10_qty = ask10_qty_eob
    
    below_qty_eob, above_qty_eob = compute_banded_quantities(final_state, p_ref_eob)
    for band in BANDS:
        bar.eob_below_qty[band] = below_qty_eob[band]
        bar.eob_above_qty[band] = above_qty_eob[band]
    
    cdi_eob = compute_cdi(below_qty_eob, above_qty_eob)
    for band in BANDS:
        bar.eob_cdi[band] = cdi_eob[band]
    
    below_frac_eob, above_frac_eob = compute_banded_fractions(below_qty_eob, above_qty_eob, bid10_qty_eob, ask10_qty_eob)
    for band in BANDS:
        bar.eob_below_frac[band] = below_frac_eob[band]
        bar.eob_above_frac[band] = above_frac_eob[band]
    
    bar.eob_bid_sz[:] = final_state.bid_sz
    bar.eob_ask_sz[:] = final_state.ask_sz
    bar.eob_bid_ct[:] = final_state.bid_ct
    bar.eob_ask_ct[:] = final_state.ask_ct
    bar.eob_bid_px[:] = final_state.bid_px
    bar.eob_ask_px[:] = final_state.ask_px
    
    ladder = compute_ladder_features(final_state.bid_px, final_state.ask_px)
    
    bid_sz_frac = compute_shape_fractions(final_state.bid_sz)
    ask_sz_frac = compute_shape_fractions(final_state.ask_sz)
    bid_ct_frac = compute_shape_fractions(final_state.bid_ct)
    ask_ct_frac = compute_shape_fractions(final_state.ask_ct)
    
    wall_bid = compute_wall_features(final_state.bid_sz, final_state.bid_px, p_ref_eob, is_bid=True)
    wall_ask = compute_wall_features(final_state.ask_sz, final_state.ask_px, p_ref_eob, is_bid=False)
    
    bar_dict = {
        "bar_ts": bar.bar_start_ns,
        "symbol": symbol,
        "bar5s_meta_msg_cnt_sum": bar.meta_msg_cnt,
        "bar5s_meta_clear_cnt_sum": bar.meta_clear_cnt,
        "bar5s_meta_add_cnt_sum": bar.meta_add_cnt,
        "bar5s_meta_cancel_cnt_sum": bar.meta_cancel_cnt,
        "bar5s_meta_modify_cnt_sum": bar.meta_modify_cnt,
        "bar5s_meta_trade_cnt_sum": bar.meta_trade_cnt,
        "bar5s_state_spread_pts_twa": bar.twa_spread_pts,
        "bar5s_state_spread_pts_eob": bar.eob_spread_pts,
        "bar5s_state_obi0_twa": bar.twa_obi0,
        "bar5s_state_obi0_eob": bar.eob_obi0,
        "bar5s_state_obi10_twa": bar.twa_obi10,
        "bar5s_state_obi10_eob": bar.eob_obi10,
    }
    
    for band in BANDS:
        bar_dict[f"bar5s_state_cdi_{band}_twa"] = bar.twa_cdi[band]
        bar_dict[f"bar5s_state_cdi_{band}_eob"] = bar.eob_cdi[band]
    
    bar_dict["bar5s_depth_bid10_qty_twa"] = bar.twa_bid10_qty
    bar_dict["bar5s_depth_bid10_qty_eob"] = bar.eob_bid10_qty
    bar_dict["bar5s_depth_ask10_qty_twa"] = bar.twa_ask10_qty
    bar_dict["bar5s_depth_ask10_qty_eob"] = bar.eob_ask10_qty
    
    for band in BANDS:
        bar_dict[f"bar5s_depth_below_{band}_qty_twa"] = bar.twa_below_qty[band]
        bar_dict[f"bar5s_depth_below_{band}_qty_eob"] = bar.eob_below_qty[band]
        bar_dict[f"bar5s_depth_above_{band}_qty_twa"] = bar.twa_above_qty[band]
        bar_dict[f"bar5s_depth_above_{band}_qty_eob"] = bar.eob_above_qty[band]
    
    for band in BANDS:
        bar_dict[f"bar5s_depth_below_{band}_frac_twa"] = bar.twa_below_frac[band]
        bar_dict[f"bar5s_depth_below_{band}_frac_eob"] = bar.eob_below_frac[band]
        bar_dict[f"bar5s_depth_above_{band}_frac_twa"] = bar.twa_above_frac[band]
        bar_dict[f"bar5s_depth_above_{band}_frac_eob"] = bar.eob_above_frac[band]
    
    bar_dict["bar5s_ladder_ask_gap_max_pts_eob"] = ladder["ask_gap_max_pts"]
    bar_dict["bar5s_ladder_ask_gap_mean_pts_eob"] = ladder["ask_gap_mean_pts"]
    bar_dict["bar5s_ladder_bid_gap_max_pts_eob"] = ladder["bid_gap_max_pts"]
    bar_dict["bar5s_ladder_bid_gap_mean_pts_eob"] = ladder["bid_gap_mean_pts"]
    
    for i in range(10):
        bar_dict[f"bar5s_shape_bid_sz_l{i:02d}_eob"] = float(bar.eob_bid_sz[i])
        bar_dict[f"bar5s_shape_ask_sz_l{i:02d}_eob"] = float(bar.eob_ask_sz[i])
    
    for i in range(10):
        bar_dict[f"bar5s_shape_bid_ct_l{i:02d}_eob"] = float(bar.eob_bid_ct[i])
        bar_dict[f"bar5s_shape_ask_ct_l{i:02d}_eob"] = float(bar.eob_ask_ct[i])
    
    for i in range(10):
        bar_dict[f"bar5s_shape_bid_sz_frac_l{i:02d}_eob"] = float(bid_sz_frac[i])
        bar_dict[f"bar5s_shape_ask_sz_frac_l{i:02d}_eob"] = float(ask_sz_frac[i])
    
    for i in range(10):
        bar_dict[f"bar5s_shape_bid_ct_frac_l{i:02d}_eob"] = float(bid_ct_frac[i])
        bar_dict[f"bar5s_shape_ask_ct_frac_l{i:02d}_eob"] = float(ask_ct_frac[i])
    
    for side in ["bid", "ask"]:
        for band in BANDS:
            key = f"{side}_{band}"
            bar_dict[f"bar5s_flow_add_vol_{side}_{band}_sum"] = bar.flow_add_vol[key]
            bar_dict[f"bar5s_flow_rem_vol_{side}_{band}_sum"] = bar.flow_rem_vol[key]
            bar_dict[f"bar5s_flow_net_vol_{side}_{band}_sum"] = bar.flow_net_vol[key]
    
    for side in ["bid", "ask"]:
        for band in BANDS:
            key = f"{side}_{band}"
            bar_dict[f"bar5s_flow_cnt_add_{side}_{band}_sum"] = bar.flow_cnt_add[key]
            bar_dict[f"bar5s_flow_cnt_cancel_{side}_{band}_sum"] = bar.flow_cnt_cancel[key]
            bar_dict[f"bar5s_flow_cnt_modify_{side}_{band}_sum"] = bar.flow_cnt_modify[key]
    
    for side in ["bid", "ask"]:
        for band in BANDS:
            key = f"{side}_{band}"
            net_vol = bar.flow_net_vol[key]
            if side == "bid":
                twa_qty = bar.twa_below_qty[band]
            else:
                twa_qty = bar.twa_above_qty[band]
            norm = net_vol / max(twa_qty, 1.0)
            bar_dict[f"bar5s_flow_net_volnorm_{side}_{band}_sum"] = norm
    
    bar_dict["bar5s_trade_cnt_sum"] = bar.trade_cnt
    bar_dict["bar5s_trade_vol_sum"] = bar.trade_vol
    bar_dict["bar5s_trade_aggbuy_vol_sum"] = bar.trade_aggbuy_vol
    bar_dict["bar5s_trade_aggsell_vol_sum"] = bar.trade_aggsell_vol
    bar_dict["bar5s_trade_signed_vol_sum"] = bar.trade_aggbuy_vol - bar.trade_aggsell_vol
    
    bar_dict["bar5s_wall_bid_maxz_eob"] = wall_bid["maxz"]
    bar_dict["bar5s_wall_ask_maxz_eob"] = wall_ask["maxz"]
    bar_dict["bar5s_wall_bid_maxz_levelidx_eob"] = wall_bid["maxz_levelidx"]
    bar_dict["bar5s_wall_ask_maxz_levelidx_eob"] = wall_ask["maxz_levelidx"]
    bar_dict["bar5s_wall_bid_nearest_strong_dist_pts_eob"] = wall_bid["nearest_strong_dist_pts"]
    bar_dict["bar5s_wall_ask_nearest_strong_dist_pts_eob"] = wall_ask["nearest_strong_dist_pts"]
    bar_dict["bar5s_wall_bid_nearest_strong_levelidx_eob"] = wall_bid["nearest_strong_levelidx"]
    bar_dict["bar5s_wall_ask_nearest_strong_levelidx_eob"] = wall_ask["nearest_strong_levelidx"]
    
    bars.append(bar_dict)
