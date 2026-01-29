from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numba import float64, int8, int64, jit, types
from numba.typed import Dict as NumbaDict
from numba.typed import List as NumbaList

# Constants
PRICE_SCALE = 1e-9
WINDOW_NS = 1_000_000_000
REST_NS = 500_000_000
EPS_QTY = 1.0

# Mapped Constants
ACT_MAP = {"A": 1, "C": 2, "M": 3, "R": 4, "F": 5, "T": 6, "N": 0}
SIDE_MAP = {"A": 0, "B": 1, "N": -1}

@dataclass
class OptionOrderState:
    instrument_id: int
    side: str
    price_int: int
    qty: int
    ts_enter: int
    order_id: int

class OptionsBookEngine:
    def __init__(self, window_ns: int = WINDOW_NS, rest_ns: int = REST_NS):
        self.window_ns = window_ns
        self.rest_ns = rest_ns
        
        # State:
        self.orders_meta = NumbaDict.empty(int64, types.Tuple((int64, int8, int64, int64, int64)))
        self.depth = NumbaDict.empty(types.Tuple((int64, int8, int64)), int64)
        self.levels_bid = NumbaDict.empty(int64, types.ListType(int64))
        self.levels_ask = NumbaDict.empty(int64, types.ListType(int64))
        self.known_iids = NumbaDict.empty(int64, int8)
        
    def process_batch(self, df_mbo: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df_mbo.empty:
            return pd.DataFrame(), pd.DataFrame()

        df_mbo = df_mbo.sort_values(["ts_event", "sequence"])
        
        ts_arr = df_mbo["ts_event"].values.astype(np.int64)
        iid_arr = df_mbo["instrument_id"].values.astype(np.int64)
        act_arr = df_mbo["action"].map(ACT_MAP).fillna(0).values.astype(np.int8)
        side_arr = df_mbo["side"].map(SIDE_MAP).fillna(-1).values.astype(np.int8)
        px_arr = df_mbo["price"].values.astype(np.int64)
        sz_arr = df_mbo["size"].values.astype(np.int64)
        oid_arr = df_mbo["order_id"].values.astype(np.int64)
        flags_arr = df_mbo["flags"].values.astype(np.int64)
        
        # Separate accumulators
        dt = types.Tuple((int64, int8, int64))
        win_acc_add = NumbaDict.empty(dt, int64)
        win_acc_pull = NumbaDict.empty(dt, int64)
        win_acc_pull_rest = NumbaDict.empty(dt, int64)
        win_acc_fill = NumbaDict.empty(dt, int64)
        
        # Scratch dict for active keys (Set replacement)
        active_keys = NumbaDict.empty(dt, int8)

        # Pre-allocated dict for resting depth calculation
        local_depth_rest = NumbaDict.empty(dt, int64)
        
        (
            out_win_end, out_iid, out_side, out_px, 
            out_depth_total, out_depth_rest, out_add_qty, out_pull_qty, out_pull_rest, out_fill_qty,
            bbo_ts, bbo_iid, bbo_bid, bbo_ask, bbo_mid
        ) = _process_options_mbo(
            ts_arr, iid_arr, act_arr, side_arr, px_arr, sz_arr, oid_arr, flags_arr,
            self.orders_meta, self.depth, 
            win_acc_add, win_acc_pull, win_acc_pull_rest, win_acc_fill, active_keys,
            self.levels_bid, self.levels_ask, self.known_iids,
            self.window_ns, self.rest_ns, local_depth_rest
        )
        
        # Flow DF
        if len(out_win_end) > 0:
            df_out = pd.DataFrame({
                "window_end_ts_ns": list(out_win_end),
                "instrument_id": list(out_iid),
                "side_int": list(out_side),
                "price_int": list(out_px),
                "depth_total": list(out_depth_total),
                "depth_rest": list(out_depth_rest),
                "add_qty": list(out_add_qty),
                "pull_qty": list(out_pull_qty),
                "pull_rest_qty": list(out_pull_rest),
                "fill_qty": list(out_fill_qty)
            })
            inv_side_map = {0: "A", 1: "B", -1: "N"}
            df_out["side"] = df_out["side_int"].map(inv_side_map)
            df_out = df_out.drop(columns=["side_int"])
        else:
            df_out = pd.DataFrame()

        # BBO DF
        if len(bbo_ts) > 0:
            df_bbo = pd.DataFrame({
                "window_end_ts_ns": list(bbo_ts),
                "instrument_id": list(bbo_iid),
                "bid_price_int": list(bbo_bid),
                "ask_price_int": list(bbo_ask),
                "mid_price_int": list(bbo_mid)
            })
        else:
            df_bbo = pd.DataFrame()
            
        return df_out, df_bbo

@jit(nopython=True)
def _process_options_mbo(
    ts_arr, iid_arr, act_arr, side_arr, px_arr, sz_arr, oid_arr, flags_arr,
    orders_meta, depth, 
    win_acc_add, win_acc_pull, win_acc_pull_rest, win_acc_fill, active_keys,
    levels_bid, levels_ask, known_iids,
    window_ns, rest_ns, local_depth_rest
):
    out_win_end = NumbaList.empty_list(int64)
    # ... (rest of lists)
    out_iid = NumbaList.empty_list(int64)
    out_side = NumbaList.empty_list(int8)
    out_px = NumbaList.empty_list(int64)
    out_depth_total = NumbaList.empty_list(int64)
    out_depth_rest = NumbaList.empty_list(int64)
    out_add_qty = NumbaList.empty_list(int64)
    out_pull_qty = NumbaList.empty_list(int64)
    out_pull_rest = NumbaList.empty_list(int64)
    out_fill_qty = NumbaList.empty_list(int64)
    
    bbo_ts = NumbaList.empty_list(int64)
    bbo_iid = NumbaList.empty_list(int64)
    bbo_bid = NumbaList.empty_list(int64)
    bbo_ask = NumbaList.empty_list(int64)
    bbo_mid = NumbaList.empty_list(float64)
    
    curr_window = -1
    window_end = 0
    
    n = len(ts_arr)
    for i in range(n):
        ts = ts_arr[i]
        w_id = ts // window_ns
        
        if curr_window == -1:
            curr_window = w_id
            window_end = (w_id + 1) * window_ns
        
        if w_id > curr_window:
            # Emit Window Data
            # Build active keys from flow + depth to ensure we capture resting liquidity.
            for k in win_acc_add:
                active_keys[k] = 1
            for k in win_acc_pull:
                active_keys[k] = 1
            for k in win_acc_fill:
                active_keys[k] = 1
            for k in depth:
                if depth.get(k, 0) > 0:
                    active_keys[k] = 1

            # Compute resting depth for this window end
            local_depth_rest.clear()
            for oid in orders_meta:
                 _iid, _side, _px, _qty, _ts = orders_meta[oid]
                 if window_end - _ts >= rest_ns:
                     k_ord = (_iid, _side, _px)
                     local_depth_rest[k_ord] = local_depth_rest.get(k_ord, 0) + _qty
            
            for k in active_keys:
                iid_k, side_k, px_k = k
                add = win_acc_add.get(k, 0)
                pull = win_acc_pull.get(k, 0)
                pull_rest = win_acc_pull_rest.get(k, 0)
                fill = win_acc_fill.get(k, 0)
                
                d_total = depth.get(k, 0)
                d_rest = local_depth_rest.get(k, 0)
                
                if d_total <= 0 and add == 0 and pull == 0 and pull_rest == 0 and fill == 0:
                    continue
                
                out_win_end.append(window_end)
                out_iid.append(iid_k)
                out_side.append(side_k)
                out_px.append(px_k)
                out_depth_total.append(d_total)
                out_depth_rest.append(d_rest)
                out_add_qty.append(add)
                out_pull_qty.append(pull)
                out_pull_rest.append(pull_rest)
                out_fill_qty.append(fill)
            
            # Reset
            win_acc_add.clear()
            win_acc_pull.clear()
            win_acc_pull_rest.clear()
            win_acc_fill.clear()
            active_keys.clear() # Clear scratch
            
            # Emit BBO
            while curr_window < w_id:
                for iid in known_iids:
                    bb = 0
                    if iid in levels_bid:
                         lb = levels_bid[iid]
                         if len(lb) > 0: bb = lb[-1]
                    ba = 0
                    if iid in levels_ask:
                         la = levels_ask[iid]
                         if len(la) > 0: ba = la[0]
                    if bb > 0 and ba > 0 and ba > bb:
                        mid = (bb + ba) * 0.5
                        bbo_ts.append(window_end)
                        bbo_iid.append(iid)
                        bbo_bid.append(bb)
                        bbo_ask.append(ba)
                        bbo_mid.append(mid)
                curr_window += 1
                window_end = (curr_window + 1) * window_ns
        
        act = act_arr[i]
        iid = iid_arr[i]
        oid = oid_arr[i]
        sz = sz_arr[i] # Extract size here for all actions
        known_iids[iid] = 1
        
        if act == 1: # ADD
            side = side_arr[i]
            px = px_arr[i]
            # sz extracted above
            orders_meta[oid] = (iid, side, px, sz, ts)
            dk = (iid, side, px)
            depth[dk] = depth.get(dk, 0) + sz
            
            if dk in win_acc_add:
                win_acc_add[dk] += sz
            else:
                win_acc_add[dk] = sz
            
            if side == 1: 
                if iid not in levels_bid: levels_bid[iid] = NumbaList.empty_list(int64)
                _insort(levels_bid[iid], px)
            else:
                if iid not in levels_ask: levels_ask[iid] = NumbaList.empty_list(int64)
                _insort(levels_ask[iid], px)
            
        elif act == 5: # FILL
            if oid in orders_meta:
                _iid, _side, _px, _qty, _ts = orders_meta[oid]
                rem = _qty - sz
                if rem <= 0:
                    del orders_meta[oid]
                else:
                    orders_meta[oid] = (_iid, _side, _px, rem, _ts)
                    
                dk = (_iid, _side, _px)
                d = depth.get(dk, 0)
                depth[dk] = max(0, d - sz)
                
                if dk in win_acc_fill: win_acc_fill[dk] += sz
                else: win_acc_fill[dk] = sz
                
                if depth[dk] <= 0:
                    if _side == 1: _remove_val(levels_bid[_iid], _px)
                    else: _remove_val(levels_ask[_iid], _px)
                
        elif act == 2: # CANCEL
            if oid in orders_meta:
                _iid, _side, _px, _qty, _ts = orders_meta[oid]
                rem = _qty - sz
                if rem <= 0:
                    del orders_meta[oid]
                else:
                    orders_meta[oid] = (_iid, _side, _px, rem, _ts)
                
                dk = (_iid, _side, _px)
                d = depth.get(dk, 0)
                depth[dk] = max(0, d - sz)
                
                is_resting = (ts - _ts) >= rest_ns
                
                if dk in win_acc_pull: 
                    win_acc_pull[dk] += sz
                else: 
                    win_acc_pull[dk] = sz
                
                if is_resting:
                    if dk in win_acc_pull_rest: win_acc_pull_rest[dk] += sz
                    else: win_acc_pull_rest[dk] = sz
                
                if depth[dk] <= 0:
                    if _side == 1: _remove_val(levels_bid[_iid], _px)
                    else: _remove_val(levels_ask[_iid], _px)
                
        elif act == 3: # MODIFY
            if oid in orders_meta:
                _iid, _side, _px, _qty, _ts = orders_meta[oid]
                dk_old = (_iid, _side, _px)
                d_old = depth.get(dk_old, 0)
                depth[dk_old] = max(0, d_old - _qty)
                is_resting = (ts - _ts) >= rest_ns
                
                if px != _px:
                    if dk_old in win_acc_pull: win_acc_pull[dk_old] += _qty
                    else: win_acc_pull[dk_old] = _qty
                    if is_resting:
                        if dk_old in win_acc_pull_rest: win_acc_pull_rest[dk_old] += _qty
                        else: win_acc_pull_rest[dk_old] = _qty
                        
                    if depth[dk_old] <= 0:
                        if _side == 1: _remove_val(levels_bid[_iid], _px)
                        else: _remove_val(levels_ask[_iid], _px)
                    
                    dk_new = (_iid, side, px)
                    depth[dk_new] = depth.get(dk_new, 0) + sz
                    orders_meta[oid] = (_iid, side, px, sz, ts)
                    
                    if dk_new in win_acc_add: win_acc_add[dk_new] += sz
                    else: win_acc_add[dk_new] = sz
                    
                    if side == 1: 
                        if iid not in levels_bid: levels_bid[iid] = NumbaList.empty_list(int64)
                        _insort(levels_bid[iid], px)
                    else:
                        if iid not in levels_ask: levels_ask[iid] = NumbaList.empty_list(int64)
                        _insort(levels_ask[iid], px)
                else:
                    dk = dk_old
                    if sz < _qty:
                         dec = _qty - sz
                         if dk in win_acc_pull: win_acc_pull[dk] += dec
                         else: win_acc_pull[dk] = dec
                         if is_resting:
                             if dk in win_acc_pull_rest: win_acc_pull_rest[dk] += dec
                             else: win_acc_pull_rest[dk] = dec
                         orders_meta[oid] = (_iid, _side, _px, sz, _ts)
                    else:
                         inc = sz - _qty
                         if dk in win_acc_add: win_acc_add[dk] += inc
                         else: win_acc_add[dk] = inc
                         orders_meta[oid] = (_iid, _side, _px, sz, ts)
                         
                    depth[dk] = max(0, depth.get(dk, 0) - _qty + sz)
                    
    # Emit Last Window
    if curr_window != -1:
        # Use passed active_keys (it was cleared)
        for k in win_acc_add:
            active_keys[k] = 1
        for k in win_acc_pull:
            active_keys[k] = 1
        for k in win_acc_fill:
            active_keys[k] = 1
        for k in depth:
            if depth.get(k, 0) > 0:
                active_keys[k] = 1
        
        # Compute resting depth
        local_depth_rest.clear()
        for oid in orders_meta:
             _iid, _side, _px, _qty, _ts = orders_meta[oid]
             if window_end - _ts >= rest_ns:
                 k_ord = (_iid, _side, _px)
                 local_depth_rest[k_ord] = local_depth_rest.get(k_ord, 0) + _qty

        for k in active_keys:
            iid_k, side_k, px_k = k
            add = win_acc_add.get(k, 0)
            pull = win_acc_pull.get(k, 0)
            pull_rest = win_acc_pull_rest.get(k, 0)
            fill = win_acc_fill.get(k, 0)
            d_total = depth.get(k, 0)
            d_rest = local_depth_rest.get(k, 0)
            
            if d_total <= 0 and add == 0 and pull == 0 and pull_rest == 0 and fill == 0:
                continue
            
            out_win_end.append(window_end)
            out_iid.append(iid_k)
            out_side.append(side_k)
            out_px.append(px_k)
            out_depth_total.append(d_total)
            out_depth_rest.append(d_rest)
            out_add_qty.append(add)
            out_pull_qty.append(pull)
            out_pull_rest.append(pull_rest)
            out_fill_qty.append(fill)
            
        for iid in known_iids:
            bb = 0
            if iid in levels_bid:
                 lb = levels_bid[iid]
                 if len(lb) > 0: bb = lb[-1]
            ba = 0
            if iid in levels_ask:
                 la = levels_ask[iid]
                 if len(la) > 0: ba = la[0]
            if bb > 0 and ba > 0 and ba > bb:
                mid = (bb + ba) * 0.5
                bbo_ts.append(window_end)
                bbo_iid.append(iid)
                bbo_bid.append(bb)
                bbo_ask.append(ba)
                bbo_mid.append(mid)

    return (
        out_win_end, out_iid, out_side, out_px, 
        out_depth_total, out_depth_rest, out_add_qty, out_pull_qty, out_pull_rest, out_fill_qty,
        bbo_ts, bbo_iid, bbo_bid, bbo_ask, bbo_mid
    )

@jit(nopython=True)
def _insort(l, val):
    found = False
    for i in range(len(l)):
        if l[i] == val:
            found = True
            break
    if found: return
    lo = 0
    hi = len(l)
    while lo < hi:
        mid = (lo + hi) // 2
        if l[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    l.insert(lo, val)

@jit(nopython=True)
def _remove_val(l, val):
    idx = -1
    for i in range(len(l)):
        if l[i] == val:
            idx = i
            break
    if idx != -1:
        l.pop(idx)
