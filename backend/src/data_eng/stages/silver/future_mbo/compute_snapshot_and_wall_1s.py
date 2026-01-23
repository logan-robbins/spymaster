from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numba import boolean, float64, int64, int8, jit, types
from numba.typed import Dict as NumbaDict, List as NumbaList

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    read_partition,
    write_partition,
)

PRICE_SCALE = 1e-9
TICK_SIZE = 0.25
TICK_INT = int(round(TICK_SIZE / PRICE_SCALE))
WINDOW_NS = 1_000_000_000  # 1 second
REST_NS = 500_000_000
EPS_QTY = 1.0
HUD_MAX_TICKS = 600

# Action constants
ACTION_ADD = "A"
ACTION_CANCEL = "C"
ACTION_MODIFY = "M"
ACTION_CLEAR = "R"
ACTION_TRADE = "T"
ACTION_FILL = "F"
ACTION_NONE = "N"

# Mapped constants for Numba
ACT_ADD = 1
ACT_CANCEL = 2
ACT_MODIFY = 3
ACT_CLEAR = 4
ACT_FILL = 5
ACT_TRADE = 6

SIDE_ASK = 0
SIDE_BID = 1

F_SNAPSHOT = 128
F_LAST = 256


class SilverComputeSnapshotAndWall1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_snapshot_and_wall_1s",
            io=StageIO(
                inputs=["bronze.future_mbo.mbo"],
                output="silver.future_mbo.book_snapshot_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_key_snap = "silver.future_mbo.book_snapshot_1s"
        out_key_wall = "silver.future_mbo.wall_surface_1s"

        ref_snap = partition_ref(cfg, out_key_snap, symbol, dt)
        ref_wall = partition_ref(cfg, out_key_wall, symbol, dt)

        if is_partition_complete(ref_snap) and is_partition_complete(ref_wall):
            return

        input_key = self.io.inputs[0]
        in_ref = partition_ref(cfg, input_key, symbol, dt)
        if not is_partition_complete(in_ref):
            raise FileNotFoundError(f"Input not ready: {input_key} dt={dt}")

        in_contract_path = repo_root / cfg.dataset(input_key).contract
        in_contract = load_avro_contract(in_contract_path)
        df_in = read_partition(in_ref)
        df_in = enforce_contract(df_in, in_contract)

        df_snap, df_wall = self.transform(df_in, dt)

        contract_snap_path = repo_root / cfg.dataset(out_key_snap).contract
        contract_wall_path = repo_root / cfg.dataset(out_key_wall).contract

        contract_snap = load_avro_contract(contract_snap_path)
        contract_wall = load_avro_contract(contract_wall_path)

        df_snap = enforce_contract(df_snap, contract_snap)
        df_wall = enforce_contract(df_wall, contract_wall)

        lineage = [{"dataset": in_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(in_ref)}]

        if not is_partition_complete(ref_snap):
            write_partition(
                cfg=cfg,
                dataset_key=out_key_snap,
                symbol=symbol,
                dt=dt,
                df=df_snap,
                contract_path=contract_snap_path,
                inputs=lineage,
                stage=self.name,
            )

        if not is_partition_complete(ref_wall):
            write_partition(
                cfg=cfg,
                dataset_key=out_key_wall,
                symbol=symbol,
                dt=dt,
                df=df_wall,
                contract_path=contract_wall_path,
                inputs=lineage,
                stage=self.name,
            )

    def transform(self, df: pd.DataFrame, dt: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        required_cols = {"ts_event", "action", "side", "price", "size", "order_id", "sequence"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

        df = df.sort_values(["ts_event", "sequence"], ascending=[True, True])

        # Prepare for Numba
        ts_arr = df["ts_event"].to_numpy(dtype=np.int64)
        
        act_map = {"A": 1, "C": 2, "M": 3, "R": 4, "F": 5, "T": 6, "N": 0}
        act_arr = df["action"].map(act_map).fillna(0).to_numpy(dtype=np.int8)

        side_map = {"A": 0, "B": 1, "N": -1}
        side_arr = df["side"].map(side_map).fillna(-1).to_numpy(dtype=np.int8)

        px_arr = df["price"].to_numpy(dtype=np.int64)
        sz_arr = df["size"].to_numpy(dtype=np.int64)
        oid_arr = df["order_id"].to_numpy(dtype=np.int64)
        flags_arr = df["flags"].to_numpy(dtype=np.int64) if "flags" in df.columns else np.zeros(len(df), dtype=np.int64)

        # Free memory
        del df

        # Numba Execution
        snap_rows, wall_rows = _numba_compute_snapshot_wall(
            ts_arr, act_arr, side_arr, px_arr, sz_arr, oid_arr, flags_arr, WINDOW_NS, TICK_INT, HUD_MAX_TICKS, REST_NS
        )

        # Construct DF Snap
        df_snap_out = pd.DataFrame(snap_rows, columns=[
            "window_start_ts_ns", "window_end_ts_ns", "best_bid_price_int", "best_bid_qty",
            "best_ask_price_int", "best_ask_qty", "mid_price", "mid_price_int",
            "last_trade_price_int", "spot_ref_price_int", "book_valid"
        ])

        # Construct DF Wall
        df_wall_out = pd.DataFrame(wall_rows, columns=[
            "window_start_ts_ns", "window_end_ts_ns", "price_int", "side", "spot_ref_price_int",
            "rel_ticks", "depth_qty_start", "depth_qty_end", "add_qty", "pull_qty_total",
            "depth_qty_rest", "pull_qty_rest", "fill_qty", "window_valid"
        ])
        
        # Side mapping for output
        df_wall_out["side"] = df_wall_out["side"].map({1: "B", 0: "A"})
        
        # Derivatives
        if not df_wall_out.empty:
            df_wall_out = _calc_derivatives(df_wall_out)
        else:
            cols = ["window_start_ts_ns", "window_end_ts_ns", "price_int", "side", "spot_ref_price_int", "rel_ticks", 
                    "depth_qty_start", "depth_qty_end", "add_qty", "pull_qty_total", "depth_qty_rest", "pull_qty_rest", 
                    "fill_qty", "d1_depth_qty", "d2_depth_qty", "d3_depth_qty", "window_valid"]
            df_wall_out = pd.DataFrame(columns=cols)

        return df_snap_out, df_wall_out


@jit(nopython=True)
def _numba_compute_snapshot_wall(ts_arr, act_arr, side_arr, px_arr, sz_arr, oid_arr, flags_arr, window_ns, tick_int, hud_max_ticks, rest_ns):
    # Output Lists (Flat)
    # Snap
    s_start = NumbaList.empty_list(int64)
    s_end = NumbaList.empty_list(int64)
    s_bb_p = NumbaList.empty_list(int64)
    s_bb_q = NumbaList.empty_list(int64)
    s_ba_p = NumbaList.empty_list(int64)
    s_ba_q = NumbaList.empty_list(int64)
    s_mid = NumbaList.empty_list(float64)
    s_mid_i = NumbaList.empty_list(int64)
    s_last = NumbaList.empty_list(int64)
    s_spot = NumbaList.empty_list(int64)
    s_valid = NumbaList.empty_list(boolean)

    # Wall
    w_start = NumbaList.empty_list(int64)
    w_end = NumbaList.empty_list(int64)
    w_px = NumbaList.empty_list(int64)
    w_side = NumbaList.empty_list(int64)
    w_spot = NumbaList.empty_list(int64)
    w_rel = NumbaList.empty_list(int64)
    w_d_start = NumbaList.empty_list(float64)
    w_d_end = NumbaList.empty_list(float64)
    w_add = NumbaList.empty_list(float64)
    w_pull = NumbaList.empty_list(float64)
    w_rest = NumbaList.empty_list(float64)
    w_p_rest = NumbaList.empty_list(float64)
    w_fill = NumbaList.empty_list(float64)
    w_valid = NumbaList.empty_list(boolean)
    
    # State
    ord_side = NumbaDict.empty(int64, int8)
    ord_px = NumbaDict.empty(int64, int64)
    ord_qty = NumbaDict.empty(int64, int64)
    ord_ts = NumbaDict.empty(int64, int64)
    
    # Depth: Price -> Qty
    depth_b = NumbaDict.empty(int64, int64)
    depth_a = NumbaDict.empty(int64, int64)
    
    # Accumulators: Price -> Value (Separate Bid/Ask)
    # Using separate dicts avoids mixed keys
    acc_add_b = NumbaDict.empty(int64, float64)
    acc_add_a = NumbaDict.empty(int64, float64)
    
    acc_pull_b = NumbaDict.empty(int64, float64)
    acc_pull_a = NumbaDict.empty(int64, float64)
    
    acc_prest_b = NumbaDict.empty(int64, float64)
    acc_prest_a = NumbaDict.empty(int64, float64)
    
    acc_fill_b = NumbaDict.empty(int64, float64)
    acc_fill_a = NumbaDict.empty(int64, float64)
    
    curr_window = -1
    window_start = 0
    window_end = 0
    
    last_trade_price = 0
    book_valid = False
    in_snapshot_load = False
    
    n = len(ts_arr)
    for i in range(n):
        ts = ts_arr[i]
        w_id = ts // window_ns
        
        if curr_window == -1:
            curr_window = w_id
            window_start = w_id * window_ns
            window_end = window_start + window_ns
            
        if w_id != curr_window:
            # Emit
            # Best Bid
            bb_p = 0
            bb_q = 0
            if len(depth_b) > 0:
                max_p = -1
                for k in depth_b:
                     if max_p == -1 or k > max_p: max_p = k
                if max_p != -1:
                    bb_p = max_p
                    bb_q = depth_b[max_p]
            
            # Best Ask
            ba_p = 0
            ba_q = 0
            if len(depth_a) > 0:
                min_p = -1
                for k in depth_a:
                    if min_p == -1 or k < min_p: min_p = k
                if min_p != -1:
                    ba_p = min_p
                    ba_q = depth_a[min_p]
            
            mid = 0.0
            mid_i = 0
            if bb_p > 0 and ba_p > 0:
                mid = (bb_p + ba_p) * 0.5 * 1e-9
                mid_i = int(round((bb_p + ba_p) * 0.5))
            
            spot_ref = last_trade_price
            if spot_ref == 0:
                if book_valid and bb_p > 0:
                    spot_ref = bb_p
            
            # Snap Output
            s_start.append(window_start)
            s_end.append(window_end)
            s_bb_p.append(bb_p)
            s_bb_q.append(bb_q)
            s_ba_p.append(ba_p)
            s_ba_q.append(ba_q)
            s_mid.append(mid)
            s_mid_i.append(mid_i)
            s_last.append(last_trade_price)
            s_spot.append(spot_ref)
            s_valid.append(book_valid)
            
            # Wall Output
            if spot_ref > 0:
                min_p = spot_ref - hud_max_ticks * tick_int
                max_p = spot_ref + hud_max_ticks * tick_int
                
                # Collect Keys
                unique_b = NumbaList.empty_list(int64)
                unique_a = NumbaList.empty_list(int64)
                
                # Bids
                for k in depth_b:
                    if k >= min_p and k <= max_p: unique_b.append(k)
                for k in acc_add_b:
                    if k >= min_p and k <= max_p: unique_b.append(k)
                
                # Asks
                for k in depth_a:
                    if k >= min_p and k <= max_p: unique_a.append(k)
                for k in acc_add_a:
                    if k >= min_p and k <= max_p: unique_a.append(k)
                
                # Unique logic via temporary dict?
                # Or just iterate? Duplicates ok? 
                # If duplicates, we get multiple rows for same price/side.
                # Must dedupe.
                # Numba has no set() easily. Use Dict(int64, bool)
                
                seen_b = NumbaDict.empty(int64, boolean)
                seen_a = NumbaDict.empty(int64, boolean)
                
                for k in unique_b: seen_b[k] = True
                for k in unique_a: seen_a[k] = True

                # Rest Depth Calc
                res_depth_b = NumbaDict.empty(int64, float64)
                res_depth_a = NumbaDict.empty(int64, float64)
                
                for oid_k in ord_px:
                     opx = ord_px[oid_k]
                     if opx >= min_p and opx <= max_p:
                         ots = ord_ts[oid_k]
                         if (window_end - ots) >= rest_ns:
                             osd = ord_side[oid_k]
                             oqt = ord_qty[oid_k]
                             if osd == 1:
                                 if opx in res_depth_b: res_depth_b[opx] += oqt
                                 else: res_depth_b[opx] = oqt
                             else:
                                 if opx in res_depth_a: res_depth_a[opx] += oqt
                                 else: res_depth_a[opx] = oqt

                # Emit Bids
                for px in seen_b:
                    sd = 1
                    d_end = 0.0
                    if px in depth_b: d_end = float(depth_b[px])
                    
                    add_q = 0.0
                    if px in acc_add_b: add_q = acc_add_b[px]
                    pull_q = 0.0
                    if px in acc_pull_b: pull_q = acc_pull_b[px]
                    fill_q = 0.0
                    if px in acc_fill_b: fill_q = acc_fill_b[px]
                    prest_q = 0.0
                    if px in acc_prest_b: prest_q = acc_prest_b[px]
                    
                    d_start = d_end - add_q + pull_q + fill_q
                    if d_start < 0: d_start = 0.0
                    rel = int(round((px - spot_ref) / tick_int))
                    d_rest = 0.0
                    if px in res_depth_b: d_rest = res_depth_b[px]
                    
                    w_start.append(window_start)
                    w_end.append(window_end)
                    w_px.append(px)
                    w_side.append(sd)
                    # ... append rest
                    w_spot.append(spot_ref)
                    w_rel.append(rel)
                    w_d_start.append(d_start)
                    w_d_end.append(d_end)
                    w_add.append(add_q)
                    w_pull.append(pull_q)
                    w_rest.append(d_rest)
                    w_p_rest.append(prest_q)
                    w_fill.append(fill_q)
                    w_valid.append(book_valid)
                    
                # Emit Asks
                for px in seen_a:
                    sd = 0
                    d_end = 0.0
                    if px in depth_a: d_end = float(depth_a[px])
                    
                    add_q = 0.0
                    if px in acc_add_a: add_q = acc_add_a[px]
                    pull_q = 0.0
                    if px in acc_pull_a: pull_q = acc_pull_a[px]
                    fill_q = 0.0
                    if px in acc_fill_a: fill_q = acc_fill_a[px]
                    prest_q = 0.0
                    if px in acc_prest_a: prest_q = acc_prest_a[px]
                    
                    d_start = d_end - add_q + pull_q + fill_q
                    if d_start < 0: d_start = 0.0
                    rel = int(round((px - spot_ref) / tick_int))
                    d_rest = 0.0
                    if px in res_depth_a: d_rest = res_depth_a[px]
                    
                    w_start.append(window_start)
                    w_end.append(window_end)
                    w_px.append(px)
                    w_side.append(sd)
                    w_spot.append(spot_ref)
                    w_rel.append(rel)
                    w_d_start.append(d_start)
                    w_d_end.append(d_end)
                    w_add.append(add_q)
                    w_pull.append(pull_q)
                    w_rest.append(d_rest)
                    w_p_rest.append(prest_q)
                    w_fill.append(fill_q)
                    w_valid.append(book_valid)

            # Reset Accumulators
            acc_add_b = NumbaDict.empty(int64, float64)
            acc_add_a = NumbaDict.empty(int64, float64)
            acc_pull_b = NumbaDict.empty(int64, float64)
            acc_pull_a = NumbaDict.empty(int64, float64)
            acc_prest_b = NumbaDict.empty(int64, float64)
            acc_prest_a = NumbaDict.empty(int64, float64)
            acc_fill_b = NumbaDict.empty(int64, float64)
            acc_fill_a = NumbaDict.empty(int64, float64)
            
            curr_window = w_id
            window_start = curr_window * window_ns
            window_end = window_start + window_ns
            
        # Process Actions
        act = act_arr[i]
        oid = oid_arr[i]
        flg = flags_arr[i]
        
        if act == ACT_CLEAR:
             ord_side = NumbaDict.empty(int64, int8)
             ord_px = NumbaDict.empty(int64, int64)
             ord_qty = NumbaDict.empty(int64, int64)
             ord_ts = NumbaDict.empty(int64, int64)
             depth_b = NumbaDict.empty(int64, int64)
             depth_a = NumbaDict.empty(int64, int64)
             
             if (flg & F_SNAPSHOT) > 0:
                 in_snapshot_load = True
                 book_valid = False
             else:
                 in_snapshot_load = True
                 book_valid = False
             continue
             
        if in_snapshot_load:
            if (flg & F_LAST) > 0:
                in_snapshot_load = False
                book_valid = True
        
        if act == ACT_ADD:
            sd = side_arr[i]
            px = px_arr[i]
            sz = sz_arr[i]
            
            ord_side[oid] = sd
            ord_px[oid] = px
            ord_qty[oid] = sz
            ord_ts[oid] = ts
            
            if sd == 1:
                if px in depth_b: depth_b[px] += sz
                else: depth_b[px] = sz
                if px in acc_add_b: acc_add_b[px] += sz
                else: acc_add_b[px] = sz
            else:
                if px in depth_a: depth_a[px] += sz
                else: depth_a[px] = sz
                if px in acc_add_a: acc_add_a[px] += sz
                else: acc_add_a[px] = sz
            
        elif act == ACT_MODIFY:
             if oid in ord_px:
                 o_sd = ord_side[oid]
                 o_px = ord_px[oid]
                 o_qt = ord_qty[oid]
                 o_ts = ord_ts[oid]
                 
                 n_px = px_arr[i]
                 n_sz = sz_arr[i]
                 
                 # Remove Old
                 if o_sd == 1:
                     if o_px in depth_b:
                         depth_b[o_px] -= o_qt
                         if depth_b[o_px] <= 0: del depth_b[o_px]
                 else:
                     if o_px in depth_a:
                         depth_a[o_px] -= o_qt
                         if depth_a[o_px] <= 0: del depth_a[o_px]
                 
                 # Add New
                 if o_sd == 1:
                     if n_px in depth_b: depth_b[n_px] += n_sz
                     else: depth_b[n_px] = n_sz
                 else:
                     if n_px in depth_a: depth_a[n_px] += n_sz
                     else: depth_a[n_px] = n_sz
                     
                 # Update Order
                 ord_px[oid] = n_px
                 ord_qty[oid] = n_sz
                 if n_px == o_px: ord_ts[oid] = o_ts
                 else: ord_ts[oid] = ts
                 
                 # Accums
                 if n_px != o_px:
                      # Pull Full Old
                      if o_sd == 1:
                          if o_px in acc_pull_b: acc_pull_b[o_px] += o_qt
                          else: acc_pull_b[o_px] = o_qt
                          if (ts - o_ts) >= rest_ns:
                              if o_px in acc_prest_b: acc_prest_b[o_px] += o_qt
                              else: acc_prest_b[o_px] = o_qt
                          # Add Full New
                          if n_px in acc_add_b: acc_add_b[n_px] += n_sz
                          else: acc_add_b[n_px] = n_sz
                      else:
                          if o_px in acc_pull_a: acc_pull_a[o_px] += o_qt
                          else: acc_pull_a[o_px] = o_qt
                          if (ts - o_ts) >= rest_ns:
                              if o_px in acc_prest_a: acc_prest_a[o_px] += o_qt
                              else: acc_prest_a[o_px] = o_qt
                          if n_px in acc_add_a: acc_add_a[n_px] += n_sz
                          else: acc_add_a[n_px] = n_sz
                 else:
                      # Size change
                      if n_sz < o_qt:
                          delta = o_qt - n_sz
                          if o_sd == 1:
                              if o_px in acc_pull_b: acc_pull_b[o_px] += delta
                              else: acc_pull_b[o_px] = delta
                              if (ts - o_ts) >= rest_ns:
                                  if o_px in acc_prest_b: acc_prest_b[o_px] += delta
                                  else: acc_prest_b[o_px] = delta
                          else:
                              if o_px in acc_pull_a: acc_pull_a[o_px] += delta
                              else: acc_pull_a[o_px] = delta
                              if (ts - o_ts) >= rest_ns:
                                  if o_px in acc_prest_a: acc_prest_a[o_px] += delta
                                  else: acc_prest_a[o_px] = delta
                      elif n_sz > o_qt:
                          delta = n_sz - o_qt
                          if o_sd == 1:
                              if o_px in acc_add_b: acc_add_b[o_px] += delta
                              else: acc_add_b[o_px] = delta
                          else:
                              if o_px in acc_add_a: acc_add_a[o_px] += delta
                              else: acc_add_a[o_px] = delta
                          
        elif act == ACT_CANCEL:
             if oid in ord_px:
                 o_sd = ord_side[oid]
                 o_px = ord_px[oid]
                 o_qt = ord_qty[oid]
                 o_ts = ord_ts[oid]
                 
                 if o_sd == 1:
                     if o_px in depth_b:
                         depth_b[o_px] -= o_qt
                         if depth_b[o_px] <= 0: del depth_b[o_px]
                     # Accum
                     if o_px in acc_pull_b: acc_pull_b[o_px] += o_qt
                     else: acc_pull_b[o_px] = o_qt
                     if (ts - o_ts) >= rest_ns:
                          if o_px in acc_prest_b: acc_prest_b[o_px] += o_qt
                          else: acc_prest_b[o_px] = o_qt
                 else:
                     if o_px in depth_a:
                         depth_a[o_px] -= o_qt
                         if depth_a[o_px] <= 0: del depth_a[o_px]
                     # Accum
                     if o_px in acc_pull_a: acc_pull_a[o_px] += o_qt
                     else: acc_pull_a[o_px] = o_qt
                     if (ts - o_ts) >= rest_ns:
                          if o_px in acc_prest_a: acc_prest_a[o_px] += o_qt
                          else: acc_prest_a[o_px] = o_qt
                 
                 del ord_px[oid]
                 del ord_side[oid]
                 del ord_qty[oid]
                 del ord_ts[oid]
        
        elif act == ACT_FILL:
             if oid in ord_px:
                 o_sd = ord_side[oid]
                 o_px = ord_px[oid]
                 o_qt = ord_qty[oid]
                 
                 f_sz = sz_arr[i]
                 rem_sz = o_qt - f_sz
                 
                 if o_sd == 1:
                     if o_px in depth_b:
                         depth_b[o_px] -= f_sz
                         if depth_b[o_px] <= 0: del depth_b[o_px]
                     # Accum
                     if o_px in acc_fill_b: acc_fill_b[o_px] += f_sz
                     else: acc_fill_b[o_px] = f_sz
                 else:
                     if o_px in depth_a:
                         depth_a[o_px] -= f_sz
                         if depth_a[o_px] <= 0: del depth_a[o_px]
                     # Accum
                     if o_px in acc_fill_a: acc_fill_a[o_px] += f_sz
                     else: acc_fill_a[o_px] = f_sz
                 
                 if rem_sz <= 0:
                     del ord_px[oid]
                     del ord_side[oid]
                     del ord_qty[oid]
                     del ord_ts[oid]
                 else:
                     ord_qty[oid] = rem_sz
                     
        elif act == ACT_TRADE:
             px = px_arr[i]
             if px > 0:
                 last_trade_price = px
                 
    # Last Flush logic
    if curr_window != -1:
         # Best Bid
            bb_p = 0
            bb_q = 0
            if len(depth_b) > 0:
                max_p = -1
                for k in depth_b:
                     if max_p == -1 or k > max_p: max_p = k
                if max_p != -1:
                    bb_p = max_p
                    bb_q = depth_b[max_p]
            
            # Best Ask
            ba_p = 0
            ba_q = 0
            if len(depth_a) > 0:
                min_p = -1
                for k in depth_a:
                    if min_p == -1 or k < min_p: min_p = k
                if min_p != -1:
                    ba_p = min_p
                    ba_q = depth_a[min_p]
            
            mid = 0.0
            mid_i = 0
            if bb_p > 0 and ba_p > 0:
                mid = (bb_p + ba_p) * 0.5 * 1e-9
                mid_i = int(round((bb_p + ba_p) * 0.5))
            
            spot_ref = last_trade_price
            if spot_ref == 0:
                if book_valid and bb_p > 0:
                    spot_ref = bb_p
            
            s_start.append(window_start)
            s_end.append(window_end)
            s_bb_p.append(bb_p)
            s_bb_q.append(bb_q)
            s_ba_p.append(ba_p)
            s_ba_q.append(ba_q)
            s_mid.append(mid)
            s_mid_i.append(mid_i)
            s_last.append(last_trade_price)
            s_spot.append(spot_ref)
            s_valid.append(book_valid)
            
            if spot_ref > 0:
                min_p = spot_ref - hud_max_ticks * tick_int
                max_p = spot_ref + hud_max_ticks * tick_int
                
                unique_b = NumbaList.empty_list(int64)
                unique_a = NumbaList.empty_list(int64)
                for k in depth_b:
                    if k >= min_p and k <= max_p: unique_b.append(k)
                for k in acc_add_b:
                    if k >= min_p and k <= max_p: unique_b.append(k)
                for k in depth_a:
                    if k >= min_p and k <= max_p: unique_a.append(k)
                for k in acc_add_a:
                    if k >= min_p and k <= max_p: unique_a.append(k)
                
                seen_b = NumbaDict.empty(int64, boolean)
                seen_a = NumbaDict.empty(int64, boolean)
                for k in unique_b: seen_b[k] = True
                for k in unique_a: seen_a[k] = True

                res_depth_b = NumbaDict.empty(int64, float64)
                res_depth_a = NumbaDict.empty(int64, float64)
                for oid_k in ord_px:
                     opx = ord_px[oid_k]
                     if opx >= min_p and opx <= max_p:
                         ots = ord_ts[oid_k]
                         if (window_end - ots) >= rest_ns:
                             osd = ord_side[oid_k]
                             oqt = ord_qty[oid_k]
                             if osd == 1:
                                 if opx in res_depth_b: res_depth_b[opx] += oqt
                                 else: res_depth_b[opx] = oqt
                             else:
                                 if opx in res_depth_a: res_depth_a[opx] += oqt
                                 else: res_depth_a[opx] = oqt

                # Emit Bids
                for px in seen_b:
                    sd = 1
                    d_end = 0.0
                    if px in depth_b: d_end = float(depth_b[px])
                    add_q = 0.0
                    if px in acc_add_b: add_q = acc_add_b[px]
                    pull_q = 0.0
                    if px in acc_pull_b: pull_q = acc_pull_b[px]
                    fill_q = 0.0
                    if px in acc_fill_b: fill_q = acc_fill_b[px]
                    prest_q = 0.0
                    if px in acc_prest_b: prest_q = acc_prest_b[px]
                    
                    d_start = d_end - add_q + pull_q + fill_q
                    if d_start < 0: d_start = 0.0
                    rel = int(round((px - spot_ref) / tick_int))
                    d_rest = 0.0
                    if px in res_depth_b: d_rest = res_depth_b[px]
                    
                    w_start.append(window_start)
                    w_end.append(window_end)
                    w_px.append(px)
                    w_side.append(sd)
                    w_spot.append(spot_ref)
                    w_rel.append(rel)
                    w_d_start.append(d_start)
                    w_d_end.append(d_end)
                    w_add.append(add_q)
                    w_pull.append(pull_q)
                    w_rest.append(d_rest)
                    w_p_rest.append(prest_q)
                    w_fill.append(fill_q)
                    w_valid.append(book_valid)
                    
                # Emit Asks
                for px in seen_a:
                    sd = 0
                    d_end = 0.0
                    if px in depth_a: d_end = float(depth_a[px])
                    add_q = 0.0
                    if px in acc_add_a: add_q = acc_add_a[px]
                    pull_q = 0.0
                    if px in acc_pull_a: pull_q = acc_pull_a[px]
                    fill_q = 0.0
                    if px in acc_fill_a: fill_q = acc_fill_a[px]
                    prest_q = 0.0
                    if px in acc_prest_a: prest_q = acc_prest_a[px]
                    d_start = d_end - add_q + pull_q + fill_q
                    if d_start < 0: d_start = 0.0
                    rel = int(round((px - spot_ref) / tick_int))
                    d_rest = 0.0
                    if px in res_depth_a: d_rest = res_depth_a[px]
                    w_start.append(window_start)
                    w_end.append(window_end)
                    w_px.append(px)
                    w_side.append(sd)
                    w_spot.append(spot_ref)
                    w_rel.append(rel)
                    w_d_start.append(d_start)
                    w_d_end.append(d_end)
                    w_add.append(add_q)
                    w_pull.append(pull_q)
                    w_rest.append(d_rest)
                    w_p_rest.append(prest_q)
                    w_fill.append(fill_q)
                    w_valid.append(book_valid)

    snap_rows = []
    for i in range(len(s_start)):
        snap_rows.append((s_start[i], s_end[i], s_bb_p[i], s_bb_q[i], s_ba_p[i], s_ba_q[i], s_mid[i], s_mid_i[i], s_last[i], s_spot[i], s_valid[i]))
        
    wall_rows = []
    for i in range(len(w_start)):
        wall_rows.append((w_start[i], w_end[i], w_px[i], w_side[i], w_spot[i], w_rel[i], w_d_start[i], w_d_end[i], w_add[i], w_pull[i], w_rest[i], w_p_rest[i], w_fill[i], w_valid[i]))

    return snap_rows, wall_rows


def _calc_derivatives(df):
    df = df.sort_values(["side", "price_int", "window_end_ts_ns"])
    groups = df.groupby(["side", "price_int"])["depth_qty_end"]
    df["d1_depth_qty"] = groups.diff().fillna(0.0)
    df["d2_depth_qty"] = df.groupby(["side", "price_int"])["d1_depth_qty"].diff().fillna(0.0)
    df["d3_depth_qty"] = df.groupby(["side", "price_int"])["d2_depth_qty"].diff().fillna(0.0)
    return df
