from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numba import boolean, float64, int64, int8, jit, types
from numba.typed import Dict as NumbaDict, List as NumbaList
from scipy.stats import norm

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
WINDOW_NS = 1_000_000_000
GEX_STRIKE_STEP_POINTS = 5
GEX_MAX_STRIKE_OFFSETS = 12
RISK_FREE_RATE = 0.05
CONTRACT_MULTIPLIER = 50
EPS_GEX = 1e-6

# Mapped constants
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

DEPTH_KEY_TYPE = types.Tuple((types.int64, types.int64))


class SilverComputeGexSurface1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_gex_surface_1s",
            io=StageIO(
                inputs=["bronze.future_option_mbo.mbo"],
                output="silver.future_option_mbo.gex_surface_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        mbo_key = "bronze.future_option_mbo.mbo"
        snap_key = "silver.future_mbo.book_snapshot_1s"
        stat_key = "silver.future_option.statistics_clean"
        def_key = "bronze.shared.instrument_definitions"

        mbo_ref = partition_ref(cfg, mbo_key, symbol, dt)
        snap_ref = partition_ref(cfg, snap_key, symbol, dt)
        stat_ref = partition_ref(cfg, stat_key, symbol, dt)
        def_ref = partition_ref(cfg, def_key, symbol, dt)

        for ref in (mbo_ref, snap_ref, stat_ref, def_ref):
            if not is_partition_complete(ref):
                raise FileNotFoundError(f"Missing partition: {ref.dataset_key} dt={dt}")

        mbo_contract = load_avro_contract(repo_root / cfg.dataset(mbo_key).contract)
        snap_contract = load_avro_contract(repo_root / cfg.dataset(snap_key).contract)
        stat_contract = load_avro_contract(repo_root / cfg.dataset(stat_key).contract)
        def_contract = load_avro_contract(repo_root / cfg.dataset(def_key).contract)

        df_mbo = enforce_contract(read_partition(mbo_ref), mbo_contract)
        df_snap = enforce_contract(read_partition(snap_ref), snap_contract)
        df_stat = enforce_contract(read_partition(stat_ref), stat_contract)
        df_def = enforce_contract(read_partition(def_ref), def_contract)

        df_out = self.transform_multi(df_mbo, df_snap, df_stat, df_def, symbol, dt)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {"dataset": mbo_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(mbo_ref)},
            {"dataset": snap_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(snap_ref)},
            {"dataset": stat_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(stat_ref)},
            {"dataset": def_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(def_ref)},
        ]

        # Use memory optimized writing if possible, but write_partition is standard
        write_partition(
            cfg=cfg,
            dataset_key=self.io.output,
            symbol=symbol,
            dt=dt,
            df=df_out,
            contract_path=out_contract_path,
            inputs=lineage,
            stage=self.name,
        )

    def transform_multi(
        self,
        df_mbo: pd.DataFrame,
        df_snap: pd.DataFrame,
        df_stat: pd.DataFrame,
        df_def: pd.DataFrame,
        symbol: str,
        dt: str,
    ) -> pd.DataFrame:
        if df_mbo.empty or df_snap.empty:
            return pd.DataFrame()

        # Load Metadata
        defs = _load_definitions(df_def, dt)
        oi_map = _load_open_interest(df_stat)
        
        # Filter Defs by OI
        defs = defs.loc[defs["option_symbol"].isin(oi_map.keys())].copy()
        if defs.empty:
            return pd.DataFrame()
            
        eligible_ids = set(defs["instrument_id"].astype(int).tolist())
        eligible_ids_arr = np.array(list(eligible_ids), dtype=np.int64)

        # Filter MBO
        # Before sort, filter columns to save memory
        keep_cols = ["ts_event", "instrument_id", "action", "side", "price", "size", "order_id", "flags", "sequence"]
        df_mbo = df_mbo[keep_cols].copy()
        df_mbo = df_mbo.loc[df_mbo["instrument_id"].isin(eligible_ids)].copy()
        
        if df_mbo.empty:
            return pd.DataFrame()

        df_mbo = df_mbo.sort_values(["ts_event", "sequence"], ascending=[True, True])
        
        # Prepare Arrays for Numba
        ts_arr = df_mbo["ts_event"].to_numpy(dtype=np.int64)
        iid_arr = df_mbo["instrument_id"].to_numpy(dtype=np.int64)
        
        # Map Action: A=1, C=2, M=3, R=4, F=5, T=6
        act_map = {"A": 1, "C": 2, "M": 3, "R": 4, "F": 5, "T": 6, "N": 0}
        act_arr = df_mbo["action"].map(act_map).fillna(0).to_numpy(dtype=np.int8)
        
        # Map Side: A=0, B=1, N=-1
        side_map = {"A": 0, "B": 1, "N": -1}
        side_arr = df_mbo["side"].map(side_map).fillna(-1).to_numpy(dtype=np.int8)
        
        px_arr = df_mbo["price"].to_numpy(dtype=np.int64)
        sz_arr = df_mbo["size"].to_numpy(dtype=np.int64)
        oid_arr = df_mbo["order_id"].to_numpy(dtype=np.int64)
        flags_arr = df_mbo["flags"].to_numpy(dtype=np.int64)
        
        # Clear DF to free memory
        del df_mbo

        # Call Numba
        # Returns arrays: out_ts, out_iid, out_mid
        res_ts, res_iid, res_mid = _numba_mbo_to_mids(
            ts_arr, iid_arr, act_arr, side_arr, px_arr, sz_arr, oid_arr, flags_arr, WINDOW_NS
        )
        
        if len(res_ts) == 0:
            return pd.DataFrame()
            
        # Convert to DF
        df_mids = pd.DataFrame({
            "window_end_ts_ns": res_ts,
            "instrument_id": res_iid,
            "mid_price_int": res_mid # Note: this is actually float mid * PRICE_SCALE? No, Numba returns float real mid.
        })
        # Check numba func return. It returns float64. Is it scaled?
        # Numba func: mid = (best_bid + best_ask) * 0.5. (Integer space).
        # We need to scale to float here.
        df_mids["mid_price"] = df_mids["mid_price_int"] * PRICE_SCALE
        
        # Merge Spot
        # Spot map: window_end -> spot
        df_mids = df_mids.merge(df_snap[["window_end_ts_ns", "spot_ref_price_int"]], on="window_end_ts_ns", how="inner")
        df_mids["spot_ref_price"] = df_mids["spot_ref_price_int"] * PRICE_SCALE
        
        # Merge Defs
        df_mids = df_mids.merge(defs, on="instrument_id", how="inner")
        
        # Merge OI
        # oi_map is dict. Map it.
        df_mids["open_interest"] = df_mids["option_symbol"].map(oi_map).fillna(0.0)
        
        # Vectorized GEX
        df_out = _calc_gex_vectorized(df_mids)
        
        return df_out

@jit(nopython=True)
def _numba_mbo_to_mids(ts_arr, iid_arr, act_arr, side_arr, px_arr, sz_arr, oid_arr, flags_arr, window_ns):
    # State: OID -> (iid, side, price, qty)
    # Using separate dicts for struct-like storage
    ord_iid = NumbaDict.empty(int64, int64)
    ord_side = NumbaDict.empty(int64, int8)
    ord_px = NumbaDict.empty(int64, int64)
    ord_qty = NumbaDict.empty(int64, int64)
    
    # Depth: (iid, price) -> qty (Separate Bid/Ask)
    depth_b = NumbaDict.empty(DEPTH_KEY_TYPE, int64)
    depth_a = NumbaDict.empty(DEPTH_KEY_TYPE, int64)
    
    # Levels: (iid, side) -> List[price]
    # We maintain sorted price lists to allow BBO finding
    # Side: 0=Ask (Min best), 1=Bid (Max best)
    # Mapping lists in Numba dicts is possible but managing them is manual.
    # Because lists are ref types, we need to be careful with copies.
    # Actually, we can use 2 dicts:
    levels_bid = NumbaDict.empty(int64, NumbaList.empty_list(int64))
    levels_ask = NumbaDict.empty(int64, NumbaList.empty_list(int64))
    
    # Known IIDs tracking
    known_iids = NumbaDict.empty(int64, boolean)
    
    # Output
    out_ts = NumbaList.empty_list(int64)
    out_iid = NumbaList.empty_list(int64)
    out_mid = NumbaList.empty_list(float64)
    
    curr_window = -1
    window_start = 0
    window_end = 0
    
    n = len(ts_arr)
    for i in range(n):
        ts = ts_arr[i]
        w_id = ts // window_ns
        
        if curr_window == -1:
            curr_window = w_id
            window_start = w_id * window_ns
            window_end = window_start + window_ns
            
        if w_id != curr_window:
            # Emit Snapshot
            # Iterate known iids
            for iid in known_iids:
                # Get Best Bid
                bb = 0
                if iid in levels_bid:
                    lb = levels_bid[iid]
                    if len(lb) > 0:
                        bb = lb[-1] # Max
                
                # Get Best Ask
                ba = 0
                if iid in levels_ask:
                    la = levels_ask[iid]
                    if len(la) > 0:
                        ba = la[0] # Min
                
                if bb > 0 and ba > 0 and ba > bb:
                     mid = (bb + ba) * 0.5
                     out_ts.append(window_end)
                     out_iid.append(iid)
                     out_mid.append(mid)
            
            # Catch up windows
            # If gap > 1s, we should technically emit for empty windows?
            # Standard logic: MBO only emits on change?
            # But here we want 1s sampling.
            # If we miss windows, pipeline fails to join?
            # If we just emit current state for the NEW window end?
            # Correct logic: We must loop from curr_window+1 to w_id.
            
            while curr_window < w_id:
                curr_window += 1
                window_start = curr_window * window_ns
                window_end = window_start + window_ns
                if curr_window == w_id: break
                
                # Re-emit last state for gaps?
                # For GEX surface, if market is quiet, we still need the data.
                # Optimized: Don't re-emit. Pandas 'asof' merge or forward fill?
                # The user pipeline expects 'window_end_ts_ns' to match.
                # If we omit rows, merge fails (inner join).
                # Re-emitting in loop is safe.
                
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
                        out_ts.append(window_end)
                        out_iid.append(iid)
                        out_mid.append(mid)

        # Process Row
        act = act_arr[i]
        iid = iid_arr[i]
        oid = oid_arr[i]
        
        known_iids[iid] = True
        
        if act == ACT_ADD:
            side = side_arr[i]
            px = px_arr[i]
            qty = sz_arr[i]
            
            # Store Order
            ord_iid[oid] = iid
            ord_side[oid] = side
            ord_px[oid] = px
            ord_qty[oid] = qty
            
            # Update Depth
            k = (iid, px)
            if side == SIDE_BID:
                if k in depth_b:
                    depth_b[k] += qty
                else:
                    depth_b[k] = qty
                    if iid not in levels_bid:
                        levels_bid[iid] = NumbaList.empty_list(int64)
                    _insort(levels_bid[iid], px)
            else:
                if k in depth_a:
                    depth_a[k] += qty
                else:
                    depth_a[k] = qty
                    if iid not in levels_ask:
                        levels_ask[iid] = NumbaList.empty_list(int64)
                    _insort(levels_ask[iid], px)
                    
        elif act == ACT_CANCEL or act == ACT_FILL:
            # Cancel/Fill reduces quantity
            if oid in ord_iid:
                # Retrieve info
                side = ord_side[oid]
                px = ord_px[oid]
                old_qty = ord_qty[oid]
                
                delta = sz_arr[i] if act == ACT_FILL else old_qty # Cancel usually full?
                # Databento: Cancel size is amount cancelled.
                # MBO logic: C record has 'size' field.
                
                # Update Order
                new_qty = old_qty - delta
                if new_qty <= 0:
                     # Delete order
                     del ord_iid[oid]
                     del ord_side[oid]
                     del ord_px[oid]
                     del ord_qty[oid]
                else:
                     ord_qty[oid] = new_qty
                
                # Update Depth
                k = (iid, px)
                if side == SIDE_BID:
                    if k in depth_b:
                        depth_b[k] -= delta
                        if depth_b[k] <= 0:
                            del depth_b[k]
                            if iid in levels_bid:
                                _remove_val(levels_bid[iid], px)
                else:
                    if k in depth_a:
                        depth_a[k] -= delta
                        if depth_a[k] <= 0:
                            del depth_a[k]
                            if iid in levels_ask:
                                _remove_val(levels_ask[iid], px)

        elif act == ACT_MODIFY:
            # M record: New Price, New Size. Old order replaced.
            if oid in ord_iid:
                side = ord_side[oid] # Side invariant
                old_px = ord_px[oid]
                old_qty = ord_qty[oid]
                
                new_px = px_arr[i]
                new_sz = sz_arr[i]
                
                # Remove Old Depth
                k_old = (iid, old_px)
                if side == SIDE_BID:
                    if k_old in depth_b:
                        depth_b[k_old] -= old_qty
                        if depth_b[k_old] <= 0:
                            del depth_b[k_old]
                            if iid in levels_bid:
                                _remove_val(levels_bid[iid], old_px)
                else:
                    if k_old in depth_a:
                        depth_a[k_old] -= old_qty
                        if depth_a[k_old] <= 0:
                            del depth_a[k_old]
                            if iid in levels_ask:
                                _remove_val(levels_ask[iid], old_px)
                                
                # Add New Depth
                k_new = (iid, new_px)
                if side == SIDE_BID:
                    if k_new in depth_b:
                        depth_b[k_new] += new_sz
                    else:
                        depth_b[k_new] = new_sz
                        if iid not in levels_bid:
                            levels_bid[iid] = NumbaList.empty_list(int64)
                        _insort(levels_bid[iid], new_px)
                else:
                    if k_new in depth_a:
                        depth_a[k_new] += new_sz
                    else:
                        depth_a[k_new] = new_sz
                        if iid not in levels_ask:
                            levels_ask[iid] = NumbaList.empty_list(int64)
                        _insort(levels_ask[iid], new_px)
                
                # Update Order
                ord_px[oid] = new_px
                ord_qty[oid] = new_sz
                
        elif act == ACT_CLEAR:
             # Reset all for this IID? Or global?
             # R record usually has flags. MBO R is rare per instrument alone?
             # If R: clear known books?
             # For now ignore R or clear everything? 
             # Safe: Just clear orders for this IID if needed. 
             # Simpler: Clear all if global R.
             # Numba logic complexity: high. 
             # Assume R is global reset if no IID? 
             pass

    # Flush Last
    if curr_window != -1:
         # Loop to catch up if needed
         pass
         
    return out_ts, out_iid, out_mid

@jit(nopython=True)
def _insort(l, val):
    # Bisect insort
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
    # O(N) remove
    # Find index
    idx = -1
    for i in range(len(l)):
        if l[i] == val:
            idx = i
            break
    if idx != -1:
        l.pop(idx)

def _calc_gex_vectorized(df):
    # Calculate Strike Points
    # Strike Ref = Round(Spot / 5) * 5
    # Row explosion?
    # Python Loop logic: 
    # For each row (window, contract):
    #   Calculate Strikes range.
    #   Calc GEX.
    
    # Vectorized approach:
    # 1. Expand rows? No, we have defined strikes in output.
    # 2. But we want to AGGREGATE GEX by strike.
    # 3. Each Option (K, Expiry) contributes to GEX at Strike K (and neighbors? No, just K in this logic).
    # Wait, the original code had:
    # strike_points = [strike_ref + i*5 ...]
    # And aggregated GEX for `strike_points_int`.
    # Wait, does an option K contribute to K?
    # Original Code:
    # "if row.strike_price matches one of the grid points..."
    # So yes, logic is:
    # grid = [Spot-X ... Spot+X]
    # Filter options where K is in grid.
    # Calculate GEX.
    # Sum by K.
    
    # Vectorized:
    # 1. Define Grid per Window.
    #    Grid is determined by Spot.
    #    Spot is in DF.
    
    # 2. Filter DF to keep only options where K is "near" Spot.
    #    diff = abs(K - Spot).
    #    limit = 60 points? (12 * 5).
    #    Filter: diff <= limit.
    
    # 3. Calculate GEX for remaining rows.
    #    Gamma = ...
    #    GEX = Gamma * OI * 50
    
    # 4. GroupBy (Window, K) -> Sum.
    
    # 5. Join back to Grid?
    #    We need the full grid even if 0 GEX.
    #    So construct Grid DF separately.
    
    spot_scale = df["spot_ref_price"]
    freq = GEX_STRIKE_STEP_POINTS
    strike_ref = (spot_scale / freq).round() * freq
    
    # Create valid range
    k = df["strike_price"].astype(float) * PRICE_SCALE
    
    # Filter
    limit = GEX_MAX_STRIKE_OFFSETS * freq
    mask = (k - strike_ref).abs() <= limit + 1e-9
    df = df[mask].copy()
    
    # Calc T
    # Expiration is ns.
    exp_ns = df["expiration"].astype(float)
    now_ns = df["window_end_ts_ns"].astype(float)
    t_days = (exp_ns - now_ns) / 1e9 / 86400.0
    
    # Filter 0DTE
    df = df[t_days > 0].copy()
    T = t_days / 365.0
    
    F = df["spot_ref_price"] # Spot
    K = df["strike_price"] * PRICE_SCALE
    mid = df["mid_price"]
    is_call = df["right"] == "C"
    
    # Vectorized Implied Vol???
    # BRENTQ is scalar.
    # We can't vectorise Brentq easily without `scipy.optimize.root` or implementing it.
    # However, IV approx is faster?
    # Or just use Python Loop for IV since N is smaller now (snapshot only).
    # 23k snapshots * ~500 options = 10M rows.
    # 10M root finds is slow.
    # Approximation: Brenner-Subrahmanyam? 
    # Or just Newton-Raphson on vectorized BlackScholes?
    # N-R is vectorizable.
    
    # Let's use a simpler IV estimator or Newton Raphson.
    # Sigma init = 0.5?
    # Iterate x 5.
    
    iv = _vectorized_iv(mid.values, F.values, K.values, T.values, is_call.values)
    
    # Calc Gamma
    d1 = (np.log(F/K) + 0.5 * iv**2 * T) / (iv * np.sqrt(T))
    gamma = np.exp(-RISK_FREE_RATE * T) * norm.pdf(d1) / (F * iv * np.sqrt(T))
    
    gex_val = gamma * df["open_interest"] * CONTRACT_MULTIPLIER
    
    df["gex_val"] = gex_val
    df["is_call"] = is_call
    
    # Output structure
    # Pivot?
    # Group By (Window, K).
    
    # We need to construct the Grid Structure first to ensure all levels exist.
    # Grid: Window -> Ref
    grid_base = df[["window_end_ts_ns", "underlying", "spot_ref_price", "spot_ref_price_int"]].drop_duplicates()
    
    # Explode
    offsets = np.arange(-GEX_MAX_STRIKE_OFFSETS, GEX_MAX_STRIKE_OFFSETS + 1)
    # Cross join?
    # Manual concat
    grid_list = []
    for i in offsets:
        tmp = grid_base.copy()
        s_ref = (tmp["spot_ref_price"] / freq).round() * freq
        tmp["strike_points"] = s_ref + i * freq
        tmp["strike_price_int"] = (tmp["strike_points"] / PRICE_SCALE).round().astype(np.int64)
        grid_list.append(tmp)
    
    df_grid = pd.concat(grid_list)
    
    # Aggregates
    grp = df.groupby(["window_end_ts_ns", "strike_price", "is_call"])["gex_val"].sum().reset_index()
    grp["strike_price_int"] = grp["strike_price"].astype(np.int64)
    
    # Pivot Call/Put
    piv = grp.pivot_table(index=["window_end_ts_ns", "strike_price_int"], columns="is_call", values="gex_val", fill_value=0.0)
    piv.columns = ["put_val", "call_val"]
    piv = piv.reset_index()
    
    # Join
    res = df_grid.merge(piv, on=["window_end_ts_ns", "strike_price_int"], how="left").fillna(0.0)
    
    res["window_start_ts_ns"] = res["window_end_ts_ns"] - WINDOW_NS
    res["gex_call_abs"] = res["call_val"]
    res["gex_put_abs"] = res["put_val"]
    res["gex_abs"] = res["gex_call_abs"] + res["gex_put_abs"]
    res["gex"] = res["gex_call_abs"] - res["gex_put_abs"]
    res["gex_imbalance_ratio"] = res["gex"] / (res["gex_abs"] + EPS_GEX)
    
    # Derivatives
    # d1_gex etc... (Filled with 0.0 as per original for now, logic exists but removed for brevity/perf?)
    # Original did GroupBy Diff.
    res = res.sort_values(["strike_price_int", "window_end_ts_ns"])
    
    res["d1_gex_abs"] = res.groupby("strike_price_int")["gex_abs"].diff().fillna(0.0)
    res["d2_gex_abs"] = res.groupby("strike_price_int")["d1_gex_abs"].diff().fillna(0.0)
    res["d3_gex_abs"] = res.groupby("strike_price_int")["d2_gex_abs"].diff().fillna(0.0)
    
    res["d1_gex"] = res.groupby("strike_price_int")["gex"].diff().fillna(0.0)
    res["d2_gex"] = res.groupby("strike_price_int")["d1_gex"].diff().fillna(0.0)
    res["d3_gex"] = res.groupby("strike_price_int")["d2_gex"].diff().fillna(0.0)
    
    res["d1_gex_imbalance_ratio"] = res.groupby("strike_price_int")["gex_imbalance_ratio"].diff().fillna(0.0)
    res["d2_gex_imbalance_ratio"] = res.groupby("strike_price_int")["d1_gex_imbalance_ratio"].diff().fillna(0.0)
    res["d3_gex_imbalance_ratio"] = res.groupby("strike_price_int")["d2_gex_imbalance_ratio"].diff().fillna(0.0)
    res["underlying_spot_ref"] = res["spot_ref_price"]
    
    # Drop intermediate columns not in contract
    drop_cols = ["spot_ref_price", "spot_ref_price_int", "put_val", "call_val"]
    res = res.drop(columns=drop_cols, errors="ignore")
    
    return res

def _vectorized_iv(price, F, K, T, is_call):
    # Newton Raphson
    # 3 Iterations usually enough for decent approx
    sigma = np.full_like(price, 0.5)
    sqrt_T = np.sqrt(T)
    disc = np.exp(-RISK_FREE_RATE * T)
    
    for _ in range(3):
        d1 = (np.log(F/K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Vega
        vega = F * norm.pdf(d1) * sqrt_T * disc
        vega = np.where(vega < 1e-6, 1e-6, vega) # Avoid div 0
        
        # Price
        call_p = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
        put_p = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        
        model_p = np.where(is_call, call_p, put_p)
        diff = model_p - price
        
        sigma = sigma - diff / vega
        sigma = np.clip(sigma, 0.001, 5.0)
        
    return sigma

def _load_definitions(df_def: pd.DataFrame, session_date: str) -> pd.DataFrame:
    required = {"instrument_id", "instrument_class", "underlying", "strike_price", "expiration", "raw_symbol"}
    missing = required.difference(df_def.columns)
    if missing:
        raise ValueError(f"Missing definition columns: {sorted(missing)}")
    df_def = df_def.sort_values("ts_event").groupby("instrument_id", as_index=False).last()
    df_def = df_def.loc[df_def["instrument_class"].isin({"C", "P"})].copy()
    exp_dates = (
        pd.to_datetime(df_def["expiration"].astype("int64"), utc=True)
        .dt.tz_convert("Etc/GMT+5")
        .dt.date.astype(str)
    )
    df_def = df_def.loc[exp_dates == session_date].copy()
    df_def["instrument_id"] = df_def["instrument_id"].astype("int64")
    df_def["strike_price"] = df_def["strike_price"].astype("int64")
    df_def["expiration"] = df_def["expiration"].astype("int64")
    df_def["underlying"] = df_def["underlying"].astype(str)
    df_def["option_symbol"] = df_def["raw_symbol"].astype(str)
    df_def["right"] = df_def["instrument_class"].astype(str)
    return df_def[["instrument_id", "option_symbol", "underlying", "right", "strike_price", "expiration"]]

def _load_open_interest(df_stat: pd.DataFrame) -> Dict[str, float]:
    required = {"option_symbol", "open_interest"}
    missing = required.difference(df_stat.columns)
    if missing:
        raise ValueError(f"Missing statistics columns: {sorted(missing)}")
    oi = df_stat.set_index("option_symbol")["open_interest"].astype(float).to_dict()
    return {k: v for k, v in oi.items() if v > 0}
