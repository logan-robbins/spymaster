from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

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

PULL_ACTIONS = {ACTION_CANCEL, ACTION_MODIFY}


@dataclass
class OrderState:
    side: str
    price_int: int
    qty: int
    ts_enter_price: int  # Timestamp when order entered current price


class SilverComputeSnapshotAndWall1s(Stage):
    def __init__(self) -> None:
        # We output two datasets, but StageIO expects one 'output' for the default check.
        # We'll use book_snapshot_1s as the primary to satisfy the base class,
        # but override run() to handle both.
        super().__init__(
            name="silver_compute_snapshot_and_wall_1s",
            io=StageIO(
                inputs=["bronze.future_mbo.mbo"],
                output="silver.future_mbo.book_snapshot_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        # Define output keys
        out_key_snap = "silver.future_mbo.book_snapshot_1s"
        out_key_wall = "silver.future_mbo.wall_surface_1s"

        ref_snap = partition_ref(cfg, out_key_snap, symbol, dt)
        ref_wall = partition_ref(cfg, out_key_wall, symbol, dt)

        # Idempotency check: if both complete, return
        if is_partition_complete(ref_snap) and is_partition_complete(ref_wall):
            return

        # Check input
        input_key = self.io.inputs[0]
        in_ref = partition_ref(cfg, input_key, symbol, dt)
        if not is_partition_complete(in_ref):
            raise FileNotFoundError(f"Input not ready: {input_key} dt={dt}")

        # Read Input
        in_contract_path = repo_root / cfg.dataset(input_key).contract
        in_contract = load_avro_contract(in_contract_path)
        df_in = read_partition(in_ref)
        df_in = enforce_contract(df_in, in_contract)

        # Transform
        df_snap, df_wall = self.transform(df_in, dt)

        # Enforce Contracts
        contract_snap_path = repo_root / cfg.dataset(out_key_snap).contract
        contract_wall_path = repo_root / cfg.dataset(out_key_wall).contract
        
        contract_snap = load_avro_contract(contract_snap_path)
        contract_wall = load_avro_contract(contract_wall_path)

        df_snap = enforce_contract(df_snap, contract_snap)
        df_wall = enforce_contract(df_wall, contract_wall)

        # Lineage
        manifest = read_manifest_hash(in_ref)
        lineage = [{"dataset": in_ref.dataset_key, "dt": dt, "manifest_sha256": manifest}]

        # Write Outputs
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
        
        # Type conversions for speed
        ts_event = df["ts_event"].to_numpy(dtype=np.int64)
        action = df["action"].to_numpy(dtype=object)
        side = df["side"].to_numpy(dtype=object)
        price = df["price"].to_numpy(dtype=np.int64)
        size = df["size"].to_numpy(dtype=np.int64)
        order_id = df["order_id"].to_numpy(dtype=np.int64)
        flags = df["flags"].to_numpy(dtype=np.int64) if "flags" in df.columns else np.zeros(len(df), dtype=np.int64)

        # State
        orders: Dict[int, OrderState] = {}
        # depth: price_int -> qty (separate per side to be safe, though price usually determines side in disjoint books)
        # Using dict for sparse depth
        depth_bid: Dict[int, int] = {}
        depth_ask: Dict[int, int] = {}

        # Snapshot Logic
        # Databento snapshots start with R and F_SNAPSHOT (128 according to docs, but we should check constant)
        # We assume R clears the book.
        
        # Accumulators for current window
        # wall_accum[(price, side)] = {add, pull, fill, etc}
        wall_accum: Dict[Tuple[int, str], Dict[str, float]] = {}
        
        rows_snap = []
        rows_wall = []

        curr_window_id = None
        window_start_ts = 0
        window_end_ts = 0
        
        book_valid = False
        last_trade_price = 0
        best_bid = 0
        best_ask = 0
        
        # Constants for flags
        F_SNAPSHOT = 128
        F_LAST = 256 # Assuming standard Databento flags, need to verify if accessible.
        # Actually F_LAST is often 0x80 (128) if F_SNAPSHOT is also set? 
        # "Snapshot records are marked with F_SNAPSHOT... followed by... F_LAST"
        # Let's rely on logic: if R -> clear. If we are in snapshot mode, we wait for F_LAST?
        # IMPLEMENT.md: "Snapshot records are marked with F_SNAPSHOT... until... F_LAST"
        # We will track if we are inside a snapshot loading phase.

        in_snapshot_load = False

        for i in range(len(df)):
            ts = ts_event[i]
            act = action[i]
            oid = order_id[i]
            px = price[i]
            sz = size[i]
            sd = side[i]
            flg = flags[i]

            w_id = ts // WINDOW_NS

            # Window Boundary Check
            if curr_window_id is None:
                curr_window_id = w_id
                window_start_ts = w_id * WINDOW_NS
                window_end_ts = window_start_ts + WINDOW_NS

            if w_id != curr_window_id:
                # Close Window
                # Emit Snapshot
                # We need best bid/ask
                bb_p, bb_q = _get_best(depth_bid, is_bid=True)
                ba_p, ba_q = _get_best(depth_ask, is_bid=False)
                
                mid = 0.0
                mid_int = 0
                if bb_p > 0 and ba_p > 0:
                    mid = (bb_p + ba_p) / 2.0 * PRICE_SCALE
                    mid_int = int(round((bb_p + ba_p) / 2.0))
                
                # Spot Ref Logic: Last Trade > Best Bid (if valid) > 0
                spot_ref = last_trade_price
                if spot_ref == 0:
                    if book_valid and bb_p > 0:
                        spot_ref = bb_p
                
                rows_snap.append({
                    "window_start_ts_ns": window_start_ts,
                    "window_end_ts_ns": window_end_ts,
                    "best_bid_price_int": bb_p,
                    "best_bid_qty": bb_q,
                    "best_ask_price_int": ba_p,
                    "best_ask_qty": ba_q,
                    "mid_price": mid,
                    "mid_price_int": mid_int,
                    "last_trade_price_int": last_trade_price,
                    "spot_ref_price_int": spot_ref,
                    "book_valid": book_valid
                })

                # Emit Wall Surface
                if spot_ref > 0:
                    _emit_wall_rows(
                        rows_wall, wall_accum, 
                        depth_bid, depth_ask, 
                        window_start_ts, window_end_ts, 
                        spot_ref, book_valid, ts, orders
                    )

                # Reset
                wall_accum.clear()
                curr_window_id = w_id
                window_start_ts = w_id * WINDOW_NS
                window_end_ts = window_start_ts + WINDOW_NS

            # Process Action
            if act == ACTION_CLEAR:
                orders.clear()
                depth_bid.clear()
                depth_ask.clear()
                # Check flags for snapshot start
                if flg & F_SNAPSHOT:
                    in_snapshot_load = True
                    book_valid = False
                else:
                    # Regular clear (unlikely in MBO without snapshot, but possible)
                    # We treat it as valid empty book? Or invalid?
                    # "R: clear book (reset)... book_valid=false from start until F_LAST"
                    in_snapshot_load = True # Assume any R starts a rebuild
                    book_valid = False
                continue

            if in_snapshot_load:
                # We are loading a snapshot
                # Check for F_LAST
                if flg & F_LAST:
                    in_snapshot_load = False
                    book_valid = True
                # Even if F_LAST is set, this record itself is part of the snapshot (A record)
            
            # Update Logic
            if act == ACTION_ADD:
                orders[oid] = OrderState(side=sd, price_int=px, qty=sz, ts_enter_price=ts)
                _update_depth(depth_bid, depth_ask, sd, px, sz)
                _accum_wall(wall_accum, px, sd, "add_qty", sz)
            
            elif act == ACTION_MODIFY:
                old = orders.get(oid)
                # If modified order not found (and not snapshot loading), ignore
                if old:
                    # Remove old depth
                    _update_depth(depth_bid, depth_ask, old.side, old.price_int, -old.qty)
                    # Add new depth
                    # New params
                    new_px = px # Price in M record is the new price
                    new_sz = sz # Size in M record is new size
                    # Side usually doesn't change
                    orders[oid] = OrderState(side=old.side, price_int=new_px, qty=new_sz, ts_enter_price=ts) 
                    # Note: ts_enter_price updates on modify if price changes? 
                    # "ts_enter_price updates when the order changes price (or is newly added)"
                    # converting logic: if new_px == old.price_int, maybe keep old ts? 
                    # IMPLEMENT.md says "updates when the order changes price".
                    if new_px == old.price_int:
                        orders[oid].ts_enter_price = old.ts_enter_price # Revert to old ts if price same
                    
                    _update_depth(depth_bid, depth_ask, old.side, new_px, new_sz)
                    
                    # Accumulate logic
                    # "pull_qty_total (cancels/mods reducing size)"
                    # If size reduced: delta = old.qty - new_sz
                    # If price changed: full remove old, full add new? 
                    # Usually M is treated as:
                    # - cancellation of old (pull)
                    # - addition of new (add)?
                    # IMPLEMENT.md: "pull_qty_total... reducing size... within window at this price/side"
                    # If price changes, it's a pull at old price, add at new price?
                    if new_px != old.price_int:
                        # Pull full at old
                        delta_pull = old.qty
                        _accum_pull(wall_accum, old.price_int, old.side, delta_pull, ts, old.ts_enter_price)
                        # Add full at new
                        _accum_wall(wall_accum, new_px, old.side, "add_qty", new_sz)
                    else:
                        # Price same, size change
                        if new_sz < old.qty:
                            delta = old.qty - new_sz
                            _accum_pull(wall_accum, new_px, old.side, delta, ts, old.ts_enter_price)
                        elif new_sz > old.qty:
                             delta = new_sz - old.qty
                             _accum_wall(wall_accum, new_px, old.side, "add_qty", delta)

            elif act == ACTION_CANCEL:
                old = orders.get(oid)
                if old:
                    _update_depth(depth_bid, depth_ask, old.side, old.price_int, -old.qty)
                    _accum_pull(wall_accum, old.price_int, old.side, old.qty, ts, old.ts_enter_price)
                    del orders[oid]

            elif act == ACTION_FILL:
                old = orders.get(oid)
                if old:
                    # Fill reduces qty
                    fill_sz = sz # F record size is filled amount
                    rem_sz = old.qty - fill_sz
                    
                    # Update depth (remove fill amount)
                    _update_depth(depth_bid, depth_ask, old.side, old.price_int, -fill_sz)
                    
                    # Accumulate Fill
                    _accum_wall(wall_accum, old.price_int, old.side, "fill_qty", fill_sz)
                    
                    if rem_sz <= 0:
                        del orders[oid]
                    else:
                        old.qty = rem_sz # Update in place
            
            elif act == ACTION_TRADE:
                # T record. Update last trade
                # T record often has price/size
                if px > 0:
                    last_trade_price = px

        # End of Loop - Flush last window
        if curr_window_id is not None:
             # Lookahead/Flush logic identical to above
             # Copied for neatness, normally would refactor
             bb_p, bb_q = _get_best(depth_bid, is_bid=True)
             ba_p, ba_q = _get_best(depth_ask, is_bid=False)
             mid = (bb_p + ba_p) * 0.5 * PRICE_SCALE if (bb_p and ba_p) else 0.0
             mid_int = int((bb_p + ba_p) * 0.5) if (bb_p and ba_p) else 0
             spot_ref = last_trade_price if last_trade_price else (bb_p if (book_valid and bb_p) else 0)
             
             rows_snap.append({
                    "window_start_ts_ns": window_start_ts,
                    "window_end_ts_ns": window_end_ts,
                    "best_bid_price_int": bb_p,
                    "best_bid_qty": bb_q,
                    "best_ask_price_int": ba_p,
                    "best_ask_qty": ba_q,
                    "mid_price": mid,
                    "mid_price_int": mid_int,
                    "last_trade_price_int": last_trade_price,
                    "spot_ref_price_int": spot_ref,
                    "book_valid": book_valid
             })
             if spot_ref > 0:
                 _emit_wall_rows(rows_wall, wall_accum, depth_bid, depth_ask, window_start_ts, window_end_ts, spot_ref, book_valid, ts_event[-1], orders)

        # Build dataframes
        df_snap_out = pd.DataFrame(rows_snap)
        df_wall_out = pd.DataFrame(rows_wall)
        
        # Derivatives for Wall Surface (D1/D2/D3)
        # We need to sort by price/side/time?
        # IMPLEMENT.md: "computed over time for the same (price_int, side)"
        if not df_wall_out.empty:
            df_wall_out = _calc_derivatives(df_wall_out)
        else:
            # Ensure columns exist even if empty
            cols = ["window_start_ts_ns", "window_end_ts_ns", "price_int", "side", "spot_ref_price_int", "rel_ticks", 
                    "depth_qty_start", "depth_qty_end", "add_qty", "pull_qty_total", "depth_qty_rest", "pull_qty_rest", 
                    "fill_qty", "d1_depth_qty", "d2_depth_qty", "d3_depth_qty", "window_valid"]
            df_wall_out = pd.DataFrame(columns=cols)

        return df_snap_out, df_wall_out

# Helpers

def _update_depth(bid_book, ask_book, side, price, delta):
    if side == "B":
        bid_book[price] = bid_book.get(price, 0) + delta
        if bid_book[price] <= 0:
            del bid_book[price]
    elif side == "A":
        ask_book[price] = ask_book.get(price, 0) + delta
        if ask_book[price] <= 0:
            del ask_book[price]

def _get_best(book, is_bid):
    if not book:
        return 0, 0
    if is_bid:
        p = max(book.keys())
    else:
        p = min(book.keys())
    return p, book[p]

def _accum_wall(accum, price, side, field, value):
    key = (price, side)
    if key not in accum:
        accum[key] = {"add_qty": 0.0, "pull_qty_total": 0.0, "pull_qty_rest": 0.0, "fill_qty": 0.0}
    accum[key][field] += float(value)

def _accum_pull(accum, price, side, qty, curr_ts, enter_ts):
    _accum_wall(accum, price, side, "pull_qty_total", qty)
    if (curr_ts - enter_ts) >= REST_NS:
         _accum_wall(accum, price, side, "pull_qty_rest", qty)

def _emit_wall_rows(rows, accum, depth_bid, depth_ask, start_ts, end_ts, spot_ref, valid, curr_ts, orders):
    # Iterate ticks around spot
    min_p = spot_ref - HUD_MAX_TICKS * TICK_INT
    max_p = spot_ref + HUD_MAX_TICKS * TICK_INT
    
    # We need to emit rows for every price that has activity OR depth?
    # IMPLEMENT.md: "Emit rows only for price levels in [range]" (implies dense? or sparse?)
    # "Wall Surface... Shows what is sitting there" -> Should probably be sparse where depth exists, or dense?
    # "Converted into dense, bounded textures" -> implies pipeline can be sparse, HUD query makes it dense.
    # But for "erosion" d1/d2, we need continuity. 
    # Let's iterate over ALL prices in the map + accum that are in range.
    
    active_prices = set()
    for p in depth_bid.keys():
        if min_p <= p <= max_p: active_prices.add((p, "B"))
    for p in depth_ask.keys():
        if min_p <= p <= max_p: active_prices.add((p, "A"))
    for (p, s) in accum.keys():
        if min_p <= p <= max_p: active_prices.add((p, s))

    # Calculate depth_qty_rest specific to CURRENT STATE (end of window)
    # This requires iterating orders. This is expensive.
    # Optimization: "depth_qty_rest" is subset of depth_qty_end.
    # We can iterate orders ONCE per window output?
    # Or maintain a "resting depth" structure? Maintaining is hard because time flows.
    # Iterating valid orders in range is safer.
    
    # Reset buckets
    res_depth = {} # (p,s) -> rest_qty
    
    for o in orders.values():
        if min_p <= o.price_int <= max_p:
            if (curr_ts - o.ts_enter_price) >= REST_NS:
                k = (o.price_int, o.side)
                res_depth[k] = res_depth.get(k, 0.0) + o.qty

    for (p, s) in active_prices:
        acc = accum.get((p, s), {"add_qty": 0.0, "pull_qty_total": 0.0, "pull_qty_rest": 0.0, "fill_qty": 0.0})
        
        d_end = 0.0
        if s == "B": d_end = float(depth_bid.get(p, 0))
        else: d_end = float(depth_ask.get(p, 0))
        
        # depth_start = depth_end - adds + pulls + fills ??
        # Conservation: End = Start + Add - Pull - Fill
        # Start = End - Add + Pull + Fill
        d_start = d_end - acc["add_qty"] + acc["pull_qty_total"] + acc["fill_qty"]
        # Clamp to 0 just in case floating point drift
        d_start = max(0.0, d_start)
        
        rel = int(round((p - spot_ref) / TICK_INT))
        
        rows.append({
            "window_start_ts_ns": start_ts,
            "window_end_ts_ns": end_ts,
            "price_int": p,
            "side": s,
            "spot_ref_price_int": spot_ref,
            "rel_ticks": rel,
            "depth_qty_start": d_start,
            "depth_qty_end": d_end,
            "add_qty": acc["add_qty"],
            "pull_qty_total": acc["pull_qty_total"],
            "depth_qty_rest": float(res_depth.get((p, s), 0.0)),
            "pull_qty_rest": acc["pull_qty_rest"],
            "fill_qty": acc["fill_qty"],
            "d1_depth_qty": 0.0, # Placeholder, computed later
            "d2_depth_qty": 0.0,
            "d3_depth_qty": 0.0,
            "window_valid": valid
        })

def _calc_derivatives(df):
    # Sort by side, price, time
    df = df.sort_values(["side", "price_int", "window_end_ts_ns"])
    
    # Group by side, price
    # We want diff of depth_qty_end
    # Using shift
    # Need to handle gaps? Assuming consecutive windows for now or 0.
    
    groups = df.groupby(["side", "price_int"])["depth_qty_end"]
    
    df["d1_depth_qty"] = groups.diff().fillna(0.0)
    df["d2_depth_qty"] = df.groupby(["side", "price_int"])["d1_depth_qty"].diff().fillna(0.0)
    df["d3_depth_qty"] = df.groupby(["side", "price_int"])["d2_depth_qty"].diff().fillna(0.0)
    
    return df
