import pandas as pd
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import defaultdict
import databento as db
import json

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants (Matching Pipeline)
PRICE_SCALE = 1e-9
TICK_SIZE = 0.25
TICK_INT = int(round(TICK_SIZE / PRICE_SCALE))
WINDOW_NS = 5_000_000_000 # 5 seconds
GRID_RANGE = 5.0 # +/- $5.00

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

@dataclass
class OrderState:
    side: str
    price_int: int
    qty: int
    bucket_enter_ts: int

def calculate_vacuum_clean():
    # 1. Load Clean Bronze Data
    # Path: lake/bronze/source=databento/product_type=future_mbo/symbol=ESH6/table=mbo/dt=2026-01-15/*.parquet
    BRONZE_DIR = Path("lake/bronze/source=databento/product_type=future_mbo/symbol=ESH6/table=mbo/dt=2026-01-15")
    
    if not BRONZE_DIR.exists():
        logger.error(f"Bronze data not found at {BRONZE_DIR}")
        return

    files = list(BRONZE_DIR.glob("*.parquet"))
    if not files:
        logger.error("No parquet files found.")
        return

    logger.info(f"Loading {len(files)} parquet files...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")
            
    if not dfs:
        logger.error("No data loaded.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["ts_event", "sequence"])
    
    # 2. Filter Time Window (09:30 - 10:00)
    # 2026-01-15
    start_ns = pd.Timestamp("2026-01-15 09:30:00", tz="America/New_York").value
    end_ns = pd.Timestamp("2026-01-15 09:36:00", tz="America/New_York").value
    
    df = df[(df["ts_event"] >= start_ns) & (df["ts_event"] <= end_ns)]
    if df.empty:
        logger.error("No data in window.")
        return
        
    logger.info(f"Processing {len(df)} events...")

    # 3. Processing Loop (Silver Logic)
    orders: dict[int, OrderState] = {}
    
    # Erosion Tracking (Reset every window)
    # Price -> Qty
    ask_erosion_qty = defaultdict(float)
    bid_erosion_qty = defaultdict(float)
    
    current_window_start = (df["ts_event"].iloc[0] // WINDOW_NS) * WINDOW_NS
    
    output_frames = []
    
    # VWAP State
    total_vol = 0
    total_pv = 0.0
    last_trade_price = 0.0
    
    # Pre-calc Ints
    start_time = pd.Timestamp.now()
    
    for row in df.itertuples(index=False):
        ts = row.ts_event
        window_start = (ts // WINDOW_NS) * WINDOW_NS
        
        # New Window?
        if window_start > current_window_start:
            # Snapshot!
            if total_vol > 0:
                vwap = total_pv / total_vol / 1e9 
            else:
                vwap = 0.0
            
            # Use last trade price if available, else vwap
            curr_price = last_trade_price if last_trade_price > 0 else vwap
                
            # Build Grid
            if vwap > 0:
                grid_frames = _build_grid_snapshot(orders, ask_erosion_qty, bid_erosion_qty, vwap, curr_price, current_window_start)
                output_frames.append(grid_frames)
            
            # Reset Erosion
            ask_erosion_qty.clear()
            bid_erosion_qty.clear()
            current_window_start = window_start
            
            if len(output_frames) % 120 == 0:
                logger.info(f"Processed 10m. Current: {pd.Timestamp(ts, unit='ns')}")

        action = row.action
        oid = int(row.order_id)
        price_int = int(row.price)
        size = int(row.size)
        side = row.side
        
        # State Update Logic
        old = orders.get(oid)
        
        # Trade / VWAP
        if action == "T":
            total_vol += size
            total_pv += (price_int * size)
            last_trade_price = float(price_int) / 1e9
            
            # Aggressor Logic for Erosion
            # If Aggressor Buy -> Eats Ask
            if side == "B": 
                ask_erosion_qty[price_int] += size
            # If Aggressor Sell -> Eats Bid
            elif side == "A":
                bid_erosion_qty[price_int] += size
            continue
            
        # Cancel / Erosion
        if action == "C" and old:
            # Cancel is strictly erosion of LIQUIDITY
            if old.side == "A":
                ask_erosion_qty[old.price_int] += size # Assuming size is cancel amt
            elif old.side == "B":
                bid_erosion_qty[old.price_int] += size
                
            # Remove
            orders.pop(oid, None) # Simple removal for C? 
            # Note: Pipeline uses complex diff logic for partials, but std C is usually kill or reduce.
            # BronzeIngest seems to treat C as size reduction?
            # Pipeline: "If action == 'C', new_order = None". Treating C as full cancel/kill.
            # NOTE: Check Ingest. If BronzeIngest keeps size as 'amount cancelled', then we reduce.
            # But the Pipeline logic: `elif action == "C": new_order = None`. It kills it.
            # So let's follow Pipeline: C kills the order (or the specific key).
            
        elif action == "A":
            orders[oid] = OrderState(side=side, price_int=price_int, qty=size, bucket_enter_ts=ts)
            
        elif action == "M":
            # Modify - Treat as Cancel Old + Add New for book integrity
            if old:
                # Remove old depth (implicitly handled by snapshot reading current orders)
                # But we must update the 'orders' dict
                
                # Is it erosion?
                # If size decreases, it's a partial cancel.
                if size < old.qty:
                    diff = old.qty - size
                    if old.side == "A": ask_erosion_qty[old.price_int] += diff
                    if old.side == "B": bid_erosion_qty[old.price_int] += diff
                
                # Update Order
                orders[oid] = OrderState(side=old.side, price_int=price_int, qty=size, bucket_enter_ts=ts)

    # Final Save
    # Save to the same directory as the script
    SCRIPT_DIR = Path(__file__).parent
    out_file = SCRIPT_DIR / "vacuum_heatmap_clean.json"
    
    if not output_frames:
        logger.warning("No frames generated!")
    
    with open(out_file, "w") as f:
        json.dump(output_frames, f, cls=NpEncoder)
    logger.info(f"Saved {len(output_frames)} frames to {out_file}")

def _build_grid_snapshot(orders, ask_erosion, bid_erosion, vwap, price, ts):
    # Determine Grid
    tick = 0.25
    center = round(vwap / tick) * tick
    min_p = center - 5.0
    max_p = center + 5.0
    
    # Convert Orders to Depth Map
    # PriceInt -> Qty
    ask_depth = defaultdict(int)
    bid_depth = defaultdict(int)
    
    for o in orders.values():
        if o.side == "A": ask_depth[o.price_int] += o.qty
        if o.side == "B": bid_depth[o.price_int] += o.qty
        
    prices = np.arange(min_p, max_p + 0.001, tick)
    
    frame = {
        "ts": ts,
        "vwap": vwap,
        "price": price,
        "prices": [],
        "vacuum_up": [],
        "vacuum_down": []
    }
    
    for p in prices:
        p_int = int(round(p * 1e9))
        
        # Depth
        a_qty = ask_depth[p_int]
        b_qty = bid_depth[p_int]
        
        # Erosion
        a_ero = ask_erosion[p_int]
        b_ero = bid_erosion[p_int]
        
        # Features
        vac_up = a_ero / (a_qty + 1.0)
        vac_down = b_ero / (b_qty + 1.0)
        
        frame["prices"].append(p)
        frame["vacuum_up"].append(float(f"{vac_up:.4f}"))
        frame["vacuum_down"].append(float(f"{vac_down:.4f}"))
        
    return frame

if __name__ == "__main__":
    calculate_vacuum_clean()
