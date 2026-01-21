
import pandas as pd
from pathlib import Path
import sys

# Update this path to match your environment if needed
bronze_path = "lake/bronze/source=databento/product_type=future_mbo/symbol=ESH6/table=mbo/dt=2026-01-07"

try:
    df = pd.read_parquet(bronze_path)
    print(f"Loaded Bronze DF: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check pre-market trades
    # _premarket_window_ns: 2025-10-01 05:00 - 09:30 NY
    dt = "2025-10-01"
    start_local = pd.Timestamp(f"{dt} 05:00:00", tz="America/New_York")
    end_local = pd.Timestamp(f"{dt} 09:30:00", tz="America/New_York")
    start_ns = int(start_local.tz_convert("UTC").value)
    end_ns = int(end_local.tz_convert("UTC").value)
    
    print(f"Window: {start_ns} to {end_ns}")
    
    df["ts_event"] = df["ts_event"].astype("int64")
    trade_action = "T"
    is_trade = df["action"] == trade_action
    in_window = (df["ts_event"] >= start_ns) & (df["ts_event"] < end_ns)
    mask = is_trade & in_window
    
    trades = df[mask]
    print(f"Premarket Trades: {len(trades)}")
    
    if len(trades) > 0:
        prices = trades["price"].to_numpy()
        prices = prices[prices > 0]
        print(f"Valid Prices: {len(prices)}")
        if len(prices) > 0:
            print(f"Max Price (int): {prices.max()}")
    
except Exception as e:
    print(f"Error: {e}")
