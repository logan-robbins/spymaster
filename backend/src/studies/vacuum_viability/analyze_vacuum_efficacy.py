
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_efficacy():
    # 1. Load Pressure Data
    pressure_path = "temp_pressure_manual.parquet"
    if not Path(pressure_path).exists():
        print("Pressure data not found.")
        return

    df_pressure = pd.read_parquet(pressure_path)
    print(f"Loaded Pressure: {len(df_pressure)} rows")
    
    # 2. Load Bronze Trades for Price
    bronze_path = "lake/bronze/source=databento/product_type=future_mbo/symbol=ESH6/table=mbo/dt=2026-01-07"
    df_trades = pd.read_parquet(bronze_path)
    # Filter for trades
    df_trades = df_trades[df_trades["action"] == "T"].copy()
    df_trades["ts_event"] = df_trades["ts_event"].astype("int64")
    df_trades["price"] = df_trades["price"].astype("int64") / 1e9 # Convert to float price
    df_trades = df_trades[["ts_event", "price"]].sort_values("ts_event")
    
    print(f"Loaded Trades: {len(df_trades)} rows")

    # 3. Align Pressure to Forward Price
    # Pressure timestamp is 'ts_end_ns' (end connected window).
    # We want price at T (trigger) and price at T + 1min.
    
    # Merge using asof
    df_pressure = df_pressure.sort_values("ts_end_ns")
    
    # Price AT signal
    df_merged = pd.merge_asof(
        df_pressure, 
        df_trades, 
        left_on="ts_end_ns", 
        right_on="ts_event", 
        direction="backward"
    ).rename(columns={"price": "price_at_signal"})
    
    # Price 60s LATER
    df_trades["ts_1m"] = df_trades["ts_event"] - 60_000_000_000 # Shift trades back so we can find 'future' trade by looking back from shifted time?
    # No, easy way: T_target = T_signal + 60s. Find trade nearest T_target.
    
    df_merged["target_ts"] = df_merged["ts_end_ns"] + 60_000_000_000
    
    df_merged = pd.merge_asof(
        df_merged,
        df_trades,
        left_on="target_ts",
        right_on="ts_event",
        direction="forward", # Find subsequent trade
        tolerance=10_000_000_000 # 10s tolerance
    ).rename(columns={"price": "price_1m_later"})
    
    df_merged["ret_1m_ticks"] = (df_merged["price_1m_later"] - df_merged["price_at_signal"]) / 0.25
    
    # 4. Filter for Valid Signals (Active Vacuum)
    # Filter approach_up
    up_candidates = df_merged[df_merged["approach_dir"] == "approach_up"].copy()
    # Filter approach_down
    down_candidates = df_merged[df_merged["approach_dir"] == "approach_down"].copy()
    
    print(f"Candidates: Up={len(up_candidates)}, Down={len(down_candidates)}")
    
    # 5. Efficacy Stats
    # High Vacuum vs Low Vacuum
    for name, df in [("UP", up_candidates), ("DOWN", down_candidates)]:
        if len(df) == 0:
            continue
            
        print(f"\n--- {name} APPROACH ---")
        
        # Quantiles of Vacuum Score
        df["vac_bucket"] = pd.qcut(df["vacuum_score"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        print(df.groupby("vac_bucket", observed=True)["ret_1m_ticks"].describe()[["count", "mean", "std"]])
        
        # Gated vs Ungated? We only have the final score.
        # Check pure correlation
        corr = df["vacuum_score"].corr(df["ret_1m_ticks"])
        print(f"Correlation (Score vs Return): {corr:.4f}")
        
        # Check Active Signal (>0.7) efficiency
        active = df[df["vacuum_score"] > 0.7]
        passive = df[df["vacuum_score"] < 0.3]
        
        print(f"Mean Return (Score > 0.7): {active['ret_1m_ticks'].mean():.2f} ticks")
        print(f"Mean Return (Score < 0.3): {passive['ret_1m_ticks'].mean():.2f} ticks")

if __name__ == "__main__":
    analyze_efficacy()
