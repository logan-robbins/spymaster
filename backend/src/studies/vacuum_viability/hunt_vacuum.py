
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import glob
import os

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data_eng.stages.silver.future_mbo.compute_level_vacuum_5s import compute_mbo_level_vacuum_5s, _load_p_ref
from src.data_eng.stages.gold.future_mbo.build_pressure_stream import _pressure_scores

def hunt_vacuum():
    # 1. Find all Bronze partitions
    lake_root = Path(__file__).parent.parent.parent.parent.parent / "backend" / "lake"
    pattern = str(lake_root / "bronze/source=databento/product_type=future_mbo/symbol=*/table=mbo/dt=*")
    paths = glob.glob(pattern)
    paths.sort() 

    print(f"Found {len(paths)} partitions. Hunting for Vacuum > 0.0...")
    
    hits = []

    for p in paths:
        try:
            # Parse symbol/dt from path
            parts = p.split("/")
            dt = parts[-1].replace("dt=", "")
            symbol = parts[-3].replace("symbol=", "")
            
            # Filter Front Month
            # Oct-Dec 2025 -> ESZ5
            if dt.startswith("2025-10") or dt.startswith("2025-11") or dt.startswith("2025-12"):
                if symbol != "ESZ5": continue
            # Jan-Mar 2026 -> ESH6
            elif dt.startswith("2026-01") or dt.startswith("2026-02") or dt.startswith("2026-03"):
                if symbol != "ESH6": continue
            else:
                continue

            # print(f"Checking {symbol} {dt}...")
            
            df = pd.read_parquet(p)
            if len(df) < 1000:
                continue

            # 2. Check PM High Interaction
            try:
                p_ref = _load_p_ref(df, dt)
            except ValueError:
                continue
                
            # Check Max RTH Price (09:30 onwards)
            start_local = pd.Timestamp(f"{dt} 09:30:00", tz="America/New_York")
            start_ns = int(start_local.tz_convert("UTC").value)
            
            rth_df = df[df["ts_event"] >= start_ns]
            if len(rth_df) == 0:
                continue
                
            max_rth = rth_df[rth_df["action"] == "T"]["price"].max() * 1e-9
            
            # Dist check
            gap = abs(max_rth - p_ref)
            
            # If we didn't get within 8 ticks (2.0 points), skip expensive compute
            if gap > 2.0: 
                 continue
            
            print(f"[{dt} {symbol}] PM High: {p_ref:.2f}, Max RTH: {max_rth:.2f}, Gap: {gap:.2f}")

            # 3. Compute Vacuum
            df_vac = compute_mbo_level_vacuum_5s(df, p_ref, symbol)
            
            # 4. Compute Scores
            max_score = 0.0
            trigger_row = None
            
            for row in df_vac.itertuples():
                scores = _pressure_scores(row, row.approach_dir)
                if scores:
                    s = scores.get("vacuum_score")
                    if s is not None and s > max_score:
                        max_score = s
                        trigger_row = row
            
            print(f"  -> Max Vacuum Score: {max_score:.4f}")
            
            if max_score > 0.0:
                # Capture trigger time safely
                trigger_time_str = "Unknown"
                if trigger_row:
                    try:
                        ts_ns = int(trigger_row.window_start_ts_ns)
                        # Convert to NY time
                        ts_pd = pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert("America/New_York")
                        trigger_time_str = ts_pd.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception as e:
                        trigger_time_str = f"Error: {e}"

                hits.append({
                    "date": dt,
                    "symbol": symbol,
                    "p_ref": p_ref,
                    "gap": gap,
                    "max_score": max_score,
                    "trigger_time": trigger_time_str
                })
                
        except Exception as e:
            print(f"  -> Error processing {p}: {e}")
            continue

    print("\n--- SUMMARY ---")
    if not hits:
        print("No hits found.")
    else:
        print(f"Found {len(hits)} days with interaction:")
        df_hits = pd.DataFrame(hits)
        print(df_hits.to_string())

if __name__ == "__main__":
    hunt_vacuum()
