
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import math

# Add backend to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data_eng.stages.gold.future_mbo.build_pressure_stream import _build_pressure_stream, OUTPUT_COLUMNS
from src.data_eng.contracts import load_avro_contract

def run_pressure_manual_for_day(symbol: str, dt: str):
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    
    # 1. Load Silver Vacuum Data
    vacuum_path = repo_root / f"backend/lake/silver/product_type=future_mbo/symbol={symbol}/table=mbo_level_vacuum_5s/dt={dt}"
    if not vacuum_path.exists():
        # Try finding the parquet file directly if the directory structure implies partition
        print(f"Vacuum path not found: {vacuum_path}")
        return None

    df_vacuum = pd.read_parquet(vacuum_path)
    print(f"Loaded Vacuum DF: {len(df_vacuum)} rows")

    # 2. Mock Trigger Signals (Empty)
    # We need the contract to know columns, or just use expected columns for empty DF
    # The _build_pressure_stream function expects df_triggers to have 'trigger_ts', 'fire_flag', etc.
    # Let's inspect what _build_pressure_stream uses.
    # It specifically iterates df_triggers tuples.
    # If len(df_triggers) == 0, it uses default retrieval (fire_flag=0, signal="NONE").
    # So an empty DataFrame is sufficient IF it passes any contract checks? 
    # The function itself doesn't enforce contract, the Stage does.
    # So passing an empty DataFrame is fine.
    
    df_triggers = pd.DataFrame(columns=["trigger_ts", "fire_flag", "signal", "p_break", "p_reject", "p_chop", "margin", "risk_q80_ticks", "resolve_rate", "whipsaw_rate"])
    
    # 3. Load Level ID (Env var needed by _load_level_id inside build_pressure_stream?)
    # Yes, _build_pressure_stream calls _load_level_id() which checks os.environ["LEVEL_ID"]
    import os
    os.environ["LEVEL_ID"] = "pm_high" # Hardcode for test

    # 4. Run Logic
    print("Computing Pressure Stream...")
    df_pressure = _build_pressure_stream(
        df_vacuum=df_vacuum,
        df_triggers=df_triggers,
        symbol=symbol,
        session_date=dt
    )
    
    print(f"Computed Pressure Stream: {len(df_pressure)} rows")
    return df_pressure

if __name__ == "__main__":
    df = run_pressure_manual_for_day("ESH6", "2026-01-07")
    if df is not None:
        print(df[["ts_end_ns", "vacuum_score", "pressure_above_retreat", "pressure_below_retreat_or_recede"]].head())
        # Save to temp
        out_path = Path("temp_pressure_manual.parquet")
        df.to_parquet(out_path)
        print(f"Saved to {out_path}")
