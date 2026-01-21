
import pandas as pd
from pathlib import Path

def check_pref():
    # Path to the Silver output we generated
    path = "lake/silver/product_type=future_mbo/symbol=ESH6/table=mbo_level_vacuum_5s/dt=2026-01-07"
    
    try:
        df = pd.read_parquet(path)
        if "P_ref" in df.columns:
            p_ref = df["P_ref"].iloc[0]
            print(f"P_ref (PM High): {p_ref}")
            print(f"P_ref (Int): {df['P_REF_INT'].iloc[0]}")
        else:
            print("P_ref column not found")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_pref()
