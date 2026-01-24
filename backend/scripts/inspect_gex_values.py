import pandas as pd
import numpy as np
from pathlib import Path

# Define path to the verified GEX file
FILE_PATH = "lake/silver/product_type=future_option_mbo/symbol=ESH6/table=gex_surface_1s/dt=2026-01-06"

def inspect_gex_values():
    p = Path(FILE_PATH)
    files = list(p.glob("*.parquet"))
    if not files:
        print(f"ERROR: No parquet files found in {FILE_PATH}")
        return

    print(f"Loading {files[0]}...")
    df = pd.read_parquet(files[0])
    
    print(f"Shape: {df.shape}")
    print("-" * 50)

    # 1. Check for NaNs
    print("MISSING / INF CHECK:")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("PASS: No NaNs found.")
    else:
        print(f"WARN: Columns with NaNs:\n{null_counts[null_counts > 0]}")
    print("-" * 50)

    # 2. Check Zeros vs Non-Zeros
    print("VALUE DISTRIBUTION:")
    gex_abs = df["gex_abs"]
    non_zero = (gex_abs.abs() > 1e-9).sum()
    print(f"Row Count: {len(df)}")
    print(f"Non-Zero GEX Rows: {non_zero} ({non_zero/len(df):.1%})")
    print(f"Max GEX: {gex_abs.max():.2f}")
    print(f"Min GEX: {gex_abs.min():.2f}")
    print("-" * 50)

    # 3. Check Spot Reference
    print("SPOT REFERENCE CHECK:")
    spots = df["underlying_spot_ref"].unique()
    print(f"Unique Spot Prices: {len(spots)}")
    print(f"Spot Range: {spots.min():.2f} - {spots.max():.2f}")
    
    # 4. Check Strikes
    print("-" * 50)
    print("STRIKE CHECK:")
    strikes = df["strike_price_int"]
    print(f"Strike Range: {strikes.min()} - {strikes.max()}")
    
    # 5. Sample
    print("-" * 50)
    print("SAMPLE ROWS (Top GEX):")
    print(df.sort_values("gex_abs", ascending=False)[["window_end_ts_ns", "underlying_spot_ref", "strike_price_int", "gex_abs", "gex"]].head())

if __name__ == "__main__":
    inspect_gex_values()
