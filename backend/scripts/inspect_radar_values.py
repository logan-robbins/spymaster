import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Define path to the specific parquet file we just generated
# We know it's in backend/lake/silver/product_type=future_mbo/symbol=ESH6/table=radar_vacuum_1s/dt=2026-01-06
FILE_PATH = "lake/silver/product_type=future_mbo/symbol=ESH6/table=radar_vacuum_1s/dt=2026-01-06"

def inspect_radar_values():
    # Find the parquet file
    p = Path(FILE_PATH)
    files = list(p.glob("*.parquet"))
    if not files:
        print(f"ERROR: No parquet files found in {FILE_PATH}")
        return

    print(f"Loading {files[0]}...")
    df = pd.read_parquet(files[0])
    
    print(f"Shape: {df.shape}")
    print("-" * 50)

    # 1. Check for NaNs/Infs
    print("MISSING / INF CHECK:")
    null_counts = df.isnull().sum()
    inf_counts = df.apply(lambda x: np.isinf(x) if np.issubdtype(x.dtype, np.number) else False).sum()
    
    missing_cols = null_counts[null_counts > 0]
    inf_cols = inf_counts[inf_counts > 0]
    
    if missing_cols.empty and inf_cols.empty:
        print("PASS: No NaNs or Infs found.")
    else:
        if not missing_cols.empty:
            print(f"WARN: Columns with NaNs:\n{missing_cols}")
        if not inf_cols.empty:
            print(f"WARN: Columns with Infs:\n{inf_cols}")
    print("-" * 50)

    # 2. Check Zeros
    print("ZERO VALUE CHECK (Top 10 cols with most zeros):")
    zero_counts = (df == 0).sum().sort_values(ascending=False).head(10)
    print(zero_counts)
    print("-" * 50)

    # 3. Check Normalization (Share features should be 0-1)
    print("NORMALIZATION CHECK (Share/Ratio features [0,1]):")
    share_cols = [c for c in df.columns if "share" in c]
    out_of_bounds = {}
    for c in share_cols:
        vmin = df[c].min()
        vmax = df[c].max()
        if vmin < -1e-9 or vmax > 1.0 + 1e-9:
            out_of_bounds[c] = (vmin, vmax)
    
    if not out_of_bounds:
        print("PASS: All 'share' features are within [0, 1].")
    else:
        print("WARN: Features out of [0, 1] bounds:")
        for c, (vmin, vmax) in out_of_bounds.items():
            print(f"  {c}: min={vmin:.4f}, max={vmax:.4f}")
    
    print("-" * 50)

    # 4. Check Log Features checks
    print("LOG FEATURE CHECK (Should be reasonable range, e.g. -20 to 20):")
    log_cols = [c for c in df.columns if "log" in c]
    weird_logs = {}
    for c in log_cols:
        vmin = df[c].min()
        vmax = df[c].max()
        # Arbitrary "weird" threshold for log features
        if vmin < -50 or vmax > 50:
            weird_logs[c] = (vmin, vmax)
            
    if not weird_logs:
        print("PASS: Log features seem within reasonable bounds (-50, 50).")
    else:
        print("WARN: Log features with extreme values:")
        for c, (vmin, vmax) in weird_logs.items():
            print(f"  {c}: min={vmin:.4f}, max={vmax:.4f}")

    print("-" * 50)

    # 5. Spot Price Check
    spot_min = df["spot_ref_price"].min()
    spot_max = df["spot_ref_price"].max()
    print(f"SPOT PRICE RANGE: {spot_min} - {spot_max}")
    
    # 6. Specific Feature Spot Check
    print("-" * 50)
    print("SAMPLE ROWS (First 5):")
    print(df[["spot_ref_price", "f1_ask_com_disp_log", "f5_vacuum_expansion_log"]].head())

if __name__ == "__main__":
    inspect_radar_values()
