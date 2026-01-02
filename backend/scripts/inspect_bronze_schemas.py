
import pandas as pd
import os
import glob
from pathlib import Path

DATA_ROOT = "/Users/loganrobbins/research/qmachina/spymaster/backend/data"

def inspect_schema(category, sub, partition_key, value):
    path = Path(DATA_ROOT) / "bronze" / category / sub / f"{partition_key}={value}"
    print(f"\n--- Checking {category}/{sub} ---")
    
    if not path.exists():
        print(f"Path does not exist: {path}")
        return

    # Find first parquet file
    files = list(path.glob("**/*.parquet"))
    if not files:
        print(f"No parquet files found in {path}")
        return
        
    try:
        # Read schema only
        df = pd.read_parquet(files[0])
        print(f"File: {files[0].name}")
        print(f"Rows: {len(df)}")
        print("Columns:")
        for col in df.columns:
            dtype = df[col].dtype
            print(f"  - {col} ({dtype})")
            
        # Specific check for Market Tide requirements
        if category == 'options' and sub == 'trades':
            print("\n  [Market Tide Check]")
            print(f"  Has 'aggressor'? {'aggressor' in df.columns}")
            print(f"  Has 'size'? {'size' in df.columns}")
            print(f"  Has 'price'? {'price' in df.columns}")
            print(f"  Has 'option_symbol'? {'option_symbol' in df.columns}")
            
    except Exception as e:
        print(f"Error reading {files[0]}: {e}")

def main():
    # 1. Futures
    # inspect_schema("futures", "trades", "symbol", "ES")
    # inspect_schema("futures", "mbp10", "symbol", "ES")
    
    # 2. Options
    # Check a general file
    inspect_schema("options", "statistics", "underlying", "ES")
    
    # Check October specifically (the problem date)
    # Check representative dates for each month
    dates = [
        "2025-06-25",
        "2025-07-25",
        "2025-08-25",
        "2025-09-23",
        "2025-09-30"  # Last available date
    ]
    
    for date in dates:
        print(f"\n[{date}]")
        inspect_schema("options", "trades", "underlying", f"ES/date={date}")
        inspect_schema("options", "nbbo", "underlying", f"ES/date={date}")


if __name__ == "__main__":
    main()
