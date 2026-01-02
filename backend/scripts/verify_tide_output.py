
import pandas as pd
import sys

try:
    path = "/Users/loganrobbins/research/qmachina/spymaster/backend/data/silver/features/es_pipeline/version=4.5.0/date=2025-10-29/signals.parquet"
    df = pd.read_parquet(path)
    
    print(f"Loaded {len(df)} rows from {path}")
    
    if 'call_tide' in df.columns:
        ct = df['call_tide']
        pt = df['put_tide']
        print(f"Call Tide: Mean={ct.mean():.4f}, Max={ct.max():.4f}, Sum={ct.sum():.4f}")
        print(f"Put Tide:  Mean={pt.mean():.4f}, Max={pt.max():.4f}, Sum={pt.sum():.4f}")
        
        non_zero = (ct != 0) | (pt != 0)
        print(f"Rows with non-zero tide: {non_zero.sum()}")
        
        if 'force_proxy' in df.columns:
             print(f"Force Proxy Mean: {df['force_proxy'].mean()}")
    else:
        print("Tide columns missing!")

except Exception as e:
    print(f"Error: {e}")
