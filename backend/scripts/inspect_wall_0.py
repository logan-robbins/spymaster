
import pandas as pd
from pathlib import Path

# Path to Wall Surface
# Assuming standard path structure
wall_path = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake/silver/product_type=future_mbo/symbol=ESH6/table=wall_surface_1s/dt=2026-01-06")

if not wall_path.exists():
    print(f"Path not found: {wall_path}")
    exit(1)

parquet_files = list(wall_path.glob("*.parquet"))
if not parquet_files:
    print("No parquet files found.")
    exit(1)

print(f"Loading {parquet_files[0]}...")
df = pd.read_parquet(parquet_files[0])

# Inspect rel_ticks = 0
print("\n=== Wall Surface at rel_ticks = 0 ===")
df_0 = df[df["rel_ticks"] == 0]
print(f"Total Rows with rel_ticks=0: {len(df_0)}")

if not df_0.empty:
    print("\nSample Rows (rel_ticks=0):")
    print(df_0[["window_end_ts_ns", "spot_ref_price_int", "price_int", "depth_qty_rest", "side"]].head(10))
    
    print("\nStats for rel_ticks=0:")
    print(df_0["depth_qty_rest"].describe())
    
    # Check if we have consistent timestamps
    timestamps = df["window_end_ts_ns"].unique()
    timestamps_0 = df_0["window_end_ts_ns"].unique()
    print(f"\nTotal Timestamps: {len(timestamps)}")
    print(f"Timestamps with rel_ticks=0: {len(timestamps_0)}")
    
    missing_ts = set(timestamps) - set(timestamps_0)
    print(f"Timestamps MISSING rel_ticks=0: {len(missing_ts)}")

print("\n=== Neighbor Analysis ===")
for r_tick in [-1, 0, 1]:
    df_r = df[df["rel_ticks"] == r_tick]
    n = len(df_r)
    n_unique = df_r["window_end_ts_ns"].nunique()
    if not df_r.empty:
        mean_depth = df_r["depth_qty_rest"].mean()
        zeros = (df_r["depth_qty_rest"] == 0).sum()
        pct_zeros = (zeros / n) * 100
        print(f"RelTick {r_tick}: {n} rows, {n_unique} timestamps. Mean Depth: {mean_depth:.2f}. Zeros: {zeros} ({pct_zeros:.1f}%)")
    else:
        print(f"RelTick {r_tick}: NO ROWS")

# Inspect Bucket Radar if possible
radar_path = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake/silver/product_type=future_mbo/symbol=ESH6/table=bucket_radar_surface_1s/dt=2026-01-06")
if radar_path.exists():
    radar_files = list(radar_path.glob("*.parquet"))
    if radar_files:
        print(f"\nLoading Radar: {radar_files[0]}...")
        df_r = pd.read_parquet(radar_files[0])
        print("\n=== Bucket Radar at bucket_rel = 0 ===")
        df_r0 = df_r[df_r["bucket_rel"] == 0]
        print(f"Total Rows with bucket_rel=0: {len(df_r0)}")
        if not df_r0.empty:
            print(df_r0[["window_end_ts_ns", "blocked_level", "cavitation"]].head(10))
            print(df_r0["blocked_level"].describe())
        
        # Check Transparency
        # If blocked_level=0 and cavitation=0 -> Transparent
        df_trans = df_r0[(df_r0["blocked_level"] == 0) & (df_r0["cavitation"] < 0.01)]
        print(f"\nRows with bucket_rel=0 that are effectively TRANSPARENT: {len(df_trans)}")
