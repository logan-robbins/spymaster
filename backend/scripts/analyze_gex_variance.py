
import pandas as pd
from pathlib import Path

# Path to GEX Surface
gex_path = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake/silver/product_type=future_option_mbo/symbol=ESH6/table=gex_surface_1s/dt=2026-01-06")

if not gex_path.exists():
    print(f"Path not found: {gex_path}")
    exit(1)

parquet_files = list(gex_path.glob("*.parquet"))
if not parquet_files:
    print("No parquet files found.")
    exit(1)

print(f"Loading {parquet_files[0]}...")
df = pd.read_parquet(parquet_files[0])

# Focus on ATM strikes (most gamma sensitivity)
# Find spot reference
spot_ref = df["spot_ref_price_int"].median()
print(f"Median Spot Ref: {spot_ref}")

# Filter to nearby strikes (+- 200 ticks = 50 points = 10 GEX rows)
df_atm = df[abs(df["strike_price_int"] - spot_ref) < 200 * 250_000_000] # ticks * tick_int
print(f"Focused ATM Rows: {len(df_atm)}")

if df_atm.empty:
    print("No ATM rows found!")
    exit(1)

# Group by Strike and Check Variance of GEX over time
print("\n=== Variance Analysis by Strike (Top 5 ATM) ===")
stats = df_atm.groupby("strike_price_int")["gex_abs"].agg(["mean", "std", "min", "max", "count"])
stats["std_pct"] = (stats["std"] / stats["mean"]) * 100
print(stats.sort_values("mean", ascending=False).head(10))

# Check Window-to-Window Change
print("\n=== Window-to-Window Change (Max Delta) ===")
# Pivot: Index=Time, Col=Strike, Val=GEX
piv = df_atm.pivot(index="window_end_ts_ns", columns="strike_price_int", values="gex_abs")
diffs = piv.diff().abs()
max_diff = diffs.max().max()
mean_diff = diffs.mean().mean()

print(f"Max Single-Step Change: {max_diff:e}")
print(f"Mean Single-Step Change: {mean_diff:e}")

if max_diff == 0:
    print("CRITICAL: GEX IS STATIC. Zero change over time.")
else:
    print("GEX shows evolution over time.")
