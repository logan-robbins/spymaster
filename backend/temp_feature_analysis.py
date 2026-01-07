"""Temporary script to analyze final silver output features one batch at a time."""
from pathlib import Path
import pandas as pd
import numpy as np

# Load sample data from one approach table
lake_root = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake")
sample_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_pm_high_approach/dt=2025-06-04"

if not sample_path.exists():
    print(f"ERROR: Path does not exist: {sample_path}")
    exit(1)

df = pd.read_parquet(sample_path)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"\nFirst episode: {df['episode_id'].iloc[0]}")
print(f"Date range: {pd.to_datetime(df['bar_ts'].min(), unit='ns')} to {pd.to_datetime(df['bar_ts'].max(), unit='ns')}")

# Define all field batches
all_fields = df.columns.tolist()
print(f"\nTotal fields: {len(all_fields)}")

# Batch 1: First 10 metadata fields
batch_1_fields = [
    "bar_ts",
    "symbol",
    "episode_id",
    "touch_id",
    "level_type",
    "level_price",
    "trigger_bar_ts",
    "bar_index_in_episode",
    "bar_index_in_touch",
    "bars_to_trigger",
]

print("\n" + "="*80)
print("BATCH 1: Metadata Fields")
print("="*80)

for field in batch_1_fields:
    if field not in df.columns:
        print(f"\n{field}: MISSING FROM DATA")
        continue
    
    col = df[field]
    n_total = len(col)
    n_null = col.isna().sum()
    n_zero = (col == 0).sum() if col.dtype in ['int64', 'float64'] else 0
    
    print(f"\n{field}:")
    print(f"  dtype: {col.dtype}")
    print(f"  null: {n_null}/{n_total} ({100*n_null/n_total:.1f}%)")
    
    if col.dtype in ['int64', 'float64', 'int32', 'float32']:
        print(f"  zero: {n_zero}/{n_total} ({100*n_zero/n_total:.1f}%)")
        print(f"  min: {col.min()}")
        print(f"  max: {col.max()}")
        print(f"  mean: {col.mean():.2f}")
        print(f"  std: {col.std():.2f}")
        print(f"  unique: {col.nunique()}")
    else:
        print(f"  unique: {col.nunique()}")
        print(f"  sample values: {col.unique()[:5].tolist()}")
    
    # Flag suspicious patterns
    if col.dtype in ['int64', 'float64'] and n_zero / n_total > 0.5:
        print(f"  ⚠️  WARNING: >50% zeros")
    if n_null / n_total > 0.5:
        print(f"  ⚠️  WARNING: >50% nulls")
    if col.dtype in ['int64', 'float64'] and col.std() == 0:
        print(f"  ⚠️  WARNING: Zero variance (constant value)")

print("\n" + "="*80)
print("Analysis complete for Batch 1")
print("="*80)

