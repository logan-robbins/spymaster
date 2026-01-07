"""Debug why size_at_level setup features are zero."""
from pathlib import Path
import pandas as pd
import numpy as np

lake_root = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake")
approach_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_pm_high_approach/dt=2025-06-04"

df = pd.read_parquet(approach_path)

# Check if the features exist and what they contain
print("Checking size_at_level features:")
print(f"\nbar5s_lvl_total_size_at_level_eob:")
print(f"  Non-zero: {(df['bar5s_lvl_total_size_at_level_eob'] > 0).sum()}/{ len(df)}")
print(f"  Min: {df['bar5s_lvl_total_size_at_level_eob'].min()}")
print(f"  Max: {df['bar5s_lvl_total_size_at_level_eob'].max()}")
print(f"  Mean: {df['bar5s_lvl_total_size_at_level_eob'].mean():.2f}")

print(f"\nbar5s_setup_size_at_level_start:")
print(f"  All values: {df['bar5s_setup_size_at_level_start'].unique()}")
print(f"  Non-zero: {(df['bar5s_setup_size_at_level_start'] > 0).sum()}")

print(f"\nbar5s_setup_size_at_level_end:")
print(f"  All values: {df['bar5s_setup_size_at_level_end'].unique()}")
print(f"  Non-zero: {(df['bar5s_setup_size_at_level_end'] > 0).sum()}")

print(f"\nbar5s_setup_size_at_level_max:")
print(f"  All values: {df['bar5s_setup_size_at_level_max'].unique()}")
print(f"  Non-zero: {(df['bar5s_setup_size_at_level_max'] > 0).sum()}")

# Let's check one episode manually
first_ep = df['episode_id'].iloc[0]
ep_df = df[df['episode_id'] == first_ep].copy()

print(f"\n{'='*80}")
print(f"Manual check of first episode: {first_ep}")
print(f"{'='*80}")
print(f"Episode length: {len(ep_df)} bars")
print(f"\nSize at level over time:")
print(ep_df[['bar_index_in_episode', 'bar5s_lvl_total_size_at_level_eob', 
             'bar5s_setup_size_at_level_start', 'bar5s_setup_size_at_level_end']].head(20))

# Check if setup features are computed correctly
first_val = ep_df['bar5s_lvl_total_size_at_level_eob'].iloc[0]
last_val = ep_df['bar5s_lvl_total_size_at_level_eob'].iloc[-1]
max_val = ep_df['bar5s_lvl_total_size_at_level_eob'].max()

print(f"\nExpected values:")
print(f"  start (first): {first_val}")
print(f"  end (last): {last_val}")
print(f"  max: {max_val}")

print(f"\nActual values in setup features:")
print(f"  start: {ep_df['bar5s_setup_size_at_level_start'].iloc[0]}")
print(f"  end: {ep_df['bar5s_setup_size_at_level_end'].iloc[0]}")
print(f"  max: {ep_df['bar5s_setup_size_at_level_max'].iloc[0]}")

if ep_df['bar5s_setup_size_at_level_end'].iloc[0] == 0 and last_val > 0:
    print(f"\n🔴 BUG CONFIRMED: Setup features show 0 but should show {last_val}")
elif ep_df['bar5s_setup_size_at_level_end'].iloc[0] == 0 and last_val == 0:
    print(f"\n✅ Correct: Both are zero (level not visible at end)")
else:
    print(f"\n✅ Correct: Setup features match source data")

