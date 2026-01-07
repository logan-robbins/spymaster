"""Check which bars are included in setup computations."""
from pathlib import Path
import pandas as pd

lake_root = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake")
approach_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_pm_high_approach/dt=2025-06-04"

df = pd.read_parquet(approach_path)

# Check the structure of episodes
print("Episode structure:")
print(f"Total bars: {len(df)}")
print(f"Pre-trigger bars: {df['is_pre_trigger'].sum()}")
print(f"Trigger bars: {df['is_trigger_bar'].sum()}")
print(f"Post-trigger bars: {df['is_post_trigger'].sum()}")

# Check one full episode
first_ep = df['episode_id'].iloc[0]
ep_df = df[df['episode_id'] == first_ep].copy()

print(f"\nFirst episode: {first_ep}")
print(f"  Total bars: {len(ep_df)}")
print(f"  Pre-trigger: {ep_df['is_pre_trigger'].sum()}")
print(f"  Trigger: {ep_df['is_trigger_bar'].sum()}")
print(f"  Post-trigger: {ep_df['is_post_trigger'].sum()}")

# Look at the actual size at level around trigger
trigger_idx = ep_df[ep_df['is_trigger_bar']].index[0]
start_idx = max(0, trigger_idx - 5)
end_idx = min(len(ep_df), trigger_idx + 6)

print(f"\nAround trigger bar (bars {start_idx}-{end_idx}):")
print(ep_df.loc[start_idx:end_idx, ['bar_index_in_episode', 'bars_to_trigger', 
                                      'is_pre_trigger', 'is_trigger_bar', 'is_post_trigger',
                                      'bar5s_lvl_total_size_at_level_eob']])

# The key insight: Setup features use groupby on "touch_id" which groups ALL bars in episode
# But maybe the _end value is computed from pre-trigger only?
print(f"\n{'='*80}")
print("Hypothesis: Are setup features computed only from PRE-TRIGGER bars?")
print(f"{'='*80}")

# Check if that would explain it
pre_trigger_df = ep_df[ep_df['is_pre_trigger']].copy()
if len(pre_trigger_df) > 0:
    first_pre = pre_trigger_df['bar5s_lvl_total_size_at_level_eob'].iloc[0]
    last_pre = pre_trigger_df['bar5s_lvl_total_size_at_level_eob'].iloc[-1]
    max_pre = pre_trigger_df['bar5s_lvl_total_size_at_level_eob'].max()
    
    print(f"\nPre-trigger bars only:")
    print(f"  First: {first_pre}")
    print(f"  Last: {last_pre}")
    print(f"  Max: {max_pre}")
    
    setup_start = ep_df['bar5s_setup_size_at_level_start'].iloc[0]
    setup_end = ep_df['bar5s_setup_size_at_level_end'].iloc[0]
    setup_max = ep_df['bar5s_setup_size_at_level_max'].iloc[0]
    
    print(f"\nActual setup features:")
    print(f"  Start: {setup_start}")
    print(f"  End: {setup_end}")
    print(f"  Max: {setup_max}")
    
    if setup_end == last_pre:
        print(f"\n✅ CONFIRMED: _end uses LAST PRE-TRIGGER bar")
    elif setup_end == 0 and last_pre != 0:
        print(f"\n🔴 BUG: _end is 0 but last pre-trigger is {last_pre}")

