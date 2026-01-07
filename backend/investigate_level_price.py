"""Investigate level_price issue."""
from pathlib import Path
import pandas as pd

def investigate_level_price():
    # Load the silver approach data
    lake_root = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake")
    approach_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_pm_high_approach/dt=2025-06-04"
    df_approach = pd.read_parquet(approach_path)

    print("APPROACH DATA INVESTIGATION:")
    print(f"Total episodes: {len(df_approach)}")
    print(f"Unique level_price values: {df_approach['level_price'].nunique()}")
    print(f"Level_price value: {df_approach['level_price'].iloc[0]}")
    print(f"Level type: {df_approach['level_type'].iloc[0]}")

    # Check if we have episode level data
    episodes_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_pm_high_episodes/dt=2025-06-04"
    if episodes_path.exists():
        df_episodes = pd.read_parquet(episodes_path)
        print(f"\nEPISODES DATA INVESTIGATION:")
        print(f"Total episodes: {len(df_episodes)}")
        print(f"Unique level_price values in episodes: {df_episodes['level_price'].nunique()}")
        print(f"Level_price values in episodes: {df_episodes['level_price'].unique()}")

    # Check the levels data
    levels_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_with_levels/dt=2025-06-04"
    if levels_path.exists():
        df_levels = pd.read_parquet(levels_path)
        print("\nLEVELS DATA INVESTIGATION:")
        print(f"Total level records: {len(df_levels)}")
        pm_high_levels = df_levels[df_levels['level_type'] == 'PM_HIGH']
        print(f"PM_HIGH level records: {len(pm_high_levels)}")
        if len(pm_high_levels) > 0:
            print(f"PM_HIGH level prices: {pm_high_levels['level_price'].unique()}")

    # Check what PM_HIGH means - should be previous day high
    bar5s_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_bar5s/dt=2025-06-04"
    if bar5s_path.exists():
        df_bar5s = pd.read_parquet(bar5s_path)
        print("\nBAR5S DATA INVESTIGATION:")
        print(f"Total bars: {len(df_bar5s)}")

        # Look for PM_HIGH calculation
        if 'pm_high' in df_bar5s.columns:
            pm_high_val = df_bar5s['pm_high'].iloc[0]
            print(f"PM_HIGH from bar5s: {pm_high_val}")

        # Look for high prices in the data
        if 'bar5s_shape_ask_px_l00_eob' in df_bar5s.columns:
            max_price = df_bar5s['bar5s_shape_ask_px_l00_eob'].max()
            min_price = df_bar5s['bar5s_shape_bid_px_l00_eob'].min()
            print(f"Price range in data: {min_price} - {max_price}")

if __name__ == "__main__":
    investigate_level_price()
