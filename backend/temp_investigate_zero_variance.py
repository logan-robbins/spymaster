"""Investigate the 4 zero-variance fields."""
from pathlib import Path
import pandas as pd

lake_root = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake")
sample_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_pm_high_approach/dt=2025-06-04"

df = pd.read_parquet(sample_path)

fields_to_check = [
    "bar5s_setup_size_at_level_end",
    "bar5s_setup_size_at_level_recent12_sum",
    "rvol_flow_net_bid_ratio",
    "rvol_flow_net_ask_ratio",
]

print("="*80)
print("INVESTIGATING ZERO-VARIANCE FIELDS")
print("="*80)

for field in fields_to_check:
    print(f"\n{field}:")
    col = df[field]
    
    print(f"  dtype: {col.dtype}")
    print(f"  unique values: {col.nunique()}")
    print(f"  constant value: {col.iloc[0]}")
    print(f"  min: {col.min()}")
    print(f"  max: {col.max()}")
    print(f"  nulls: {col.isna().sum()}/{len(col)}")
    
    # Context check - what are related fields doing?
    if "size_at_level" in field:
        # Check the size at level visibility feature
        visible = df["bar5s_lvl_level_is_visible"]
        print(f"\n  Context: bar5s_lvl_level_is_visible")
        print(f"    Values: {visible.unique()}")
        print(f"    Counts: {visible.value_counts().to_dict()}")
        
        # Check total size at level
        total_size = df["bar5s_lvl_total_size_at_level_eob"]
        print(f"\n  Context: bar5s_lvl_total_size_at_level_eob")
        print(f"    Non-zero: {(total_size > 0).sum()}/{len(total_size)}")
        print(f"    Min: {total_size.min()}, Max: {total_size.max()}")
        print(f"    Mean: {total_size.mean():.2f}")
        
        if (total_size > 0).sum() == 0:
            print(f"\n  ✅ EXPLANATION: Level (PM_HIGH at 6050.5) is never visible in MBP-10")
            print(f"     book during this day. It's outside the 10-level range.")
            print(f"     Zero variance is EXPECTED.")
    
    elif "rvol_flow_net" in field:
        # Check if this is a calculation issue or data issue
        print(f"\n  Checking source data...")
        
        # Check raw flow values
        flow_bid_cols = [c for c in df.columns if c.startswith("bar5s_flow_net_vol_bid_")]
        flow_ask_cols = [c for c in df.columns if c.startswith("bar5s_flow_net_vol_ask_")]
        
        if flow_bid_cols:
            net_bid_total = sum(df[c].sum() for c in flow_bid_cols)
            print(f"    Total net bid flow across all bands: {net_bid_total:.2f}")
        
        if flow_ask_cols:
            net_ask_total = sum(df[c].sum() for c in flow_ask_cols)
            print(f"    Total net ask flow across all bands: {net_ask_total:.2f}")
        
        # The rvol_flow_net features compare to volume profile
        # If the profile mean is zero, ratio would be 1.0 (the constant)
        if col.iloc[0] == 1.0:
            print(f"\n  ✅ LIKELY EXPLANATION: Volume profile has zero mean net flow,")
            print(f"     so ratio defaults to 1.0. This could be correct behavior.")

print(f"\n{'='*80}")
print("INVESTIGATION COMPLETE")
print(f"{'='*80}")

