"""Investigate bar5s_meta_clear_cnt_sum - why is it always zero?"""
from pathlib import Path
import pandas as pd

# Trace back to source: bar5s features come from market_by_price_10_bar5s
# Which comes from market_by_price_10_clean (the bronze/silver source)

lake_root = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake")

# Check the bar5s table (intermediate stage before approach)
bar5s_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_bar5s/dt=2025-06-04"
print(f"Checking bar5s source: {bar5s_path}")

df_bar5s = pd.read_parquet(bar5s_path)
print(f"Loaded {len(df_bar5s)} rows")

# Check if meta_clear_cnt_sum exists and what its values are
if "bar5s_meta_clear_cnt_sum" in df_bar5s.columns:
    clear_cnt = df_bar5s["bar5s_meta_clear_cnt_sum"]
    print(f"\nbar5s_meta_clear_cnt_sum stats:")
    print(f"  Total rows: {len(clear_cnt)}")
    print(f"  Zeros: {(clear_cnt == 0).sum()} ({100*(clear_cnt == 0).sum()/len(clear_cnt):.1f}%)")
    print(f"  Non-zeros: {(clear_cnt != 0).sum()}")
    print(f"  Min: {clear_cnt.min()}")
    print(f"  Max: {clear_cnt.max()}")
    print(f"  Mean: {clear_cnt.mean()}")
    print(f"  Sum: {clear_cnt.sum()}")
    
    if clear_cnt.sum() == 0:
        print(f"\n⚠️  CONFIRMED: bar5s_meta_clear_cnt_sum is always zero in source data")
        print(f"    This suggests either:")
        print(f"    1. No Clear events in the MBP-10 data (possible - Clear events are rare)")
        print(f"    2. Clear events are not being counted correctly in compute_bar5s_features.py")
        
        # Check the raw MBP-10 data to see if Clear events exist
        mbp10_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_clean/dt=2025-06-04"
        print(f"\nChecking raw MBP-10 data: {mbp10_path}")
        
        df_mbp10 = pd.read_parquet(mbp10_path)
        print(f"Loaded {len(df_mbp10)} raw MBP-10 rows")
        
        if "action" in df_mbp10.columns:
            action_counts = df_mbp10["action"].value_counts()
            print(f"\nAction type distribution in raw MBP-10:")
            for action, count in action_counts.items():
                print(f"  {action}: {count} ({100*count/len(df_mbp10):.2f}%)")
            
            clear_count = (df_mbp10["action"] == "C").sum()
            if clear_count == 0:
                print(f"\n✅ EXPLANATION: No Clear ('C') events exist in raw MBP-10 data")
                print(f"    This is NORMAL - Clear events are rare in ES futures")
            else:
                print(f"\n🔴 PROBLEM: {clear_count} Clear events exist but not counted!")
else:
    print(f"ERROR: bar5s_meta_clear_cnt_sum not found in bar5s data")

