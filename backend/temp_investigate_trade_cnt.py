"""Quick check: Is 70% zeros in trade count expected?"""
from pathlib import Path
import pandas as pd

lake_root = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake")

# Check raw MBP-10 data
mbp10_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_clean/dt=2025-06-04"
df_mbp10 = pd.read_parquet(mbp10_path)

# ES futures have 5-second bars, so 12 bars per minute, 720 bars per hour
# During a trading day, many bars may not have trades (especially in quieter periods)

trade_count = (df_mbp10["action"] == "T").sum()
total_rows = len(df_mbp10)

print(f"Raw MBP-10 data:")
print(f"  Total events: {total_rows:,}")
print(f"  Trade events: {trade_count:,} ({100*trade_count/total_rows:.2f}%)")

# Check bar5s
bar5s_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_bar5s/dt=2025-06-04"
df_bar5s = pd.read_parquet(bar5s_path)

trade_cnt_sum = df_bar5s["bar5s_meta_trade_cnt_sum"]
bars_with_trades = (trade_cnt_sum > 0).sum()
bars_without_trades = (trade_cnt_sum == 0).sum()
total_bars = len(df_bar5s)

print(f"\nBar5s aggregated:")
print(f"  Total bars: {total_bars:,}")
print(f"  Bars WITH trades: {bars_with_trades:,} ({100*bars_with_trades/total_bars:.1f}%)")
print(f"  Bars WITHOUT trades: {bars_without_trades:,} ({100*bars_without_trades/total_bars:.1f}%)")
print(f"  Total trades across all bars: {trade_cnt_sum.sum():.0f}")

if trade_cnt_sum.sum() == trade_count:
    print(f"\n✅ Trade counting is CORRECT: {trade_cnt_sum.sum():.0f} matches raw data")
else:
    print(f"\n⚠️  Trade count mismatch:")
    print(f"    Raw data: {trade_count}")
    print(f"    Bar5s sum: {trade_cnt_sum.sum():.0f}")
    print(f"    Difference: {abs(trade_cnt_sum.sum() - trade_count):.0f}")

# Now check the approach features (filtered to first 3 hours)
approach_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_pm_high_approach/dt=2025-06-04"
df_approach = pd.read_parquet(approach_path)

approach_trade_cnt = df_approach["bar5s_meta_trade_cnt_sum"]
approach_bars_with_trades = (approach_trade_cnt > 0).sum()
approach_bars_without_trades = (approach_trade_cnt == 0).sum()
approach_total_bars = len(df_approach)

print(f"\nApproach features (PM_HIGH episodes only):")
print(f"  Total bars: {approach_total_bars:,}")
print(f"  Bars WITH trades: {approach_bars_with_trades:,} ({100*approach_bars_with_trades/approach_total_bars:.1f}%)")
print(f"  Bars WITHOUT trades: {approach_bars_without_trades:,} ({100*approach_bars_without_trades/approach_total_bars:.1f}%)")

print(f"\n💡 CONCLUSION:")
if bars_without_trades / total_bars > 0.5:
    print(f"   70% zeros is EXPECTED - many 5-second bars have no trades")
    print(f"   This is NORMAL market behavior, especially in liquid futures")
else:
    print(f"   Unexpected pattern - needs investigation")

