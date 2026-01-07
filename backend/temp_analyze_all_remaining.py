"""Analyze all remaining fields comprehensively - report only issues."""
from pathlib import Path
import pandas as pd
import numpy as np

lake_root = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake")
sample_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_pm_high_approach/dt=2025-06-04"

df = pd.read_parquet(sample_path)
print(f"Analyzing {len(df)} rows, {len(df.columns)} columns\n")

# Fields already analyzed (batches 1-10)
analyzed_fields = set([
    # Batch 1
    "bar_ts", "symbol", "episode_id", "touch_id", "level_type", "level_price",
    "trigger_bar_ts", "bar_index_in_episode", "bar_index_in_touch", "bars_to_trigger",
    # Batch 2
    "is_pre_trigger", "is_pre_touch", "is_trigger_bar", "is_post_trigger", "is_post_touch",
    "approach_direction", "is_standard_approach", "dist_to_level_pts", "signed_dist_pts", "outcome",
    # Batch 3
    "outcome_score", "is_truncated_lookback", "is_truncated_forward", "is_extended_forward",
    "extension_count", "bar5s_microprice_eob", "bar5s_midprice_eob", "bar5s_meta_msg_cnt_sum",
    "bar5s_meta_clear_cnt_sum", "bar5s_meta_add_cnt_sum",
    # Batch 4
    "bar5s_meta_cancel_cnt_sum", "bar5s_meta_modify_cnt_sum", "bar5s_meta_trade_cnt_sum",
    "bar5s_state_spread_pts_twa", "bar5s_state_spread_pts_eob", "bar5s_state_obi0_twa",
    "bar5s_state_obi0_eob", "bar5s_state_obi10_twa", "bar5s_state_obi10_eob", "bar5s_state_cdi_p0_1_twa",
    # Batch 5
    "bar5s_depth_below_p0_1_qty_eob", "bar5s_depth_below_p1_2_qty_twa", "bar5s_depth_below_p1_2_qty_eob",
    "bar5s_depth_below_p2_3_qty_twa", "bar5s_depth_below_p2_3_qty_eob", "bar5s_depth_above_p0_1_qty_twa",
    "bar5s_depth_above_p0_1_qty_eob", "bar5s_depth_above_p1_2_qty_twa", "bar5s_depth_above_p1_2_qty_eob",
    "bar5s_depth_above_p2_3_qty_twa",
    # Batch 6
    "bar5s_depth_above_p2_3_qty_eob", "bar5s_depth_below_p0_1_frac_twa", "bar5s_depth_below_p0_1_frac_eob",
    "bar5s_depth_below_p1_2_frac_twa", "bar5s_depth_below_p1_2_frac_eob", "bar5s_depth_below_p2_3_frac_twa",
    "bar5s_depth_below_p2_3_frac_eob", "bar5s_depth_above_p0_1_frac_twa", "bar5s_depth_above_p0_1_frac_eob",
    "bar5s_depth_above_p1_2_frac_twa",
    # Batch 7
    "bar5s_depth_above_p1_2_frac_eob", "bar5s_depth_above_p2_3_frac_twa", "bar5s_depth_above_p2_3_frac_eob",
    "bar5s_ladder_ask_gap_max_pts_eob", "bar5s_ladder_ask_gap_mean_pts_eob", "bar5s_ladder_bid_gap_max_pts_eob",
    "bar5s_ladder_bid_gap_mean_pts_eob", "bar5s_shape_bid_px_l00_eob", "bar5s_shape_bid_px_l01_eob",
    "bar5s_shape_bid_px_l02_eob",
    # Batch 8
    "bar5s_shape_bid_px_l03_eob", "bar5s_shape_bid_px_l04_eob", "bar5s_shape_bid_px_l05_eob",
    "bar5s_shape_bid_px_l06_eob", "bar5s_shape_bid_px_l07_eob", "bar5s_shape_bid_px_l08_eob",
    "bar5s_shape_bid_px_l09_eob", "bar5s_shape_ask_px_l00_eob", "bar5s_shape_ask_px_l01_eob",
    "bar5s_shape_ask_px_l02_eob",
    # Batch 9
    "bar5s_shape_ask_px_l03_eob", "bar5s_shape_ask_px_l04_eob", "bar5s_shape_ask_px_l05_eob",
    "bar5s_shape_ask_px_l06_eob", "bar5s_shape_ask_px_l07_eob", "bar5s_shape_ask_px_l08_eob",
    "bar5s_shape_ask_px_l09_eob", "bar5s_shape_bid_sz_l00_eob", "bar5s_shape_bid_sz_l01_eob",
    "bar5s_shape_bid_sz_l02_eob",
    # Batch 10
    "bar5s_shape_bid_sz_l03_eob", "bar5s_shape_bid_sz_l04_eob", "bar5s_shape_bid_sz_l05_eob",
    "bar5s_shape_bid_sz_l06_eob", "bar5s_shape_bid_sz_l07_eob", "bar5s_shape_bid_sz_l08_eob",
    "bar5s_shape_bid_sz_l09_eob", "bar5s_shape_ask_sz_l00_eob", "bar5s_shape_ask_sz_l01_eob",
    "bar5s_shape_ask_sz_l02_eob",
])

remaining_fields = [f for f in df.columns if f not in analyzed_fields]

print(f"Total fields: {len(df.columns)}")
print(f"Already analyzed: {len(analyzed_fields)}")
print(f"Remaining to analyze: {len(remaining_fields)}\n")

print("="*80)
print("ANALYZING ALL REMAINING FIELDS")
print("="*80)

issues_found = []
fields_ok = 0

for i, field in enumerate(remaining_fields, 1):
    col = df[field]
    n_total = len(col)
    n_null = col.isna().sum()
    pct_null = 100 * n_null / n_total
    
    issue = None
    
    # Check for problematic patterns
    if pct_null > 95:
        issue = f">{pct_null:.0f}% nulls"
    elif col.dtype in ['int64', 'float64', 'int32', 'float32']:
        n_zero = (col == 0).sum()
        pct_zero = 100 * n_zero / n_total
        std = col.std()
        
        if std == 0 and field not in ["symbol", "level_type", "level_price"]:
            issue = "Zero variance"
        elif pct_zero > 98:
            issue = f">{pct_zero:.0f}% zeros"
        elif np.isinf(col).any():
            issue = "Contains inf values"
        elif col.min() == col.max() and col.nunique() == 1:
            issue = f"Constant value: {col.iloc[0]}"
    
    if issue:
        print(f"⚠️  {i:3d}. {field}: {issue}")
        issues_found.append({"field": field, "issue": issue})
    else:
        fields_ok += 1
        if i % 50 == 0:
            print(f"✅  Checked {i}/{len(remaining_fields)} fields - {fields_ok} OK, {len(issues_found)} issues")

print(f"\n{'='*80}")
print(f"FINAL RESULTS")
print(f"{'='*80}")
print(f"Total remaining fields analyzed: {len(remaining_fields)}")
print(f"Fields OK: {fields_ok}")
print(f"Potential issues found: {len(issues_found)}")

if issues_found:
    print(f"\n{'='*80}")
    print("FIELDS REQUIRING INVESTIGATION:")
    print(f"{'='*80}")
    for item in issues_found:
        print(f"  - {item['field']}: {item['issue']}")
else:
    print("\n✅ ALL REMAINING FIELDS LOOK GOOD!")

print(f"\n{'='*80}")
print(f"GRAND TOTAL: {len(analyzed_fields) + len(remaining_fields)} fields analyzed")
print(f"{'='*80}")

