"""Comprehensive feature analysis - batch by batch."""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict

lake_root = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake")
sample_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_pm_high_approach/dt=2025-06-04"

df = pd.read_parquet(sample_path)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

all_fields = df.columns.tolist()
print(f"Total fields: {len(all_fields)}\n")

def analyze_field(df: pd.DataFrame, field: str) -> Dict:
    """Analyze a single field and return stats."""
    if field not in df.columns:
        return {"status": "MISSING", "issue": "Field not in data"}
    
    col = df[field]
    n_total = len(col)
    n_null = col.isna().sum()
    pct_null = 100 * n_null / n_total
    
    result = {
        "field": field,
        "dtype": str(col.dtype),
        "null_count": n_null,
        "null_pct": pct_null,
        "status": "OK",
        "issue": None,
    }
    
    if col.dtype in ['int64', 'float64', 'int32', 'float32']:
        n_zero = (col == 0).sum()
        pct_zero = 100 * n_zero / n_total
        
        result.update({
            "zero_count": n_zero,
            "zero_pct": pct_zero,
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "unique": int(col.nunique()),
        })
        
        # Flag issues
        if pct_zero > 50:
            result["status"] = "WARN"
            result["issue"] = f"{pct_zero:.1f}% zeros"
        if col.std() == 0:
            # Check if this is expected (metadata fields with constant values)
            if field not in ["symbol", "level_type", "level_price"]:
                result["status"] = "WARN"
                result["issue"] = "Zero variance"
    else:
        result["unique"] = int(col.nunique())
    
    if pct_null > 50:
        result["status"] = "WARN"
        result["issue"] = f"{pct_null:.1f}% nulls"
    
    return result

# Define batches - first 5 batches for now
batches = [
    {
        "name": "Batch 2: Status Flags (10 fields)",
        "fields": [
            "is_pre_trigger",
            "is_pre_touch",
            "is_trigger_bar",
            "is_post_trigger",
            "is_post_touch",
            "approach_direction",
            "is_standard_approach",
            "dist_to_level_pts",
            "signed_dist_pts",
            "outcome",
        ]
    },
    {
        "name": "Batch 3: Outcome and Prices (10 fields)",
        "fields": [
            "outcome_score",
            "is_truncated_lookback",
            "is_truncated_forward",
            "is_extended_forward",
            "extension_count",
            "bar5s_microprice_eob",
            "bar5s_midprice_eob",
            "bar5s_meta_msg_cnt_sum",
            "bar5s_meta_clear_cnt_sum",
            "bar5s_meta_add_cnt_sum",
        ]
    },
    {
        "name": "Batch 4: Meta Counts and State (10 fields)",
        "fields": [
            "bar5s_meta_cancel_cnt_sum",
            "bar5s_meta_modify_cnt_sum",
            "bar5s_meta_trade_cnt_sum",
            "bar5s_state_spread_pts_twa",
            "bar5s_state_spread_pts_eob",
            "bar5s_state_obi0_twa",
            "bar5s_state_obi0_eob",
            "bar5s_state_obi10_twa",
            "bar5s_state_obi10_eob",
            "bar5s_state_cdi_p0_1_twa",
        ]
    },
    {
        "name": "Batch 5: CDI and Depth (10 fields)",
        "fields": [
            "bar5s_state_cdi_p0_1_eob",
            "bar5s_state_cdi_p1_2_twa",
            "bar5s_state_cdi_p1_2_eob",
            "bar5s_state_cdi_p2_3_twa",
            "bar5s_state_cdi_p2_3_eob",
            "bar5s_depth_bid10_qty_twa",
            "bar5s_depth_bid10_qty_eob",
            "bar5s_depth_ask10_qty_twa",
            "bar5s_depth_ask10_qty_eob",
            "bar5s_depth_below_p0_1_qty_twa",
        ]
    },
]

print("="*80)
print("COMPREHENSIVE FEATURE ANALYSIS")
print("="*80)

issues_found = []

for batch in batches:
    print(f"\n{'='*80}")
    print(batch["name"])
    print(f"{'='*80}")
    
    for field in batch["fields"]:
        stats = analyze_field(df, field)
        
        status_symbol = "✅" if stats["status"] == "OK" else "⚠️"
        print(f"\n{status_symbol} {field}:")
        print(f"    dtype: {stats['dtype']}, nulls: {stats['null_pct']:.1f}%", end="")
        
        if stats.get("zero_pct") is not None:
            print(f", zeros: {stats['zero_pct']:.1f}%")
            print(f"    range: [{stats['min']:.2f}, {stats['max']:.2f}], mean: {stats['mean']:.2f}, std: {stats['std']:.2f}")
            print(f"    unique: {stats['unique']}")
        else:
            print(f", unique: {stats['unique']}")
        
        if stats["issue"]:
            print(f"    🔍 ISSUE: {stats['issue']}")
            issues_found.append({
                "field": field,
                "batch": batch["name"],
                "issue": stats["issue"],
                "stats": stats
            })

print(f"\n{'='*80}")
print(f"SUMMARY: {len(issues_found)} issues found across {len(batches)} batches")
print(f"{'='*80}")

if issues_found:
    print("\nISSUES TO INVESTIGATE:")
    for issue in issues_found:
        print(f"  - {issue['field']}: {issue['issue']}")
else:
    print("\n✅ All features look good!")

print(f"\nAnalyzed {sum(len(b['fields']) for b in batches)} fields total")

