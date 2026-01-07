"""Continue comprehensive analysis from Batch 5 onwards."""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict

lake_root = Path("/Users/loganrobbins/research/qmachina/spymaster/backend/lake")
sample_path = lake_root / "silver/product_type=future/symbol=ESU5/table=market_by_price_10_pm_high_approach/dt=2025-06-04"

df = pd.read_parquet(sample_path)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns\n")

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
        
        # More permissive thresholds - only flag truly suspicious patterns
        if pct_zero > 95:  # Almost all zeros
            result["status"] = "WARN"
            result["issue"] = f"{pct_zero:.1f}% zeros"
        if col.std() == 0 and field not in ["symbol", "level_type", "level_price"]:
            result["status"] = "WARN"
            result["issue"] = "Zero variance"
    else:
        result["unique"] = int(col.nunique())
    
    if pct_null > 90:  # Almost all nulls
        result["status"] = "WARN"
        result["issue"] = f"{pct_null:.1f}% nulls"
    
    return result

# Continue from Batch 5-10
batches = [
    {
        "name": "Batch 5: Depth Quantities (10 fields)",
        "fields": [
            "bar5s_depth_below_p0_1_qty_eob",
            "bar5s_depth_below_p1_2_qty_twa",
            "bar5s_depth_below_p1_2_qty_eob",
            "bar5s_depth_below_p2_3_qty_twa",
            "bar5s_depth_below_p2_3_qty_eob",
            "bar5s_depth_above_p0_1_qty_twa",
            "bar5s_depth_above_p0_1_qty_eob",
            "bar5s_depth_above_p1_2_qty_twa",
            "bar5s_depth_above_p1_2_qty_eob",
            "bar5s_depth_above_p2_3_qty_twa",
        ]
    },
    {
        "name": "Batch 6: Depth Fractions (10 fields)",
        "fields": [
            "bar5s_depth_above_p2_3_qty_eob",
            "bar5s_depth_below_p0_1_frac_twa",
            "bar5s_depth_below_p0_1_frac_eob",
            "bar5s_depth_below_p1_2_frac_twa",
            "bar5s_depth_below_p1_2_frac_eob",
            "bar5s_depth_below_p2_3_frac_twa",
            "bar5s_depth_below_p2_3_frac_eob",
            "bar5s_depth_above_p0_1_frac_twa",
            "bar5s_depth_above_p0_1_frac_eob",
            "bar5s_depth_above_p1_2_frac_twa",
        ]
    },
    {
        "name": "Batch 7: More Depth Fractions (10 fields)",
        "fields": [
            "bar5s_depth_above_p1_2_frac_eob",
            "bar5s_depth_above_p2_3_frac_twa",
            "bar5s_depth_above_p2_3_frac_eob",
            "bar5s_ladder_ask_gap_max_pts_eob",
            "bar5s_ladder_ask_gap_mean_pts_eob",
            "bar5s_ladder_bid_gap_max_pts_eob",
            "bar5s_ladder_bid_gap_mean_pts_eob",
            "bar5s_shape_bid_px_l00_eob",
            "bar5s_shape_bid_px_l01_eob",
            "bar5s_shape_bid_px_l02_eob",
        ]
    },
    {
        "name": "Batch 8: Bid Shape Prices (10 fields)",
        "fields": [
            "bar5s_shape_bid_px_l03_eob",
            "bar5s_shape_bid_px_l04_eob",
            "bar5s_shape_bid_px_l05_eob",
            "bar5s_shape_bid_px_l06_eob",
            "bar5s_shape_bid_px_l07_eob",
            "bar5s_shape_bid_px_l08_eob",
            "bar5s_shape_bid_px_l09_eob",
            "bar5s_shape_ask_px_l00_eob",
            "bar5s_shape_ask_px_l01_eob",
            "bar5s_shape_ask_px_l02_eob",
        ]
    },
    {
        "name": "Batch 9: Ask Shape Prices (10 fields)",
        "fields": [
            "bar5s_shape_ask_px_l03_eob",
            "bar5s_shape_ask_px_l04_eob",
            "bar5s_shape_ask_px_l05_eob",
            "bar5s_shape_ask_px_l06_eob",
            "bar5s_shape_ask_px_l07_eob",
            "bar5s_shape_ask_px_l08_eob",
            "bar5s_shape_ask_px_l09_eob",
            "bar5s_shape_bid_sz_l00_eob",
            "bar5s_shape_bid_sz_l01_eob",
            "bar5s_shape_bid_sz_l02_eob",
        ]
    },
    {
        "name": "Batch 10: Bid Shape Sizes (10 fields)",
        "fields": [
            "bar5s_shape_bid_sz_l03_eob",
            "bar5s_shape_bid_sz_l04_eob",
            "bar5s_shape_bid_sz_l05_eob",
            "bar5s_shape_bid_sz_l06_eob",
            "bar5s_shape_bid_sz_l07_eob",
            "bar5s_shape_bid_sz_l08_eob",
            "bar5s_shape_bid_sz_l09_eob",
            "bar5s_shape_ask_sz_l00_eob",
            "bar5s_shape_ask_sz_l01_eob",
            "bar5s_shape_ask_sz_l02_eob",
        ]
    },
]

print("="*80)
print("CONTINUED COMPREHENSIVE FEATURE ANALYSIS")
print("="*80)

issues_found = []
fields_analyzed = 0

for batch in batches:
    print(f"\n{'='*80}")
    print(batch["name"])
    print(f"{'='*80}")
    
    for field in batch["fields"]:
        fields_analyzed += 1
        stats = analyze_field(df, field)
        
        status_symbol = "✅" if stats["status"] == "OK" else "⚠️"
        print(f"{status_symbol} {field}", end="")
        
        if stats.get("issue"):
            print(f" - ⚠️  {stats['issue']}")
            issues_found.append({
                "field": field,
                "batch": batch["name"],
                "issue": stats["issue"],
            })
        else:
            # Just show OK, keep output compact
            print(" - OK")

print(f"\n{'='*80}")
print(f"SUMMARY: {len(issues_found)} potential issues, {fields_analyzed} fields analyzed")
print(f"{'='*80}")

if issues_found:
    print("\nFields to investigate:")
    for issue in issues_found:
        print(f"  - {issue['field']}: {issue['issue']}")
else:
    print("\n✅ All features look good!")

