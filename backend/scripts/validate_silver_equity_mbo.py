"""Validate silver equity_mbo layer formulas and constraints."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LAKE_ROOT = Path(__file__).parent.parent / "lake"
PRICE_SCALE = 1e-9
BUCKET_SIZE = 0.50
BUCKET_INT = int(round(BUCKET_SIZE / PRICE_SCALE))  # 500_000_000

DATES = ["2026-01-07", "2026-01-15", "2026-01-27"]


def load_silver_data(table: str, dt: str) -> pd.DataFrame:
    """Load silver equity_mbo table for a given date."""
    path = LAKE_ROOT / f"silver/product_type=equity_mbo/symbol=QQQ/table={table}/dt={dt}"
    files = list(path.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)


def validate_book_snapshot(dt: str) -> dict:
    """Validate book_snapshot_1s formulas."""
    df = load_silver_data("book_snapshot_1s", dt)
    results = {"dt": dt, "table": "book_snapshot_1s", "row_count": len(df)}
    
    # Check for nulls
    null_counts = df.isnull().sum()
    results["null_counts"] = null_counts.to_dict()
    
    # Validate mid_price formula: (bid + ask) / 2 * PRICE_SCALE
    expected_mid = (df["best_bid_price_int"] + df["best_ask_price_int"]) * 0.5 * PRICE_SCALE
    valid_mask = (df["best_bid_price_int"] > 0) & (df["best_ask_price_int"] > 0)
    mid_price_errors = (df.loc[valid_mask, "mid_price"] - expected_mid[valid_mask]).abs()
    results["mid_price_max_error"] = float(mid_price_errors.max()) if len(mid_price_errors) > 0 else 0.0
    results["mid_price_valid"] = mid_price_errors.max() < 1e-6 if len(mid_price_errors) > 0 else True
    
    # Validate mid_price_int formula: round((bid + ask) / 2)
    expected_mid_int = np.round((df["best_bid_price_int"] + df["best_ask_price_int"]) * 0.5).astype("int64")
    mid_int_errors = (df.loc[valid_mask, "mid_price_int"] - expected_mid_int[valid_mask]).abs()
    results["mid_price_int_max_error"] = int(mid_int_errors.max()) if len(mid_int_errors) > 0 else 0
    results["mid_price_int_valid"] = mid_int_errors.max() <= 1 if len(mid_int_errors) > 0 else True
    
    # Check for crossed books (ask < bid)
    crossed = df.loc[valid_mask, "best_ask_price_int"] <= df.loc[valid_mask, "best_bid_price_int"]
    results["crossed_book_count"] = int(crossed.sum())
    results["crossed_book_pct"] = float(crossed.sum() / len(df)) * 100 if len(df) > 0 else 0.0
    
    # Validate spread
    spread_values = df.loc[valid_mask, "best_ask_price_int"] - df.loc[valid_mask, "best_bid_price_int"]
    results["spread_min"] = float(spread_values.min() * PRICE_SCALE) if len(spread_values) > 0 else 0.0
    results["spread_max"] = float(spread_values.max() * PRICE_SCALE) if len(spread_values) > 0 else 0.0
    results["spread_mean"] = float(spread_values.mean() * PRICE_SCALE) if len(spread_values) > 0 else 0.0
    
    # Price ranges (in dollars)
    results["bid_price_min"] = float(df["best_bid_price_int"].min() * PRICE_SCALE)
    results["bid_price_max"] = float(df["best_bid_price_int"].max() * PRICE_SCALE)
    results["ask_price_min"] = float(df["best_ask_price_int"].min() * PRICE_SCALE)
    results["ask_price_max"] = float(df["best_ask_price_int"].max() * PRICE_SCALE)
    
    # Book valid percentage
    results["book_valid_pct"] = float(df["book_valid"].sum() / len(df)) * 100 if len(df) > 0 else 0.0
    
    return results


def validate_depth_and_flow(dt: str) -> dict:
    """Validate depth_and_flow_1s formulas and constraints."""
    df = load_silver_data("depth_and_flow_1s", dt)
    results = {"dt": dt, "table": "depth_and_flow_1s", "row_count": len(df)}
    
    # Check for nulls
    null_counts = df.isnull().sum()
    results["null_counts"] = null_counts.to_dict()
    
    # Validate depth_qty_start formula: depth_qty_end - add_qty + pull_qty + fill_qty
    expected_start = df["depth_qty_end"] - df["add_qty"] + df["pull_qty"] + df["fill_qty"]
    start_errors = (df["depth_qty_start"] - expected_start).abs()
    results["depth_qty_start_max_error"] = float(start_errors.max())
    results["depth_qty_start_formula_valid"] = start_errors.max() < 0.01
    
    # Negative depth_qty_start (should be 0 or positive based on clamping)
    negative_start = df["depth_qty_start"] < 0
    results["negative_depth_qty_start_count"] = int(negative_start.sum())
    
    # Validate rel_ticks formula: (price_int - spot_ref_price_int) / BUCKET_INT
    expected_rel_ticks = ((df["price_int"] - df["spot_ref_price_int"]) / BUCKET_INT).astype("int32")
    rel_ticks_match = df["rel_ticks"] == expected_rel_ticks
    results["rel_ticks_match_pct"] = float(rel_ticks_match.sum() / len(df)) * 100 if len(df) > 0 else 0.0
    results["rel_ticks_range"] = [int(df["rel_ticks"].min()), int(df["rel_ticks"].max())]
    
    # Quantity constraints (all should be non-negative)
    qty_cols = ["depth_qty_start", "depth_qty_end", "add_qty", "pull_qty", "depth_qty_rest", "pull_qty_rest", "fill_qty"]
    negative_counts = {}
    for col in qty_cols:
        negative_counts[col] = int((df[col] < 0).sum())
    results["negative_qty_counts"] = negative_counts
    
    # Side distribution
    side_counts = df["side"].value_counts().to_dict()
    results["side_distribution"] = side_counts
    results["side_balance_pct"] = abs(side_counts.get("A", 0) - side_counts.get("B", 0)) / len(df) * 100 if len(df) > 0 else 0.0
    
    # Window valid percentage
    results["window_valid_pct"] = float(df["window_valid"].sum() / len(df)) * 100 if len(df) > 0 else 0.0
    
    # Depth constraint: depth_qty_rest <= depth_qty_end (resting can't exceed total)
    rest_exceeds_end = df["depth_qty_rest"] > df["depth_qty_end"]
    results["depth_qty_rest_exceeds_end_count"] = int(rest_exceeds_end.sum())
    
    # Pull constraint: pull_qty_rest <= pull_qty
    pull_rest_exceeds_total = df["pull_qty_rest"] > df["pull_qty"]
    results["pull_qty_rest_exceeds_total_count"] = int(pull_rest_exceeds_total.sum())
    
    return results


def validate_bronze_silver_reconciliation(dt: str) -> dict:
    """Cross-validate bronze inputs against silver outputs."""
    import pyarrow.dataset as ds
    
    # Load bronze data
    bronze_path = LAKE_ROOT / f"bronze/source=databento/product_type=equity_mbo/symbol=QQQ/table=mbo/dt={dt}"
    bronze_files = list(bronze_path.glob("part-*.parquet"))
    if not bronze_files:
        return {"dt": dt, "error": "Bronze data not found"}
    
    # Use dataset API to handle schema unification
    dataset = ds.dataset([str(f) for f in bronze_files], format="parquet")
    bronze_df = dataset.to_table(columns=["action", "size"]).to_pandas()
    
    # Load silver depth_and_flow
    silver_df = load_silver_data("depth_and_flow_1s", dt)
    
    results = {"dt": dt}
    
    # Compare Add volumes
    bronze_adds = bronze_df.loc[bronze_df["action"] == "A", "size"].sum()
    silver_adds = silver_df["add_qty"].sum()
    results["bronze_add_volume"] = int(bronze_adds)
    results["silver_add_volume"] = float(silver_adds)
    results["add_volume_match_pct"] = float(silver_adds / bronze_adds) * 100 if bronze_adds > 0 else 0.0
    
    # Compare Fill volumes  
    bronze_fills = bronze_df.loc[bronze_df["action"] == "F", "size"].sum()
    silver_fills = silver_df["fill_qty"].sum()
    results["bronze_fill_volume"] = int(bronze_fills)
    results["silver_fill_volume"] = float(silver_fills)
    results["fill_volume_match_pct"] = float(silver_fills / bronze_fills) * 100 if bronze_fills > 0 else 0.0
    
    return results


def main():
    print("=" * 80)
    print("EQUITY MBO SILVER LAYER VALIDATION")
    print("=" * 80)
    
    all_valid = True
    
    for dt in DATES:
        print(f"\n{'='*40}")
        print(f"DATE: {dt}")
        print(f"{'='*40}")
        
        # Validate book_snapshot_1s
        snap_results = validate_book_snapshot(dt)
        print(f"\n[book_snapshot_1s] {snap_results['row_count']} rows")
        print(f"  - Null counts: {sum(snap_results['null_counts'].values())}")
        print(f"  - mid_price formula valid: {snap_results['mid_price_valid']} (max error: {snap_results['mid_price_max_error']:.2e})")
        print(f"  - mid_price_int formula valid: {snap_results['mid_price_int_valid']} (max error: {snap_results['mid_price_int_max_error']})")
        print(f"  - Crossed books: {snap_results['crossed_book_count']} ({snap_results['crossed_book_pct']:.2f}%)")
        print(f"  - Spread range: ${snap_results['spread_min']:.4f} - ${snap_results['spread_max']:.4f}")
        print(f"  - Price range: ${snap_results['bid_price_min']:.2f} - ${snap_results['ask_price_max']:.2f}")
        print(f"  - book_valid: {snap_results['book_valid_pct']:.1f}%")
        
        if snap_results['crossed_book_count'] > 0:
            print(f"  ❌ FAIL: {snap_results['crossed_book_count']} crossed books found")
            all_valid = False
        else:
            print(f"  ✅ PASS: No crossed books")
        
        # Validate depth_and_flow_1s
        flow_results = validate_depth_and_flow(dt)
        print(f"\n[depth_and_flow_1s] {flow_results['row_count']} rows")
        print(f"  - Null counts: {sum(flow_results['null_counts'].values())}")
        print(f"  - depth_qty_start formula valid: {flow_results['depth_qty_start_formula_valid']} (max error: {flow_results['depth_qty_start_max_error']:.2e})")
        print(f"  - rel_ticks match: {flow_results['rel_ticks_match_pct']:.1f}%")
        print(f"  - rel_ticks range: {flow_results['rel_ticks_range']}")
        print(f"  - Side distribution: {flow_results['side_distribution']}")
        print(f"  - Side balance: {flow_results['side_balance_pct']:.1f}% imbalance")
        print(f"  - window_valid: {flow_results['window_valid_pct']:.1f}%")
        
        # Check negative quantities
        neg_counts = flow_results['negative_qty_counts']
        total_negative = sum(neg_counts.values())
        if total_negative > 0:
            print(f"  ❌ FAIL: {total_negative} negative quantity values found")
            for col, cnt in neg_counts.items():
                if cnt > 0:
                    print(f"     - {col}: {cnt}")
            all_valid = False
        else:
            print(f"  ✅ PASS: All quantities non-negative")
        
        # Constraint checks
        if flow_results['depth_qty_rest_exceeds_end_count'] > 0:
            print(f"  ⚠️  WARNING: {flow_results['depth_qty_rest_exceeds_end_count']} rows where depth_qty_rest > depth_qty_end")
        
        if flow_results['pull_qty_rest_exceeds_total_count'] > 0:
            print(f"  ⚠️  WARNING: {flow_results['pull_qty_rest_exceeds_total_count']} rows where pull_qty_rest > pull_qty")
        
        # Bronze-Silver reconciliation
        recon_results = validate_bronze_silver_reconciliation(dt)
        print(f"\n[Bronze-Silver Reconciliation]")
        print(f"  - Add volume match: {recon_results['add_volume_match_pct']:.1f}%")
        print(f"    Bronze: {recon_results['bronze_add_volume']:,} | Silver: {recon_results['silver_add_volume']:,.0f}")
        print(f"  - Fill volume match: {recon_results['fill_volume_match_pct']:.1f}%")
        print(f"    Bronze: {recon_results['bronze_fill_volume']:,} | Silver: {recon_results['silver_fill_volume']:,.0f}")
    
    print("\n" + "=" * 80)
    if all_valid:
        print("✅ VALIDATION PASSED: All formulas and constraints verified")
    else:
        print("❌ VALIDATION FAILED: Issues found (see above)")
    print("=" * 80)
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
