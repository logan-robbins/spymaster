#!/usr/bin/env python3
"""
Institution-Grade Silver Layer Audit for EQUITY_MBO
Performs thorough semantic and statistical analysis of book_snapshot_1s and depth_and_flow_1s.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Configuration
PRICE_SCALE = 1e-9
BUCKET_SIZE = 0.50
BUCKET_INT = int(round(BUCKET_SIZE / PRICE_SCALE))  # 500_000_000
TOLERANCE = 1.0  # Allow small floating point tolerance

# Paths
BOOK_SNAPSHOT_PATH = Path("lake/silver/product_type=equity_mbo/symbol=QQQ/table=book_snapshot_1s/dt=2026-01-08/part-00000.parquet")
DEPTH_FLOW_PATH = Path("lake/silver/product_type=equity_mbo/symbol=QQQ/table=depth_and_flow_1s/dt=2026-01-08/part-00000.parquet")


def print_header(title: str) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_section(title: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}\n")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both parquet files."""
    print_header("1. LOADING AND INSPECTING DATA")
    
    df_snap = pd.read_parquet(BOOK_SNAPSHOT_PATH)
    df_depth = pd.read_parquet(DEPTH_FLOW_PATH)
    
    print(f"book_snapshot_1s:")
    print(f"  - Rows: {len(df_snap):,}")
    print(f"  - File size: {BOOK_SNAPSHOT_PATH.stat().st_size:,} bytes")
    print(f"  - Columns: {list(df_snap.columns)}")
    print(f"\ndepth_and_flow_1s:")
    print(f"  - Rows: {len(df_depth):,}")
    print(f"  - File size: {DEPTH_FLOW_PATH.stat().st_size:,} bytes")
    print(f"  - Columns: {list(df_depth.columns)}")
    
    return df_snap, df_depth


def describe_schema(df: pd.DataFrame, name: str) -> None:
    """Print schema details."""
    print_section(f"Schema: {name}")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isna().sum()
        print(f"  {col}: {dtype}, nulls={null_count}")


def analyze_book_snapshot_semantics(df: pd.DataFrame) -> dict:
    """Semantic analysis of book_snapshot_1s features."""
    print_header("2. SEMANTIC ANALYSIS: book_snapshot_1s")
    
    issues = []
    
    # window_start_ts_ns, window_end_ts_ns
    print_section("Time Boundaries (window_start_ts_ns, window_end_ts_ns)")
    ts_start = df['window_start_ts_ns'].iloc[0]
    ts_end = df['window_end_ts_ns'].iloc[-1]
    dt_start = datetime.fromtimestamp(ts_start / 1e9, tz=timezone.utc)
    dt_end = datetime.fromtimestamp(ts_end / 1e9, tz=timezone.utc)
    print(f"  First window: {dt_start}")
    print(f"  Last window:  {dt_end}")
    print(f"  Total windows: {len(df):,}")
    
    window_durations = df['window_end_ts_ns'] - df['window_start_ts_ns']
    expected_duration = 1_000_000_000  # 1 second in ns
    bad_durations = (window_durations != expected_duration).sum()
    print(f"  Window duration (expected 1s): {window_durations.iloc[0] / 1e9:.3f}s")
    if bad_durations > 0:
        issues.append(f"ISSUE: {bad_durations} windows with non-1s duration")
        print(f"  ⚠️ {bad_durations} windows with incorrect duration")
    else:
        print(f"  ✅ All windows have 1s duration")
    
    # Check for gaps
    gaps = df['window_start_ts_ns'].iloc[1:].values - df['window_end_ts_ns'].iloc[:-1].values
    gap_count = (gaps != 0).sum()
    if gap_count > 0:
        issues.append(f"ISSUE: {gap_count} gaps between consecutive windows")
        print(f"  ⚠️ {gap_count} gaps between windows")
    else:
        print(f"  ✅ No gaps between consecutive windows")
    
    # BBO prices
    print_section("BBO Prices (best_bid_price_int, best_ask_price_int)")
    valid_mask = df['book_valid'] == True
    valid_df = df[valid_mask]
    
    bid_prices_real = valid_df['best_bid_price_int'] * PRICE_SCALE
    ask_prices_real = valid_df['best_ask_price_int'] * PRICE_SCALE
    
    print(f"  best_bid_price_int range: [{valid_df['best_bid_price_int'].min():,}, {valid_df['best_bid_price_int'].max():,}]")
    print(f"  best_bid_price ($): [{bid_prices_real.min():.2f}, {bid_prices_real.max():.2f}]")
    print(f"  best_ask_price_int range: [{valid_df['best_ask_price_int'].min():,}, {valid_df['best_ask_price_int'].max():,}]")
    print(f"  best_ask_price ($): [{ask_prices_real.min():.2f}, {ask_prices_real.max():.2f}]")
    
    # Check crossed book
    crossed = (valid_df['best_bid_price_int'] >= valid_df['best_ask_price_int']).sum()
    if crossed > 0:
        issues.append(f"CRITICAL: {crossed} windows with crossed book (bid >= ask)")
        print(f"  ⚠️ CROSSED BOOK: {crossed} windows where bid >= ask")
    else:
        print(f"  ✅ No crossed books")
    
    # Spread analysis
    spread_int = valid_df['best_ask_price_int'] - valid_df['best_bid_price_int']
    spread_dollars = spread_int * PRICE_SCALE
    print(f"  Spread ($): min={spread_dollars.min():.4f}, max={spread_dollars.max():.4f}, mean={spread_dollars.mean():.4f}")
    
    # BBO quantities
    print_section("BBO Quantities (best_bid_qty, best_ask_qty)")
    print(f"  best_bid_qty range: [{valid_df['best_bid_qty'].min()}, {valid_df['best_bid_qty'].max():,}]")
    print(f"  best_ask_qty range: [{valid_df['best_ask_qty'].min()}, {valid_df['best_ask_qty'].max():,}]")
    print(f"  best_bid_qty mean: {valid_df['best_bid_qty'].mean():.1f}")
    print(f"  best_ask_qty mean: {valid_df['best_ask_qty'].mean():.1f}")
    
    neg_bid_qty = (valid_df['best_bid_qty'] < 0).sum()
    neg_ask_qty = (valid_df['best_ask_qty'] < 0).sum()
    if neg_bid_qty > 0 or neg_ask_qty > 0:
        issues.append(f"CRITICAL: Negative quantities - bid: {neg_bid_qty}, ask: {neg_ask_qty}")
        print(f"  ⚠️ Negative quantities: bid={neg_bid_qty}, ask={neg_ask_qty}")
    else:
        print(f"  ✅ All quantities >= 0")
    
    # Mid price
    print_section("Mid Price (mid_price, mid_price_int)")
    print(f"  mid_price range ($): [{valid_df['mid_price'].min():.4f}, {valid_df['mid_price'].max():.4f}]")
    print(f"  mid_price_int range: [{valid_df['mid_price_int'].min():,}, {valid_df['mid_price_int'].max():,}]")
    
    # Verify mid_price_int calculation
    expected_mid_int = ((valid_df['best_bid_price_int'] + valid_df['best_ask_price_int']) / 2).round().astype(int)
    mid_mismatch = (valid_df['mid_price_int'] != expected_mid_int).sum()
    if mid_mismatch > 0:
        issues.append(f"ISSUE: {mid_mismatch} mid_price_int mismatches")
        print(f"  ⚠️ {mid_mismatch} mid_price_int != (bid + ask) / 2")
    else:
        print(f"  ✅ mid_price_int = (bid + ask) / 2 verified")
    
    # Verify mid_price vs mid_price_int
    expected_mid_real = valid_df['mid_price_int'] * PRICE_SCALE
    mid_real_diff = (valid_df['mid_price'] - expected_mid_real).abs()
    bad_mid_real = (mid_real_diff > 1e-10).sum()
    if bad_mid_real > 0:
        issues.append(f"ISSUE: {bad_mid_real} mid_price vs mid_price_int inconsistencies")
        print(f"  ⚠️ {bad_mid_real} mid_price inconsistent with mid_price_int")
    else:
        print(f"  ✅ mid_price = mid_price_int * PRICE_SCALE verified")
    
    # Last trade price
    print_section("Last Trade Price (last_trade_price_int)")
    print(f"  last_trade_price_int range: [{df['last_trade_price_int'].min():,}, {df['last_trade_price_int'].max():,}]")
    last_trade_real = df['last_trade_price_int'] * PRICE_SCALE
    print(f"  last_trade_price ($): [{last_trade_real.min():.2f}, {last_trade_real.max():.2f}]")
    
    zero_last_trade = (df['last_trade_price_int'] == 0).sum()
    if zero_last_trade > 0:
        print(f"  Note: {zero_last_trade} windows with last_trade_price_int = 0 (no trades yet)")
    
    # Spot reference price
    print_section("Spot Reference (spot_ref_price_int)")
    print(f"  spot_ref_price_int range: [{df['spot_ref_price_int'].min():,}, {df['spot_ref_price_int'].max():,}]")
    spot_real = df['spot_ref_price_int'] * PRICE_SCALE
    print(f"  spot_ref_price ($): [{spot_real.min():.2f}, {spot_real.max():.2f}]")
    
    # Verify bucketing ($0.50 grid)
    spot_mod = df['spot_ref_price_int'] % BUCKET_INT
    non_bucketed = (spot_mod != 0).sum()
    if non_bucketed > 0:
        issues.append(f"ISSUE: {non_bucketed} spot_ref_price_int not on $0.50 grid")
        print(f"  ⚠️ {non_bucketed} spot_ref_price_int not on $0.50 bucket grid")
    else:
        print(f"  ✅ All spot_ref_price_int on $0.50 bucket grid")
    
    # book_valid flag
    print_section("Book Valid Flag (book_valid)")
    valid_count = df['book_valid'].sum()
    invalid_count = len(df) - valid_count
    print(f"  book_valid=True: {valid_count:,} ({100*valid_count/len(df):.1f}%)")
    print(f"  book_valid=False: {invalid_count:,} ({100*invalid_count/len(df):.1f}%)")
    
    return {"issues": issues}


def analyze_depth_flow_semantics(df: pd.DataFrame) -> dict:
    """Semantic analysis of depth_and_flow_1s features."""
    print_header("2. SEMANTIC ANALYSIS: depth_and_flow_1s")
    
    issues = []
    
    # Basic info
    print_section("Basic Statistics")
    print(f"  Total rows: {len(df):,}")
    unique_windows = df['window_start_ts_ns'].nunique()
    print(f"  Unique time windows: {unique_windows:,}")
    unique_prices = df['price_int'].nunique()
    print(f"  Unique price levels: {unique_prices:,}")
    
    # Side distribution
    print_section("Side Distribution")
    side_counts = df['side'].value_counts()
    for side, count in side_counts.items():
        print(f"  {side}: {count:,} ({100*count/len(df):.1f}%)")
    
    # Price levels
    print_section("Price Levels (price_int)")
    print(f"  price_int range: [{df['price_int'].min():,}, {df['price_int'].max():,}]")
    price_real = df['price_int'] * PRICE_SCALE
    print(f"  price ($): [{price_real.min():.2f}, {price_real.max():.2f}]")
    
    # Verify bucketing
    price_mod = df['price_int'] % BUCKET_INT
    non_bucketed = (price_mod != 0).sum()
    if non_bucketed > 0:
        issues.append(f"ISSUE: {non_bucketed} price_int not on $0.50 grid")
        print(f"  ⚠️ {non_bucketed} price_int not on $0.50 bucket grid")
    else:
        print(f"  ✅ All price_int on $0.50 bucket grid")
    
    # Spot reference
    print_section("Spot Reference (spot_ref_price_int)")
    print(f"  spot_ref_price_int range: [{df['spot_ref_price_int'].min():,}, {df['spot_ref_price_int'].max():,}]")
    
    spot_mod = df['spot_ref_price_int'] % BUCKET_INT
    spot_non_bucketed = (spot_mod != 0).sum()
    if spot_non_bucketed > 0:
        issues.append(f"ISSUE: {spot_non_bucketed} spot_ref_price_int not on $0.50 grid")
        print(f"  ⚠️ {spot_non_bucketed} spot_ref_price_int not on $0.50 bucket grid")
    else:
        print(f"  ✅ All spot_ref_price_int on $0.50 bucket grid")
    
    # Relative ticks
    print_section("Relative Ticks (rel_ticks, rel_ticks_side)")
    print(f"  rel_ticks range: [{df['rel_ticks'].min()}, {df['rel_ticks'].max()}]")
    print(f"  rel_ticks_side range: [{df['rel_ticks_side'].min()}, {df['rel_ticks_side'].max()}]")
    
    # Verify rel_ticks calculation
    expected_rel_ticks = (df['price_int'] - df['spot_ref_price_int']) // BUCKET_INT
    rel_ticks_mismatch = (df['rel_ticks'] != expected_rel_ticks).sum()
    if rel_ticks_mismatch > 0:
        issues.append(f"ISSUE: {rel_ticks_mismatch} rel_ticks mismatches")
        print(f"  ⚠️ {rel_ticks_mismatch} rel_ticks != (price - spot_ref) / bucket")
    else:
        print(f"  ✅ rel_ticks = (price_int - spot_ref_price_int) / BUCKET_INT verified")
    
    # Depth quantities
    print_section("Depth Quantities")
    for col in ['depth_qty_start', 'depth_qty_end', 'add_qty', 'pull_qty', 'fill_qty', 'depth_qty_rest', 'pull_qty_rest']:
        col_data = df[col]
        neg_count = (col_data < 0).sum()
        print(f"  {col}: min={col_data.min():.1f}, max={col_data.max():.1f}, mean={col_data.mean():.1f}, negatives={neg_count}")
        if neg_count > 0:
            issues.append(f"ISSUE: {col} has {neg_count} negative values")
    
    # window_valid flag
    print_section("Window Valid Flag (window_valid)")
    valid_count = df['window_valid'].sum()
    invalid_count = len(df) - valid_count
    print(f"  window_valid=True: {valid_count:,} ({100*valid_count/len(df):.1f}%)")
    print(f"  window_valid=False: {invalid_count:,} ({100*invalid_count/len(df):.1f}%)")
    
    return {"issues": issues}


def statistical_analysis(df_snap: pd.DataFrame, df_depth: pd.DataFrame) -> dict:
    """Statistical validation checks."""
    print_header("3. STATISTICAL ANALYSIS")
    
    issues = []
    
    # ACCOUNTING IDENTITY
    print_section("Accounting Identity: depth_qty_start + add_qty - pull_qty - fill_qty = depth_qty_end")
    
    computed_end = df_depth['depth_qty_start'] + df_depth['add_qty'] - df_depth['pull_qty'] - df_depth['fill_qty']
    residual = (computed_end - df_depth['depth_qty_end']).abs()
    
    print(f"  Residual stats:")
    print(f"    min: {residual.min():.6f}")
    print(f"    max: {residual.max():.6f}")
    print(f"    mean: {residual.mean():.6f}")
    print(f"    median: {residual.median():.6f}")
    
    violations = (residual > TOLERANCE).sum()
    if violations > 0:
        issues.append(f"CRITICAL: {violations} accounting identity violations (tolerance={TOLERANCE})")
        print(f"  ⚠️ {violations} rows violate accounting identity (tolerance={TOLERANCE})")
        # Show some examples
        bad_rows = df_depth[residual > TOLERANCE].head(5)
        print(f"  Example violations:")
        for idx, row in bad_rows.iterrows():
            calc = row['depth_qty_start'] + row['add_qty'] - row['pull_qty'] - row['fill_qty']
            print(f"    Row {idx}: start={row['depth_qty_start']:.1f}, add={row['add_qty']:.1f}, pull={row['pull_qty']:.1f}, fill={row['fill_qty']:.1f}, expected_end={calc:.1f}, actual_end={row['depth_qty_end']:.1f}")
    else:
        print(f"  ✅ All rows satisfy accounting identity")
    
    # Quantity constraints
    print_section("Quantity Constraints (all >= 0)")
    qty_cols = ['depth_qty_start', 'depth_qty_end', 'add_qty', 'pull_qty', 'fill_qty', 'depth_qty_rest', 'pull_qty_rest']
    for col in qty_cols:
        neg_count = (df_depth[col] < 0).sum()
        if neg_count > 0:
            issues.append(f"CRITICAL: {col} has {neg_count} negative values")
            print(f"  ⚠️ {col}: {neg_count} negative values")
        else:
            print(f"  ✅ {col}: all >= 0")
    
    # Mid price verification (on valid rows)
    print_section("Mid Price Verification (book_valid rows)")
    valid_snap = df_snap[df_snap['book_valid'] == True]
    expected_mid = ((valid_snap['best_bid_price_int'] + valid_snap['best_ask_price_int']) / 2).round().astype(int)
    mid_matches = (valid_snap['mid_price_int'] == expected_mid).sum()
    mid_total = len(valid_snap)
    print(f"  mid_price_int matches: {mid_matches}/{mid_total} ({100*mid_matches/mid_total:.1f}%)")
    if mid_matches < mid_total:
        issues.append(f"ISSUE: {mid_total - mid_matches} mid_price_int calculation errors")
    else:
        print(f"  ✅ All mid_price_int correctly calculated")
    
    # Spread positive when book_valid
    print_section("Spread Positive When book_valid")
    spread = valid_snap['best_ask_price_int'] - valid_snap['best_bid_price_int']
    non_positive_spread = (spread <= 0).sum()
    if non_positive_spread > 0:
        issues.append(f"CRITICAL: {non_positive_spread} non-positive spreads when book_valid")
        print(f"  ⚠️ {non_positive_spread} rows with spread <= 0")
    else:
        print(f"  ✅ All spreads positive when book_valid")
    
    # depth_qty_rest <= depth_qty_end
    print_section("Resting Depth Constraint: depth_qty_rest <= depth_qty_end")
    rest_violations = (df_depth['depth_qty_rest'] > df_depth['depth_qty_end'] + TOLERANCE).sum()
    if rest_violations > 0:
        issues.append(f"ISSUE: {rest_violations} rows where depth_qty_rest > depth_qty_end")
        print(f"  ⚠️ {rest_violations} rows violate depth_qty_rest <= depth_qty_end")
    else:
        print(f"  ✅ depth_qty_rest <= depth_qty_end verified")
    
    # pull_qty_rest <= pull_qty
    print_section("Pull Rest Constraint: pull_qty_rest <= pull_qty")
    pull_rest_violations = (df_depth['pull_qty_rest'] > df_depth['pull_qty'] + TOLERANCE).sum()
    if pull_rest_violations > 0:
        issues.append(f"ISSUE: {pull_rest_violations} rows where pull_qty_rest > pull_qty")
        print(f"  ⚠️ {pull_rest_violations} rows violate pull_qty_rest <= pull_qty")
    else:
        print(f"  ✅ pull_qty_rest <= pull_qty verified")
    
    # Bucket grid consistency
    print_section("Bucket Grid Consistency ($0.50)")
    
    # Check price_int on grid
    depth_price_mod = df_depth['price_int'] % BUCKET_INT
    depth_off_grid = (depth_price_mod != 0).sum()
    print(f"  depth_and_flow price_int on $0.50 grid: {len(df_depth) - depth_off_grid}/{len(df_depth)}")
    
    # Check spot_ref on grid
    snap_spot_mod = df_snap['spot_ref_price_int'] % BUCKET_INT
    snap_off_grid = (snap_spot_mod != 0).sum()
    print(f"  book_snapshot spot_ref_price_int on $0.50 grid: {len(df_snap) - snap_off_grid}/{len(df_snap)}")
    
    if depth_off_grid > 0 or snap_off_grid > 0:
        issues.append(f"ISSUE: Grid alignment issues - depth: {depth_off_grid}, snap: {snap_off_grid}")
    else:
        print(f"  ✅ All prices on $0.50 bucket grid")
    
    return {"issues": issues}


def data_quality_report(df_snap: pd.DataFrame, df_depth: pd.DataFrame) -> dict:
    """Data quality analysis."""
    print_header("4. DATA QUALITY REPORT")
    
    issues = []
    
    # Valid flag counts
    print_section("Validity Flag Summary")
    print(f"  book_snapshot_1s:")
    print(f"    book_valid=True: {df_snap['book_valid'].sum():,}")
    print(f"    book_valid=False: {(~df_snap['book_valid']).sum():,}")
    
    print(f"\n  depth_and_flow_1s:")
    print(f"    window_valid=True: {df_depth['window_valid'].sum():,}")
    print(f"    window_valid=False: {(~df_depth['window_valid']).sum():,}")
    
    # NaN detection
    print_section("NaN Detection")
    snap_nans = df_snap.isna().sum()
    if snap_nans.sum() > 0:
        print(f"  book_snapshot_1s NaNs:")
        for col, count in snap_nans.items():
            if count > 0:
                print(f"    {col}: {count}")
                issues.append(f"ISSUE: NaN in book_snapshot.{col}: {count}")
    else:
        print(f"  book_snapshot_1s: No NaNs ✅")
    
    depth_nans = df_depth.isna().sum()
    if depth_nans.sum() > 0:
        print(f"  depth_and_flow_1s NaNs:")
        for col, count in depth_nans.items():
            if count > 0:
                print(f"    {col}: {count}")
                issues.append(f"ISSUE: NaN in depth_and_flow.{col}: {count}")
    else:
        print(f"  depth_and_flow_1s: No NaNs ✅")
    
    # Negative value detection
    print_section("Negative Value Detection")
    
    qty_cols_snap = ['best_bid_qty', 'best_ask_qty']
    for col in qty_cols_snap:
        neg = (df_snap[col] < 0).sum()
        if neg > 0:
            print(f"  book_snapshot.{col}: {neg} negatives ⚠️")
            issues.append(f"CRITICAL: Negative {col}: {neg}")
        else:
            print(f"  book_snapshot.{col}: No negatives ✅")
    
    qty_cols_depth = ['depth_qty_start', 'depth_qty_end', 'add_qty', 'pull_qty', 'fill_qty', 'depth_qty_rest', 'pull_qty_rest']
    for col in qty_cols_depth:
        neg = (df_depth[col] < 0).sum()
        if neg > 0:
            print(f"  depth_and_flow.{col}: {neg} negatives ⚠️")
            issues.append(f"CRITICAL: Negative {col}: {neg}")
        else:
            print(f"  depth_and_flow.{col}: No negatives ✅")
    
    # Impossible states
    print_section("Impossible State Detection")
    
    # Crossed book when valid
    valid_snap = df_snap[df_snap['book_valid'] == True]
    crossed = (valid_snap['best_bid_price_int'] >= valid_snap['best_ask_price_int']).sum()
    if crossed > 0:
        print(f"  Crossed books (bid >= ask): {crossed} ⚠️")
        issues.append(f"CRITICAL: {crossed} crossed books")
    else:
        print(f"  Crossed books: None ✅")
    
    # Zero prices when valid
    zero_bid = (valid_snap['best_bid_price_int'] == 0).sum()
    zero_ask = (valid_snap['best_ask_price_int'] == 0).sum()
    if zero_bid > 0 or zero_ask > 0:
        print(f"  Zero prices when book_valid: bid={zero_bid}, ask={zero_ask} ⚠️")
        issues.append(f"ISSUE: Zero prices when valid: bid={zero_bid}, ask={zero_ask}")
    else:
        print(f"  Zero prices when book_valid: None ✅")
    
    return {"issues": issues}


def generate_summary(all_issues: list[str]) -> str:
    """Generate final summary with grade."""
    print_header("5. FINAL SUMMARY")
    
    critical_issues = [i for i in all_issues if "CRITICAL" in i]
    other_issues = [i for i in all_issues if "CRITICAL" not in i]
    
    # Grading criteria
    # A: No issues
    # B: Minor issues only (< 5)
    # C: Some issues (5-10) or 1 critical
    # D: Many issues or multiple criticals
    # F: Data is unusable
    
    if len(all_issues) == 0:
        grade = "A"
        assessment = "Excellent - No issues found"
    elif len(critical_issues) == 0 and len(other_issues) < 5:
        grade = "B"
        assessment = "Good - Minor issues only"
    elif len(critical_issues) <= 1 and len(all_issues) <= 10:
        grade = "C"
        assessment = "Acceptable - Some issues need attention"
    elif len(critical_issues) <= 3:
        grade = "D"
        assessment = "Poor - Significant issues"
    else:
        grade = "F"
        assessment = "Failing - Data quality unacceptable"
    
    print(f"  GRADE: {grade}")
    print(f"  Assessment: {assessment}")
    print(f"\n  Total Issues: {len(all_issues)}")
    print(f"    Critical: {len(critical_issues)}")
    print(f"    Other: {len(other_issues)}")
    
    if all_issues:
        print(f"\n  Issue List:")
        for i, issue in enumerate(all_issues, 1):
            print(f"    {i}. {issue}")
    
    return grade


def main():
    print(f"=" * 80)
    print(f"  INSTITUTION-GRADE SILVER LAYER AUDIT")
    print(f"  Product Type: equity_mbo")
    print(f"  Symbol: QQQ")
    print(f"  Date: 2026-01-08")
    print(f"  Audit Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"=" * 80)
    
    all_issues = []
    
    # 1. Load data
    df_snap, df_depth = load_data()
    
    # Print schemas
    describe_schema(df_snap, "book_snapshot_1s")
    describe_schema(df_depth, "depth_and_flow_1s")
    
    # 2. Semantic analysis
    snap_result = analyze_book_snapshot_semantics(df_snap)
    all_issues.extend(snap_result["issues"])
    
    depth_result = analyze_depth_flow_semantics(df_depth)
    all_issues.extend(depth_result["issues"])
    
    # 3. Statistical analysis
    stat_result = statistical_analysis(df_snap, df_depth)
    all_issues.extend(stat_result["issues"])
    
    # 4. Data quality report
    quality_result = data_quality_report(df_snap, df_depth)
    all_issues.extend(quality_result["issues"])
    
    # 5. Summary
    grade = generate_summary(all_issues)
    
    print(f"\n{'=' * 80}")
    print(f"  AUDIT COMPLETE")
    print(f"{'=' * 80}\n")
    
    return 0 if grade in ("A", "B") else 1


if __name__ == "__main__":
    sys.exit(main())
