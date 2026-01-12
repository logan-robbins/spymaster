#!/usr/bin/env python3
"""
Feature Validator - Main Entry Point
=====================================

Usage:
    python validate.py features.parquet --date 2024-12-18 --p-ref 6840.75
    python validate.py features.csv --date 2024-12-18 --p-ref 6840.75
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas required. Install with: pip install pandas")
    sys.exit(1)

from lib.stats import compute_all_stats
from lib.math_tests import run_all_math_tests
from lib.extremes import find_extreme_values
from lib.dashboard import generate_dashboard
from lib.utils import detect_prefix


def main():
    parser = argparse.ArgumentParser(description="Validate liquidity vacuum features")
    parser.add_argument("features", help="Feature file (parquet or csv)")
    parser.add_argument("--date", required=True, help="Session date YYYY-MM-DD")
    parser.add_argument("--p-ref", type=float, required=True, help="Reference price level")
    parser.add_argument("--output", default=".", help="Output directory (default: current)")
    args = parser.parse_args()
    
    # Load data
    path = Path(args.features)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"FEATURE VALIDATION")
    print(f"{'='*60}")
    
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    
    prefix = detect_prefix(df)
    
    print(f"Date:    {args.date}")
    print(f"P_ref:   {args.p_ref}")
    print(f"File:    {path.name}")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Format:  {'rollup (w0_*)' if prefix else 'raw'}")
    print(f"{'='*60}\n")
    
    # Run validation
    print("[1/4] Computing statistics...")
    stats = compute_all_stats(df, prefix)
    n_features = sum(len(v) for v in stats.values())
    print(f"       {n_features} features across {len(stats)} families")
    
    print("[2/4] Running math tests...")
    tests = run_all_math_tests(df, prefix)
    passed = sum(1 for t in tests if t["passed"])
    failed = len(tests) - passed
    print(f"       {passed} passed, {failed} failed")
    
    print("[3/4] Finding extreme values...")
    extremes = find_extreme_values(df, prefix)
    print(f"       {len(extremes)} key features analyzed")
    
    print("[4/4] Generating dashboard...")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    html_path = out_dir / f"validation_{args.date}.html"
    generate_dashboard(
        stats=stats,
        tests=tests,
        extremes=extremes,
        meta={"date": args.date, "p_ref": args.p_ref, "n": len(df), "cols": len(df.columns)},
        output_path=html_path
    )
    print(f"       Saved: {html_path}")
    
    # Print extremes to console
    print(f"\n{'-'*60}")
    print("EXTREME VALUES (UTC timestamps for TradingView)")
    print(f"{'-'*60}")
    for feat, data in extremes.items():
        print(f"\n{feat}:")
        for e in data["high"][:2]:
            print(f"  HIGH: {e['time']} -> {e['value']:.4f}")
        for e in data["low"][:2]:
            print(f"  LOW:  {e['time']} -> {e['value']:.4f}")
    
    print(f"\n{'='*60}")
    if failed == 0:
        print("✓ ALL MATH TESTS PASSED")
    else:
        print(f"✗ {failed} MATH TESTS FAILED")
    print(f"{'='*60}\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
