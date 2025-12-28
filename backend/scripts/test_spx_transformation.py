"""
Test SPX transformation and validate v1 Final Call implementation.

Quick smoke test to verify:
1. SPX options data is available in Bronze
2. ES front-month filtering works
3. SPX OHLCV builds correctly (no /10 division)
4. Price range is correct (SPX ~5700-5800, not ~570-580)
5. Pipeline runs without errors

Usage:
    uv run python scripts/test_spx_transformation.py --date 2025-12-16
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from src.lake.bronze_writer import BronzeReader
from src.common.utils.contract_selector import ContractSelector
from src.pipeline.stages.build_spx_ohlcv import build_spx_ohlcv_from_es
from src.pipeline.stages.load_bronze import futures_trades_from_df


def test_spx_data_available(bronze_root: str, date: str) -> bool:
    """Test that SPX options data exists."""
    print(f"\n1. Testing SPX options data availability...")
    
    spx_trades_dir = Path(bronze_root) / 'options' / 'trades' / 'underlying=SPX' / f'date={date}'
    spx_nbbo_dir = Path(bronze_root) / 'options' / 'nbbo' / 'underlying=SPX' / f'date={date}'
    
    trades_exists = spx_trades_dir.exists() and list(spx_trades_dir.rglob('*.parquet'))
    nbbo_exists = spx_nbbo_dir.exists() and list(spx_nbbo_dir.rglob('*.parquet'))
    
    if trades_exists:
        print(f"  ✅ SPX trades found: {spx_trades_dir}")
        sample_file = list(spx_trades_dir.rglob('*.parquet'))[0]
        df = pd.read_parquet(sample_file)
        print(f"     Sample size: {len(df):,} records")
        print(f"     Columns: {df.columns.tolist()}")
    else:
        print(f"  ❌ SPX trades NOT found: {spx_trades_dir}")
        return False
    
    if nbbo_exists:
        print(f"  ✅ SPX NBBO found: {spx_nbbo_dir}")
    else:
        print(f"  ⚠️  SPX NBBO NOT found: {spx_nbbo_dir}")
        print(f"     (Optional, but recommended for better GEX computation)")
    
    return True


def test_es_front_month_filtering(bronze_root: str, date: str) -> bool:
    """Test ES front-month selector."""
    print(f"\n2. Testing ES front-month filtering...")
    
    selector = ContractSelector(bronze_root)
    
    try:
        selection = selector.select_front_month(date)
        print(f"  ✅ Front-month: {selection.front_month_symbol}")
        print(f"     Dominance: {selection.dominance_ratio:.1%}")
        print(f"     Roll contaminated: {selection.roll_contaminated}")
        
        if selection.roll_contaminated:
            print(f"  ⚠️  WARNING: Roll contamination detected")
            if selection.runner_up_symbol:
                print(f"     Runner-up: {selection.runner_up_symbol} ({selection.runner_up_ratio:.1%})")
        
        return not selection.roll_contaminated
        
    except Exception as e:
        print(f"  ❌ Front-month selection failed: {e}")
        return False


def test_spx_ohlcv_conversion(bronze_root: str, date: str) -> bool:
    """Test that SPX OHLCV is in correct units (index points, not dollars)."""
    print(f"\n3. Testing SPX OHLCV conversion...")
    
    reader = BronzeReader(data_root=bronze_root.replace('/bronze', ''))
    
    try:
        # Load ES futures trades with front-month filtering
        trades_df = reader.read_futures_trades(
            symbol='ES',
            date=date,
            front_month_only=True
        )
        
        if trades_df.empty:
            print(f"  ❌ No ES trades found")
            return False
        
        print(f"  ✅ Loaded {len(trades_df):,} ES trades")
        
        # Build SPX OHLCV
        trades = futures_trades_from_df(trades_df)
        ohlcv_df = build_spx_ohlcv_from_es(trades, date, freq='1min', rth_only=True)
        
        if ohlcv_df.empty:
            print(f"  ❌ No OHLCV bars generated")
            return False
        
        print(f"  ✅ Built {len(ohlcv_df)} SPX OHLCV bars")
        
        # Check price range
        price_min = ohlcv_df['low'].min()
        price_max = ohlcv_df['high'].max()
        
        print(f"  SPX price range: {price_min:.2f} - {price_max:.2f}")
        
        # SPX should be in 5000-6000 range (Dec 2025), NOT 500-600
        if 5000 < price_min < 6000 and 5000 < price_max < 6000:
            print(f"  ✅ PASS: Prices in SPX index range (no /10 division)")
            return True
        elif 500 < price_min < 600:
            print(f"  ❌ FAIL: Prices in SPY dollar range (incorrect /10 division still applied!)")
            return False
        else:
            print(f"  ⚠️  WARNING: Unexpected price range")
            return False
            
    except Exception as e:
        print(f"  ❌ OHLCV conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_run(date: str) -> bool:
    """Test full pipeline execution."""
    print(f"\n4. Testing full pipeline execution...")
    
    try:
        from src.pipeline.pipelines.v1_0_spx_final_call import build_v1_0_spx_final_call_pipeline
        
        pipeline = build_v1_0_spx_final_call_pipeline()
        print(f"  Pipeline version: {pipeline.version}")
        print(f"  Stages: {len(pipeline.stages)}")
        
        print(f"\n  Running pipeline for {date}...")
        signals_df = pipeline.run(date)
        
        if signals_df is None or signals_df.empty:
            print(f"  ⚠️  WARNING: No signals generated (may be normal if no touches)")
            return True  # Not necessarily a failure
        
        print(f"  ✅ Generated {len(signals_df)} signals")
        
        # Check key columns exist
        expected_cols = [
            'event_id', 'level_price', 'direction', 'minutes_since_open',
            'velocity', 'acceleration', 'integrated_ofi',
            'outcome'
        ]
        
        missing = [col for col in expected_cols if col not in signals_df.columns]
        if missing:
            print(f"  ⚠️  WARNING: Missing columns: {missing}")
        else:
            print(f"  ✅ All expected columns present")
        
        # Check value ranges
        if 'level_price' in signals_df.columns:
            level_min = signals_df['level_price'].min()
            level_max = signals_df['level_price'].max()
            print(f"  Level price range: {level_min:.2f} - {level_max:.2f}")
            
            if 5000 < level_min < 6000:
                print(f"  ✅ Levels in SPX range (correct)")
            else:
                print(f"  ❌ Levels NOT in SPX range (check conversion)")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test SPX transformation')
    parser.add_argument('--date', type=str, required=True, help='Date to test (YYYY-MM-DD)')
    parser.add_argument(
        '--bronze-root',
        type=str,
        default=None,
        help='Bronze root path'
    )
    
    args = parser.parse_args()
    
    if args.bronze_root:
        bronze_root = args.bronze_root
    else:
        backend_dir = Path(__file__).parent.parent
        bronze_root = str(backend_dir / 'data' / 'lake' / 'bronze')
    
    print(f"{'='*60}")
    print(f"SPX Transformation Test: {args.date}")
    print(f"Bronze root: {bronze_root}")
    print(f"{'='*60}")
    
    # Run tests
    tests_passed = 0
    tests_total = 4
    
    if test_spx_data_available(bronze_root, args.date):
        tests_passed += 1
    
    if test_es_front_month_filtering(bronze_root, args.date):
        tests_passed += 1
    
    if test_spx_ohlcv_conversion(bronze_root, args.date):
        tests_passed += 1
    
    if test_pipeline_run(args.date):
        tests_passed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {tests_passed}/{tests_total} passed")
    print(f"{'='*60}")
    
    return 0 if tests_passed == tests_total else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

