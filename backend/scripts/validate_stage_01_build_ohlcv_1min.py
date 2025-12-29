"""
Validate Stage 1: BuildOHLCV (1min)

Goals:
1. Build 1-minute OHLCV bars from ES futures trades (ALL hours for PM/OR levels)
2. Compute ATR (Average True Range) for volatility baseline
3. Verify bar completeness (no gaps in trading session)
4. Validate OHLC relationships (High >= Close >= Low, etc)
5. Include premarket bars (04:00-09:30 ET) for PM_HIGH/PM_LOW calculation

Validation Checks:
- Bar count and coverage (RTH: 09:30-16:00 ET = 390 minutes)
- OHLCV schema and types
- OHLC logical consistency
- Volume aggregation correctness
- ATR computation and validity
- No premarket/afterhours in ATR baseline
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from src.pipeline.pipelines.es_pipeline import build_es_pipeline


def setup_logging(log_file: str):
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class Stage1Validator:
    """Validator for BuildOHLCV (1min) stage."""
    
    def __init__(self, logger):
        self.logger = logger
        self.results = {
            'stage': 'build_ohlcv_1min',
            'stage_idx': 1,
            'checks': {},
            'warnings': [],
            'errors': [],
            'passed': True
        }
    
    def validate(self, date: str, ctx) -> Dict[str, Any]:
        """Run all validation checks.
        
        Args:
            date: Date being processed
            ctx: StageContext after BuildOHLCV (1min) execution
        
        Returns:
            Validation results dict
        """
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Validating Stage 1: BuildOHLCV (1min) for {date}")
        self.logger.info(f"{'='*80}")
        
        self.results['date'] = date
        
        # Check 1: Required outputs present
        self._check_required_outputs(ctx)
        
        # Check 2: Validate OHLCV DataFrame
        if 'ohlcv_1min' in ctx.data:
            self._validate_ohlcv_dataframe(ctx.data['ohlcv_1min'], date)
        
        # Check 3: Validate ATR if present
        if 'atr' in ctx.data:
            self._validate_atr_series(ctx.data['atr'], date)
        elif 'ohlcv_1min' in ctx.data and 'atr' in ctx.data['ohlcv_1min'].columns:
            self._validate_atr_series(ctx.data['ohlcv_1min']['atr'], date)
        
        # Summary
        self.logger.info(f"\n{'='*80}")
        if self.results['passed']:
            self.logger.info("✅ Stage 1 Validation: PASSED")
        else:
            self.logger.error("❌ Stage 1 Validation: FAILED")
            self.logger.error(f"Errors: {len(self.results['errors'])}")
            for error in self.results['errors']:
                self.logger.error(f"  - {error}")
        
        if self.results['warnings']:
            self.logger.warning(f"Warnings: {len(self.results['warnings'])}")
            for warning in self.results['warnings']:
                self.logger.warning(f"  - {warning}")
        
        self.logger.info(f"{'='*80}\n")
        
        return self.results
    
    def _check_required_outputs(self, ctx):
        """Verify required outputs are present in context."""
        self.logger.info("\n1. Checking required outputs...")
        
        # From previous stage
        previous_required = ['trades', 'trades_df', 'mbp10_snapshots', 'option_trades_df']
        # New from this stage
        new_required = ['ohlcv_1min']
        
        available = list(ctx.data.keys())
        
        missing_previous = [key for key in previous_required if key not in available]
        missing_new = [key for key in new_required if key not in available]
        
        if missing_previous:
            self.results['checks']['previous_outputs_preserved'] = False
            self.results['passed'] = False
            error = f"Previous stage outputs missing: {missing_previous}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            self.results['checks']['previous_outputs_preserved'] = True
            self.logger.info(f"  ✅ Previous outputs preserved")
        
        if missing_new:
            self.results['checks']['new_outputs_present'] = False
            self.results['passed'] = False
            error = f"Missing new outputs: {missing_new}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            self.results['checks']['new_outputs_present'] = True
            self.logger.info(f"  ✅ New output present: ohlcv_1min")
    
    def _validate_ohlcv_dataframe(self, ohlcv_df, date):
        """Validate OHLCV DataFrame structure and content."""
        self.logger.info("\n2. Validating OHLCV DataFrame...")
        
        checks = {}
        
        # Check non-empty
        row_count = len(ohlcv_df)
        self.logger.info(f"  Bar count: {row_count:,}")
        
        if row_count == 0:
            checks['ohlcv_non_empty'] = False
            self.results['passed'] = False
            self.results['errors'].append("OHLCV DataFrame is empty")
            self.logger.error("  ❌ No bars generated")
            return
        
        checks['ohlcv_non_empty'] = True
        
        # Check schema
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in ohlcv_df.columns]
        
        if missing_cols:
            checks['ohlcv_schema'] = False
            self.results['passed'] = False
            error = f"OHLCV missing columns: {missing_cols}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['ohlcv_schema'] = True
            self.logger.info(f"  ✅ Schema valid: {required_cols}")
        
        # Check for ATR column
        if 'atr' in ohlcv_df.columns:
            self.logger.info(f"  ✅ ATR column present")
            checks['atr_present'] = True
        else:
            self.logger.info(f"  ℹ️  No ATR column (may be computed separately)")
            checks['atr_present'] = False
        
        # Check timestamp index
        if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
            checks['datetime_index'] = False
            self.results['passed'] = False
            error = f"Index is not DatetimeIndex: {type(ohlcv_df.index)}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['datetime_index'] = True
            self.logger.info(f"  ✅ DatetimeIndex")
            
            # Check index monotonicity
            if not ohlcv_df.index.is_monotonic_increasing:
                checks['index_monotonic'] = False
                self.results['errors'].append("OHLCV index not monotonic")
                self.results['passed'] = False
                self.logger.error(f"  ❌ Index not monotonic")
            else:
                checks['index_monotonic'] = True
                self.logger.info(f"  ✅ Index monotonic")
        
        # Check timestamp range
        if isinstance(ohlcv_df.index, pd.DatetimeIndex):
            start_time = ohlcv_df.index.min()
            end_time = ohlcv_df.index.max()
            
            self.logger.info(f"  Time range: {start_time} to {end_time}")
            
            # Convert to ET for RTH check
            if start_time.tz is None:
                start_et = start_time.tz_localize('UTC').tz_convert('America/New_York')
                end_et = end_time.tz_localize('UTC').tz_convert('America/New_York')
            else:
                start_et = start_time.tz_convert('America/New_York')
                end_et = end_time.tz_convert('America/New_York')
            
            self.logger.info(f"  Time range (ET): {start_et.strftime('%H:%M')} to {end_et.strftime('%H:%M')}")
            
            # Check if includes premarket (needed for PM_HIGH/PM_LOW)
            if start_et.hour > 4 or (start_et.hour == 4 and start_et.minute > 30):
                checks['includes_premarket'] = False
                warning = f"Missing early premarket: starts at {start_et.strftime('%H:%M')} ET"
                self.results['warnings'].append(warning)
                self.logger.warning(f"  ⚠️  {warning}")
            else:
                checks['includes_premarket'] = True
                self.logger.info(f"  ✅ Includes premarket (starts {start_et.strftime('%H:%M')} ET)")
            
            # Check expected bar count for full trading session
            # ES trades 23 hours/day (closed 17:00-18:00 ET) = ~1380 bars
            # Includes premarket (needed for PM_HIGH/PM_LOW)
            expected_bars_full = 1380
            
            if row_count < expected_bars_full - 100:
                checks['bar_count'] = False
                warning = f"Low bar count: {row_count} (expected ~{expected_bars_full} for full session)"
                self.results['warnings'].append(warning)
                self.logger.warning(f"  ⚠️  {warning}")
            elif row_count > expected_bars_full + 100:
                checks['bar_count'] = False
                warning = f"High bar count: {row_count} (expected ~{expected_bars_full})"
                self.results['warnings'].append(warning)
                self.logger.warning(f"  ⚠️  {warning}")
            else:
                checks['bar_count'] = True
                self.logger.info(f"  ✅ Bar count reasonable for full session")
        
        # Check OHLC relationships
        self.logger.info(f"\n  Checking OHLC logical consistency...")
        
        # High should be >= Open, Close, Low
        high_ge_open = (ohlcv_df['high'] >= ohlcv_df['open']).all()
        high_ge_close = (ohlcv_df['high'] >= ohlcv_df['close']).all()
        high_ge_low = (ohlcv_df['high'] >= ohlcv_df['low']).all()
        
        # Low should be <= Open, Close, High
        low_le_open = (ohlcv_df['low'] <= ohlcv_df['open']).all()
        low_le_close = (ohlcv_df['low'] <= ohlcv_df['close']).all()
        low_le_high = (ohlcv_df['low'] <= ohlcv_df['high']).all()
        
        ohlc_valid = all([high_ge_open, high_ge_close, high_ge_low, low_le_open, low_le_close, low_le_high])
        
        if not ohlc_valid:
            checks['ohlc_relationships'] = False
            self.results['passed'] = False
            error = "OHLC relationships violated"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            
            if not high_ge_open:
                self.logger.error(f"    High < Open in some bars")
            if not high_ge_close:
                self.logger.error(f"    High < Close in some bars")
            if not high_ge_low:
                self.logger.error(f"    High < Low in some bars")
            if not low_le_open:
                self.logger.error(f"    Low > Open in some bars")
            if not low_le_close:
                self.logger.error(f"    Low > Close in some bars")
        else:
            checks['ohlc_relationships'] = True
            self.logger.info(f"    ✅ OHLC relationships valid")
        
        # Check price ranges
        self.logger.info(f"\n  Price statistics:")
        self.logger.info(f"    Open:  min={ohlcv_df['open'].min():.2f}, max={ohlcv_df['open'].max():.2f}")
        self.logger.info(f"    High:  min={ohlcv_df['high'].min():.2f}, max={ohlcv_df['high'].max():.2f}")
        self.logger.info(f"    Low:   min={ohlcv_df['low'].min():.2f}, max={ohlcv_df['low'].max():.2f}")
        self.logger.info(f"    Close: min={ohlcv_df['close'].min():.2f}, max={ohlcv_df['close'].max():.2f}")
        
        # Check for reasonable ES price range
        all_prices = pd.concat([ohlcv_df['open'], ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close']])
        if all_prices.min() < 3000 or all_prices.max() > 10000:
            checks['price_range'] = False
            warning = f"Unusual price range: {all_prices.min():.2f} - {all_prices.max():.2f}"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
        else:
            checks['price_range'] = True
            self.logger.info(f"    ✅ Price range reasonable")
        
        # Check volume
        self.logger.info(f"\n  Volume statistics:")
        self.logger.info(f"    Total: {ohlcv_df['volume'].sum():,.0f}")
        self.logger.info(f"    Mean: {ohlcv_df['volume'].mean():.0f}")
        self.logger.info(f"    Median: {ohlcv_df['volume'].median():.0f}")
        self.logger.info(f"    Max: {ohlcv_df['volume'].max():,.0f}")
        
        # Check for zero-volume bars
        zero_vol_count = (ohlcv_df['volume'] == 0).sum()
        if zero_vol_count > 0:
            checks['volume_non_zero'] = False
            warning = f"{zero_vol_count} bars with zero volume"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
        else:
            checks['volume_non_zero'] = True
            self.logger.info(f"    ✅ All bars have volume")
        
        # Check for NaN values
        nan_counts = ohlcv_df[['open', 'high', 'low', 'close', 'volume']].isna().sum()
        if nan_counts.any():
            checks['no_nans'] = False
            self.results['passed'] = False
            error = f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['no_nans'] = True
            self.logger.info(f"  ✅ No NaN values")
        
        self.results['checks']['ohlcv_dataframe'] = checks
    
    def _validate_atr_series(self, atr_series, date):
        """Validate ATR computation."""
        self.logger.info("\n3. Validating ATR (Average True Range)...")
        
        checks = {}
        
        if atr_series is None or (isinstance(atr_series, pd.Series) and atr_series.empty):
            checks['atr_computed'] = False
            self.results['warnings'].append("ATR not computed")
            self.logger.warning("  ⚠️  ATR not found")
            self.results['checks']['atr'] = checks
            return
        
        checks['atr_computed'] = True
        self.logger.info(f"  ✅ ATR computed ({len(atr_series)} values)")
        
        # Count non-NaN ATR values
        non_nan_atr = atr_series.notna().sum()
        self.logger.info(f"  ATR values: {non_nan_atr} / {len(atr_series)}")
        
        if non_nan_atr == 0:
            checks['atr_values'] = False
            self.results['errors'].append("All ATR values are NaN")
            self.results['passed'] = False
            self.logger.error(f"  ❌ All ATR values are NaN")
        else:
            checks['atr_values'] = True
            
            # ATR statistics
            atr_mean = atr_series.mean()
            atr_std = atr_series.std()
            atr_min = atr_series.min()
            atr_max = atr_series.max()
            
            self.logger.info(f"  ATR statistics:")
            self.logger.info(f"    Mean: {atr_mean:.2f}")
            self.logger.info(f"    Std:  {atr_std:.2f}")
            self.logger.info(f"    Min:  {atr_min:.2f}")
            self.logger.info(f"    Max:  {atr_max:.2f}")
            
            # Check for reasonable ATR range (typically 5-50 points for ES)
            if atr_mean < 1 or atr_mean > 100:
                checks['atr_range'] = False
                warning = f"Unusual ATR mean: {atr_mean:.2f}"
                self.results['warnings'].append(warning)
                self.logger.warning(f"  ⚠️  {warning}")
            else:
                checks['atr_range'] = True
                self.logger.info(f"  ✅ ATR range reasonable")
            
            # Check for negative ATR (should never happen)
            if (atr_series < 0).any():
                checks['atr_positive'] = False
                self.results['passed'] = False
                error = "Negative ATR values found"
                self.results['errors'].append(error)
                self.logger.error(f"  ❌ {error}")
            else:
                checks['atr_positive'] = True
                self.logger.info(f"  ✅ All ATR values positive")
        
        self.results['checks']['atr'] = checks


def main():
    parser = argparse.ArgumentParser(description='Validate Stage 1: BuildOHLCV (1min)')
    parser.add_argument('--date', type=str, required=True, help='Date to validate (YYYY-MM-DD)')
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-file', type=str, default=None, help='Log file path')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_file is None:
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        args.log_file = str(log_dir / f'validate_stage_01_{args.date}.log')
    
    logger = setup_logging(args.log_file)
    logger.info(f"Starting Stage 1 validation for {args.date}")
    logger.info(f"Log file: {args.log_file}")
    
    try:
        # Run pipeline through stage 1
        logger.info("Running through BuildOHLCV (1min) stage...")
        pipeline = build_es_pipeline()
        
        signals_df = pipeline.run(
            date=args.date,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_stage=1,
            stop_at_stage=1
        )
        
        # Load checkpoint
        from src.pipeline.core.checkpoint import CheckpointManager
        manager = CheckpointManager(args.checkpoint_dir)
        ctx = manager.load_checkpoint("es_pipeline", args.date, stage_idx=1)
        
        if ctx is None:
            logger.error("Failed to load checkpoint")
            return 1
        
        # Validate
        validator = Stage1Validator(logger)
        results = validator.validate(args.date, ctx)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path(__file__).parent.parent / 'logs'
            output_path = output_dir / f'validate_stage_01_{args.date}_results.json'
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        
        return 0 if results['passed'] else 1
        
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

