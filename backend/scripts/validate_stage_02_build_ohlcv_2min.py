"""
Validate Stage 2: BuildOHLCV (2min with warmup)

Goals:
1. Build 2-minute OHLCV bars from ES futures trades
2. Include warmup days for SMA_200/400 calculation
3. Verify multi-day data concatenation
4. Ensure proper time ordering across dates
5. Validate bar structure for SMA computation

Validation Checks:
- Warmup data inclusion (multiple dates)
- Bar count (2min bars = half of 1min bars per day)
- DatetimeIndex preservation
- Multi-day timestamp ordering
- OHLC relationships
- Sufficient warmup for SMA_200/400
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
from src.common.config import CONFIG


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


class Stage2Validator:
    """Validator for BuildOHLCV (2min with warmup) stage."""
    
    def __init__(self, logger):
        self.logger = logger
        self.results = {
            'stage': 'build_ohlcv_2min',
            'stage_idx': 2,
            'checks': {},
            'warnings': [],
            'errors': [],
            'passed': True
        }
    
    def validate(self, date: str, ctx) -> Dict[str, Any]:
        """Run all validation checks.
        
        Args:
            date: Date being processed
            ctx: StageContext after BuildOHLCV (2min) execution
        
        Returns:
            Validation results dict
        """
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Validating Stage 2: BuildOHLCV (2min with warmup) for {date}")
        self.logger.info(f"{'='*80}")
        
        self.results['date'] = date
        
        # Check 1: Required outputs present
        self._check_required_outputs(ctx)
        
        # Check 2: Validate OHLCV DataFrame
        if 'ohlcv_2min' in ctx.data:
            self._validate_ohlcv_dataframe(ctx.data['ohlcv_2min'], date, ctx)
        
        # Summary
        self.logger.info(f"\n{'='*80}")
        if self.results['passed']:
            self.logger.info("✅ Stage 2 Validation: PASSED")
        else:
            self.logger.error("❌ Stage 2 Validation: FAILED")
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
        
        # From previous stages
        previous_required = ['trades', 'trades_df', 'mbp10_snapshots', 'option_trades_df', 'ohlcv_1min', 'atr', 'volatility']
        # New from this stage
        new_required = ['ohlcv_2min']
        
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
            self.logger.info(f"  ✅ New output present: ohlcv_2min")
        
        # Check for warmup metadata
        if 'warmup_dates' in ctx.data:
            warmup_dates = ctx.data['warmup_dates']
            self.logger.info(f"  ✅ Warmup dates: {warmup_dates}")
            self.results['checks']['warmup_metadata'] = True
        else:
            self.logger.info(f"  ℹ️  No warmup_dates metadata (may not be needed)")
            self.results['checks']['warmup_metadata'] = False
    
    def _validate_ohlcv_dataframe(self, ohlcv_df, date, ctx):
        """Validate 2min OHLCV DataFrame structure and content."""
        self.logger.info("\n2. Validating 2min OHLCV DataFrame...")
        
        checks = {}
        
        # Check non-empty
        row_count = len(ohlcv_df)
        self.logger.info(f"  Bar count: {row_count:,}")
        
        if row_count == 0:
            checks['ohlcv_non_empty'] = False
            self.results['passed'] = False
            self.results['errors'].append("2min OHLCV DataFrame is empty")
            self.logger.error("  ❌ No bars generated")
            return
        
        checks['ohlcv_non_empty'] = True
        
        # Check schema
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in ohlcv_df.columns]
        
        if missing_cols:
            checks['ohlcv_schema'] = False
            self.results['passed'] = False
            error = f"2min OHLCV missing columns: {missing_cols}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['ohlcv_schema'] = True
            self.logger.info(f"  ✅ Schema valid")
        
        # Check DatetimeIndex
        if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
            checks['datetime_index'] = False
            self.results['passed'] = False
            error = f"Index is not DatetimeIndex: {type(ohlcv_df.index)}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['datetime_index'] = True
            self.logger.info(f"  ✅ DatetimeIndex")
            
            # Check monotonicity
            if not ohlcv_df.index.is_monotonic_increasing:
                checks['index_monotonic'] = False
                self.results['errors'].append("2min OHLCV index not monotonic")
                self.results['passed'] = False
                self.logger.error(f"  ❌ Index not monotonic")
            else:
                checks['index_monotonic'] = True
                self.logger.info(f"  ✅ Index monotonic")
        
        # Check timestamp range and warmup
        if isinstance(ohlcv_df.index, pd.DatetimeIndex):
            start_time = ohlcv_df.index.min()
            end_time = ohlcv_df.index.max()
            
            self.logger.info(f"  Time range: {start_time} to {end_time}")
            
            # Check date span
            start_date = start_time.date()
            end_date = end_time.date()
            date_obj = pd.Timestamp(date).date()
            
            self.logger.info(f"  Date span: {start_date} to {end_date}")
            
            # Check if warmup is included
            if start_date < date_obj:
                days_of_warmup = (date_obj - start_date).days
                self.logger.info(f"  ✅ Warmup included: {days_of_warmup} days before {date}")
                checks['warmup_included'] = True
                
                # Check if sufficient for SMA_200/400
                # SMA_200 @ 2min = 200 bars = ~6.67 hours
                # SMA_400 @ 2min = 400 bars = ~13.33 hours  
                # Need ~2-3 trading days for SMA_400
                warmup_days_needed = CONFIG.SMA_WARMUP_DAYS if hasattr(CONFIG, 'SMA_WARMUP_DAYS') else 3
                
                if days_of_warmup < warmup_days_needed:
                    checks['warmup_sufficient'] = False
                    warning = f"Insufficient warmup: {days_of_warmup} days (need {warmup_days_needed} for SMA_400)"
                    self.results['warnings'].append(warning)
                    self.logger.warning(f"  ⚠️  {warning}")
                else:
                    checks['warmup_sufficient'] = True
                    self.logger.info(f"  ✅ Sufficient warmup for SMA_400")
            else:
                checks['warmup_included'] = False
                warning = "No warmup data (starts on target date)"
                self.results['warnings'].append(warning)
                self.logger.warning(f"  ⚠️  {warning}")
            
            # Check expected bar count
            # 2min bars = ~690 per day (23 hours trading)
            # With warmup, should have much more
            expected_min_bars = 690 if start_date == date_obj else 690 * (days_of_warmup + 1)
            
            if row_count < expected_min_bars - 100:
                checks['bar_count'] = False
                warning = f"Low bar count: {row_count} (expected ~{expected_min_bars})"
                self.results['warnings'].append(warning)
                self.logger.warning(f"  ⚠️  {warning}")
            else:
                checks['bar_count'] = True
                self.logger.info(f"  ✅ Bar count reasonable ({row_count:,} bars)")
        
        # Check OHLC relationships
        self.logger.info(f"\n  Checking OHLC logical consistency...")
        
        high_ge_open = (ohlcv_df['high'] >= ohlcv_df['open']).all()
        high_ge_close = (ohlcv_df['high'] >= ohlcv_df['close']).all()
        high_ge_low = (ohlcv_df['high'] >= ohlcv_df['low']).all()
        low_le_open = (ohlcv_df['low'] <= ohlcv_df['open']).all()
        low_le_close = (ohlcv_df['low'] <= ohlcv_df['close']).all()
        
        ohlc_valid = all([high_ge_open, high_ge_close, high_ge_low, low_le_open, low_le_close])
        
        if not ohlc_valid:
            checks['ohlc_relationships'] = False
            self.results['passed'] = False
            error = "2min OHLC relationships violated"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['ohlc_relationships'] = True
            self.logger.info(f"    ✅ OHLC relationships valid")
        
        # Price statistics
        self.logger.info(f"\n  Price statistics:")
        self.logger.info(f"    Open:  min={ohlcv_df['open'].min():.2f}, max={ohlcv_df['open'].max():.2f}")
        self.logger.info(f"    High:  min={ohlcv_df['high'].min():.2f}, max={ohlcv_df['high'].max():.2f}")
        self.logger.info(f"    Low:   min={ohlcv_df['low'].min():.2f}, max={ohlcv_df['low'].max():.2f}")
        self.logger.info(f"    Close: min={ohlcv_df['close'].min():.2f}, max={ohlcv_df['close'].max():.2f}")
        
        # Volume statistics
        self.logger.info(f"\n  Volume statistics:")
        self.logger.info(f"    Total: {ohlcv_df['volume'].sum():,.0f}")
        self.logger.info(f"    Mean: {ohlcv_df['volume'].mean():.0f}")
        self.logger.info(f"    Median: {ohlcv_df['volume'].median():.0f}")
        
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
        
        # Verify 2min bars align with 1min bars
        if 'ohlcv_1min' in ctx.data:
            self._compare_with_1min(ohlcv_df, ctx.data['ohlcv_1min'], date, checks)
        
        self.results['checks']['ohlcv_dataframe'] = checks
    
    def _compare_with_1min(self, ohlcv_2min, ohlcv_1min, date, checks):
        """Compare 2min bars with 1min bars for consistency."""
        self.logger.info(f"\n  Comparing with 1min bars...")
        
        # Filter both to target date only for comparison
        date_obj = pd.Timestamp(date, tz='UTC')
        
        ohlcv_1min_date = ohlcv_1min[ohlcv_1min.index.date == date_obj.date()]
        ohlcv_2min_date = ohlcv_2min[ohlcv_2min.index.date == date_obj.date()]
        
        bars_1min = len(ohlcv_1min_date)
        bars_2min = len(ohlcv_2min_date)
        
        self.logger.info(f"    1min bars (date only): {bars_1min}")
        self.logger.info(f"    2min bars (date only): {bars_2min}")
        
        # 2min should be roughly half of 1min
        expected_2min = bars_1min / 2
        
        if abs(bars_2min - expected_2min) > 10:
            checks['bar_ratio'] = False
            warning = f"2min/1min bar ratio off: {bars_2min} vs expected ~{expected_2min:.0f}"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
        else:
            checks['bar_ratio'] = True
            self.logger.info(f"    ✅ 2min bars ≈ 1min bars / 2")
        
        # Check that 2min high >= 1min high (aggregation check)
        # Sample a few aligned timestamps
        if not ohlcv_2min_date.empty and not ohlcv_1min_date.empty:
            # Find a 2min bar timestamp
            sample_2min_ts = ohlcv_2min_date.index[len(ohlcv_2min_date)//2]
            
            # Get corresponding 1min bars (2 bars that compose this 2min bar)
            start_1min = sample_2min_ts
            end_1min = sample_2min_ts + pd.Timedelta(minutes=2)
            
            bars_1min_slice = ohlcv_1min_date[(ohlcv_1min_date.index >= start_1min) & 
                                              (ohlcv_1min_date.index < end_1min)]
            
            if len(bars_1min_slice) >= 2:
                # 2min high should be >= max of 1min highs
                bars_2min_slice = ohlcv_2min_date[ohlcv_2min_date.index == start_1min]
                
                if not bars_2min_slice.empty:
                    max_1min_high = bars_1min_slice['high'].max()
                    bar_2min_high = bars_2min_slice['high'].iloc[0]
                    
                    if bar_2min_high >= max_1min_high - 0.25:  # Allow 1 tick tolerance
                        self.logger.info(f"    ✅ 2min aggregation consistent with 1min")
                        checks['aggregation_consistent'] = True
                    else:
                        checks['aggregation_consistent'] = False
                        warning = f"2min high ({bar_2min_high}) < max 1min high ({max_1min_high})"
                        self.results['warnings'].append(warning)
                        self.logger.warning(f"  ⚠️  {warning}")


def main():
    parser = argparse.ArgumentParser(description='Validate Stage 2: BuildOHLCV (2min)')
    parser.add_argument('--date', type=str, required=True, help='Date to validate (YYYY-MM-DD)')
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-file', type=str, default=None, help='Log file path')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_file is None:
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        args.log_file = str(log_dir / f'validate_stage_02_{args.date}.log')
    
    logger = setup_logging(args.log_file)
    logger.info(f"Starting Stage 2 validation for {args.date}")
    logger.info(f"Log file: {args.log_file}")
    
    try:
        # Run pipeline through stage 2
        logger.info("Running through BuildOHLCV (2min) stage...")
        pipeline = build_es_pipeline()
        
        signals_df = pipeline.run(
            date=args.date,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_stage=2,
            stop_at_stage=2
        )
        
        # Load checkpoint
        from src.pipeline.core.checkpoint import CheckpointManager
        manager = CheckpointManager(args.checkpoint_dir)
        ctx = manager.load_checkpoint("es_pipeline", args.date, stage_idx=2)
        
        if ctx is None:
            logger.error("Failed to load checkpoint")
            return 1
        
        # Validate
        validator = Stage2Validator(logger)
        results = validator.validate(args.date, ctx)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path(__file__).parent.parent / 'logs'
            output_path = output_dir / f'validate_stage_02_{args.date}_results.json'
        
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

