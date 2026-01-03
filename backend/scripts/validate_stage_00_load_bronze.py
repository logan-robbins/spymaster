"""
Validate Stage 0: LoadBronze

Goals:
1. Load ES futures trades from Bronze
2. Load ES MBP-10 snapshots (downsampled)
3. Load ES 0DTE options trades
4. Verify front-month filtering
5. Ensure data quality (no gaps, correct timestamps, valid values)

Validation Checks:
- Data coverage: Expected row counts based on trading hours
- Timestamp monotonicity and range
- Front-month purity (single contract dominance)
- Schema compliance (correct columns and types)
- Value ranges (prices, sizes, strikes)
- 0DTE filtering for options (exp_date == session_date)
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from src.pipeline.pipelines.bronze_to_silver import build_bronze_to_silver_pipeline
from src.common.config import CONFIG


# Setup logging
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


class Stage0Validator:
    """Validator for LoadBronze stage."""
    
    def __init__(self, logger):
        self.logger = logger
        self.results = {
            'stage': 'load_bronze',
            'stage_idx': 0,
            'checks': {},
            'warnings': [],
            'errors': [],
            'passed': True
        }
    
    def validate(self, date: str, ctx) -> Dict[str, Any]:
        """Run all validation checks.
        
        Args:
            date: Date being processed
            ctx: StageContext after LoadBronze execution
        
        Returns:
            Validation results dict
        """
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Validating Stage 0: LoadBronze for {date}")
        self.logger.info(f"{'='*80}")
        
        self.results['date'] = date
        
        # Check 1: Required outputs present
        self._check_required_outputs(ctx)
        
        # Check 2: ES Futures Trades
        if 'trades' in ctx.data and 'trades_df' in ctx.data:
            self._validate_futures_trades(ctx.data['trades'], ctx.data['trades_df'], date)
        
        # Check 3: ES MBP-10 Snapshots
        if 'mbp10_snapshots' in ctx.data:
            self._validate_mbp10_snapshots(ctx.data['mbp10_snapshots'], date)
        
        # Check 4: ES Options Trades
        if 'option_trades_df' in ctx.data:
            self._validate_option_trades(ctx.data['option_trades_df'], date)
        
        # Summary
        self.logger.info(f"\n{'='*80}")
        if self.results['passed']:
            self.logger.info("✅ Stage 0 Validation: PASSED")
        else:
            self.logger.error("❌ Stage 0 Validation: FAILED")
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
        
        required = ['trades', 'trades_df', 'mbp10_snapshots', 'option_trades_df']
        available = list(ctx.data.keys())
        
        missing = [key for key in required if key not in available]
        
        if missing:
            self.results['checks']['required_outputs'] = False
            self.results['passed'] = False
            error = f"Missing required outputs: {missing}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            self.results['checks']['required_outputs'] = True
            self.logger.info(f"  ✅ All required outputs present: {required}")
    
    def _validate_futures_trades(self, trades, trades_df, date):
        """Validate ES futures trades."""
        self.logger.info("\n2. Validating ES Futures Trades...")
        
        checks = {}
        
        # Check row count
        row_count = len(trades_df)
        self.logger.info(f"  Row count: {row_count:,}")
        
        if row_count == 0:
            checks['trades_non_empty'] = False
            self.results['passed'] = False
            self.results['errors'].append("ES trades DataFrame is empty")
            self.logger.error("  ❌ No trades loaded")
            return
        
        checks['trades_non_empty'] = True
        
        # Check schema
        expected_cols = ['ts_event_ns', 'ts_recv_ns', 'source', 'symbol', 'price', 'size', 'aggressor', 'exchange', 'seq']
        missing_cols = [col for col in expected_cols if col not in trades_df.columns]
        
        if missing_cols:
            checks['trades_schema'] = False
            self.results['passed'] = False
            error = f"Trades missing columns: {missing_cols}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['trades_schema'] = True
            self.logger.info(f"  ✅ Schema valid")
        
        # Check timestamp range
        min_ts = pd.to_datetime(trades_df['ts_event_ns'], unit='ns')
        max_ts = pd.to_datetime(trades_df['ts_event_ns'], unit='ns')
        
        expected_date = pd.Timestamp(date, tz='America/New_York').tz_convert('UTC')
        min_ts_val = min_ts.min()
        max_ts_val = max_ts.max()
        
        self.logger.info(f"  Timestamp range: {min_ts_val} to {max_ts_val}")
        
        # Verify timestamps are from correct date (allow premarket/afterhours)
        if min_ts_val.date() != expected_date.date() and min_ts_val.date() != (expected_date - pd.Timedelta(days=1)).date():
            checks['trades_timestamp_range'] = False
            self.results['warnings'].append(f"Trades start date mismatch: {min_ts_val.date()} != {expected_date.date()}")
            self.logger.warning(f"  ⚠️  Start date mismatch")
        else:
            checks['trades_timestamp_range'] = True
        
        # Check monotonicity
        if not trades_df['ts_event_ns'].is_monotonic_increasing:
            checks['trades_monotonic'] = False
            self.results['warnings'].append("Trades timestamps not monotonic")
            self.logger.warning(f"  ⚠️  Timestamps not monotonic (should be sorted)")
        else:
            checks['trades_monotonic'] = True
            self.logger.info(f"  ✅ Timestamps monotonic")
        
        # Check contract symbols
        unique_symbols = trades_df['symbol'].unique()
        self.logger.info(f"  Unique symbols: {list(unique_symbols)}")
        
        # Check for front-month dominance
        symbol_counts = trades_df['symbol'].value_counts()
        if len(symbol_counts) > 1:
            dominant_symbol = symbol_counts.index[0]
            dominant_pct = symbol_counts.iloc[0] / len(trades_df)
            
            if dominant_pct < 0.95:
                checks['trades_front_month'] = False
                self.results['warnings'].append(f"Front-month dominance only {dominant_pct:.1%} ({dominant_symbol})")
                self.logger.warning(f"  ⚠️  Weak front-month dominance: {dominant_pct:.1%}")
            else:
                checks['trades_front_month'] = True
                self.logger.info(f"  ✅ Front-month purity: {dominant_pct:.1%} ({dominant_symbol})")
        else:
            checks['trades_front_month'] = True
            self.logger.info(f"  ✅ Single contract: {unique_symbols[0]}")
        
        # Check price ranges
        prices = trades_df['price']
        self.logger.info(f"  Price range: {prices.min():.2f} - {prices.max():.2f}")
        
        if prices.min() < 3000 or prices.max() > 10000:
            checks['trades_price_range'] = False
            self.results['warnings'].append(f"Unusual price range: {prices.min():.2f} - {prices.max():.2f}")
            self.logger.warning(f"  ⚠️  Unusual price range")
        else:
            checks['trades_price_range'] = True
            self.logger.info(f"  ✅ Price range reasonable")
        
        # Check sizes
        sizes = trades_df['size']
        self.logger.info(f"  Size range: {sizes.min()} - {sizes.max()}, median: {sizes.median()}")
        
        if sizes.min() < 0 or sizes.max() > 10000:
            checks['trades_size_range'] = False
            self.results['errors'].append(f"Invalid size range: {sizes.min()} - {sizes.max()}")
            self.results['passed'] = False
            self.logger.error(f"  ❌ Invalid size range")
        else:
            checks['trades_size_range'] = True
            self.logger.info(f"  ✅ Size range valid")
        
        # Check list<FuturesTrade> length matches DataFrame
        if len(trades) != len(trades_df):
            checks['trades_list_match'] = False
            self.results['warnings'].append(f"trades list length ({len(trades)}) != trades_df length ({len(trades_df)})")
            self.logger.warning(f"  ⚠️  trades/trades_df length mismatch")
        else:
            checks['trades_list_match'] = True
            self.logger.info(f"  ✅ trades list matches DataFrame")
        
        self.results['checks']['futures_trades'] = checks
    
    def _validate_mbp10_snapshots(self, mbp10_snapshots, date):
        """Validate ES MBP-10 snapshots."""
        self.logger.info("\n3. Validating ES MBP-10 Snapshots...")
        
        checks = {}
        
        # Check count
        snapshot_count = len(mbp10_snapshots)
        self.logger.info(f"  Snapshot count: {snapshot_count:,}")
        
        if snapshot_count == 0:
            checks['mbp10_non_empty'] = False
            self.results['passed'] = False
            self.results['errors'].append("MBP-10 snapshots list is empty")
            self.logger.error("  ❌ No snapshots loaded")
            return
        
        checks['mbp10_non_empty'] = True
        
        # Sample first and last snapshots
        first = mbp10_snapshots[0]
        last = mbp10_snapshots[-1]
        
        first_ts = pd.to_datetime(first.ts_event_ns, unit='ns')
        last_ts = pd.to_datetime(last.ts_event_ns, unit='ns')
        
        self.logger.info(f"  Timestamp range: {first_ts} to {last_ts}")
        duration = (last_ts - first_ts).total_seconds() / 3600
        self.logger.info(f"  Duration: {duration:.2f} hours")
        
        # Check expected duration (should be ~6.5+ hours for RTH with buffer)
        if duration < 5:
            checks['mbp10_duration'] = False
            self.results['warnings'].append(f"Short MBP-10 duration: {duration:.2f} hours")
            self.logger.warning(f"  ⚠️  Short duration")
        else:
            checks['mbp10_duration'] = True
            self.logger.info(f"  ✅ Duration reasonable")
        
        # Check bid/ask levels
        if hasattr(first, 'levels') and first.levels:
            num_levels = len(first.levels)
            self.logger.info(f"  Levels per snapshot: {num_levels}")
            
            if num_levels != 10:
                checks['mbp10_levels'] = False
                self.results['warnings'].append(f"Expected 10 levels, got {num_levels}")
                self.logger.warning(f"  ⚠️  Expected 10 levels")
            else:
                checks['mbp10_levels'] = True
                self.logger.info(f"  ✅ 10 levels present")
            
            # Check first level values
            top_level = first.levels[0]
            self.logger.info(f"  Top bid: {top_level.bid_px} x {top_level.bid_sz}")
            self.logger.info(f"  Top ask: {top_level.ask_px} x {top_level.ask_sz}")
            
            if top_level.bid_px <= 0 or top_level.ask_px <= 0:
                checks['mbp10_prices'] = False
                self.results['errors'].append("Invalid bid/ask prices in MBP-10")
                self.results['passed'] = False
                self.logger.error(f"  ❌ Invalid prices")
            else:
                checks['mbp10_prices'] = True
                self.logger.info(f"  ✅ Prices valid")
        
        self.results['checks']['mbp10_snapshots'] = checks
    
    def _validate_option_trades(self, option_trades_df, date):
        """Validate ES options trades."""
        self.logger.info("\n4. Validating ES Options Trades...")
        
        checks = {}
        
        if option_trades_df is None or option_trades_df.empty:
            checks['options_non_empty'] = False
            self.results['warnings'].append("No ES options trades found (may be legitimate for low-volume days)")
            self.logger.warning("  ⚠️  No options trades (may be legitimate)")
            self.results['checks']['option_trades'] = checks
            return
        
        checks['options_non_empty'] = True
        
        # Check row count
        row_count = len(option_trades_df)
        self.logger.info(f"  Row count: {row_count:,}")
        
        # Check schema
        expected_cols = ['ts_event_ns', 'underlying', 'option_symbol', 'exp_date', 'strike', 'right', 'price', 'size']
        missing_cols = [col for col in expected_cols if col not in option_trades_df.columns]
        
        if missing_cols:
            checks['options_schema'] = False
            self.results['passed'] = False
            error = f"Options missing columns: {missing_cols}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['options_schema'] = True
            self.logger.info(f"  ✅ Schema valid")
        
        # Check underlying
        underlyings = option_trades_df['underlying'].unique()
        self.logger.info(f"  Underlyings: {list(underlyings)}")
        
        if 'ES' not in underlyings:
            checks['options_underlying'] = False
            self.results['warnings'].append(f"Expected ES underlying, found: {underlyings}")
            self.logger.warning(f"  ⚠️  No ES underlying found")
        else:
            checks['options_underlying'] = True
            self.logger.info(f"  ✅ ES underlying present")
        
        # Check 0DTE filtering
        exp_dates = option_trades_df['exp_date'].unique()
        self.logger.info(f"  Expiration dates: {list(exp_dates)[:5]}")
        
        # Verify all options are 0DTE (exp_date == session_date)
        expected_exp_date = date.replace('-', '')  # YYYYMMDD format
        non_0dte = [d for d in exp_dates if str(d) != expected_exp_date]
        
        if non_0dte:
            checks['options_0dte'] = False
            self.results['warnings'].append(f"Non-0DTE options found: {non_0dte[:5]}")
            self.logger.warning(f"  ⚠️  Non-0DTE options present")
        else:
            checks['options_0dte'] = True
            self.logger.info(f"  ✅ All options are 0DTE")
        
        # Check strikes
        strikes = option_trades_df['strike'].unique()
        self.logger.info(f"  Strike range: {strikes.min():.0f} - {strikes.max():.0f}")
        self.logger.info(f"  Unique strikes: {len(strikes)}")
        
        # Check strike spacing (should be multiples of 5 for ATM)
        strike_diffs = np.diff(sorted(strikes))
        common_spacing = pd.Series(strike_diffs).mode().values[0] if len(strike_diffs) > 0 else None
        
        if common_spacing:
            self.logger.info(f"  Common strike spacing: {common_spacing:.0f} points")
            
            if common_spacing not in [5, 10, 25, 50]:
                checks['options_strike_spacing'] = False
                self.results['warnings'].append(f"Unusual strike spacing: {common_spacing:.0f}")
                self.logger.warning(f"  ⚠️  Unusual strike spacing")
            else:
                checks['options_strike_spacing'] = True
                self.logger.info(f"  ✅ Strike spacing reasonable")
        
        # Check call/put distribution
        rights = option_trades_df['right'].value_counts()
        self.logger.info(f"  Call/Put distribution:")
        for right, count in rights.items():
            pct = count / len(option_trades_df) * 100
            self.logger.info(f"    {right}: {count:,} ({pct:.1f}%)")
        
        self.results['checks']['option_trades'] = checks


def main():
    parser = argparse.ArgumentParser(description='Validate Stage 0: LoadBronze')
    parser.add_argument('--date', type=str, required=True, help='Date to validate (YYYY-MM-DD)')
    parser.add_argument('--level', type=str, default='PM_HIGH', help='Level type (PM_HIGH, PM_LOW, OR_HIGH, OR_LOW, SMA_90)')
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints', help='Checkpoint directory')
    parser.add_argument('--canonical-version', type=str, default='4.0.0', help='Canonical version')
    parser.add_argument('--log-file', type=str, default=None, help='Log file path')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_file is None:
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        args.log_file = str(log_dir / f'validate_stage_00_{args.date}.log')
    
    logger = setup_logging(args.log_file)
    logger.info(f"Starting Stage 0 validation for {args.date}")
    logger.info(f"Log file: {args.log_file}")
    
    try:
        # Load checkpoint from stage 0 (should already exist from pipeline run)
        logger.info(f"Loading checkpoint from stage 0 for level {args.level}...")
        from src.pipeline.core.checkpoint import CheckpointManager
        manager = CheckpointManager(args.checkpoint_dir)
        ctx = manager.load_checkpoint("bronze_to_silver", args.date, stage_idx=0, level=args.level)
        
        if ctx is None:
            logger.error("Failed to load checkpoint")
            return 1
        
        # Validate
        validator = Stage0Validator(logger)
        results = validator.validate(args.date, ctx)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path(__file__).parent.parent / 'logs'
            output_path = output_dir / f'validate_stage_00_{args.date}_results.json'
        
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

