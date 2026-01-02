"""
Validate Stage 3: InitMarketState

Goals:
1. Initialize MarketState with ES trades and MBP-10 data
2. Compute Greeks (delta/gamma) for ES options using Black-76
3. Load options into MarketState
4. Determine spot price from ES futures
5. Verify MarketState buffers are populated

Validation Checks:
- MarketState instance created
- Trades loaded into MarketState buffers
- MBP-10 snapshots loaded
- Options DataFrame enriched with delta/gamma
- Greeks values reasonable (delta: -1 to 1, gamma: positive)
- Spot price determined correctly
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from src.pipeline.pipelines.bronze_to_silver import build_bronze_to_silver_pipeline
from src.core.market_state import MarketState


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


class Stage3Validator:
    """Validator for InitMarketState stage."""
    
    def __init__(self, logger):
        self.logger = logger
        self.results = {
            'stage': 'init_market_state',
            'stage_idx': 3,
            'checks': {},
            'warnings': [],
            'errors': [],
            'passed': True
        }
    
    def validate(self, date: str, ctx) -> Dict[str, Any]:
        """Run all validation checks."""
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Validating Stage 3: InitMarketState for {date}")
        self.logger.info(f"{'='*80}")
        
        self.results['date'] = date
        
        # Check 1: Required outputs present
        self._check_required_outputs(ctx)
        
        # Check 2: Validate MarketState
        if 'market_state' in ctx.data:
            self._validate_market_state(ctx.data['market_state'], date)
        
        # Check 3: Validate Greeks
        if 'option_trades_df' in ctx.data:
            self._validate_greeks(ctx.data['option_trades_df'], date)
        
        # Check 4: Validate spot price
        if 'spot_price' in ctx.data:
            self._validate_spot_price(ctx.data['spot_price'])
        
        # Summary
        self.logger.info(f"\n{'='*80}")
        if self.results['passed']:
            self.logger.info("✅ Stage 3 Validation: PASSED")
        else:
            self.logger.error("❌ Stage 3 Validation: FAILED")
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
        previous_required = ['trades', 'trades_df', 'mbp10_snapshots', 'option_trades_df', 
                           'ohlcv_1min', 'ohlcv_2min', 'atr', 'volatility']
        # New from this stage
        new_required = ['market_state', 'spot_price']
        
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
            self.logger.info(f"  ✅ New outputs present: market_state, spot_price")
    
    def _validate_market_state(self, market_state, date):
        """Validate MarketState instance."""
        self.logger.info("\n2. Validating MarketState...")
        
        checks = {}
        
        # Check instance type
        if not isinstance(market_state, MarketState):
            checks['market_state_type'] = False
            self.results['passed'] = False
            error = f"market_state is not MarketState instance: {type(market_state)}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            return
        
        checks['market_state_type'] = True
        self.logger.info(f"  ✅ MarketState instance created")
        
        # Check ES trades buffer
        trades_count = len(market_state.es_trades_buffer)
        self.logger.info(f"  ES trades in buffer: {trades_count:,}")
        
        if trades_count == 0:
            checks['trades_loaded'] = False
            self.results['errors'].append("No ES trades loaded into MarketState")
            self.results['passed'] = False
            self.logger.error(f"  ❌ No trades in buffer")
        else:
            checks['trades_loaded'] = True
            self.logger.info(f"  ✅ Trades loaded")
        
        # Check MBP-10 buffer
        mbp_count = len(market_state.es_mbp10_buffer)
        self.logger.info(f"  MBP-10 snapshots in buffer: {mbp_count:,}")
        
        if mbp_count == 0:
            checks['mbp10_loaded'] = False
            self.results['warnings'].append("No MBP-10 snapshots in MarketState")
            self.logger.warning(f"  ⚠️  No MBP-10 in buffer")
        else:
            checks['mbp10_loaded'] = True
            self.logger.info(f"  ✅ MBP-10 loaded")
        
        # Check option flows
        if hasattr(market_state, 'option_flows'):
            flows_count = len(market_state.option_flows)
            self.logger.info(f"  Option flows in buffer: {flows_count:,}")
            
            if flows_count == 0:
                checks['options_loaded'] = False
                self.results['warnings'].append("No option flows in MarketState")
                self.logger.warning(f"  ⚠️  No option flows")
            else:
                checks['options_loaded'] = True
                self.logger.info(f"  ✅ Option flows loaded")
        
        # Check buffer window from ring buffer
        if hasattr(market_state.es_trades_buffer, 'max_window_ns'):
            buffer_window = market_state.es_trades_buffer.max_window_ns / 1e9
            self.logger.info(f"  Buffer window: {buffer_window:.0f} seconds")
            
            if buffer_window < 240:
                checks['buffer_window'] = False
                warning = f"Small buffer window: {buffer_window}s (need >= 240s for barrier)"
                self.results['warnings'].append(warning)
                self.logger.warning(f"  ⚠️  {warning}")
            else:
                checks['buffer_window'] = True
                self.logger.info(f"  ✅ Buffer window sufficient")
        
        self.results['checks']['market_state'] = checks
    
    def _validate_greeks(self, option_trades_df, date):
        """Validate Greeks computation."""
        self.logger.info("\n3. Validating Greeks (delta/gamma)...")
        
        checks = {}
        
        if option_trades_df is None or option_trades_df.empty:
            checks['options_present'] = False
            self.results['warnings'].append("No options trades (may be legitimate)")
            self.logger.warning("  ⚠️  No options trades")
            self.results['checks']['greeks'] = checks
            return
        
        checks['options_present'] = True
        
        # Check for delta/gamma columns
        has_delta = 'delta' in option_trades_df.columns
        has_gamma = 'gamma' in option_trades_df.columns
        
        if not has_delta or not has_gamma:
            checks['greeks_computed'] = False
            self.results['passed'] = False
            error = f"Greeks not computed: delta={has_delta}, gamma={has_gamma}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            self.results['checks']['greeks'] = checks
            return
        
        checks['greeks_computed'] = True
        self.logger.info(f"  ✅ Delta and gamma columns present")
        
        # Validate delta values (-1 to 1)
        delta_min = option_trades_df['delta'].min()
        delta_max = option_trades_df['delta'].max()
        
        self.logger.info(f"  Delta range: {delta_min:.4f} to {delta_max:.4f}")
        
        if delta_min < -1.1 or delta_max > 1.1:
            checks['delta_range'] = False
            error = f"Delta out of range: {delta_min:.4f} to {delta_max:.4f}"
            self.results['errors'].append(error)
            self.results['passed'] = False
            self.logger.error(f"  ❌ {error}")
        else:
            checks['delta_range'] = True
            self.logger.info(f"  ✅ Delta range valid")
        
        # Check delta by call/put
        if 'right' in option_trades_df.columns:
            calls = option_trades_df[option_trades_df['right'] == 'C']
            puts = option_trades_df[option_trades_df['right'] == 'P']
            
            if not calls.empty:
                call_delta_mean = calls['delta'].mean()
                self.logger.info(f"  Call delta mean: {call_delta_mean:.4f} (should be positive)")
                
                if call_delta_mean < 0:
                    checks['call_delta_sign'] = False
                    warning = f"Call delta mean is negative: {call_delta_mean:.4f}"
                    self.results['warnings'].append(warning)
                    self.logger.warning(f"  ⚠️  {warning}")
                else:
                    checks['call_delta_sign'] = True
            
            if not puts.empty:
                put_delta_mean = puts['delta'].mean()
                self.logger.info(f"  Put delta mean: {put_delta_mean:.4f} (should be negative)")
                
                if put_delta_mean > 0:
                    checks['put_delta_sign'] = False
                    warning = f"Put delta mean is positive: {put_delta_mean:.4f}"
                    self.results['warnings'].append(warning)
                    self.logger.warning(f"  ⚠️  {warning}")
                else:
                    checks['put_delta_sign'] = True
        
        # Validate gamma values (always positive)
        gamma_min = option_trades_df['gamma'].min()
        gamma_max = option_trades_df['gamma'].max()
        gamma_mean = option_trades_df['gamma'].mean()
        
        self.logger.info(f"  Gamma range: {gamma_min:.6f} to {gamma_max:.6f}")
        self.logger.info(f"  Gamma mean: {gamma_mean:.6f}")
        
        if gamma_min < 0:
            checks['gamma_positive'] = False
            error = f"Negative gamma values found: min={gamma_min:.6f}"
            self.results['errors'].append(error)
            self.results['passed'] = False
            self.logger.error(f"  ❌ {error}")
        else:
            checks['gamma_positive'] = True
            self.logger.info(f"  ✅ All gamma values positive")
        
        # Check for NaN Greeks
        nan_delta = option_trades_df['delta'].isna().sum()
        nan_gamma = option_trades_df['gamma'].isna().sum()
        
        if nan_delta > 0 or nan_gamma > 0:
            checks['greeks_complete'] = False
            warning = f"NaN Greeks: delta={nan_delta}, gamma={nan_gamma}"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
        else:
            checks['greeks_complete'] = True
            self.logger.info(f"  ✅ No NaN Greeks")
        
        self.results['checks']['greeks'] = checks
    
    def _validate_spot_price(self, spot_price):
        """Validate spot price determination."""
        self.logger.info("\n4. Validating Spot Price...")
        
        checks = {}
        
        self.logger.info(f"  Spot price: {spot_price:.2f}")
        
        # Check reasonable ES price range
        if spot_price < 3000 or spot_price > 10000:
            checks['spot_price_range'] = False
            error = f"Spot price out of reasonable range: {spot_price:.2f}"
            self.results['errors'].append(error)
            self.results['passed'] = False
            self.logger.error(f"  ❌ {error}")
        else:
            checks['spot_price_range'] = True
            self.logger.info(f"  ✅ Spot price reasonable")
        
        self.results['checks']['spot_price'] = checks


def main():
    parser = argparse.ArgumentParser(description='Validate Stage 3: InitMarketState')
    parser.add_argument('--date', type=str, required=True, help='Date to validate (YYYY-MM-DD)')
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints', help='Checkpoint directory')
    parser.add_argument('--canonical-version', type=str, default='4.0.0', help='Canonical version')
    parser.add_argument('--log-file', type=str, default=None, help='Log file path')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_file is None:
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        args.log_file = str(log_dir / f'validate_stage_03_{args.date}.log')
    
    logger = setup_logging(args.log_file)
    logger.info(f"Starting Stage 3 validation for {args.date}")
    logger.info(f"Log file: {args.log_file}")
    
    try:# Load checkpoint from stage (should already exist from pipeline run)
        from src.pipeline.core.checkpoint import CheckpointManager
        manager = CheckpointManager(args.checkpoint_dir)
        ctx = manager.load_checkpoint("bronze_to_silver", args.date, stage_idx=3)
        
        if ctx is None:
            logger.error("Failed to load checkpoint")
            return 1
        
        # Validate
        validator = Stage3Validator(logger)
        results = validator.validate(args.date, ctx)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path(__file__).parent.parent / 'logs'
            output_path = output_dir / f'validate_stage_03_{args.date}_results.json'
        
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

