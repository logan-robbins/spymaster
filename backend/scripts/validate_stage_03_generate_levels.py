"""
Validate Stage 4: GenerateLevels

Goals:
1. Generate PM_HIGH/PM_LOW from premarket bars (04:00-09:30 ET)
2. Generate OR_HIGH/OR_LOW from opening range (09:30-09:45 ET)
3. Generate SMA_90/EMA_20 from 2min bars with warmup
4. Create static_level_info (static levels only)
5. Create dynamic_levels (per-bar level values)

Validation Checks:
- All 6 level kinds present (PM_HIGH/LOW, OR_HIGH/LOW, SMA_90/EMA_20)
- PM levels computed from premarket data
- OR levels from first 15 minutes
- SMA values reasonable (within price range)
- Dynamic levels have time series for each level kind
- Level prices are monotonic where expected (PM_HIGH > PM_LOW, etc)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from src.pipeline.pipelines.bronze_to_silver import build_bronze_to_silver_pipeline


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


class Stage4Validator:
    """Validator for GenerateLevels stage."""
    
    def __init__(self, logger):
        self.logger = logger
        self.results = {
            'stage': 'generate_levels',
            'stage_idx': 3,
            'checks': {},
            'warnings': [],
            'errors': [],
            'passed': True
        }
    
    def validate(self, date: str, ctx) -> Dict[str, Any]:
        """Run all validation checks."""
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Validating Stage 4: GenerateLevels for {date}")
        self.logger.info(f"{'='*80}")
        
        self.results['date'] = date
        
        # Check 1: Required outputs present
        self._check_required_outputs(ctx)
        
        # Check 2: Validate level_info
        if 'level_info' in ctx.data:
            self._validate_level_info(ctx.data['level_info'], date, ctx)
        
        # Check 3: Validate static_level_info
        if 'static_level_info' in ctx.data:
            self._validate_static_level_info(ctx.data['static_level_info'])
        
        # Check 4: Validate dynamic_levels
        if 'dynamic_levels' in ctx.data:
            self._validate_dynamic_levels(ctx.data['dynamic_levels'], date, ctx)
        
        # Summary
        self.logger.info(f"\n{'='*80}")
        if self.results['passed']:
            self.logger.info("✅ Stage 4 Validation: PASSED")
        else:
            self.logger.error("❌ Stage 4 Validation: FAILED")
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
        
        # New from this stage
        new_required = ['level_info', 'static_level_info', 'dynamic_levels']
        
        available = list(ctx.data.keys())
        missing_new = [key for key in new_required if key not in available]
        
        if missing_new:
            self.results['checks']['new_outputs_present'] = False
            self.results['passed'] = False
            error = f"Missing new outputs: {missing_new}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            self.results['checks']['new_outputs_present'] = True
            self.logger.info(f"  ✅ New outputs present: level_info, static_level_info, dynamic_levels")
    
    def _validate_level_info(self, level_info, date, ctx):
        """Validate level_info structure and values."""
        self.logger.info("\n2. Validating Level Universe...")
        
        checks = {}
        
        # Check structure
        if not hasattr(level_info, 'prices') or not hasattr(level_info, 'kind_names'):
            checks['level_info_structure'] = False
            self.results['passed'] = False
            error = "level_info missing prices or kind_names attributes"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            return
        
        checks['level_info_structure'] = True
        
        # Check level count
        num_levels = len(level_info.prices)
        self.logger.info(f"  Total levels: {num_levels}")
        self.logger.info(f"  Level kinds: {level_info.kind_names}")
        
        if num_levels == 0:
            checks['levels_generated'] = False
            self.results['passed'] = False
            error = "No levels generated"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            return
        
        checks['levels_generated'] = True
        
        # Check for required level kinds
        expected_kinds = ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_90', 'EMA_20']
        found_kinds = set(level_info.kind_names)
        
        missing_kinds = [k for k in expected_kinds if k not in found_kinds]
        
        if missing_kinds:
            checks['all_kinds_present'] = False
            warning = f"Missing level kinds: {missing_kinds}"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
        else:
            checks['all_kinds_present'] = True
            self.logger.info(f"  ✅ All 6 level kinds present")
        
        # Display level values
        self.logger.info(f"\n  Level values:")
        for i, (price, name) in enumerate(zip(level_info.prices, level_info.kind_names)):
            self.logger.info(f"    {name:<12}: {price:>8.2f}")
        
        # Check PM_HIGH > PM_LOW
        pm_high_prices = [p for p, n in zip(level_info.prices, level_info.kind_names) if n == 'PM_HIGH']
        pm_low_prices = [p for p, n in zip(level_info.prices, level_info.kind_names) if n == 'PM_LOW']
        
        if pm_high_prices and pm_low_prices:
            if pm_high_prices[0] <= pm_low_prices[0]:
                checks['pm_ordering'] = False
                error = f"PM_HIGH ({pm_high_prices[0]:.2f}) <= PM_LOW ({pm_low_prices[0]:.2f})"
                self.results['errors'].append(error)
                self.results['passed'] = False
                self.logger.error(f"  ❌ {error}")
            else:
                checks['pm_ordering'] = True
                spread = pm_high_prices[0] - pm_low_prices[0]
                self.logger.info(f"  ✅ PM_HIGH > PM_LOW (spread: {spread:.2f} points)")
        
        # Check OR_HIGH > OR_LOW
        or_high_prices = [p for p, n in zip(level_info.prices, level_info.kind_names) if n == 'OR_HIGH']
        or_low_prices = [p for p, n in zip(level_info.prices, level_info.kind_names) if n == 'OR_LOW']
        
        if or_high_prices and or_low_prices:
            if or_high_prices[0] <= or_low_prices[0]:
                checks['or_ordering'] = False
                error = f"OR_HIGH ({or_high_prices[0]:.2f}) <= OR_LOW ({or_low_prices[0]:.2f})"
                self.results['errors'].append(error)
                self.results['passed'] = False
                self.logger.error(f"  ❌ {error}")
            else:
                checks['or_ordering'] = True
                spread = or_high_prices[0] - or_low_prices[0]
                self.logger.info(f"  ✅ OR_HIGH > OR_LOW (spread: {spread:.2f} points)")
        
        # Check SMA values are within reasonable range
        sma_90_prices = [p for p, n in zip(level_info.prices, level_info.kind_names) if n == 'SMA_90']
        ema_20_prices = [p for p, n in zip(level_info.prices, level_info.kind_names) if n == 'EMA_20']
        
        # Get price range from ohlcv for reference
        if 'ohlcv_1min' in ctx.data:
            ohlcv_1min = ctx.data['ohlcv_1min']
            price_min = ohlcv_1min['low'].min()
            price_max = ohlcv_1min['high'].max()
            
            if sma_90_prices:
                sma_90 = sma_90_prices[0]
                if sma_90 < price_min - 50 or sma_90 > price_max + 50:
                    checks['sma_90_range'] = False
                    warning = f"SMA_90 ({sma_90:.2f}) far from price range ({price_min:.2f}-{price_max:.2f})"
                    self.results['warnings'].append(warning)
                    self.logger.warning(f"  ⚠️  {warning}")
                else:
                    checks['sma_90_range'] = True
                    self.logger.info(f"  ✅ SMA_90 within reasonable range")
            
            if ema_20_prices:
                ema_20 = ema_20_prices[0]
                if ema_20 < price_min - 50 or ema_20 > price_max + 50:
                    checks['ema_20_range'] = False
                    warning = f"EMA_20 ({ema_20:.2f}) far from price range ({price_min:.2f}-{price_max:.2f})"
                    self.results['warnings'].append(warning)
                    self.logger.warning(f"  ⚠️  {warning}")
                else:
                    checks['ema_20_range'] = True
                    self.logger.info(f"  ✅ EMA_20 within reasonable range")
        
        self.results['checks']['level_info'] = checks
    
    def _validate_static_level_info(self, static_level_info):
        """Validate static_level_info (for future static levels if any)."""
        self.logger.info("\n3. Validating Static Level Info...")
        
        checks = {}
        
        num_static = len(static_level_info.prices)
        self.logger.info(f"  Static levels: {num_static}")
        
        if num_static > 0:
            self.logger.info(f"  Static level kinds: {static_level_info.kind_names}")
            checks['static_levels_present'] = True
        else:
            self.logger.info(f"  ℹ️  No static levels (all levels are dynamic)")
            checks['static_levels_present'] = False
        
        self.results['checks']['static_level_info'] = checks
    
    def _validate_dynamic_levels(self, dynamic_levels, date, ctx):
        """Validate dynamic_levels time series."""
        self.logger.info("\n4. Validating Dynamic Levels Series...")
        
        checks = {}
        
        if not isinstance(dynamic_levels, dict):
            checks['dynamic_levels_type'] = False
            self.results['passed'] = False
            error = f"dynamic_levels is not dict: {type(dynamic_levels)}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            return
        
        checks['dynamic_levels_type'] = True
        
        # Check keys
        level_keys = list(dynamic_levels.keys())
        self.logger.info(f"  Dynamic level series: {level_keys}")
        
        # Expected series
        expected_series = ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_90', 'EMA_20']
        missing_series = [k for k in expected_series if k not in level_keys]
        
        if missing_series:
            checks['expected_series_present'] = False
            warning = f"Missing dynamic level series: {missing_series}"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
        else:
            checks['expected_series_present'] = True
            self.logger.info(f"  ✅ All expected series present")
        
        # Validate each series
        for key, series in dynamic_levels.items():
            if not isinstance(series, pd.Series):
                checks[f'{key}_is_series'] = False
                warning = f"{key} is not pd.Series: {type(series)}"
                self.results['warnings'].append(warning)
                self.logger.warning(f"  ⚠️  {warning}")
                continue
            
            # Check length matches ohlcv_1min
            if 'ohlcv_1min' in ctx.data:
                expected_len = len(ctx.data['ohlcv_1min'])
                actual_len = len(series)
                
                if actual_len != expected_len:
                    checks[f'{key}_length'] = False
                    warning = f"{key} length mismatch: {actual_len} vs {expected_len} (ohlcv_1min)"
                    self.results['warnings'].append(warning)
                    self.logger.warning(f"  ⚠️  {warning}")
                else:
                    self.logger.info(f"  ✅ {key}: {actual_len} values")
        
        self.results['checks']['dynamic_levels'] = checks


def main():
    parser = argparse.ArgumentParser(description='Validate Stage 4: GenerateLevels')
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
        args.log_file = str(log_dir / f'validate_stage_04_{args.date}.log')
    
    logger = setup_logging(args.log_file)
    logger.info(f"Starting Stage 4 validation for {args.date}")
    logger.info(f"Log file: {args.log_file}")
    
    try:# Load checkpoint from stage (should already exist from pipeline run)
        from src.pipeline.core.checkpoint import CheckpointManager
        manager = CheckpointManager(args.checkpoint_dir)
        ctx = manager.load_checkpoint("bronze_to_silver", args.date, stage_idx=3)
        
        if ctx is None:
            logger.error("Failed to load checkpoint")
            return 1
        
        # Validate
        validator = Stage4Validator(logger)
        results = validator.validate(args.date, ctx)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path(__file__).parent.parent / 'logs'
            output_path = output_dir / f'validate_stage_04_{args.date}_results.json'
        
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
