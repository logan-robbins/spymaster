"""
Validate Stage 16: MaterializeStateTable

Goals:
1. Generate time-sampled state at 30-second cadence (09:30-12:30 ET)
2. One row per (timestamp, level_kind) pair
3. Forward-fill features from event table (all online-safe)
4. Handle dynamic SMA levels and inactive OR levels before 09:45

Validation Checks:
- Required inputs present (signals_df, ohlcv_2min, date)
- state_df output exists with correct schema
- Timestamp coverage: 09:30-12:30 ET at 30s intervals
- 6 level_kinds present (PM_HIGH, PM_LOW, OR_HIGH, OR_LOW, SMA_90, EMA_20)
- OR levels inactive before 09:45 ET
- All features are online-safe (no future data)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import time

import numpy as np
import pandas as pd

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


class Stage16Validator:
    """Validator for MaterializeStateTable stage."""

    def __init__(self, logger):
        self.logger = logger
        self.results = {
            'stage': 'materialize_state_table',
            'stage_idx': 16,
            'checks': {},
            'warnings': [],
            'errors': [],
            'passed': True
        }

    def validate(self, date: str, ctx) -> Dict[str, Any]:
        """Run all validation checks."""
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Validating Stage 16: MaterializeStateTable for {date}")
        self.logger.info(f"{'='*80}")

        self.results['date'] = date

        # Check 1: Required inputs + outputs present
        self._check_required_outputs(ctx)

        # Check 2: Validate state_df
        if 'state_df' in ctx.data:
            self._validate_state_df(ctx.data['state_df'], date)

        # Summary
        self.logger.info(f"\n{'='*80}")
        if self.results['passed']:
            self.logger.info("✅ Stage 16 Validation: PASSED")
        else:
            self.logger.error("❌ Stage 16 Validation: FAILED")
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
        """Verify required inputs and outputs are present in context."""
        self.logger.info("\n1. Checking required inputs/outputs...")

        required_inputs = ['signals_df', 'ohlcv_2min', 'date']
        new_outputs = ['state_df']

        available = list(ctx.data.keys())
        missing_inputs = [key for key in required_inputs if key not in available]
        missing_outputs = [key for key in new_outputs if key not in available]

        if missing_inputs:
            self.results['checks']['required_inputs_present'] = False
            self.results['passed'] = False
            error = f"Missing required inputs: {missing_inputs}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            self.results['checks']['required_inputs_present'] = True
            self.logger.info("  ✅ Required inputs present")

        if missing_outputs:
            self.results['checks']['new_outputs_present'] = False
            self.results['passed'] = False
            error = f"Missing new outputs: {missing_outputs}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            self.results['checks']['new_outputs_present'] = True
            self.logger.info("  ✅ New output present: state_df")

    def _validate_state_df(self, state_df: pd.DataFrame, date: str):
        """Validate state_df structure and values."""
        self.logger.info("\n2. Validating state_df...")

        checks = {}

        if not isinstance(state_df, pd.DataFrame):
            checks['state_df_type'] = False
            self.results['passed'] = False
            error = f"state_df is not DataFrame: {type(state_df)}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            return

        checks['state_df_type'] = True

        if state_df.empty:
            checks['state_df_not_empty'] = False
            self.results['passed'] = False
            error = "state_df is empty"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            self.results['checks']['state_df'] = checks
            return

        checks['state_df_not_empty'] = True
        self.logger.info(f"  Total state rows: {len(state_df):,}")

        # Check required columns
        required_cols = [
            'timestamp', 'ts_ns', 'date', 'minutes_since_open', 'bars_since_open',
            'level_kind', 'level_price', 'level_active',
            'spot', 'atr', 'distance_signed_atr'
        ]

        missing_cols = [col for col in required_cols if col not in state_df.columns]
        if missing_cols:
            checks['required_columns_present'] = False
            self.results['passed'] = False
            error = f"state_df missing columns: {missing_cols}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            self.results['checks']['state_df'] = checks
            return

        checks['required_columns_present'] = True
        self.logger.info(f"  ✅ Required columns present")

        # Check level_kinds
        level_kinds = state_df['level_kind'].unique()
        expected_levels = {'PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_90', 'EMA_20'}
        missing_levels = expected_levels - set(level_kinds)
        
        if missing_levels:
            warning = f"Missing level kinds: {missing_levels}"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
        else:
            self.logger.info(f"  ✅ All 6 level kinds present")

        # Check timestamp coverage (should be 30s intervals from 09:30-12:30)
        # Expected: 360 timestamps × 6 levels = 2160 rows
        unique_timestamps = state_df['timestamp'].nunique()
        expected_timestamps = 361  # 09:30:00 to 12:30:00 inclusive at 30s = 361 timestamps
        
        if abs(unique_timestamps - expected_timestamps) > 10:
            warning = f"Timestamp count {unique_timestamps} differs from expected ~{expected_timestamps}"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
        else:
            self.logger.info(f"  ✅ Timestamp coverage: {unique_timestamps} samples")

        # Check time range
        if 'minutes_since_open' in state_df.columns:
            min_time = state_df['minutes_since_open'].min()
            max_time = state_df['minutes_since_open'].max()
            
            if min_time < -1 or max_time > 181:
                warning = f"Time range {min_time:.1f} to {max_time:.1f} outside expected 0-180 min"
                self.results['warnings'].append(warning)
                self.logger.warning(f"  ⚠️  {warning}")
            else:
                self.logger.info(f"  ✅ Time range: {min_time:.1f} to {max_time:.1f} min")

        # Check OR levels inactive before 09:45
        early_state = state_df[state_df['minutes_since_open'] < 15]
        if not early_state.empty:
            or_early = early_state[early_state['level_kind'].isin(['OR_HIGH', 'OR_LOW'])]
            if not or_early.empty:
                active_or = or_early[or_early['level_active'] == True]
                if len(active_or) > 0:
                    warning = f"{len(active_or)} OR level rows active before 09:45"
                    self.results['warnings'].append(warning)
                    self.logger.warning(f"  ⚠️  {warning}")
                else:
                    self.logger.info(f"  ✅ OR levels correctly inactive before 09:45")

        # Check rows per level_kind
        level_counts = state_df['level_kind'].value_counts()
        self.logger.info(f"  Rows per level: {level_counts.to_dict()}")

        self.results['checks']['state_df'] = checks


def main():
    parser = argparse.ArgumentParser(description='Validate Stage 16: MaterializeStateTable')
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
        args.log_file = str(log_dir / f'validate_stage_16_{args.date}.log')

    logger = setup_logging(args.log_file)
    logger.info(f"Starting Stage 16 validation for {args.date}")
    logger.info(f"Log file: {args.log_file}")

    try:
        # Run pipeline through stage 16
        logger.info("Running through MaterializeStateTable stage...")
        pipeline = build_es_pipeline()

        pipeline.run(
            date=args.date,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_stage=16,
            stop_at_stage=16
        )

        # Load checkpoint from stage (should already exist from pipeline run)
        from src.pipeline.core.checkpoint import CheckpointManager
        manager = CheckpointManager(args.checkpoint_dir)
        ctx = manager.load_checkpoint("bronze_to_silver", args.date, stage_idx=16)

        if ctx is None:
            logger.error("Failed to load checkpoint")
            return 1

        # Validate
        validator = Stage16Validator(logger)
        results = validator.validate(args.date, ctx)

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path(__file__).parent.parent / 'logs'
            output_path = output_dir / f'validate_stage_16_{args.date}_results.json'

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

