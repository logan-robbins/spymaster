"""
Validate Stage 10: ComputeLevelDistances

Goals:
1. Compute signed distance features to structural levels
2. Add ATR-normalized variants
3. Add level stacking counts

Validation Checks:
- Required inputs present (signals_df, dynamic_levels, atr)
- signals_df output exists and has required distance columns
- Row count matches touches_df
- Distance columns numeric (warn if all-NaN)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

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


class Stage10Validator:
    """Validator for ComputeLevelDistances stage."""

    def __init__(self, logger):
        self.logger = logger
        self.results = {
            'stage': 'compute_level_distances',
            'stage_idx': 9,
            'checks': {},
            'warnings': [],
            'errors': [],
            'passed': True
        }

    def validate(self, date: str, ctx) -> Dict[str, Any]:
        """Run all validation checks."""
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Validating Stage 10: ComputeLevelDistances for {date}")
        self.logger.info(f"{'='*80}")

        self.results['date'] = date

        # Check 1: Required inputs + outputs present
        self._check_required_outputs(ctx)

        # Check 2: Validate signals_df
        if 'signals_df' in ctx.data:
            self._validate_signals_df(ctx.data['signals_df'], ctx)

        # Summary
        self.logger.info(f"\n{'='*80}")
        if self.results['passed']:
            self.logger.info("✅ Stage 10 Validation: PASSED")
        else:
            self.logger.error("❌ Stage 10 Validation: FAILED")
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

        required_inputs = ['signals_df', 'dynamic_levels', 'atr']
        new_outputs = ['signals_df']

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
            self.logger.info("  ✅ New output present: signals_df")

    def _validate_signals_df(self, signals_df: pd.DataFrame, ctx):
        """Validate signals_df structure and values."""
        self.logger.info("\n2. Validating signals_df...")

        checks = {}

        if not isinstance(signals_df, pd.DataFrame):
            checks['signals_df_type'] = False
            self.results['passed'] = False
            error = f"signals_df is not DataFrame: {type(signals_df)}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            return

        checks['signals_df_type'] = True

        if signals_df.empty:
            warning = "signals_df is empty (no level distances computed)"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
            self.results['checks']['signals_df'] = checks
            return

        self.logger.info(f"  Total signals: {len(signals_df):,}")

        level_types = ['pm_high', 'pm_low', 'or_high', 'or_low', 'sma_90', 'ema_20']
        distance_cols = []
        for level_type in level_types:
            distance_cols.append(f'dist_to_{level_type}')
            distance_cols.append(f'dist_to_{level_type}_atr')

        extra_cols = [
            'level_stacking_2pt',
            'level_stacking_5pt',
            'level_stacking_10pt'
        ]

        required_cols = distance_cols + extra_cols

        missing_cols = [col for col in required_cols if col not in signals_df.columns]
        if missing_cols:
            checks['required_columns_present'] = False
            self.results['passed'] = False
            error = f"signals_df missing distance columns: {missing_cols}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            self.results['checks']['signals_df'] = checks
            return

        checks['required_columns_present'] = True

        # Row count matches touches_df
        touches_df = ctx.data.get('touches_df')
        if isinstance(touches_df, pd.DataFrame) and not touches_df.empty:
            if len(signals_df) != len(touches_df):
                checks['row_count_match'] = False
                self.results['passed'] = False
                error = f"signals_df length {len(signals_df)} != touches_df length {len(touches_df)}"
                self.results['errors'].append(error)
                self.logger.error(f"  ❌ {error}")
            else:
                checks['row_count_match'] = True
                self.logger.info("  ✅ signals_df row count matches touches_df")

        # Numeric distance columns validation
        for col in required_cols:
            values = pd.to_numeric(signals_df[col], errors='coerce')
            if values.isna().any():
                checks[f'{col}_nan'] = False
                warning = f"{col} has NaN values"
                self.results['warnings'].append(warning)
                self.logger.warning(f"  ⚠️  {warning}")
            else:
                checks[f'{col}_nan'] = True

        # Warn if all base distance columns are NaN (likely missing bar_idx alignment)
        base_dist = signals_df[[f'dist_to_{lt}' for lt in level_types]]
        if base_dist.isna().all().all():
            warning = "All base distance columns are NaN (check bar_idx alignment)"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")

        self.results['checks']['signals_df'] = checks


def main():
    parser = argparse.ArgumentParser(description='Validate Stage 10: ComputeLevelDistances')
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
        args.log_file = str(log_dir / f'validate_stage_10_{args.date}.log')

    logger = setup_logging(args.log_file)
    logger.info(f"Starting Stage 10 validation for {args.date}")
    logger.info(f"Log file: {args.log_file}")

    try:
        # Run pipeline through stage 10
        logger.info("Running through ComputeLevelDistances stage...")
        pipeline = build_es_pipeline()

        pipeline.run(
            date=args.date,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_stage=10,
            stop_at_stage=10
        )

        # Load checkpoint from stage (should already exist from pipeline run)
        from src.pipeline.core.checkpoint import CheckpointManager
        manager = CheckpointManager(args.checkpoint_dir)
        ctx = manager.load_checkpoint("bronze_to_silver", args.date, stage_idx=9)

        if ctx is None:
            logger.error("Failed to load checkpoint")
            return 1

        # Validate
        validator = Stage10Validator(logger)
        results = validator.validate(args.date, ctx)

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path(__file__).parent.parent / 'logs'
            output_path = output_dir / f'validate_stage_10_{args.date}_results.json'

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
