"""
Validate Stage 15: FilterRTH

Goals:
1. Filter signals to first 4 hours (09:30-13:30 ET)
2. Output final 'signals' DataFrame

Validation Checks:
- Required inputs present (signals_df)
- signals output exists and is DataFrame
- ts_ns within RTH bounds
- bar_idx column removed
- Schema validation passes (informational)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.common.config import CONFIG
from src.common.schemas.silver_features import validate_silver_features
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


class Stage15Validator:
    """Validator for FilterRTH stage."""

    def __init__(self, logger):
        self.logger = logger
        self.results = {
            'stage': 'filter_rth',
            'stage_idx': 15,
            'checks': {},
            'warnings': [],
            'errors': [],
            'passed': True
        }

    def validate(self, date: str, ctx) -> Dict[str, Any]:
        """Run all validation checks."""
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Validating Stage 15: FilterRTH for {date}")
        self.logger.info(f"{'='*80}")

        self.results['date'] = date

        # Check 1: Required inputs + outputs present
        self._check_required_outputs(ctx)

        # Check 2: Validate signals
        if 'signals' in ctx.data:
            self._validate_signals(ctx.data['signals'], date, ctx)

        # Summary
        self.logger.info(f"\n{'='*80}")
        if self.results['passed']:
            self.logger.info("✅ Stage 15 Validation: PASSED")
        else:
            self.logger.error("❌ Stage 15 Validation: FAILED")
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

        required_inputs = ['signals_df']
        new_outputs = ['signals']

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
            self.logger.info("  ✅ New output present: signals")

    def _validate_signals(self, signals: pd.DataFrame, date: str, ctx):
        """Validate filtered signals."""
        self.logger.info("\n2. Validating signals...")

        checks = {}

        if not isinstance(signals, pd.DataFrame):
            checks['signals_type'] = False
            self.results['passed'] = False
            error = f"signals is not DataFrame: {type(signals)}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            return

        checks['signals_type'] = True

        if signals.empty:
            warning = "signals is empty after RTH filtering"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
            self.results['checks']['signals'] = checks
            return

        self.logger.info(f"  Filtered signals: {len(signals):,}")

        # Compare counts if pre-filter data present
        signals_df = ctx.data.get('signals_df')
        if isinstance(signals_df, pd.DataFrame) and not signals_df.empty:
            if len(signals) > len(signals_df):
                checks['row_count'] = False
                self.results['passed'] = False
                error = "Filtered signals larger than pre-filter signals_df"
                self.results['errors'].append(error)
                self.logger.error(f"  ❌ {error}")
            else:
                checks['row_count'] = True
                self.logger.info(f"  ✅ Row count reduced: {len(signals_df)} → {len(signals)}")

        # RTH bounds
        session_start = pd.Timestamp(date, tz="America/New_York") + pd.Timedelta(
            hours=CONFIG.RTH_START_HOUR,
            minutes=CONFIG.RTH_START_MINUTE
        )
        session_end = pd.Timestamp(date, tz="America/New_York") + pd.Timedelta(
            hours=CONFIG.RTH_END_HOUR,
            minutes=CONFIG.RTH_END_MINUTE
        )
        start_ns = session_start.tz_convert("UTC").value
        end_ns = session_end.tz_convert("UTC").value

        ts_ns = pd.to_numeric(signals['ts_ns'], errors='coerce')
        if ts_ns.isna().any():
            checks['ts_ns_nan'] = False
            self.results['passed'] = False
            error = "signals ts_ns has NaN values"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['ts_ns_nan'] = True
            out_of_bounds = ((ts_ns < start_ns) | (ts_ns > end_ns)).sum()
            if out_of_bounds > 0:
                checks['rth_bounds'] = False
                self.results['passed'] = False
                error = f"{out_of_bounds} signals outside RTH bounds"
                self.results['errors'].append(error)
                self.logger.error(f"  ❌ {error}")
            else:
                checks['rth_bounds'] = True
                self.logger.info("  ✅ All signals within RTH bounds")

        # Ensure bar_idx dropped
        if 'bar_idx' in signals.columns:
            checks['bar_idx_removed'] = False
            self.results['passed'] = False
            error = "bar_idx column present after filter_rth"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['bar_idx_removed'] = True
            self.logger.info("  ✅ bar_idx column removed")

        # Schema validation (informational)
        try:
            validate_silver_features(signals)
            self.logger.info("  ✅ Silver schema validation passed")
        except ValueError as e:
            warning = f"Silver schema validation failed: {e}"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")

        self.results['checks']['signals'] = checks


def main():
    parser = argparse.ArgumentParser(description='Validate Stage 15: FilterRTH')
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
        args.log_file = str(log_dir / f'validate_stage_15_{args.date}.log')

    logger = setup_logging(args.log_file)
    logger.info(f"Starting Stage 15 validation for {args.date}")
    logger.info(f"Log file: {args.log_file}")

    try:
        # Run pipeline through stage 15
        logger.info("Running through FilterRTH stage...")
        pipeline = build_es_pipeline()

        pipeline.run(
            date=args.date,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_stage=15,
            stop_at_stage=15
        )

        # Load checkpoint from stage (should already exist from pipeline run)
        from src.pipeline.core.checkpoint import CheckpointManager
        manager = CheckpointManager(args.checkpoint_dir)
        ctx = manager.load_checkpoint("bronze_to_silver", args.date, stage_idx=15)

        if ctx is None:
            logger.error("Failed to load checkpoint")
            return 1

        # Validate
        validator = Stage15Validator(logger)
        results = validator.validate(args.date, ctx)

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path(__file__).parent.parent / 'logs'
            output_path = output_dir / f'validate_stage_15_{args.date}_results.json'

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
