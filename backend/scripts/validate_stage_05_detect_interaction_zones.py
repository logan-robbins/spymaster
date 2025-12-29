"""
Validate Stage 5: DetectInteractionZones

Goals:
1. Detect zone entry events around structural levels
2. Assign direction based on approach side (UP/DOWN)
3. Generate deterministic event IDs for retrieval
4. Output touches_df with required columns and sane values

Validation Checks:
- Required inputs present (ohlcv_1min, level_info, atr)
- touches_df output exists and has required columns
- Event IDs deterministic and unique
- Direction values are valid
- Zone width respects CONFIG.MONITOR_BAND floor
- Level kind/name mapping is consistent with LevelInfo
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.common.config import CONFIG
from src.pipeline.pipelines.es_pipeline import build_es_pipeline
from src.pipeline.stages.detect_interaction_zones import compute_deterministic_event_id


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


class Stage5Validator:
    """Validator for DetectInteractionZones stage."""

    def __init__(self, logger):
        self.logger = logger
        self.results = {
            'stage': 'detect_interaction_zones',
            'stage_idx': 5,
            'checks': {},
            'warnings': [],
            'errors': [],
            'passed': True
        }

    def validate(self, date: str, ctx) -> Dict[str, Any]:
        """Run all validation checks."""
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Validating Stage 5: DetectInteractionZones for {date}")
        self.logger.info(f"{'='*80}")

        self.results['date'] = date

        # Check 1: Required inputs + outputs present
        self._check_required_outputs(ctx)

        # Check 2: Validate touches_df
        if 'touches_df' in ctx.data:
            self._validate_touches_df(ctx.data['touches_df'], ctx, date)

        # Summary
        self.logger.info(f"\n{'='*80}")
        if self.results['passed']:
            self.logger.info("✅ Stage 5 Validation: PASSED")
        else:
            self.logger.error("❌ Stage 5 Validation: FAILED")
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

        required_inputs = ['ohlcv_1min', 'level_info', 'atr']
        new_outputs = ['touches_df']

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
            self.logger.info("  ✅ New output present: touches_df")

    def _validate_touches_df(self, touches_df: pd.DataFrame, ctx, date: str):
        """Validate touches_df structure and values."""
        self.logger.info("\n2. Validating touches_df...")

        checks = {}

        if not isinstance(touches_df, pd.DataFrame):
            checks['touches_df_type'] = False
            self.results['passed'] = False
            error = f"touches_df is not DataFrame: {type(touches_df)}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            return

        checks['touches_df_type'] = True

        if touches_df.empty:
            warning = "touches_df is empty (no interaction events detected)"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
            self.results['checks']['touches_df'] = checks
            return

        self.logger.info(f"  Total events: {len(touches_df):,}")

        required_cols = [
            'event_id', 'ts_ns', 'timestamp', 'bar_idx', 'level_price',
            'level_kind', 'level_kind_name', 'direction', 'entry_price',
            'spot', 'zone_width', 'date'
        ]
        missing_cols = [col for col in required_cols if col not in touches_df.columns]
        if missing_cols:
            checks['required_columns_present'] = False
            self.results['passed'] = False
            error = f"touches_df missing required columns: {missing_cols}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            self.results['checks']['touches_df'] = checks
            return

        checks['required_columns_present'] = True

        # Basic distribution logging
        level_counts = touches_df['level_kind_name'].value_counts().to_dict()
        direction_counts = touches_df['direction'].value_counts().to_dict()
        self.logger.info(f"  Level distribution: {level_counts}")
        self.logger.info(f"  Direction distribution: {direction_counts}")

        # Validate direction values
        valid_directions = {'UP', 'DOWN'}
        invalid_directions = sorted(set(touches_df['direction']) - valid_directions)
        if invalid_directions:
            checks['direction_values'] = False
            self.results['passed'] = False
            error = f"Invalid direction values: {invalid_directions}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['direction_values'] = True
            self.logger.info("  ✅ Direction values valid (UP/DOWN)")

        # Validate zone width floor
        zone_width = pd.to_numeric(touches_df['zone_width'], errors='coerce')
        if zone_width.isna().any():
            checks['zone_width_nan'] = False
            self.results['passed'] = False
            error = "zone_width has NaN values"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['zone_width_nan'] = True

        min_zone = float(CONFIG.MONITOR_BAND)
        if (zone_width < (min_zone - 1e-9)).any():
            checks['zone_width_floor'] = False
            self.results['passed'] = False
            error = f"zone_width below CONFIG.MONITOR_BAND ({min_zone})"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['zone_width_floor'] = True
            self.logger.info(f"  ✅ Zone width respects MONITOR_BAND >= {min_zone}")

        # Validate event IDs
        self._validate_event_ids(touches_df, date, checks)

        # Validate level kind mapping
        self._validate_level_mapping(touches_df, ctx, checks)

        # Validate timestamps ordering
        ts_ns = pd.to_numeric(touches_df['ts_ns'], errors='coerce')
        if ts_ns.isna().any():
            checks['ts_ns_nan'] = False
            self.results['passed'] = False
            error = "ts_ns has NaN values"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['ts_ns_nan'] = True
            if not ts_ns.is_monotonic_increasing:
                checks['ts_ns_ordered'] = False
                warning = "ts_ns not monotonic increasing (events may be unsorted)"
                self.results['warnings'].append(warning)
                self.logger.warning(f"  ⚠️  {warning}")
            else:
                checks['ts_ns_ordered'] = True
                self.logger.info("  ✅ Events sorted by ts_ns")

        # Validate date column
        date_values = touches_df['date'].astype(str).unique().tolist()
        if len(date_values) != 1 or date_values[0] != date:
            checks['date_column'] = False
            warning = f"touches_df date column mismatch: {date_values}"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
        else:
            checks['date_column'] = True
            self.logger.info("  ✅ Date column matches requested date")

        self.results['checks']['touches_df'] = checks

    def _validate_event_ids(self, touches_df: pd.DataFrame, date: str, checks: Dict[str, Any]):
        """Validate deterministic event IDs and uniqueness."""
        event_ids = touches_df['event_id'].astype(str)

        if event_ids.isna().any():
            checks['event_id_nan'] = False
            self.results['passed'] = False
            error = "event_id has NaN values"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            return

        checks['event_id_nan'] = True

        # Uniqueness
        if event_ids.duplicated().any():
            checks['event_id_unique'] = False
            self.results['passed'] = False
            error = "Duplicate event_id values found"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['event_id_unique'] = True
            self.logger.info("  ✅ event_id values are unique")

        # Prefix check
        if not event_ids.str.startswith(f"{date}_").all():
            warning = "Some event_id values do not start with the date prefix"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")

        # Deterministic check on sample
        sample = touches_df.head(min(25, len(touches_df)))
        mismatches = 0
        for _, row in sample.iterrows():
            expected = compute_deterministic_event_id(
                date=date,
                level_kind=str(row['level_kind_name']),
                level_price=float(row['level_price']),
                anchor_ts_ns=int(row['ts_ns']),
                direction=str(row['direction'])
            )
            if str(row['event_id']) != expected:
                mismatches += 1

        if mismatches > 0:
            checks['event_id_deterministic'] = False
            self.results['passed'] = False
            error = f"{mismatches} event_id mismatches in deterministic check"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['event_id_deterministic'] = True
            self.logger.info("  ✅ event_id deterministic check passed")

    def _validate_level_mapping(self, touches_df: pd.DataFrame, ctx, checks: Dict[str, Any]):
        """Validate level kind/name mapping against LevelInfo."""
        level_info = ctx.data.get('level_info')
        if level_info is None:
            warning = "level_info missing; skipping level mapping checks"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
            return

        kind_names = list(level_info.kind_names)
        kind_codes = list(level_info.kinds)

        if len(set(kind_names)) != len(kind_names):
            warning = "Duplicate level kind names in level_info; skipping mapping check"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
            return

        mapping = dict(zip(kind_names, kind_codes))

        invalid_names = sorted(set(touches_df['level_kind_name']) - set(kind_names))
        if invalid_names:
            checks['level_kind_name_valid'] = False
            self.results['passed'] = False
            error = f"Invalid level_kind_name values: {invalid_names}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['level_kind_name_valid'] = True
            self.logger.info("  ✅ level_kind_name values match LevelInfo")

        mismatched = touches_df[
            touches_df['level_kind_name'].map(mapping) != touches_df['level_kind']
        ]
        if not mismatched.empty:
            checks['level_kind_code_match'] = False
            self.results['passed'] = False
            error = f"{len(mismatched)} rows have level_kind code mismatch"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['level_kind_code_match'] = True
            self.logger.info("  ✅ level_kind codes match LevelInfo")


def main():
    parser = argparse.ArgumentParser(description='Validate Stage 5: DetectInteractionZones')
    parser.add_argument('--date', type=str, required=True, help='Date to validate (YYYY-MM-DD)')
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-file', type=str, default=None, help='Log file path')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')

    args = parser.parse_args()

    # Setup logging
    if args.log_file is None:
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        args.log_file = str(log_dir / f'validate_stage_05_{args.date}.log')

    logger = setup_logging(args.log_file)
    logger.info(f"Starting Stage 5 validation for {args.date}")
    logger.info(f"Log file: {args.log_file}")

    try:
        # Run pipeline through stage 5
        logger.info("Running through DetectInteractionZones stage...")
        pipeline = build_es_pipeline()

        pipeline.run(
            date=args.date,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_stage=5,
            stop_at_stage=5
        )

        # Load checkpoint
        from src.pipeline.core.checkpoint import CheckpointManager
        manager = CheckpointManager(args.checkpoint_dir)
        ctx = manager.load_checkpoint("es_pipeline", args.date, stage_idx=5)

        if ctx is None:
            logger.error("Failed to load checkpoint")
            return 1

        # Validate
        validator = Stage5Validator(logger)
        results = validator.validate(args.date, ctx)

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path(__file__).parent.parent / 'logs'
            output_path = output_dir / f'validate_stage_05_{args.date}_results.json'

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
