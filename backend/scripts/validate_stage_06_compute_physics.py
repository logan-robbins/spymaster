"""
Validate Stage 6: ComputePhysics

Goals:
1. Compute barrier/tape/fuel physics for each interaction event
2. Preserve event identity columns from touches_df
3. Output signals_df with physics columns populated

Validation Checks:
- Required inputs present (touches_df, market_state, trades, mbp10_snapshots)
- signals_df output exists and has required columns
- Row count matches touches_df
- Barrier state values valid
- Fuel effect values valid
- Numeric physics columns finite and non-null
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.pipeline.pipelines.es_pipeline import build_es_pipeline
from src.core.barrier_engine import BarrierState
from src.core.fuel_engine import FuelEffect


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


class Stage6Validator:
    """Validator for ComputePhysics stage."""

    def __init__(self, logger):
        self.logger = logger
        self.results = {
            'stage': 'compute_physics',
            'stage_idx': 6,
            'checks': {},
            'warnings': [],
            'errors': [],
            'passed': True
        }

    def validate(self, date: str, ctx) -> Dict[str, Any]:
        """Run all validation checks."""
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Validating Stage 6: ComputePhysics for {date}")
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
            self.logger.info("✅ Stage 6 Validation: PASSED")
        else:
            self.logger.error("❌ Stage 6 Validation: FAILED")
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

        required_inputs = ['touches_df', 'market_state', 'trades', 'mbp10_snapshots']
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
            warning = "signals_df is empty (no physics computed)"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")
            self.results['checks']['signals_df'] = checks
            return

        self.logger.info(f"  Total signals: {len(signals_df):,}")

        base_cols = [
            'event_id', 'ts_ns', 'timestamp', 'level_price', 'level_kind',
            'level_kind_name', 'direction', 'entry_price', 'zone_width', 'date'
        ]
        physics_cols = [
            'barrier_state', 'barrier_delta_liq', 'barrier_replenishment_ratio',
            'wall_ratio', 'tape_imbalance', 'tape_buy_vol', 'tape_sell_vol',
            'tape_velocity', 'sweep_detected', 'fuel_effect', 'gamma_exposure'
        ]
        required_cols = base_cols + physics_cols

        missing_cols = [col for col in required_cols if col not in signals_df.columns]
        if missing_cols:
            checks['required_columns_present'] = False
            self.results['passed'] = False
            error = f"signals_df missing required columns: {missing_cols}"
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

        # Barrier state validation
        barrier_states = set(signals_df['barrier_state'].astype(str))
        valid_barrier = {state.value for state in BarrierState}
        invalid_barrier = sorted(barrier_states - valid_barrier)
        if invalid_barrier:
            checks['barrier_state_values'] = False
            self.results['passed'] = False
            error = f"Invalid barrier_state values: {invalid_barrier}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['barrier_state_values'] = True
            self.logger.info(f"  ✅ Barrier states valid: {signals_df['barrier_state'].value_counts().to_dict()}")

        # Fuel effect validation
        fuel_effects = set(signals_df['fuel_effect'].astype(str))
        valid_fuel = {effect.value for effect in FuelEffect}
        invalid_fuel = sorted(fuel_effects - valid_fuel)
        if invalid_fuel:
            checks['fuel_effect_values'] = False
            self.results['passed'] = False
            error = f"Invalid fuel_effect values: {invalid_fuel}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['fuel_effect_values'] = True
            self.logger.info(f"  ✅ Fuel effects valid: {signals_df['fuel_effect'].value_counts().to_dict()}")

        # Numeric columns validation
        numeric_cols = [
            'barrier_delta_liq', 'barrier_replenishment_ratio', 'wall_ratio',
            'tape_imbalance', 'tape_buy_vol', 'tape_sell_vol',
            'tape_velocity', 'gamma_exposure'
        ]
        for col in numeric_cols:
            values = pd.to_numeric(signals_df[col], errors='coerce')
            if values.isna().any():
                checks[f'{col}_nan'] = False
                self.results['passed'] = False
                error = f"{col} has NaN values"
                self.results['errors'].append(error)
                self.logger.error(f"  ❌ {error}")
            else:
                checks[f'{col}_nan'] = True

        # Non-negative checks
        for col in ['tape_buy_vol', 'tape_sell_vol', 'wall_ratio']:
            values = pd.to_numeric(signals_df[col], errors='coerce')
            if (values < 0).any():
                checks[f'{col}_non_negative'] = False
                self.results['passed'] = False
                error = f"{col} has negative values"
                self.results['errors'].append(error)
                self.logger.error(f"  ❌ {error}")
            else:
                checks[f'{col}_non_negative'] = True

        # sweep_detected boolean check
        sweep_values = set(signals_df['sweep_detected'].dropna().unique().tolist())
        if not sweep_values.issubset({True, False}):
            checks['sweep_detected_bool'] = False
            self.results['passed'] = False
            error = f"sweep_detected has non-boolean values: {sorted(sweep_values)}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['sweep_detected_bool'] = True
            self.logger.info("  ✅ sweep_detected values are boolean")

        self.results['checks']['signals_df'] = checks


def main():
    parser = argparse.ArgumentParser(description='Validate Stage 6: ComputePhysics')
    parser.add_argument('--date', type=str, required=True, help='Date to validate (YYYY-MM-DD)')
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-file', type=str, default=None, help='Log file path')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')

    args = parser.parse_args()

    # Setup logging
    if args.log_file is None:
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        args.log_file = str(log_dir / f'validate_stage_06_{args.date}.log')

    logger = setup_logging(args.log_file)
    logger.info(f"Starting Stage 6 validation for {args.date}")
    logger.info(f"Log file: {args.log_file}")

    try:
        # Run pipeline through stage 6
        logger.info("Running through ComputePhysics stage...")
        pipeline = build_es_pipeline()

        pipeline.run(
            date=args.date,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_stage=6,
            stop_at_stage=6
        )

        # Load checkpoint
        from src.pipeline.core.checkpoint import CheckpointManager
        manager = CheckpointManager(args.checkpoint_dir)
        ctx = manager.load_checkpoint("es_pipeline", args.date, stage_idx=6)

        if ctx is None:
            logger.error("Failed to load checkpoint")
            return 1

        # Validate
        validator = Stage6Validator(logger)
        results = validator.validate(args.date, ctx)

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path(__file__).parent.parent / 'logs'
            output_path = output_dir / f'validate_stage_06_{args.date}_results.json'

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
