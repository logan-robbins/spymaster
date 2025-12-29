"""
Validate Stage 18: ConstructEpisodes

Goals:
1. Construct 111-dimensional episode vectors from events + state table
2. Apply normalization using precomputed statistics
3. Generate episode metadata with labels and emission weights
4. Output: vectors.npy + metadata.parquet (date-partitioned)

Vector Architecture (111 dims):
- Section A: Context State (26 dims)
- Section B: Multi-Scale Trajectory (37 dims)
- Section C: Micro-History (35 dims, 7 features × 5 bars)
- Section D: Derived Physics (9 dims)
- Section E: Cluster Trends (4 dims)

Validation Checks:
- Required inputs present (signals_df, state_df, date)
- episodes_vectors output exists with shape [N × 111]
- episodes_metadata output exists with required columns
- Row counts match (vectors and metadata)
- Metadata has outcome labels (outcome_2min, outcome_4min, outcome_8min)
- Metadata has emission_weight and time_bucket fields
- No NaN/inf in vectors (after normalization)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

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


class Stage18Validator:
    """Validator for ConstructEpisodes stage."""

    def __init__(self, logger):
        self.logger = logger
        self.results = {
            'stage': 'construct_episodes',
            'stage_idx': 18,
            'checks': {},
            'warnings': [],
            'errors': [],
            'passed': True
        }

    def validate(self, date: str, ctx) -> Dict[str, Any]:
        """Run all validation checks."""
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Validating Stage 18: ConstructEpisodes for {date}")
        self.logger.info(f"{'='*80}")

        self.results['date'] = date

        # Check 1: Required inputs + outputs present
        self._check_required_outputs(ctx)

        # Check 2: Validate episodes_vectors
        if 'episodes_vectors' in ctx.data:
            self._validate_vectors(ctx.data['episodes_vectors'])

        # Check 3: Validate episodes_metadata
        if 'episodes_metadata' in ctx.data:
            self._validate_metadata(ctx.data['episodes_metadata'])

        # Check 4: Cross-validate vectors and metadata
        if 'episodes_vectors' in ctx.data and 'episodes_metadata' in ctx.data:
            self._cross_validate(ctx.data['episodes_vectors'], ctx.data['episodes_metadata'])

        # Summary
        self.logger.info(f"\n{'='*80}")
        if self.results['passed']:
            self.logger.info("✅ Stage 18 Validation: PASSED")
        else:
            self.logger.error("❌ Stage 18 Validation: FAILED")
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

        required_inputs = ['signals_df', 'state_df', 'date']
        new_outputs = ['episodes_vectors', 'episodes_metadata']

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
            self.logger.info("  ✅ New outputs present: episodes_vectors, episodes_metadata")

    def _validate_vectors(self, vectors):
        """Validate episode vectors array."""
        self.logger.info("\n2. Validating episodes_vectors...")

        checks = {}

        if not isinstance(vectors, np.ndarray):
            checks['vectors_type'] = False
            self.results['passed'] = False
            error = f"episodes_vectors is not ndarray: {type(vectors)}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            return

        checks['vectors_type'] = True

        if vectors.size == 0:
            checks['vectors_not_empty'] = False
            self.results['passed'] = False
            error = "episodes_vectors is empty"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            self.results['checks']['episodes_vectors'] = checks
            return

        checks['vectors_not_empty'] = True

        # Check shape
        if len(vectors.shape) != 2:
            checks['vectors_shape'] = False
            self.results['passed'] = False
            error = f"episodes_vectors has wrong shape: {vectors.shape} (expected 2D)"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            self.results['checks']['episodes_vectors'] = checks
            return

        n_episodes, n_dims = vectors.shape

        if n_dims != 111:
            checks['vector_dimension'] = False
            self.results['passed'] = False
            error = f"episodes_vectors has {n_dims} dims (expected 111)"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['vector_dimension'] = True
            self.logger.info(f"  ✅ Vector dimension: 111")

        self.logger.info(f"  Total episodes: {n_episodes:,}")
        self.logger.info(f"  Vector shape: {vectors.shape}")

        # Check for NaN/inf
        nan_count = np.isnan(vectors).sum()
        inf_count = np.isinf(vectors).sum()

        if nan_count > 0:
            checks['no_nan'] = False
            self.results['passed'] = False
            error = f"episodes_vectors contains {nan_count} NaN values"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['no_nan'] = True
            self.logger.info(f"  ✅ No NaN values")

        if inf_count > 0:
            checks['no_inf'] = False
            self.results['passed'] = False
            error = f"episodes_vectors contains {inf_count} inf values"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            checks['no_inf'] = True
            self.logger.info(f"  ✅ No inf values")

        # Check value ranges (after normalization should be mostly [-4, 4] or [0, 1])
        vec_min = vectors.min()
        vec_max = vectors.max()
        vec_mean = vectors.mean()
        vec_std = vectors.std()

        self.logger.info(f"  Vector stats: min={vec_min:.2f}, max={vec_max:.2f}, mean={vec_mean:.2f}, std={vec_std:.2f}")

        if vec_min < -10 or vec_max > 10:
            warning = f"Vector values outside expected range: [{vec_min:.2f}, {vec_max:.2f}]"
            self.results['warnings'].append(warning)
            self.logger.warning(f"  ⚠️  {warning}")

        self.results['checks']['episodes_vectors'] = checks

    def _validate_metadata(self, metadata):
        """Validate episode metadata DataFrame."""
        self.logger.info("\n3. Validating episodes_metadata...")

        checks = {}

        if not isinstance(metadata, pd.DataFrame):
            checks['metadata_type'] = False
            self.results['passed'] = False
            error = f"episodes_metadata is not DataFrame: {type(metadata)}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            return

        checks['metadata_type'] = True

        if metadata.empty:
            checks['metadata_not_empty'] = False
            self.results['passed'] = False
            error = "episodes_metadata is empty"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            self.results['checks']['episodes_metadata'] = checks
            return

        checks['metadata_not_empty'] = True
        self.logger.info(f"  Total metadata rows: {len(metadata):,}")

        # Check required columns
        required_cols = [
            'event_id', 'date', 'timestamp', 'ts_ns',
            'level_kind', 'level_price', 'direction',
            'spot', 'atr', 'minutes_since_open', 'time_bucket',
            'outcome_2min', 'outcome_4min', 'outcome_8min',
            'excursion_favorable', 'excursion_adverse',
            'emission_weight',
            'prior_touches', 'attempt_index'
        ]

        missing_cols = [col for col in required_cols if col not in metadata.columns]
        if missing_cols:
            checks['required_columns_present'] = False
            self.results['passed'] = False
            error = f"episodes_metadata missing columns: {missing_cols}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
            self.results['checks']['episodes_metadata'] = checks
            return

        checks['required_columns_present'] = True
        self.logger.info(f"  ✅ Required columns present")

        # Check outcome labels
        for horizon in ['2min', '4min', '8min']:
            col = f'outcome_{horizon}'
            if col in metadata.columns:
                outcome_dist = metadata[col].value_counts().to_dict()
                self.logger.info(f"  {horizon} outcomes: {outcome_dist}")

                valid_outcomes = {'BREAK', 'REJECT', 'CHOP'}
                invalid = set(metadata[col].unique()) - valid_outcomes
                if invalid:
                    warning = f"Invalid outcome values in {col}: {invalid}"
                    self.results['warnings'].append(warning)
                    self.logger.warning(f"  ⚠️  {warning}")

        # Check time buckets
        if 'time_bucket' in metadata.columns:
            bucket_dist = metadata['time_bucket'].value_counts().to_dict()
            self.logger.info(f"  Time bucket distribution: {bucket_dist}")

            valid_buckets = {'T0_30', 'T30_60', 'T60_120', 'T120_180'}
            invalid = set(metadata['time_bucket'].unique()) - valid_buckets
            if invalid:
                warning = f"Invalid time_bucket values: {invalid}"
                self.results['warnings'].append(warning)
                self.logger.warning(f"  ⚠️  {warning}")

        # Check emission weights
        if 'emission_weight' in metadata.columns:
            weight_mean = metadata['emission_weight'].mean()
            weight_min = metadata['emission_weight'].min()
            weight_max = metadata['emission_weight'].max()

            self.logger.info(f"  Emission weight: mean={weight_mean:.3f}, range=[{weight_min:.3f}, {weight_max:.3f}]")

            if weight_min < 0 or weight_max > 1:
                warning = f"emission_weight outside [0, 1]: [{weight_min:.3f}, {weight_max:.3f}]"
                self.results['warnings'].append(warning)
                self.logger.warning(f"  ⚠️  {warning}")

        self.results['checks']['episodes_metadata'] = checks

    def _cross_validate(self, vectors, metadata):
        """Cross-validate vectors and metadata."""
        self.logger.info("\n4. Cross-validating vectors and metadata...")

        # Row count match
        n_vectors = len(vectors) if isinstance(vectors, np.ndarray) else 0
        n_metadata = len(metadata) if isinstance(metadata, pd.DataFrame) else 0

        if n_vectors != n_metadata:
            self.results['checks']['row_count_match'] = False
            self.results['passed'] = False
            error = f"Row count mismatch: vectors={n_vectors}, metadata={n_metadata}"
            self.results['errors'].append(error)
            self.logger.error(f"  ❌ {error}")
        else:
            self.results['checks']['row_count_match'] = True
            self.logger.info(f"  ✅ Row counts match: {n_vectors}")


def main():
    parser = argparse.ArgumentParser(description='Validate Stage 18: ConstructEpisodes')
    parser.add_argument('--date', type=str, required=True, help='Date to validate (YYYY-MM-DD)')
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-file', type=str, default=None, help='Log file path')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')

    args = parser.parse_args()

    # Setup logging
    if args.log_file is None:
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        args.log_file = str(log_dir / f'validate_stage_18_{args.date}.log')

    logger = setup_logging(args.log_file)
    logger.info(f"Starting Stage 18 validation for {args.date}")
    logger.info(f"Log file: {args.log_file}")

    try:
        # Run pipeline through stage 18
        logger.info("Running through ConstructEpisodes stage...")
        pipeline = build_es_pipeline()

        pipeline.run(
            date=args.date,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_stage=18,
            stop_at_stage=18
        )

        # Load checkpoint
        from src.pipeline.core.checkpoint import CheckpointManager
        manager = CheckpointManager(args.checkpoint_dir)
        ctx = manager.load_checkpoint("es_pipeline", args.date, stage_idx=18)

        if ctx is None:
            logger.error("Failed to load checkpoint")
            return 1

        # Validate
        validator = Stage18Validator(logger)
        results = validator.validate(args.date, ctx)

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path(__file__).parent.parent / 'logs'
            output_path = output_dir / f'validate_stage_18_{args.date}_results.json'

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

