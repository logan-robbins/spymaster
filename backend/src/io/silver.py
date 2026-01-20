"""
Silver Feature Builder - Versioned Feature Engineering.

Transforms Bronze data into versioned Silver feature sets for ML experimentation.
Implements proper separation between data cleaning (Bronze->Silver) and
feature engineering (multiple Silver versions).
"""

import os
import hashlib
import logging
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.config import CONFIG
from src.common.schemas.feature_manifest import (
    FeatureManifest,
    ExperimentRegistry,
    ExperimentRecord,
    ValidationMetrics,
)
from src.common.schemas.silver_features import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    LABEL_COLUMNS,
    SilverFeaturesESPipeline,
)
from src.io.bronze import BronzeReader

logger = logging.getLogger(__name__)


class SilverFeatureBuilder:
    """
    Builds versioned feature sets from Bronze data.
    
    Each Silver version is:
    - Defined by a manifest (features, parameters)
    - Reproducible from Bronze + manifest
    - Immutable once created
    - Independently trainable
    
    Usage:
        builder = SilverFeatureBuilder()
        
        # Create new feature version
        manifest = FeatureManifest.from_file('manifests/mechanics_only.yaml')
        builder.build_feature_set(
            manifest=manifest,
            dates=['2025-12-16', '2025-12-17']
        )
        
        # List available versions
        versions = builder.list_versions()
    """
    
    def __init__(self, data_root: Optional[str] = None):
        """
        Initialize Silver feature builder.
        
        Args:
            data_root: Root lake directory (default: CONFIG.DATA_ROOT)
        """
        self.data_root = Path(data_root or CONFIG.DATA_ROOT)
        self.bronze_reader = BronzeReader(data_root=str(self.data_root))
        
        self.silver_root = self.data_root / "silver"
        self.features_root = self.silver_root / "features"
        self.datasets_root = self.silver_root / "datasets"
        
        # Create directories
        self.features_root.mkdir(parents=True, exist_ok=True)
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        
        # Load experiment registry
        self.registry_path = self.features_root / "experiments.json"
        self.registry = ExperimentRegistry.from_file(self.registry_path)
    
    def build_feature_set(
        self,
        manifest: FeatureManifest,
        dates: List[str],
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Build a feature set for the given manifest and dates.

        Args:
            manifest: Feature manifest defining the transformation
            dates: List of dates to process (YYYY-MM-DD)
            force: If True, overwrite existing output

        Returns:
            Dictionary with build statistics
        """
        version = manifest.version
        output_dir = self.features_root / version

        logger.info(f"Building feature set: {version}")
        logger.info(f"  Dates to process: {len(dates)}")
        logger.info(f"  Output directory: {output_dir}")

        # Check if already exists
        if output_dir.exists() and not force:
            logger.warning(f"  Version {version} already exists. Use force=True to overwrite.")
            return {
                'status': 'skipped',
                'reason': f'Version {version} already exists. Use force=True to overwrite.',
                'output_dir': str(output_dir)
            }

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest.to_file(output_dir / "manifest.yaml")
        logger.debug(f"  Saved manifest to {output_dir / 'manifest.yaml'}")

        # Get versioned pipeline for this manifest
        from src.pipeline.pipelines import get_pipeline_for_version

        # Create pipeline matching manifest version
        pipeline = get_pipeline_for_version(version)
        logger.info(f"  Using pipeline: {pipeline.name} ({pipeline.version})")

        # Process each date
        all_signals = []
        stats = {
            'dates_processed': 0,
            'signals_total': 0,
            'errors': []
        }

        build_start = time.time()

        for i, date in enumerate(dates, 1):
            date_start = time.time()
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"[{i}/{len(dates)}] Processing {date} for {version}")
                logger.info(f"{'='*60}")

                # Run pipeline to get features
                signals_df = pipeline.run(date=date)

                if signals_df.empty:
                    logger.warning(f"  No signals for {date}")
                    continue

                schema_cols = set(SilverFeaturesESPipeline._arrow_schema.names)
                identity_cols = list(IDENTITY_COLUMNS)
                label_cols = list(LABEL_COLUMNS)

                feature_cols = [
                    col for col in self._get_feature_columns(manifest)
                    if col in schema_cols
                ] or list(FEATURE_COLUMNS)

                keep_cols = identity_cols + feature_cols + label_cols
                available_cols = [c for c in keep_cols if c in signals_df.columns]

                filtered_df = signals_df[available_cols].copy()

                # Write to parquet (partitioned by date)
                date_output = output_dir / f"date={date}"
                date_output.mkdir(parents=True, exist_ok=True)

                output_file = date_output / f"features_{date}.parquet"
                filtered_df.to_parquet(
                    output_file,
                    engine='pyarrow',
                    compression='zstd',
                    index=False
                )

                all_signals.append(filtered_df)
                stats['dates_processed'] += 1
                stats['signals_total'] += len(filtered_df)

                date_elapsed = time.time() - date_start
                logger.info(f"  ✅ {date}: {len(filtered_df):,} signals in {date_elapsed:.1f}s")

            except Exception as e:
                error_msg = f"Error processing {date}: {str(e)}"
                logger.error(f"  ❌ {error_msg}")
                stats['errors'].append(error_msg)
        
        # Build summary
        build_elapsed = time.time() - build_start

        logger.info(f"\n{'='*60}")
        logger.info(f"BUILD SUMMARY: {version}")
        logger.info(f"{'='*60}")
        logger.info(f"  Dates processed: {stats['dates_processed']}/{len(dates)}")
        logger.info(f"  Total signals: {stats['signals_total']:,}")
        logger.info(f"  Total time: {build_elapsed:.1f}s")
        if stats['dates_processed'] > 0:
            logger.info(f"  Avg per date: {build_elapsed/stats['dates_processed']:.1f}s")

        if stats['errors']:
            logger.warning(f"  Errors: {len(stats['errors'])}")
            for err in stats['errors']:
                logger.warning(f"    - {err}")

        # Combine all signals for validation
        if all_signals:
            logger.info(f"\nValidating combined dataset...")
            combined_df = pd.concat(all_signals, ignore_index=True)

            # Compute validation metrics
            null_rates = {
                col: float(combined_df[col].isna().mean())
                for col in combined_df.columns
                if col not in identity_cols
            }

            validation = ValidationMetrics(
                date_range={'start': dates[0], 'end': dates[-1]},
                signal_count=len(combined_df),
                feature_count=len(feature_cols),
                null_rates=null_rates,
                schema_hash=self._compute_schema_hash(combined_df)
            )

            # Update manifest with validation
            manifest.validation = validation
            manifest.to_file(output_dir / "manifest.yaml")

            # Save validation summary
            high_null_features = [col for col, rate in null_rates.items() if rate > 0.1]
            validation_summary = {
                'date_range': validation.date_range,
                'signal_count': validation.signal_count,
                'feature_count': validation.feature_count,
                'null_rates_summary': {
                    'mean': sum(null_rates.values()) / len(null_rates) if null_rates else 0,
                    'max': max(null_rates.values()) if null_rates else 0,
                    'high_null_features': high_null_features
                }
            }

            import json
            with open(output_dir / "validation.json", 'w') as f:
                json.dump(validation_summary, f, indent=2)

            logger.info(f"  Total signals: {validation.signal_count:,}")
            logger.info(f"  Features: {validation.feature_count}")
            logger.info(f"  Date range: {validation.date_range['start']} to {validation.date_range['end']}")
            if high_null_features:
                logger.warning(f"  High null rate features: {high_null_features}")
            logger.info(f"  Validation saved to: {output_dir / 'validation.json'}")

        stats['status'] = 'success'
        stats['output_dir'] = str(output_dir)
        stats['version'] = version
        stats['build_time_seconds'] = build_elapsed

        logger.info(f"\n✅ Feature set {version} built successfully!")

        return stats
    
    def _get_feature_columns(self, manifest: FeatureManifest) -> List[str]:
        """Extract all feature column names from manifest."""
        columns = []
        for group in manifest.feature_groups:
            columns.extend(group.columns)
        return columns
    
    def _compute_schema_hash(self, df: pd.DataFrame) -> str:
        """Compute MD5 hash of schema for validation."""
        schema_str = str(df.dtypes.to_dict())
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def list_versions(self) -> List[str]:
        """List all Silver feature versions."""
        if not self.features_root.exists():
            return []
        
        versions = []
        for path in self.features_root.iterdir():
            if path.is_dir() and (path / "manifest.yaml").exists():
                versions.append(path.name)
        
        return sorted(versions)
    
    def get_manifest(self, version: str) -> Optional[FeatureManifest]:
        """Load manifest for a specific version."""
        manifest_path = self.features_root / version / "manifest.yaml"
        if not manifest_path.exists():
            return None
        return FeatureManifest.from_file(manifest_path)
    
    def load_features(
        self,
        version: str,
        dates: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load features for a specific version.
        
        Args:
            version: Feature set version
            dates: Optional list of dates to load (default: all)
        
        Returns:
            DataFrame with features
        """
        version_dir = self.features_root / version
        if not version_dir.exists():
            raise ValueError(f"Version {version} not found")
        
        # Find all parquet files
        if dates:
            # Load specific dates
            dfs = []
            for date in dates:
                date_dir = version_dir / f"date={date}"
                if date_dir.exists():
                    parquet_files = list(date_dir.glob("*.parquet"))
                    for f in parquet_files:
                        dfs.append(pd.read_parquet(f))
            
            if not dfs:
                return pd.DataFrame()
            return pd.concat(dfs, ignore_index=True)
        else:
            # Load all dates
            pattern = str(version_dir / "date=*" / "*.parquet")
            return pd.read_parquet(pattern)
    
    def register_experiment(
        self,
        version: str,
        exp_id: str,
        status: str = 'completed',
        metrics: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        notes: str = ""
    ):
        """
        Register an experiment in the registry.
        
        Args:
            version: Feature set version used
            exp_id: Experiment ID (e.g., 'exp001')
            status: Experiment status
            metrics: Optional metrics dictionary
            model_path: Optional path to trained model
            notes: Optional notes
        """
        manifest = self.get_manifest(version)
        if not manifest:
            raise ValueError(f"Version {version} not found")
        
        experiment = ExperimentRecord(
            id=exp_id,
            version=version,
            created_at=datetime.utcnow().isoformat() + "Z",
            status=status,
            metrics=metrics or {},
            parent=manifest.parent_version,
            tags=manifest.tags,
            notes=notes,
            model_path=model_path
        )
        
        self.registry.add(experiment)
        self.registry.to_file(self.registry_path)
    
    def compare_versions(
        self,
        version_a: str,
        version_b: str
    ) -> Dict[str, Any]:
        """
        Compare two feature versions.
        
        Returns:
            Dictionary with comparison results
        """
        manifest_a = self.get_manifest(version_a)
        manifest_b = self.get_manifest(version_b)
        
        if not manifest_a or not manifest_b:
            raise ValueError("One or both versions not found")
        
        # Get feature columns
        cols_a = set(self._get_feature_columns(manifest_a))
        cols_b = set(self._get_feature_columns(manifest_b))
        
        return {
            'version_a': version_a,
            'version_b': version_b,
            'features_added': list(cols_b - cols_a),
            'features_removed': list(cols_a - cols_b),
            'features_common': list(cols_a & cols_b),
            'feature_count_a': len(cols_a),
            'feature_count_b': len(cols_b),
            'parameters_a': manifest_a.parameters.to_dict(),
            'parameters_b': manifest_b.parameters.to_dict(),
        }


def create_baseline_features(data_root: Optional[str] = None):
    """
    Create baseline feature sets (mechanics_only and full_ensemble).

    This is a helper function to bootstrap the Silver layer.
    """
    from src.common.schemas.feature_manifest import (
        create_mechanics_only_manifest,
        create_full_ensemble_manifest
    )

    logger.info("="*70)
    logger.info("CREATING BASELINE SILVER FEATURES")
    logger.info("="*70)

    builder = SilverFeatureBuilder(data_root=data_root)

    # Get available dates from Bronze
    bronze_reader = BronzeReader(data_root=data_root or CONFIG.DATA_ROOT)
    available_dates = bronze_reader.get_available_dates('futures/trades', 'symbol=ES')

    if not available_dates:
        logger.error("No Bronze data found. Please run backfill first.")
        return

    logger.info(f"Found {len(available_dates)} dates in Bronze")
    logger.info(f"  First: {available_dates[0]}")
    logger.info(f"  Last: {available_dates[-1]}")

    # Create mechanics-only baseline
    logger.info("\n" + "="*70)
    logger.info("BUILDING: mechanics_only")
    logger.info("="*70)
    manifest_v1 = create_mechanics_only_manifest()
    stats_v1 = builder.build_feature_set(
        manifest=manifest_v1,
        dates=available_dates,
        force=False
    )

    # Create full ensemble
    logger.info("\n" + "="*70)
    logger.info("BUILDING: full_ensemble")
    logger.info("="*70)
    manifest_v2 = create_full_ensemble_manifest()
    stats_v2 = builder.build_feature_set(
        manifest=manifest_v2,
        dates=available_dates,
        force=False
    )

    logger.info("\n" + "="*70)
    logger.info("BASELINE FEATURE SETS COMPLETE")
    logger.info("="*70)
    logger.info(f"  Available versions: {builder.list_versions()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Silver feature sets")
    parser.add_argument('--action', choices=['create_baseline', 'list', 'compare'],
                        default='create_baseline', help="Action to perform")
    parser.add_argument('--version-a', help="First version for comparison")
    parser.add_argument('--version-b', help="Second version for comparison")
    
    args = parser.parse_args()
    
    builder = SilverFeatureBuilder()
    
    if args.action == 'create_baseline':
        create_baseline_features()
    elif args.action == 'list':
        versions = builder.list_versions()
        print(f"Available Silver feature versions ({len(versions)}):")
        for v in versions:
            manifest = builder.get_manifest(v)
            if manifest:
                print(f"  - {v}")
                print(f"      Description: {manifest.description}")
                print(f"      Features: {len(builder._get_feature_columns(manifest))}")
                if manifest.validation:
                    print(f"      Signals: {manifest.validation.signal_count}")
    elif args.action == 'compare':
        if not args.version_a or not args.version_b:
            print("Error: --version-a and --version-b required for comparison")
        else:
            comparison = builder.compare_versions(args.version_a, args.version_b)
            import json
            print(json.dumps(comparison, indent=2))
