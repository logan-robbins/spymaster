#!/usr/bin/env python3
"""
Bootstrap Script for Medallion Architecture.

Creates baseline Silver feature sets and promotes best to Gold.
Run this after backfilling Bronze data.

Usage:
    cd backend
    uv run python scripts/bootstrap_medallion.py
    uv run python scripts/bootstrap_medallion.py --verbose
    uv run python scripts/bootstrap_medallion.py --debug
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from src.io.silver import SilverFeatureBuilder, create_baseline_features
from src.io.gold import GoldCurator
from src.common.schemas.feature_manifest import (
    create_mechanics_only_manifest,
    create_full_ensemble_manifest
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, debug: bool = False):
    """Configure logging for the bootstrap script."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%H:%M:%S',
        force=True
    )

    # Also set level for key modules
    for module in ['src.pipeline', 'src.lake', 'src.core', '__main__']:
        logging.getLogger(module).setLevel(level)


def main():
    """Bootstrap the Medallion architecture."""
    parser = argparse.ArgumentParser(
        description="Bootstrap Medallion Architecture - Create Silver features and promote to Gold"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (INFO level)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging (DEBUG level)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild of existing feature sets'
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose, debug=args.debug)

    start_time = time.time()

    print("=" * 70)
    print("MEDALLION ARCHITECTURE BOOTSTRAP")
    print("=" * 70)
    if args.verbose:
        print("  Mode: VERBOSE (--verbose)")
    if args.debug:
        print("  Mode: DEBUG (--debug)")
    print()

    # Step 1: Create baseline feature sets
    print("Step 1: Creating baseline Silver feature sets...")
    print("-" * 70)
    create_baseline_features()
    print()
    
    # Step 2: List created versions
    print("Step 2: Verifying Silver feature versions...")
    print("-" * 70)
    builder = SilverFeatureBuilder()
    versions = builder.list_versions()
    
    if not versions:
        print("❌ No Silver versions created. Check Bronze data availability.")
        return 1
    
    print(f"✅ Created {len(versions)} feature versions:")
    for v in versions:
        manifest = builder.get_manifest(v)
        if manifest:
            print(f"  - {v}")
            print(f"      Features: {len(builder._get_feature_columns(manifest))}")
            if manifest.validation:
                print(f"      Signals: {manifest.validation.signal_count}")
                print(f"      Dates: {manifest.validation.date_range['start']} to {manifest.validation.date_range['end']}")
    print()
    
    # Step 3: Promote full ensemble to Gold
    print("Step 3: Promoting best experiment to Gold...")
    print("-" * 70)
    curator = GoldCurator()
    
    # Promote v2.0_full_ensemble as the production dataset
    best_version = 'v2.0_full_ensemble'
    if best_version in versions:
        result = curator.promote_to_training(
            silver_version=best_version,
            dataset_name='signals_production',
            notes='Initial production dataset from full ensemble features'
        )
        
        if result['status'] == 'success':
            print(f"✅ Promoted {best_version} to Gold training")
            print(f"  Dataset: {result['dataset_name']}")
            print(f"  Signals: {result['signal_count']}")
            print(f"  Features: {result['feature_count']}")
        else:
            print(f"⚠️  Promotion status: {result['status']}")
            print(f"  Reason: {result.get('reason', 'Unknown')}")
    else:
        print(f"⚠️  Version {best_version} not found, skipping promotion")
    print()
    
    # Step 4: Validate Gold dataset
    print("Step 4: Validating Gold production dataset...")
    print("-" * 70)
    try:
        validation = curator.validate_dataset('signals_production')
        print(f"✅ Gold dataset validated")
        print(f"  Rows: {validation['row_count']:,}")
        print(f"  Columns: {validation['column_count']}")
        print(f"  Dates: {validation['date_coverage'].get('unique_dates', 0)}")
        
        if 'label_distribution' in validation and validation['label_distribution']:
            print(f"  Label distribution:")
            for label, count in validation['label_distribution'].items():
                print(f"    {label}: {count}")
    except FileNotFoundError:
        print("⚠️  Gold dataset not found (promotion may have failed)")
    print()
    
    # Summary
    elapsed = time.time() - start_time
    print("=" * 70)
    print("BOOTSTRAP COMPLETE")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print()
    print("Next steps:")
    print("  1. Train ML models using Gold data:")
    print("     uv run python -m src.ml.boosted_tree_train")
    print()
    print("  2. Create new feature experiments:")
    print("     # Edit manifest")
    print("     uv run python -m src.lake.silver_feature_builder --action create_baseline")
    print()
    print("  3. Compare feature versions:")
    print("     uv run python -m src.lake.silver_feature_builder --action compare \\")
    print("       --version-a v1.0_mechanics_only --version-b v2.0_full_ensemble")
    print()
    print("Data locations:")
    print(f"  Bronze: backend/data/bronze/")
    print(f"  Silver: backend/data/silver/features/")
    print(f"  Gold:   backend/data/gold/training/")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

