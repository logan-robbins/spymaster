"""Validate Pentaview stream output - quick health check."""
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_stream_bars(stream_bars_path: Path) -> dict:
    """
    Validate stream bars output.
    
    Checks per STREAMS.md:
    - All stream values in [-1, +1]
    - Barrier stream sign consistency with direction
    - Derivatives present
    - No NaN values
    
    Returns:
        Validation results dictionary
    """
    logger.info(f"Validating stream bars: {stream_bars_path}")
    
    if not stream_bars_path.exists():
        return {
            'status': 'FAIL',
            'error': f'File not found: {stream_bars_path}'
        }
    
    df = pd.read_parquet(stream_bars_path)
    
    logger.info(f"  Loaded {len(df):,} stream bars")
    logger.info(f"  Columns: {len(df.columns)}")
    
    results = {
        'status': 'PASS',
        'n_bars': len(df),
        'n_levels': df['level_kind'].nunique() if 'level_kind' in df.columns else 0,
        'checks': {}
    }
    
    # Check 1: Stream values bounded in [-1, +1]
    stream_cols = ['sigma_m', 'sigma_f', 'sigma_b', 'sigma_d', 'sigma_s', 'sigma_p', 'sigma_r']
    bounded_check = True
    for col in stream_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val < -1.0 or max_val > 1.0:
                logger.error(f"  ✗ {col} OUT OF BOUNDS: [{min_val:.3f}, {max_val:.3f}]")
                bounded_check = False
                results['status'] = 'FAIL'
            else:
                logger.info(f"  ✓ {col} bounded: [{min_val:.3f}, {max_val:.3f}]")
    
    results['checks']['bounded'] = bounded_check
    
    # Check 2: No NaN values
    nan_check = True
    for col in stream_cols:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                logger.error(f"  ✗ {col} has {n_nan} NaN values")
                nan_check = False
                results['status'] = 'FAIL'
    
    if nan_check:
        logger.info("  ✓ No NaN values in streams")
    
    results['checks']['no_nans'] = nan_check
    
    # Check 3: Derivatives present
    deriv_cols = ['sigma_p_slope', 'sigma_p_curvature', 'sigma_p_jerk']
    derivs_check = all(col in df.columns for col in deriv_cols)
    
    if derivs_check:
        logger.info("  ✓ Derivatives present (slope, curvature, jerk)")
    else:
        logger.error(f"  ✗ Missing derivative columns: {[c for c in deriv_cols if c not in df.columns]}")
        results['status'] = 'FAIL'
    
    results['checks']['derivatives'] = derivs_check
    
    # Check 4: Barrier sign consistency (spot check)
    if 'sigma_b' in df.columns and 'direction' in df.columns:
        # For UP direction, barrier should be more positive than for DOWN (on average)
        up_df = df[df['direction'] == 'UP']
        down_df = df[df['direction'] == 'DOWN']
        
        if len(up_df) > 0 and len(down_df) > 0:
            up_mean = up_df['sigma_b'].mean()
            down_mean = down_df['sigma_b'].mean()
            
            logger.info(f"  Barrier mean by direction: UP={up_mean:.3f}, DOWN={down_mean:.3f}")
            
            # Not a strict test, but directional bias should exist
            if abs(up_mean - down_mean) > 0.05:
                logger.info("  ✓ Barrier shows directional bias (dir_sign working)")
                results['checks']['barrier_sign'] = True
            else:
                logger.warning("  ⚠ Barrier may not be properly dir_sign adjusted")
                results['checks']['barrier_sign'] = 'UNCERTAIN'
    
    # Summary
    logger.info("\n" + "="*60)
    if results['status'] == 'PASS':
        logger.info("✅ VALIDATION PASSED")
    else:
        logger.error("❌ VALIDATION FAILED")
    logger.info("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate Pentaview stream output')
    parser.add_argument('--date', type=str, required=True,
                        help='Date to validate (YYYY-MM-DD)')
    parser.add_argument('--data-root', type=str, default='data',
                        help='Data root directory (default: data)')
    parser.add_argument('--canonical-version', type=str, default='3.1.0',
                        help='Canonical version (default: 3.1.0)')
    
    args = parser.parse_args()
    
    # Build path (find actual date partition, may have timestamp suffix)
    data_root = Path(args.data_root)
    version_dir = data_root / "gold" / "streams" / "pentaview" / f"version={args.canonical_version}"
    
    # Find date partition (may be date=YYYY-MM-DD or date=YYYY-MM-DD_HH:MM:SS)
    date_partitions = list(version_dir.glob(f"date={args.date}*"))
    
    if not date_partitions:
        logger.error(f"No date partition found for {args.date} in {version_dir}")
        return 1
    
    stream_path = date_partitions[0] / "stream_bars.parquet"
    
    # Validate
    results = validate_stream_bars(stream_path)
    
    # Return exit code
    return 0 if results['status'] == 'PASS' else 1


if __name__ == '__main__':
    exit(main())

