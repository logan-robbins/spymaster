"""
Audit Silver Feature Statistics
Investigate 'Market Tide' inversion by auditing feature distributions across time periods.

Focus:
- call_tide / put_tide
- ofi_1min / ofi_5min (Comparison)
- velocity_1min (Activity proxy)

Metrics:
- Mean, Std, Skew, Kurtosis
- % Zeros (Sparsity)
- Correlation with OFI
"""
import sys
from pathlib import Path
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_silver_data(date_range: List[str], version: str) -> pd.DataFrame:
    """Load Silver signals for a list of dates."""
    data_root = backend_dir / "data"
    dfs = []
    
    for date in date_range:
        path = data_root / "silver" / "features" / "es_pipeline" / f"version={version}" / f"date={date}" / "signals.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                df['date'] = date
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")
        else:
            logger.warning(f"Missing data for {date}")
            
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def audit_features(df: pd.DataFrame, features: List[str], period_name: str):
    """Compute and print statistics for selected features."""
    if df.empty:
        logger.warning(f"[{period_name}] No data found.")
        return

    logger.info(f"\n=== Audit: {period_name} ({len(df):,} rows) ===")
    logger.info(f"{'Feature':<20} | {'Mean':<10} | {'Std':<10} | {'Skew':<8} | {'% Zeros':<8} | {'Min':<10} | {'Max':<10}")
    logger.info("-" * 100)
    
    for feat in features:
        if feat not in df.columns:
            logger.warning(f"{feat:<20} | MISSING")
            continue
            
        series = df[feat].dropna()
        if len(series) == 0:
            logger.warning(f"{feat:<20} | EMPTY")
            continue
            
        mean = series.mean()
        std = series.std()
        skew = series.skew()
        pct_zeros = (series == 0).mean() * 100
        min_val = series.min()
        max_val = series.max()
        
        logger.info(f"{feat:<20} | {mean:<10.4f} | {std:<10.4f} | {skew:<8.2f} | {pct_zeros:<8.1f}% | {min_val:<10.2f} | {max_val:<10.2f}")

    # Correlations
    logger.info("\n--- Correlation Matrix (Pearson) ---")
    available_features = [f for f in features if f in df.columns]
    if available_features:
        corr = df[available_features].corr()
        print(corr.round(3))
    else:
        print("No features available for correlation.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="4.0.0")
    args = parser.parse_args()

    # Define Audit Periods
    periods = {
        "June (Early)": ["2025-06-05", "2025-06-06", "2025-06-09"],
        "August (Mid)": ["2025-08-20", "2025-08-21", "2025-08-22"],
        "Sept (Late)":   ["2025-09-23", "2025-09-24", "2025-09-25"]  # Sept 30 is a Tuesday
    }

    features_to_audit = [
        # Market Tide (Target)
        'call_tide', 'put_tide', 
        # Benchmark (Physics)
        'ofi_60s', 'ofi_300s',          # Updated from ofi_1min/5min
        'velocity_1min',
        # Derived
        'force_proxy', 'mass_proxy',
        'accel_residual', 'predicted_accel'  # From compute_force_mass
    ]

    for name, dates in periods.items():
        df = load_silver_data(dates, args.version)
        audit_features(df, features_to_audit, name)

if __name__ == "__main__":
    main()
