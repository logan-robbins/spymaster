"""
Run grid search for zone width exploration.

Simpler than Bayesian optimization - tests all combinations in a predefined grid.
Good for initial exploration to understand the search space.

Usage:
    # Dry run
    uv run python scripts/run_zone_grid_search.py --dry-run
    
    # Real run
    uv run python scripts/run_zone_grid_search.py \\
        --start-date 2025-11-02 \\
        --end-date 2025-11-30 \\
        --output grid_results.csv
"""

import argparse
import sys
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Dict, Any, List

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pandas as pd
import numpy as np

from src.ml.zone_objective import ZoneObjective


def generate_date_range(start_date: str, end_date: str) -> List[str]:
    """Generate list of weekdays between start and end dates."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Weekdays only
            dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates


def define_search_grid() -> Dict[str, List]:
    """Define search grid for zone parameters."""
    
    grid = {
        'monitor_band': [2.5, 5.0, 7.5, 10.0, 12.5, 15.0],  # 6 values
        'touch_band': [1.0, 2.0, 3.0, 5.0],                 # 4 values
        'outcome_strikes': [2, 3, 4],                        # 3 values
        'level_types': [
            ['PM_HIGH', 'PM_LOW'],                                              # PM only
            ['OR_HIGH', 'OR_LOW'],                                              # OR only
            ['SMA_200', 'SMA_400'],                                             # SMA only
            ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW'],                        # PM + OR
            ['PM_HIGH', 'PM_LOW', 'SMA_200', 'SMA_400'],                       # PM + SMA
            ['PM_HIGH', 'PM_LOW', 'OR_HIGH', 'OR_LOW', 'SMA_200', 'SMA_400'],  # All
        ]  # 6 combinations
    }
    
    # Total combinations: 6 × 4 × 3 × 6 = 432
    return grid


def config_to_dict(monitor: float, touch: float, strikes: int, levels: List[str]) -> Dict[str, Any]:
    """Convert grid point to config dict."""
    
    return {
        'monitor_band_pm': monitor,
        'monitor_band_or': monitor,
        'monitor_band_sma': monitor,
        'touch_band': touch,
        'outcome_strikes': strikes,
        'use_pm': any('PM' in l for l in levels),
        'use_or': any('OR' in l for l in levels),
        'use_sma_200': 'SMA_200' in levels,
        'use_sma_400': 'SMA_400' in levels,
        'config_overrides': {
            'MONITOR_BAND': monitor,
            'TOUCH_BAND': touch,
            'OUTCOME_THRESHOLD': strikes * 25.0,
        }
    }


def run_grid_search(
    train_dates: List[str],
    grid: Dict[str, List],
    dry_run: bool = False
) -> pd.DataFrame:
    """
    Run grid search over all combinations.
    
    Args:
        train_dates: Dates for training
        grid: Search grid definition
        dry_run: If True, use mock data
    
    Returns:
        DataFrame with results for each configuration
    """
    # Create objective function (reuse for all configs)
    objective_fn = ZoneObjective(
        train_dates=train_dates,
        target_events_per_day=50.0,
        dry_run=dry_run
    )
    
    results = []
    total_configs = (
        len(grid['monitor_band']) *
        len(grid['touch_band']) *
        len(grid['outcome_strikes']) *
        len(grid['level_types'])
    )
    
    print(f"\nRunning grid search over {total_configs} configurations...")
    print()
    
    # Iterate over grid
    for i, (monitor, touch, strikes, levels) in enumerate(
        product(
            grid['monitor_band'],
            grid['touch_band'],
            grid['outcome_strikes'],
            grid['level_types']
        ),
        start=1
    ):
        config = config_to_dict(monitor, touch, strikes, levels)
        
        # Create mock trial object for objective function
        class MockTrial:
            def __init__(self, config):
                self.number = i
                self._config = config
            
            def suggest_float(self, name, low, high):
                return self._config.get(name, (low + high) / 2)
            
            def suggest_int(self, name, low, high):
                return self._config.get(name, (low + high) // 2)
            
            def suggest_categorical(self, name, choices):
                return self._config.get(name, choices[0])
        
        trial = MockTrial(config)
        
        try:
            # Evaluate this configuration
            score = objective_fn(trial)
            
            # Store result
            result = {
                'config_id': i,
                'monitor_band': monitor,
                'touch_band': touch,
                'outcome_strikes': strikes,
                'level_types': ','.join(levels),
                'score': score
            }
            results.append(result)
            
            # Progress
            if i % 10 == 0 or i == total_configs:
                print(f"  [{i:3d}/{total_configs}] Config {i}: score={score:.4f}")
        
        except Exception as e:
            print(f"  [{i:3d}/{total_configs}] Config {i}: FAILED ({e})")
            results.append({
                'config_id': i,
                'monitor_band': monitor,
                'touch_band': touch,
                'outcome_strikes': strikes,
                'level_types': ','.join(levels),
                'score': 0.0,
                'error': str(e)
            })
    
    return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame):
    """Print analysis of grid search results."""
    
    print(f"\n{'='*70}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*70}\n")
    
    # Top 10 configurations
    print("Top 10 Configurations:")
    print()
    
    top10 = results_df.nlargest(10, 'score')
    
    for idx, row in top10.iterrows():
        print(f"  #{row['config_id']:3d}  Score: {row['score']:.4f}")
        print(f"         Monitor: {row['monitor_band']:5.1f}  Touch: {row['touch_band']:4.1f}  "
              f"Strikes: {row['outcome_strikes']}  Levels: {row['level_types']}")
        print()
    
    # Parameter importance (variance explained)
    print(f"\n{'='*70}")
    print("PARAMETER IMPORTANCE")
    print(f"{'='*70}\n")
    
    for param in ['monitor_band', 'touch_band', 'outcome_strikes']:
        grouped = results_df.groupby(param)['score'].agg(['mean', 'std', 'count'])
        
        print(f"{param}:")
        for val, row in grouped.iterrows():
            print(f"  {val:8}  Mean: {row['mean']:.4f}  Std: {row['std']:.4f}  N: {int(row['count'])}")
        print()
    
    # Level type importance
    print("Level Types:")
    grouped = results_df.groupby('level_types')['score'].agg(['mean', 'std', 'count'])
    for val, row in grouped.iterrows():
        print(f"  {val:50s}  Mean: {row['mean']:.4f}  Std: {row['std']:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Run grid search for zone width exploration'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='grid_search_results.csv',
        help='Output CSV file (default: grid_search_results.csv)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Use mock data for testing'
    )
    
    args = parser.parse_args()
    
    # Generate dates
    if args.dry_run:
        dates = generate_date_range('2025-11-02', '2025-11-10')
        print(f"DRY RUN MODE: Using {len(dates)} mock dates")
    else:
        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date required (or use --dry-run)")
        
        dates = generate_date_range(args.start_date, args.end_date)
        print(f"Running grid search on {len(dates)} dates")
        print(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Define grid
    grid = define_search_grid()
    
    total_configs = (
        len(grid['monitor_band']) *
        len(grid['touch_band']) *
        len(grid['outcome_strikes']) *
        len(grid['level_types'])
    )
    
    print(f"\nGrid dimensions:")
    print(f"  monitor_band:     {len(grid['monitor_band'])} values")
    print(f"  touch_band:       {len(grid['touch_band'])} values")
    print(f"  outcome_strikes:  {len(grid['outcome_strikes'])} values")
    print(f"  level_types:      {len(grid['level_types'])} combinations")
    print(f"  Total configs:    {total_configs}")
    
    if not args.dry_run:
        est_hours = (total_configs * 2) / 60  # ~2 min per config
        print(f"\nEstimated time:   ~{est_hours:.1f} hours")
    
    # Run grid search
    results_df = run_grid_search(dates, grid, args.dry_run)
    
    # Save results
    output_path = Path(args.output)
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Analyze
    analyze_results(results_df)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

