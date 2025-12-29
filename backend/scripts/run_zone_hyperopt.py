"""
Run zone width hyperparameter optimization.

Usage:
    # Dry run (no real data, test framework)
    uv run python scripts/run_zone_hyperopt.py --dry-run --n-trials 10
    
    # Real run
    uv run python scripts/run_zone_hyperopt.py \\
        --start-date 2025-11-02 \\
        --end-date 2025-11-30 \\
        --n-trials 100 \\
        --study-name zone_opt_v1
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("ERROR: optuna not installed. Run: uv add optuna")
    sys.exit(1)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: mlflow not installed. Tracking disabled. Install: uv add mlflow")

from src.ml.zone_objective import ZoneObjective


def generate_date_range(start_date: str, end_date: str) -> List[str]:
    """Generate list of weekdays between start and end dates."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = []
    current = start
    while current <= end:
        # Only weekdays (Mon-Fri)
        if current.weekday() < 5:
            dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates


def run_hyperopt(
    train_dates: List[str],
    n_trials: int = 100,
    study_name: str = 'zone_opt_v1',
    dry_run: bool = False
) -> optuna.Study:
    """
    Run hyperparameter optimization.
    
    Args:
        train_dates: Dates to use for training
        n_trials: Number of optimization trials
        study_name: Optuna study name
        dry_run: If True, use mock data instead of real pipeline
    
    Returns:
        Completed Optuna study
    """
    # Create objective function
    objective = ZoneObjective(
        train_dates=train_dates,
        val_dates=None,  # Will use last 5 train dates
        target_events_per_day=50.0,
        dry_run=dry_run
    )
    
    # Create Optuna study (maximize attribution score)
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    # Start MLflow experiment if available
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment('zone_hyperopt_v1')
        
        with mlflow.start_run(run_name=f'hyperopt_{study_name}'):
            # Run optimization
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=True,
                catch=(Exception,)
            )
            
            # Log best parameters to MLflow
            if study.best_trial:
                mlflow.log_params(study.best_params)
                mlflow.log_metric('best_attribution_score', study.best_value)
    else:
        # Run without MLflow
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            catch=(Exception,)
        )
    
    return study


def print_results(study: optuna.Study):
    """Print optimization results."""
    
    print(f"\n{'='*70}")
    print("HYPEROPT RESULTS")
    print(f"{'='*70}\n")
    
    if study.best_trial is None:
        print("No successful trials completed.")
        return
    
    print(f"Best Attribution Score: {study.best_value:.4f}")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"\nBest Parameters:")
    
    for param, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {param:25s}: {value:8.3f}")
        else:
            print(f"  {param:25s}: {value}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("OPTIMIZATION STATISTICS")
    print(f"{'='*70}\n")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"Total trials:     {len(study.trials)}")
    print(f"  Completed:      {len(completed_trials)}")
    print(f"  Pruned:         {len(pruned_trials)}")
    print(f"  Failed:         {len(failed_trials)}")
    
    if completed_trials:
        values = [t.value for t in completed_trials]
        print(f"\nScore Statistics:")
        print(f"  Min:            {min(values):.4f}")
        print(f"  Max:            {max(values):.4f}")
        print(f"  Mean:           {sum(values)/len(values):.4f}")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run zone width hyperparameter optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (test framework without real data)
    uv run python scripts/run_zone_hyperopt.py --dry-run --n-trials 10
    
    # Real optimization on November data
    uv run python scripts/run_zone_hyperopt.py \\
        --start-date 2025-11-02 \\
        --end-date 2025-11-30 \\
        --n-trials 100
        """
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
        '--n-trials',
        type=int,
        default=100,
        help='Number of optimization trials (default: 100)'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        default='zone_opt_v1',
        help='Optuna study name (default: zone_opt_v1)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Use mock data for testing (no real pipeline execution)'
    )
    
    args = parser.parse_args()
    
    # Generate dates
    if args.dry_run:
        # Mock dates for dry run
        dates = generate_date_range('2025-11-02', '2025-11-10')
        print(f"DRY RUN MODE: Using {len(dates)} mock dates")
    else:
        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date required (or use --dry-run)")
        
        dates = generate_date_range(args.start_date, args.end_date)
        print(f"Running hyperopt on {len(dates)} dates")
        print(f"Date range: {dates[0]} to {dates[-1]}")
    
    print(f"Trials: {args.n_trials}")
    print(f"Study: {args.study_name}")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN: Using mock data for framework testing")
    
    print()
    
    # Run optimization
    study = run_hyperopt(
        train_dates=dates,
        n_trials=args.n_trials,
        study_name=args.study_name,
        dry_run=args.dry_run
    )
    
    # Print results
    print_results(study)
    
    # Save study
    if not args.dry_run:
        output_dir = Path('data/ml/experiments')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        study_path = output_dir / f'{args.study_name}.pkl'
        
        try:
            import joblib
            joblib.dump(study, study_path)
            print(f"✅ Study saved to: {study_path}")
        except ImportError:
            print("Warning: joblib not installed, study not saved")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

