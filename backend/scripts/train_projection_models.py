"""
Train stream projection models with MLflow tracking.

Per STREAMS.md Section 6:
- Trains quantile polynomial regression models for stream forecasting
- Separate model per stream (pressure, flow, barrier, etc.)
- Logs experiments to MLflow with metrics
- Saves model bundles to data/ml/projection_models/

Usage:
    # Train pressure stream projection model
    uv run python -m scripts.train_projection_models \
        --stream pressure \
        --data-path data/gold/training/projection_samples \
        --epochs 200

    # Train all streams
    uv run python -m scripts.train_projection_models \
        --stream all \
        --epochs 200
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

from src.ml.stream_projector import StreamProjector, ProjectionConfig
from src.ml.tracking import tracking_run, log_metrics, log_artifacts, resolve_repo_root

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


STREAM_DISPLAY_NAMES = {
    'sigma_p': 'pressure',
    'sigma_m': 'momentum',
    'sigma_f': 'flow',
    'sigma_b': 'barrier',
    'sigma_r': 'structure'
}


def load_training_data(data_path: Path, stream_name: str, version: str = 'v1') -> List[Dict]:
    """
    Load training samples for a stream.
    
    Args:
        data_path: Path to projection_samples directory
        stream_name: Stream name (sigma_p, sigma_m, etc.)
        version: Dataset version
    
    Returns:
        List of training samples
    """
    data_file = data_path / f'projection_samples_{stream_name}_{version}.npz'
    
    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found: {data_file}")
    
    logger.info(f"Loading training data from {data_file}...")
    
    data = np.load(data_file, allow_pickle=True)
    
    n_samples = len(data['current_value'])
    logger.info(f"  Loaded {n_samples:,} training samples")
    
    # Convert back to list of dictionaries
    samples = []
    cross_stream_names = data['cross_stream_names'].tolist()
    static_feature_names = data['static_feature_names'].tolist()
    
    for i in range(n_samples):
        # Rebuild cross_streams dict
        cross_streams = {}
        for j, cs_name in enumerate(cross_stream_names):
            cross_streams[cs_name] = data['cross_streams'][i, j]
        
        # Rebuild static_features dict
        static_features = {}
        for j, sf_name in enumerate(static_feature_names):
            static_features[sf_name] = float(data['static_features'][i, j])
        
        sample = {
            'stream_hist': data['stream_hist'][i],
            'slope_hist': data['slope_hist'][i],
            'current_value': float(data['current_value'][i]),
            'future_target': data['future_target'][i],
            'setup_weight': float(data['setup_weight'][i]),
            'cross_streams': cross_streams,
            'static_features': static_features
        }
        samples.append(sample)
    
    return samples


def split_train_val(
    samples: List[Dict],
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> tuple[List[Dict], List[Dict]]:
    """
    Split samples into train/val sets.
    
    Uses random split (not temporal) since samples are already pre-filtered
    for quality and represent diverse contexts.
    
    Args:
        samples: All training samples
        val_ratio: Fraction for validation
        random_seed: Random seed
    
    Returns:
        (train_samples, val_samples)
    """
    rng = np.random.RandomState(random_seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    
    n_val = int(len(samples) * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    
    logger.info(f"  Train: {len(train_samples):,} samples")
    logger.info(f"  Val:   {len(val_samples):,} samples")
    
    return train_samples, val_samples


def evaluate_projector(
    projector: StreamProjector,
    val_samples: List[Dict]
) -> Dict[str, float]:
    """
    Evaluate projection model on validation set.
    
    Computes metrics:
    - Path MAE: Mean absolute error over projected path
    - Endpoint MAE: Error at horizon H
    - Path R^2: Coefficient of determination for path
    
    Args:
        projector: Trained projection model
        val_samples: Validation samples
    
    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating on validation set...")
    
    H = projector.config.horizon_bars
    
    # Collect predictions and targets
    all_preds = {q: [] for q in ['q10', 'q50', 'q90']}
    all_targets = []
    all_currents = []
    
    for sample in val_samples:
        coeffs = projector.predict(
            stream_hist=sample['stream_hist'],
            current_value=sample['current_value'],
            slope_hist=sample['slope_hist'],
            cross_streams=sample['cross_streams'],
            static_features=sample['static_features']
        )
        
        # Generate paths
        from src.ml.stream_projector import build_polynomial_path
        for q_str, q_coeffs in [('q10', coeffs.q10), ('q50', coeffs.q50), ('q90', coeffs.q90)]:
            path = build_polynomial_path(sample['current_value'], q_coeffs, H)
            all_preds[q_str].append(path[1:])  # Exclude current value
        
        all_targets.append(sample['future_target'])
        all_currents.append(sample['current_value'])
    
    all_targets = np.array(all_targets)  # [N, H]
    
    metrics = {}
    
    # Metrics for q50 (median forecast)
    preds_q50 = np.array(all_preds['q50'])  # [N, H]
    
    # Path MAE
    path_mae = np.mean(np.abs(preds_q50 - all_targets))
    metrics['val_path_mae_q50'] = float(path_mae)
    
    # Endpoint MAE (at horizon H)
    endpoint_mae = np.mean(np.abs(preds_q50[:, -1] - all_targets[:, -1]))
    metrics['val_endpoint_mae_q50'] = float(endpoint_mae)
    
    # Path R^2
    ss_res = np.sum((preds_q50 - all_targets)**2)
    ss_tot = np.sum((all_targets - np.mean(all_targets))**2)
    path_r2 = 1 - ss_res / (ss_tot + 1e-8)
    metrics['val_path_r2_q50'] = float(path_r2)
    
    # Horizon-specific MAE (for diagnostics)
    for h in [1, 5, 10]:
        if h <= H:
            h_mae = np.mean(np.abs(preds_q50[:, h-1] - all_targets[:, h-1]))
            metrics[f'val_mae_h{h}_q50'] = float(h_mae)
    
    # Coverage metrics for uncertainty bands
    # Check if true value falls within [q10, q90] band
    preds_q10 = np.array(all_preds['q10'])
    preds_q90 = np.array(all_preds['q90'])
    
    coverage = np.mean(
        (all_targets >= preds_q10) & (all_targets <= preds_q90)
    )
    metrics['val_coverage_80pct'] = float(coverage)
    
    # Band width (average q90 - q10)
    band_width = np.mean(preds_q90 - preds_q10)
    metrics['val_band_width'] = float(band_width)
    
    logger.info(f"  Path MAE (q50): {path_mae:.4f}")
    logger.info(f"  Endpoint MAE (q50): {endpoint_mae:.4f}")
    logger.info(f"  Path R^2 (q50): {path_r2:.4f}")
    logger.info(f"  Coverage (80%): {coverage:.4f}")
    logger.info(f"  Band width: {band_width:.4f}")
    
    return metrics


def train_stream_projector(
    stream_name: str,
    data_path: Path,
    output_dir: Path,
    config: ProjectionConfig,
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    val_ratio: float = 0.2,
    version: str = 'v1'
) -> Dict[str, float]:
    """
    Train projection model for one stream.
    
    Args:
        stream_name: Stream name (sigma_p, sigma_m, etc.)
        data_path: Path to training data directory
        output_dir: Output directory for models
        config: Projection configuration
        max_iter: Max boosting iterations
        learning_rate: Learning rate
        max_depth: Max tree depth
        val_ratio: Validation split ratio
        version: Dataset version
    
    Returns:
        Combined training and validation metrics
    """
    display_name = STREAM_DISPLAY_NAMES.get(stream_name, stream_name)
    logger.info(f"\n{'='*60}")
    logger.info(f"Training projection model: {display_name} ({stream_name})")
    logger.info(f"{'='*60}")
    
    # Load training data
    samples = load_training_data(data_path, stream_name, version)
    
    # Split train/val
    train_samples, val_samples = split_train_val(samples, val_ratio)
    
    # Initialize projector
    projector = StreamProjector(stream_name=stream_name, config=config)
    
    # Train
    train_metrics = projector.fit(
        training_samples=train_samples,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    
    # Evaluate
    val_metrics = evaluate_projector(projector, val_samples)
    
    # Save model
    model_path = output_dir / f'projection_{stream_name}_{version}.joblib'
    projector.save(model_path)
    
    # Combine metrics
    all_metrics = {**train_metrics, **val_metrics}
    
    return all_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description='Train stream projection models')
    parser.add_argument('--stream', type=str, required=True,
                       help='Stream name (sigma_p, sigma_m, etc.) or "all"')
    parser.add_argument('--data-path', type=str,
                       default='data/gold/training/projection_samples',
                       help='Path to training data directory')
    parser.add_argument('--output-dir', type=str,
                       default='data/ml/projection_models',
                       help='Output directory for models')
    parser.add_argument('--lookback', type=int, default=20, help='History window (L)')
    parser.add_argument('--horizon', type=int, default=10, help='Forecast window (H)')
    parser.add_argument('--epochs', type=int, default=200, help='Max boosting iterations')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--max-depth', type=int, default=6, help='Max tree depth')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation split')
    parser.add_argument('--version', type=str, default='v1', help='Dataset version')
    parser.add_argument('--experiment', type=str, default='stream_projection',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    repo_root = resolve_repo_root()
    
    # Projection configuration
    config = ProjectionConfig(
        lookback_bars=args.lookback,
        horizon_bars=args.horizon,
        polynomial_degree=3,
        quantiles=[0.10, 0.50, 0.90]
    )
    
    # Determine which streams to train
    if args.stream == 'all':
        stream_names = ['sigma_p', 'sigma_m', 'sigma_f', 'sigma_b', 'sigma_r']
    else:
        stream_names = [args.stream]
    
    # Train each stream
    for stream_name in stream_names:
        display_name = STREAM_DISPLAY_NAMES.get(stream_name, stream_name)
        run_name = f"projection_{display_name}_{args.version}"
        
        # Setup tracking
        params = {
            'stream_name': stream_name,
            'display_name': display_name,
            'lookback_bars': args.lookback,
            'horizon_bars': args.horizon,
            'polynomial_degree': 3,
            'max_iter': args.epochs,
            'learning_rate': args.learning_rate,
            'max_depth': args.max_depth,
            'val_ratio': args.val_ratio,
            'version': args.version
        }
        
        tags = {
            'model_type': 'stream_projection',
            'stream': stream_name,
            'version': args.version
        }
        
        wandb_tags = [
            'stream_projection',
            stream_name,
            f'v{args.version}'
        ]
        
        with tracking_run(
            run_name=run_name,
            experiment=args.experiment,
            params=params,
            tags=tags,
            wandb_tags=wandb_tags,
            project='spymaster',
            repo_root=repo_root
        ) as tracking:
            try:
                # Train model
                metrics = train_stream_projector(
                    stream_name=stream_name,
                    data_path=data_path,
                    output_dir=output_dir,
                    config=config,
                    max_iter=args.epochs,
                    learning_rate=args.learning_rate,
                    max_depth=args.max_depth,
                    val_ratio=args.val_ratio,
                    version=args.version
                )
                
                # Log metrics
                log_metrics(metrics, tracking.wandb_run)
                
                # Log model artifact
                model_path = output_dir / f'projection_{stream_name}_{args.version}.joblib'
                log_artifacts(
                    paths=[model_path],
                    name=f'projection_model_{stream_name}',
                    artifact_type='model',
                    wandb_run=tracking.wandb_run
                )
                
                logger.info(f"✓ Successfully trained {display_name} projection model")
                
            except Exception as e:
                logger.error(f"✗ Failed to train {display_name}: {e}", exc_info=True)
                return 1
    
    logger.info("\n" + "="*60)
    logger.info("All projection models trained successfully!")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

