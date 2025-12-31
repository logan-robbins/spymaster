"""
Demo script for stream projection models.

Demonstrates:
1. Loading trained projection model
2. Generating forecast curves with uncertainty bands
3. Visualizing projected trajectories

Usage:
    uv run python -m scripts.demo_projection
"""
import argparse
import logging
from pathlib import Path
import numpy as np

from src.ml.stream_projector import StreamProjector, project_stream_curves

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_projection(model_path: Path) -> None:
    """
    Demonstrate projection model inference.
    
    Args:
        model_path: Path to trained model
    """
    # Load model
    logger.info(f"Loading projection model from {model_path}...")
    projector = StreamProjector.load(model_path)
    
    logger.info(f"  Stream: {projector.stream_name}")
    logger.info(f"  Lookback: {projector.config.lookback_bars} bars")
    logger.info(f"  Horizon: {projector.config.horizon_bars} bars (20 minutes)")
    logger.info(f"  Trained on: {projector.training_history['n_samples']} samples")
    
    # Create synthetic test case
    logger.info("\nGenerating test forecast...")
    
    # Upward trending stream
    t = np.linspace(0, 1, 20)
    stream_hist = 0.3 * np.sin(2 * np.pi * t) + 0.1 * t
    slope_hist = np.gradient(stream_hist)
    current_value = float(stream_hist[-1])
    
    # Cross-stream context (other streams showing similar trend)
    cross_streams = {
        'sigma_b': 0.2 * np.sin(2 * np.pi * t[-5:]) + 0.05,
        'sigma_d': 0.1 * np.ones(5),
        'sigma_f': 0.25 * np.sin(2 * np.pi * t[-5:]) + 0.1 * t[-5:],
        'sigma_m': 0.28 * np.sin(2 * np.pi * t[-5:]) + 0.12 * t[-5:],
        'sigma_s': 0.4 * np.ones(5),
    }
    
    # Static features (typical setup)
    static_features = {
        'atr': 5.5,
        'attempt_index': 2.0,
        'direction_encoded': 1.0,  # UP
        'level_kind_encoded': 2.0,  # OR_HIGH
        'level_stacking_5pt': 3.0,
        'time_bucket_encoded': 1.0,  # T15_30
    }
    
    # Generate projection curves
    curves = project_stream_curves(
        projector=projector,
        stream_hist=stream_hist,
        current_value=current_value,
        slope_hist=slope_hist,
        cross_streams=cross_streams,
        static_features=static_features
    )
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("PROJECTION FORECAST")
    logger.info("="*70)
    logger.info(f"Current value: {current_value:.4f}")
    logger.info(f"\nTime    q10      q50      q90      Range")
    logger.info("-" * 70)
    
    for h in range(0, 11, 2):  # Show every 2 bars
        q10 = curves['q10'][h]
        q50 = curves['q50'][h]
        q90 = curves['q90'][h]
        range_width = q90 - q10
        
        if h == 0:
            time_str = "Now"
        else:
            time_str = f"+{h*2}min"
        
        logger.info(f"{time_str:6s}  {q10:7.4f}  {q50:7.4f}  {q90:7.4f}  {range_width:7.4f}")
    
    logger.info("="*70)
    
    # Compute projected slope and curvature at horizon
    from src.ml.stream_projector import compute_projected_slope, compute_projected_curvature
    
    coeffs_q50 = projector.predict(
        stream_hist=stream_hist,
        current_value=current_value,
        slope_hist=slope_hist,
        cross_streams=cross_streams,
        static_features=static_features
    ).q50
    
    slope_h5 = compute_projected_slope(coeffs_q50, horizon=5)
    curv_h5 = compute_projected_curvature(coeffs_q50, horizon=5)
    
    logger.info(f"\nProjected dynamics at +10 min (h=5):")
    logger.info(f"  Slope (velocity): {slope_h5:.4f}")
    logger.info(f"  Curvature (acceleration): {curv_h5:.4f}")
    
    logger.info("\n✓ Demo complete!")


def main() -> int:
    parser = argparse.ArgumentParser(description='Demo projection model inference')
    parser.add_argument('--model-path', type=str,
                       default='data/ml/projection_models/projection_sigma_p_test.joblib',
                       help='Path to trained projection model')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Run train_projection_models.py first to create models")
        return 1
    
    demo_projection(model_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

