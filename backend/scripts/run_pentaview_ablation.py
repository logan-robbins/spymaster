"""
Pentaview Projection Model Ablation Study

Evaluates forecast accuracy, quantile calibration, and feature importance
for the 5 Pentaview projection models across multiple dimensions.

Usage:
    uv run python scripts/run_pentaview_ablation.py --start-date 2025-11-20 --end-date 2025-12-18
    uv run python scripts/run_pentaview_ablation.py --date 2025-12-18  # Single day
    uv run python scripts/run_pentaview_ablation.py --quick  # Fast test on subset
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import joblib
from scipy import stats

# Add backend directory to path for model loading
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProjectionMetrics:
    """Metrics for evaluating trajectory forecasts."""
    # Accuracy metrics (per horizon)
    mae_by_horizon: Dict[int, float]  # {30: 0.05, 60: 0.08, ...}
    rmse_by_horizon: Dict[int, float]
    
    # Directional accuracy
    directional_accuracy: float  # % of correct up/down predictions
    direction_by_horizon: Dict[int, float]
    
    # Quantile calibration
    q10_coverage: float  # Should be ~10% below q10
    q90_coverage: float  # Should be ~10% above q90
    band_coverage: float  # Should be ~80% in [q10, q90]
    
    # Sharpness (narrower is better if calibrated)
    mean_band_width: float
    band_width_by_horizon: Dict[int, float]
    
    # Polynomial fit quality
    trajectory_r2: float  # R² of actual vs predicted trajectory
    mean_trajectory_error: float
    
    # Sample size
    n_forecasts: int
    n_valid_horizons: int


@dataclass
class AblationResult:
    """Results for one ablation configuration."""
    config_name: str
    stream_name: str  # sigma_p, sigma_m, etc.
    metrics: ProjectionMetrics
    
    # Regime breakdown
    metrics_by_regime: Dict[str, ProjectionMetrics]
    
    # Feature importance (if applicable)
    feature_ablations: Optional[Dict[str, float]] = None


class PentaviewProjectionModel:
    """Wrapper for loaded projection models with prediction logic."""
    
    def __init__(self, model_path: Path, stream_name: str):
        self.stream_name = stream_name
        model_dict = joblib.load(model_path)
        
        self.models = model_dict['models']  # {q10, q50, q90} -> sklearn model
        self.config = model_dict['config']
        self.feature_names = model_dict.get('feature_names', [])
        self.lookback = self.config.lookback_bars
        
    def _build_features(
        self,
        stream_hist: np.ndarray,
        cross_streams: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Build feature vector matching training format."""
        features = []
        
        # Primary stream history (16 bars)
        if len(stream_hist) < self.lookback:
            padded = np.zeros(self.lookback)
            padded[-len(stream_hist):] = stream_hist
            stream_hist = padded
        else:
            stream_hist = stream_hist[-self.lookback:]
        
        features.extend(stream_hist)
        
        # Compute slope history (simple diff)
        slope_hist = np.diff(stream_hist, prepend=stream_hist[0])
        features.extend(slope_hist)
        
        # Cross-stream context (last 5 bars for each of 4 other streams)
        if cross_streams:
            other_streams = [s for s in ['sigma_p', 'sigma_m', 'sigma_f', 'sigma_b', 'sigma_r'] 
                           if s != self.stream_name]
            for other_stream in other_streams:
                if other_stream in cross_streams:
                    cross_hist = cross_streams[other_stream]
                    if len(cross_hist) >= 5:
                        features.extend(cross_hist[-5:])
                    else:
                        # Pad if needed
                        padded = np.zeros(5)
                        padded[-len(cross_hist):] = cross_hist
                        features.extend(padded)
                else:
                    features.extend([0.0] * 5)  # Missing stream
        
        # Pad or truncate to match expected feature count
        features_array = np.array(features)
        expected_len = len(self.feature_names) if self.feature_names else 63
        
        if len(features_array) < expected_len:
            padded = np.zeros(expected_len)
            padded[:len(features_array)] = features_array
            features_array = padded
        elif len(features_array) > expected_len:
            features_array = features_array[:expected_len]
        
        return features_array.reshape(1, -1)
        
    def predict(
        self,
        stream_hist: np.ndarray,
        current_value: float,
        cross_streams: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, List[float]]:
        """
        Generate 10-bar forecast trajectory.
        
        Returns:
            {
                'q10': [val_30s, val_60s, ..., val_300s],
                'q50': [...],
                'q90': [...]
            }
        """
        # Build features
        X = self._build_features(stream_hist, cross_streams)
        
        # Predict polynomial coefficients for each quantile
        forecast = {}
        
        for q_name, model in self.models.items():
            # Predict coefficients: [a1, a2, a3]
            coeffs = model.predict(X)[0]  # Shape: (3,)
            
            # Expand polynomial to 10 time steps
            # v(t) = v0 + a1*t + a2*t^2 + a3*t^3
            # where t is in units of 30-second bars
            trajectory = []
            for t in range(1, 11):  # 1 to 10 bars ahead
                value = current_value + coeffs[0] * t + coeffs[1] * (t**2) + coeffs[2] * (t**3)
                # Clip to reasonable range [-1.5, 1.5] for normalized streams
                value = np.clip(value, -1.5, 1.5)
                trajectory.append(float(value))
            
            forecast[q_name] = trajectory
        
        return forecast


class PentaviewAblationStudy:
    """Main ablation study orchestrator."""
    
    def __init__(
        self,
        data_root: Path = Path("data"),
        model_dir: Path = Path("data/ml/projection_models"),
        output_dir: Path = Path("data/ml/ablation_pentaview")
    ):
        self.data_root = data_root
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stream_names = ['sigma_p', 'sigma_m', 'sigma_f', 'sigma_b', 'sigma_r']
        self.horizons = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]  # seconds
        
        # Load models
        self.models = self._load_models()
        
    def _load_models(self) -> Dict[str, PentaviewProjectionModel]:
        """Load all 5 projection models."""
        models = {}
        
        for stream in self.stream_names:
            # Expected naming: projection_sigma_p_v30s_20251115_20251215.joblib
            pattern = f"projection_{stream}_v30s_*.joblib"
            model_files = list(self.model_dir.glob(pattern))
            
            if not model_files:
                logger.warning(f"No model found for {stream}")
                continue
            
            model_path = model_files[0]  # Use most recent
            logger.info(f"Loading {stream} model: {model_path.name}")
            
            try:
                models[stream] = PentaviewProjectionModel(model_path, stream)
            except Exception as e:
                logger.error(f"Failed to load {stream} model: {e}")
        
        return models
    
    def load_stream_data(self, date: str) -> Optional[pd.DataFrame]:
        """Load stream bars for a given date."""
        date_partition = f"date={date}_00:00:00"
        stream_path = (
            self.data_root / "gold" / "streams" / "pentaview" /
            "version=3.1.0" / date_partition / "stream_bars.parquet"
        )
        
        if not stream_path.exists():
            logger.warning(f"Stream data not found: {stream_path}")
            return None
        
        df = pd.read_parquet(stream_path)
        
        # Ensure timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        logger.info(f"Loaded {len(df):,} stream bars for {date}")
        return df
    
    def compute_metrics(
        self,
        predictions: List[Dict[str, List[float]]],
        actuals: List[np.ndarray],
        current_values: List[float]
    ) -> ProjectionMetrics:
        """Compute evaluation metrics from predictions and actuals."""
        
        if len(predictions) == 0:
            # Return empty metrics
            return ProjectionMetrics(
                mae_by_horizon={},
                rmse_by_horizon={},
                directional_accuracy=0.0,
                direction_by_horizon={},
                q10_coverage=0.0,
                q90_coverage=0.0,
                band_coverage=0.0,
                mean_band_width=0.0,
                band_width_by_horizon={},
                trajectory_r2=0.0,
                mean_trajectory_error=0.0,
                n_forecasts=0,
                n_valid_horizons=0
            )
        
        n_forecasts = len(predictions)
        n_horizons = len(self.horizons)
        
        # Initialize accumulators
        mae_by_h = {h: [] for h in range(n_horizons)}
        rmse_by_h = {h: [] for h in range(n_horizons)}
        direction_by_h = {h: [] for h in range(n_horizons)}
        band_widths_by_h = {h: [] for h in range(n_horizons)}
        
        q10_below_count = 0
        q90_above_count = 0
        in_band_count = 0
        total_points = 0
        
        trajectory_errors = []
        
        for pred, actual, current in zip(predictions, actuals, current_values):
            # Handle case where actual is shorter than forecast
            valid_len = min(len(actual), n_horizons)
            
            for h_idx in range(valid_len):
                actual_val = actual[h_idx]
                q10_val = pred['q10'][h_idx]
                q50_val = pred['q50'][h_idx]
                q90_val = pred['q90'][h_idx]
                
                # Skip if any value is nan
                if np.isnan(actual_val) or np.isnan(q50_val):
                    continue
                
                # MAE/RMSE (using q50 as point forecast)
                error = abs(q50_val - actual_val)
                mae_by_h[h_idx].append(error)
                rmse_by_h[h_idx].append(error ** 2)
                
                # Directional accuracy
                pred_direction = np.sign(q50_val - current)
                actual_direction = np.sign(actual_val - current)
                direction_by_h[h_idx].append(pred_direction == actual_direction)
                
                # Quantile calibration
                if actual_val < q10_val:
                    q10_below_count += 1
                if actual_val > q90_val:
                    q90_above_count += 1
                if q10_val <= actual_val <= q90_val:
                    in_band_count += 1
                total_points += 1
                
                # Band width
                band_widths_by_h[h_idx].append(q90_val - q10_val)
            
            # Trajectory R²
            if valid_len >= 3:  # Need at least 3 points for meaningful R²
                actual_traj = actual[:valid_len]
                pred_traj = pred['q50'][:valid_len]
                
                # Filter out nans
                mask = ~(np.isnan(actual_traj) | np.isnan(pred_traj))
                if mask.sum() >= 3:
                    actual_clean = actual_traj[mask]
                    pred_clean = np.array(pred_traj)[mask]
                    
                    ss_res = np.sum((actual_clean - pred_clean) ** 2)
                    ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
                    
                    if ss_tot > 0:
                        r2 = 1 - (ss_res / ss_tot)
                        trajectory_errors.append(1 - r2)  # Convert to error
        
        # Aggregate metrics
        mae_by_horizon = {
            self.horizons[h]: np.mean(errs) if errs else np.nan
            for h, errs in mae_by_h.items()
        }
        
        rmse_by_horizon = {
            self.horizons[h]: np.sqrt(np.mean(errs)) if errs else np.nan
            for h, errs in rmse_by_h.items()
        }
        
        direction_by_horizon = {
            self.horizons[h]: np.mean(accs) if accs else np.nan
            for h, accs in direction_by_h.items()
        }
        
        band_width_by_horizon = {
            self.horizons[h]: np.mean(widths) if widths else np.nan
            for h, widths in band_widths_by_h.items()
        }
        
        # Overall directional accuracy
        all_directions = [acc for accs in direction_by_h.values() for acc in accs]
        directional_accuracy = np.mean(all_directions) if all_directions else 0.0
        
        # Quantile coverage
        q10_coverage = q10_below_count / total_points if total_points > 0 else 0.0
        q90_coverage = q90_above_count / total_points if total_points > 0 else 0.0
        band_coverage = in_band_count / total_points if total_points > 0 else 0.0
        
        # Mean band width
        all_widths = [w for widths in band_widths_by_h.values() for w in widths]
        mean_band_width = np.mean(all_widths) if all_widths else 0.0
        
        # Trajectory quality
        trajectory_r2 = 1 - np.mean(trajectory_errors) if trajectory_errors else 0.0
        mean_trajectory_error = np.mean(trajectory_errors) if trajectory_errors else np.nan
        
        return ProjectionMetrics(
            mae_by_horizon=mae_by_horizon,
            rmse_by_horizon=rmse_by_horizon,
            directional_accuracy=float(directional_accuracy),
            direction_by_horizon=direction_by_horizon,
            q10_coverage=float(q10_coverage),
            q90_coverage=float(q90_coverage),
            band_coverage=float(band_coverage),
            mean_band_width=float(mean_band_width),
            band_width_by_horizon=band_width_by_horizon,
            trajectory_r2=float(trajectory_r2),
            mean_trajectory_error=float(mean_trajectory_error),
            n_forecasts=n_forecasts,
            n_valid_horizons=sum(1 for h in mae_by_horizon.values() if not np.isnan(h))
        )
    
    def evaluate_stream(
        self,
        stream_name: str,
        stream_data: pd.DataFrame,
        min_history: int = 16
    ) -> Tuple[List[Dict], List[np.ndarray], List[float]]:
        """
        Generate predictions and collect actuals for one stream.
        
        Returns:
            (predictions, actuals, current_values)
        """
        if stream_name not in self.models:
            logger.error(f"Model not loaded for {stream_name}")
            return [], [], []
        
        model = self.models[stream_name]
        
        predictions = []
        actuals = []
        current_values = []
        
        # Iterate through data with sufficient history
        for i in range(min_history, len(stream_data) - 10):  # Need 10 bars future for actuals
            # Get history
            hist_window = stream_data.iloc[i - min_history:i]
            current_row = stream_data.iloc[i]
            future_window = stream_data.iloc[i + 1:i + 11]  # Next 10 bars
            
            # Extract stream values
            if stream_name not in hist_window.columns:
                continue
            
            stream_hist = hist_window[stream_name].values
            current_val = current_row[stream_name]
            actual_future = future_window[stream_name].values
            
            # Skip if insufficient future data or nans
            if len(actual_future) < 10 or np.isnan(current_val):
                continue
            
            # Get cross-stream context (all 4 other streams)
            cross_streams = {}
            other_stream_names = [s for s in self.stream_names if s != stream_name]
            for other in other_stream_names:
                if other in hist_window.columns:
                    cross_streams[other] = hist_window[other].values
            
            # Generate prediction
            try:
                pred = model.predict(stream_hist, current_val, cross_streams)
                predictions.append(pred)
                actuals.append(actual_future)
                current_values.append(current_val)
            except Exception as e:
                logger.warning(f"Prediction failed at index {i}: {e}")
                continue
        
        return predictions, actuals, current_values
    
    def run_baseline(self, dates: List[str]) -> Dict[str, AblationResult]:
        """Run baseline evaluation (full models, no ablations)."""
        logger.info(f"Running baseline evaluation on {len(dates)} dates")
        
        results = {}
        
        for stream_name in self.stream_names:
            if stream_name not in self.models:
                continue
            
            logger.info(f"\nEvaluating {stream_name}...")
            
            all_predictions = []
            all_actuals = []
            all_currents = []
            
            # Aggregate across all dates
            for date in dates:
                stream_data = self.load_stream_data(date)
                if stream_data is None:
                    continue
                
                preds, acts, currs = self.evaluate_stream(stream_name, stream_data)
                all_predictions.extend(preds)
                all_actuals.extend(acts)
                all_currents.extend(currs)
            
            # Compute metrics
            metrics = self.compute_metrics(all_predictions, all_actuals, all_currents)
            
            logger.info(f"  {stream_name}: {metrics.n_forecasts} forecasts")
            logger.info(f"    Directional Accuracy: {metrics.directional_accuracy:.1%}")
            logger.info(f"    MAE @ 30s: {metrics.mae_by_horizon.get(30, np.nan):.4f}")
            logger.info(f"    MAE @ 300s: {metrics.mae_by_horizon.get(300, np.nan):.4f}")
            logger.info(f"    Band Coverage: {metrics.band_coverage:.1%} (target ~80%)")
            
            results[stream_name] = AblationResult(
                config_name="baseline",
                stream_name=stream_name,
                metrics=metrics,
                metrics_by_regime={}
            )
        
        return results
    
    def save_results(self, results: Dict[str, AblationResult], output_name: str):
        """Save ablation results to JSON."""
        output_path = self.output_dir / f"{output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to serializable format
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'results': {
                stream: asdict(result) for stream, result in results.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"\n✅ Results saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Pentaview Projection Model Ablation Study")
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--date', type=str, help='Single date (YYYY-MM-DD)')
    parser.add_argument('--quick', action='store_true', help='Quick test on single day')
    parser.add_argument('--data-root', type=str, default='data', help='Data root directory')
    parser.add_argument('--output-dir', type=str, default='data/ml/ablation_pentaview', help='Output directory')
    
    args = parser.parse_args()
    
    # Determine date range
    if args.quick:
        dates = ['2025-12-18']
    elif args.date:
        dates = [args.date]
    elif args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, '%Y-%m-%d')
        end = datetime.strptime(args.end_date, '%Y-%m-%d')
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
    else:
        # Default: last 5 trading days
        dates = ['2025-12-12', '2025-12-15', '2025-12-16', '2025-12-17', '2025-12-18']
    
    logger.info("=" * 80)
    logger.info("PENTAVIEW PROJECTION MODEL ABLATION STUDY")
    logger.info("=" * 80)
    logger.info(f"Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    
    # Initialize study
    study = PentaviewAblationStudy(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir)
    )
    
    # Run baseline evaluation
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE EVALUATION")
    logger.info("=" * 80)
    
    baseline_results = study.run_baseline(dates)
    
    # Save results
    output_path = study.save_results(baseline_results, "pentaview_ablation_baseline")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    for stream_name, result in baseline_results.items():
        m = result.metrics
        logger.info(f"\n{stream_name.upper()}:")
        logger.info(f"  Forecasts: {m.n_forecasts:,}")
        logger.info(f"  Directional Accuracy: {m.directional_accuracy:.1%}")
        logger.info(f"  Trajectory R²: {m.trajectory_r2:.3f}")
        logger.info(f"  Quantile Calibration:")
        logger.info(f"    Below Q10: {m.q10_coverage:.1%} (target ~10%)")
        logger.info(f"    In Band [Q10-Q90]: {m.band_coverage:.1%} (target ~80%)")
        logger.info(f"    Above Q90: {m.q90_coverage:.1%} (target ~10%)")
        logger.info(f"  Mean Band Width: {m.mean_band_width:.4f}")
        logger.info(f"  Horizon Degradation:")
        for horizon in [30, 60, 120, 180, 240, 300]:
            mae = m.mae_by_horizon.get(horizon, np.nan)
            dir_acc = m.direction_by_horizon.get(horizon, np.nan)
            logger.info(f"    {horizon}s: MAE={mae:.4f}, Dir={dir_acc:.1%}")


if __name__ == "__main__":
    main()

