"""
Stream Projection Models - STREAMS.md Section 6.

Trains quantile polynomial regression models to forecast stream values
H=10 bars (20 minutes) ahead with uncertainty bands (q10/q50/q90).

Model outputs polynomial coefficients {a1, a2, a3} for each quantile,
ensuring smooth projected curves with direct TA interpretation:
- a1 = slope
- a2 = curvature  
- a3 = jerk
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

logger = logging.getLogger(__name__)


class MultiCoefficientPredictor:
    """
    Wrapper for 3 separate quantile regression models (a1, a2, a3).
    
    Provides sklearn-like interface but internally uses 3 models.
    This is needed because MultiOutputRegressor doesn't properly handle
    quantile loss with different quantile parameters.
    """
    
    def __init__(self, models: Dict[str, HistGradientBoostingRegressor]):
        """
        Args:
            models: {'a1': model, 'a2': model, 'a3': model}
        """
        self.models = models
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict all 3 coefficients.
        
        Args:
            X: Feature matrix [N, F]
        
        Returns:
            Coefficient matrix [N, 3] where columns are [a1, a2, a3]
        """
        a1_pred = self.models['a1'].predict(X)
        a2_pred = self.models['a2'].predict(X)
        a3_pred = self.models['a3'].predict(X)
        
        return np.column_stack([a1_pred, a2_pred, a3_pred])


@dataclass
class ProjectionConfig:
    """Configuration for stream projection models."""
    lookback_bars: int = 20  # L: history window
    horizon_bars: int = 10   # H: forecast window (20 minutes @ 2-min bars)
    polynomial_degree: int = 3  # Cubic polynomial
    quantiles: List[float] = None  # q10, q50, q90
    
    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.10, 0.50, 0.90]


@dataclass
class ProjectionCoefficients:
    """Polynomial coefficients for one stream, all quantiles."""
    q10: Dict[str, float]  # {a1, a2, a3}
    q50: Dict[str, float]
    q90: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            'q10': self.q10,
            'q50': self.q50,
            'q90': self.q90
        }


def build_polynomial_path(
    sigma_current: float,
    coeffs: Dict[str, float],
    horizon_bars: int = 10
) -> np.ndarray:
    """
    Generate smooth polynomial forecast curve.
    
    Per STREAMS.md Section 6.2:
    ŷ(h) = σ[t] + a1*h + 0.5*a2*h^2 + (1/6)*a3*h^3
    
    Args:
        sigma_current: Current stream value at time t
        coeffs: Polynomial coefficients {a1, a2, a3}
        horizon_bars: Number of future bars (H)
    
    Returns:
        Array of H+1 values [σ[t], σ[t+1], ..., σ[t+H]]
    """
    curve = np.zeros(horizon_bars + 1)
    curve[0] = sigma_current
    
    a1 = coeffs.get('a1', 0.0)
    a2 = coeffs.get('a2', 0.0)
    a3 = coeffs.get('a3', 0.0)
    
    for h in range(1, horizon_bars + 1):
        y = sigma_current + a1 * h + 0.5 * a2 * h**2 + (1.0/6.0) * a3 * h**3
        curve[h] = np.clip(y, -1.0, 1.0)
    
    return curve


def compute_projected_slope(coeffs: Dict[str, float], horizon: int) -> float:
    """
    Compute projected slope at horizon h.
    
    d/dh ŷ(h) = a1 + a2*h + 0.5*a3*h^2
    
    Args:
        coeffs: Polynomial coefficients
        horizon: Time step h
    
    Returns:
        Projected slope
    """
    a1 = coeffs.get('a1', 0.0)
    a2 = coeffs.get('a2', 0.0)
    a3 = coeffs.get('a3', 0.0)
    
    return a1 + a2 * horizon + 0.5 * a3 * horizon**2


def compute_projected_curvature(coeffs: Dict[str, float], horizon: int) -> float:
    """
    Compute projected curvature at horizon h.
    
    d^2/dh^2 ŷ(h) = a2 + a3*h
    
    Args:
        coeffs: Polynomial coefficients
        horizon: Time step h
    
    Returns:
        Projected curvature
    """
    a2 = coeffs.get('a2', 0.0)
    a3 = coeffs.get('a3', 0.0)
    
    return a2 + a3 * horizon


def fit_polynomial_coefficients(
    future_values: np.ndarray,
    current_value: float,
    horizon_bars: int = 10
) -> Dict[str, float]:
    """
    Fit polynomial coefficients to observed future trajectory.
    
    Given observed future values y[1..H], find coefficients {a1, a2, a3} that
    minimize squared error to polynomial curve starting from current_value.
    
    Args:
        future_values: Observed future stream values [H]
        current_value: Current stream value (anchor point)
        horizon_bars: Forecast horizon
    
    Returns:
        Fitted coefficients {a1, a2, a3}
    """
    if len(future_values) < horizon_bars:
        return {'a1': 0.0, 'a2': 0.0, 'a3': 0.0}
    
    # Build design matrix for polynomial regression
    # y[h] - y[0] = a1*h + 0.5*a2*h^2 + (1/6)*a3*h^3
    H = min(horizon_bars, len(future_values))
    X = np.zeros((H, 3))
    y_target = np.zeros(H)
    
    for h in range(1, H + 1):
        X[h-1, 0] = h  # a1 coefficient
        X[h-1, 1] = 0.5 * h**2  # a2 coefficient
        X[h-1, 2] = (1.0/6.0) * h**3  # a3 coefficient
        y_target[h-1] = future_values[h-1] - current_value
    
    # Solve least squares: X @ [a1, a2, a3] = y_target
    try:
        coeffs = np.linalg.lstsq(X, y_target, rcond=None)[0]
        return {
            'a1': float(coeffs[0]),
            'a2': float(coeffs[1]),
            'a3': float(coeffs[2])
        }
    except np.linalg.LinAlgError:
        return {'a1': 0.0, 'a2': 0.0, 'a3': 0.0}


class StreamProjector:
    """
    Quantile polynomial projection model for stream forecasting.
    
    Per STREAMS.md Section 6:
    - Predicts polynomial coefficients {a1, a2, a3} for q10/q50/q90
    - Guarantees smooth curves (no jagged forecasts)
    - Direct TA interpretation: slope/curvature/jerk
    """
    
    def __init__(
        self,
        stream_name: str,
        config: ProjectionConfig = None
    ):
        """
        Initialize projector.
        
        Args:
            stream_name: Name of stream ('pressure', 'flow', 'barrier', etc.)
            config: Projection configuration
        """
        self.stream_name = stream_name
        self.config = config or ProjectionConfig()
        
        # One model per quantile, each predicting 3 coefficients (a1, a2, a3)
        self.models: Dict[str, Any] = {}  # {q10, q50, q90} -> model
        self.feature_names: List[str] = []
        self.training_history: Dict[str, Any] = {}
    
    def _build_features(
        self,
        stream_hist: np.ndarray,
        slope_hist: Optional[np.ndarray] = None,
        cross_streams: Optional[Dict[str, np.ndarray]] = None,
        static_features: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Build feature vector for projection model.
        
        Per STREAMS.md Section 6.3:
        - Stream history (L bars)
        - Derivatives history (slope, curvature)
        - Cross-stream context (short)
        - Static features (level_kind, direction, time_bucket, etc.)
        
        Args:
            stream_hist: Stream values [L]
            slope_hist: Slope history [L]
            cross_streams: Other stream histories (short, e.g., last 5 bars)
            static_features: Static context features
        
        Returns:
            Feature vector [F]
        """
        features = []
        
        # Primary stream history (L bars)
        if len(stream_hist) < self.config.lookback_bars:
            # Pad with zeros if insufficient history
            padded = np.zeros(self.config.lookback_bars)
            padded[-len(stream_hist):] = stream_hist
            stream_hist = padded
        else:
            stream_hist = stream_hist[-self.config.lookback_bars:]
        
        features.extend(stream_hist)
        
        # Slope history (if available)
        if slope_hist is not None:
            if len(slope_hist) < self.config.lookback_bars:
                padded = np.zeros(self.config.lookback_bars)
                padded[-len(slope_hist):] = slope_hist
                slope_hist = padded
            else:
                slope_hist = slope_hist[-self.config.lookback_bars:]
            features.extend(slope_hist)
        
        # Cross-stream context (last 5 bars for brevity)
        if cross_streams:
            for stream_name in sorted(cross_streams.keys()):
                hist = cross_streams[stream_name]
                if len(hist) >= 5:
                    features.extend(hist[-5:])
                else:
                    padded = np.zeros(5)
                    padded[-len(hist):] = hist
                    features.extend(padded)
        
        # Static features
        if static_features:
            for key in sorted(static_features.keys()):
                features.append(static_features[key])
        
        return np.array(features, dtype=np.float32)
    
    def fit(
        self,
        training_samples: List[Dict[str, Any]],
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6
    ) -> Dict[str, float]:
        """
        Train quantile polynomial projection models.
        
        Args:
            training_samples: List of training samples (see _build_features for format)
            max_iter: Max boosting iterations
            learning_rate: Learning rate for gradient boosting
            max_depth: Max tree depth
        
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.stream_name} projector on {len(training_samples)} samples")
        
        if not training_samples:
            raise ValueError("No training samples provided")
        
        # Build feature matrix and target coefficients
        X_list = []
        y_coeffs_a1 = []
        y_coeffs_a2 = []
        y_coeffs_a3 = []
        sample_weights = []
        
        for sample in training_samples:
            # Build features
            X = self._build_features(
                stream_hist=sample['stream_hist'],
                slope_hist=sample.get('slope_hist'),
                cross_streams=sample.get('cross_streams'),
                static_features=sample.get('static_features')
            )
            X_list.append(X)
            
            # Fit polynomial to actual future trajectory
            # All quantiles will be trained on the same targets, 
            # but with quantile loss they'll learn different predictions
            future_target = sample['future_target']  # [H]
            current_value = sample['current_value']
            
            coeffs = fit_polynomial_coefficients(
                future_values=future_target,
                current_value=current_value,
                horizon_bars=self.config.horizon_bars
            )
            
            # Store coefficients separately (for 3 separate quantile models)
            y_coeffs_a1.append(coeffs['a1'])
            y_coeffs_a2.append(coeffs['a2'])
            y_coeffs_a3.append(coeffs['a3'])
            
            # Sample weight (from setup quality)
            setup_weight = sample.get('setup_weight', 1.0)
            sample_weights.append(setup_weight)
        
        X = np.vstack(X_list)
        y_a1 = np.array(y_coeffs_a1)
        y_a2 = np.array(y_coeffs_a2)
        y_a3 = np.array(y_coeffs_a3)
        sample_weights = np.array(sample_weights)
        
        # Store feature names (for diagnostics)
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Train 3 models per quantile (one for each coefficient)
        # Each uses quantile-specific loss function
        metrics = {}
        
        for q in self.config.quantiles:
            q_str = f"q{int(q*100):02d}"
            logger.info(f"  Training {q_str} model (quantile={q})...")
            
            # Train 3 separate models: one for a1, one for a2, one for a3
            # Each with quantile loss
            models_for_quantile = {}
            r2_scores = []
            
            for coeff_idx, (coeff_name, y_target) in enumerate([('a1', y_a1), ('a2', y_a2), ('a3', y_a3)]):
                # Use quantile regression
                model = HistGradientBoostingRegressor(
                    loss='quantile',
                    quantile=q,
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42
                )
                
                model.fit(X, y_target, sample_weight=sample_weights)
                models_for_quantile[coeff_name] = model
                
                # Compute training R^2
                y_pred = model.predict(X)
                ss_res = np.sum((y_target - y_pred)**2)
                ss_tot = np.sum((y_target - np.mean(y_target))**2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                r2_scores.append(r2)
                
                metrics[f'{q_str}_r2_{coeff_name}'] = float(r2)
            
            # Wrap models in a MultiCoefficientPredictor for interface compatibility
            self.models[q_str] = MultiCoefficientPredictor(models_for_quantile)
            
            metrics[f'{q_str}_r2_mean'] = np.mean(r2_scores)
            
            logger.info(f"    R^2 scores: a1={r2_scores[0]:.3f}, a2={r2_scores[1]:.3f}, a3={r2_scores[2]:.3f}")
        
        self.training_history = {
            'n_samples': len(training_samples),
            'n_features': X.shape[1],
            'metrics': metrics
        }
        
        return metrics
    
    def predict(
        self,
        stream_hist: np.ndarray,
        current_value: float,
        slope_hist: Optional[np.ndarray] = None,
        cross_streams: Optional[Dict[str, np.ndarray]] = None,
        static_features: Optional[Dict[str, float]] = None
    ) -> ProjectionCoefficients:
        """
        Predict polynomial coefficients for all quantiles.
        
        Args:
            stream_hist: Stream history [L]
            current_value: Current stream value
            slope_hist: Slope history [L]
            cross_streams: Cross-stream context
            static_features: Static features
        
        Returns:
            Projection coefficients (q10/q50/q90)
        """
        if not self.models:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Build features
        X = self._build_features(
            stream_hist=stream_hist,
            slope_hist=slope_hist,
            cross_streams=cross_streams,
            static_features=static_features
        ).reshape(1, -1)
        
        # Predict coefficients for each quantile
        coeffs_dict = {}
        for q in self.config.quantiles:
            q_str = f"q{int(q*100):02d}"
            if q_str not in self.models:
                continue
            
            pred = self.models[q_str].predict(X)[0]  # [3]
            coeffs_dict[q_str] = {
                'a1': float(pred[0]),
                'a2': float(pred[1]),
                'a3': float(pred[2])
            }
        
        return ProjectionCoefficients(
            q10=coeffs_dict.get('q10', {'a1': 0.0, 'a2': 0.0, 'a3': 0.0}),
            q50=coeffs_dict.get('q50', {'a1': 0.0, 'a2': 0.0, 'a3': 0.0}),
            q90=coeffs_dict.get('q90', {'a1': 0.0, 'a2': 0.0, 'a3': 0.0})
        )
    
    def save(self, output_path: Path) -> None:
        """Save trained model to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        bundle = {
            'stream_name': self.stream_name,
            'config': self.config,
            'models': self.models,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        joblib.dump(bundle, output_path)
        logger.info(f"Saved {self.stream_name} projector to {output_path}")
    
    @classmethod
    def load(cls, model_path: Path) -> 'StreamProjector':
        """Load trained model from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        bundle = joblib.load(model_path)
        
        projector = cls(
            stream_name=bundle['stream_name'],
            config=bundle['config']
        )
        projector.models = bundle['models']
        projector.feature_names = bundle['feature_names']
        projector.training_history = bundle['training_history']
        
        logger.info(f"Loaded {projector.stream_name} projector from {model_path}")
        return projector


def project_stream_curves(
    projector: StreamProjector,
    stream_hist: np.ndarray,
    current_value: float,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Generate projection curves with uncertainty bands.
    
    Per STREAMS.md Section 11.2:
    Returns 11-point curves (current + 10 future) for q10/q50/q90.
    
    Args:
        projector: Trained projection model
        stream_hist: Stream history
        current_value: Current stream value
        **kwargs: Additional features (slope_hist, cross_streams, static_features)
    
    Returns:
        Dictionary with 'q10', 'q50', 'q90' curves [H+1]
    """
    coeffs = projector.predict(
        stream_hist=stream_hist,
        current_value=current_value,
        **kwargs
    )
    
    H = projector.config.horizon_bars
    
    curves = {
        'q10': build_polynomial_path(current_value, coeffs.q10, H),
        'q50': build_polynomial_path(current_value, coeffs.q50, H),
        'q90': build_polynomial_path(current_value, coeffs.q90, H)
    }
    
    return curves

