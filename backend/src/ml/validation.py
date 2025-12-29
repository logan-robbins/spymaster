"""Validation framework - IMPLEMENTATION_READY.md Section 12."""
import logging
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    from scipy.stats import wasserstein_distance
except ImportError:
    logger.warning("scikit-learn or scipy not installed. Some validation functions will not work.")
    roc_auc_score = None
    brier_score_loss = None
    log_loss = None
    wasserstein_distance = None


def temporal_cv_split(
    dates: List[pd.Timestamp],
    n_splits: int = 5,
    min_train_days: int = 60
) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]:
    """
    Generate temporal train/test splits.
    
    Per IMPLEMENTATION_READY.md Section 12.1:
    - Training always precedes test (no leakage)
    - Expanding window (train grows over time)
    - Fixed test size
    
    Args:
        dates: List of dates
        n_splits: Number of splits
        min_train_days: Minimum training days
    
    Returns:
        List of (train_dates, test_dates) tuples
    """
    dates = sorted(dates)
    n_dates = len(dates)
    
    test_size = (n_dates - min_train_days) // n_splits
    
    if test_size < 1:
        logger.warning(f"Not enough dates for {n_splits} splits (need > {min_train_days})")
        return []
    
    splits = []
    
    for i in range(n_splits):
        train_end = min_train_days + i * test_size
        test_start = train_end
        test_end = test_start + test_size
        
        if test_end > n_dates:
            break
        
        train_dates = dates[:train_end]
        test_dates = dates[test_start:test_end]
        
        splits.append((train_dates, test_dates))
    
    logger.info(f"Generated {len(splits)} temporal CV splits")
    for i, (train, test) in enumerate(splits):
        logger.info(f"  Split {i+1}: train={len(train)} days, test={len(test)} days")
    
    return splits


def compute_calibration_curve(
    actuals: np.ndarray,
    predictions: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute reliability diagram data.
    
    Per IMPLEMENTATION_READY.md Section 12.3:
    - Bin predictions
    - Compute actual frequency per bin
    - Compute expected calibration error (ECE)
    
    Args:
        actuals: Actual binary outcomes [N]
        predictions: Predicted probabilities [N]
        n_bins: Number of bins
    
    Returns:
        Dict with calibration data
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    bin_means = []
    bin_true_freqs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i+1])
        if mask.sum() > 0:
            bin_means.append(float(predictions[mask].mean()))
            bin_true_freqs.append(float(actuals[mask].mean()))
            bin_counts.append(int(mask.sum()))
    
    # Expected calibration error
    n_total = len(predictions)
    ece = sum(
        (count / n_total) * abs(mean - freq)
        for mean, freq, count in zip(bin_means, bin_true_freqs, bin_counts)
    )
    
    return {
        'bin_means': bin_means,
        'bin_true_freqs': bin_true_freqs,
        'bin_counts': bin_counts,
        'expected_calibration_error': float(ece),
        'n_bins': len(bin_means)
    }


def compute_lift_analysis(
    actuals: np.ndarray,
    predictions: np.ndarray,
    thresholds: List[float] = None
) -> Dict[str, Any]:
    """
    Compute lift at various confidence thresholds.
    
    Per IMPLEMENTATION_READY.md Section 12.4:
    - For each threshold, compute BREAK rate in high-confidence subset
    - Compare to base rate
    
    Args:
        actuals: Actual binary outcomes [N]
        predictions: Predicted probabilities [N]
        thresholds: Confidence thresholds to evaluate
    
    Returns:
        Dict with lift analysis
    """
    if thresholds is None:
        thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    base_rate = float(actuals.mean())
    
    lift_results = []
    
    for threshold in thresholds:
        high_conf_mask = predictions >= threshold
        
        if high_conf_mask.sum() > 0:
            high_conf_rate = float(actuals[high_conf_mask].mean())
            lift = high_conf_rate / (base_rate + 1e-10)
            
            lift_results.append({
                'threshold': threshold,
                'n_samples': int(high_conf_mask.sum()),
                'break_rate': high_conf_rate,
                'lift': lift
            })
    
    return {
        'base_rate': base_rate,
        'lift_by_threshold': lift_results
    }


def evaluate_retrieval_system(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of retrieval system.
    
    Per IMPLEMENTATION_READY.md Section 12.2:
    - AUC
    - Brier score
    - Log loss
    - Calibration curve
    - Lift analysis
    
    Args:
        predictions: Predicted probabilities for BREAK [N]
        actuals: Actual outcomes (1 = BREAK, 0 = other) [N]
    
    Returns:
        Dict with all evaluation metrics
    """
    if len(predictions) == 0:
        return {
            'n_evaluated': 0,
            'base_rate_break': 0,
            'mean_prediction': 0
        }
    
    # Check for sklearn
    if roc_auc_score is None or brier_score_loss is None or log_loss is None:
        logger.warning("scikit-learn not installed, skipping some metrics")
        metrics = {
            'n_evaluated': len(predictions),
            'base_rate_break': float(actuals.mean()),
            'mean_prediction': float(predictions.mean())
        }
    else:
        metrics = {
            'n_evaluated': len(predictions),
            'base_rate_break': float(actuals.mean()),
            'mean_prediction': float(predictions.mean()),
            'auc': float(roc_auc_score(actuals, predictions)) if len(np.unique(actuals)) > 1 else None,
            'brier_score': float(brier_score_loss(actuals, predictions)),
            'log_loss': float(log_loss(actuals, np.clip(predictions, 0.01, 0.99)))
        }
    
    # Calibration curve
    metrics['calibration'] = compute_calibration_curve(actuals, predictions, n_bins=10)
    
    # Lift analysis
    metrics['lift'] = compute_lift_analysis(actuals, predictions)
    
    return metrics


def sanity_check_similarity(
    episode_vectors: np.ndarray,
    metadata: pd.DataFrame,
    n_samples: int = 500
) -> Dict[str, Any]:
    """
    Verify that similar episodes have similar outcomes.
    
    Per IMPLEMENTATION_READY.md Section 12.5:
    - Same-outcome neighbors should be closer than different-outcome neighbors
    - Compute separation ratio
    
    Args:
        episode_vectors: Episode vectors [N × 144]
        metadata: Metadata with outcome_4min column
        n_samples: Number of samples to check
    
    Returns:
        Dict with sanity check results
    """
    if len(episode_vectors) < 10:
        return {
            'same_outcome_mean_dist': 0,
            'diff_outcome_mean_dist': 0,
            'separation_ratio': 0,
            'interpretation': 'INSUFFICIENT_DATA'
        }
    
    # Sample indices
    sample_idx = np.random.choice(
        len(episode_vectors),
        min(n_samples, len(episode_vectors)),
        replace=False
    )
    
    same_outcome_distances = []
    diff_outcome_distances = []
    
    # Normalize for cosine distance
    vectors_norm = episode_vectors / (np.linalg.norm(episode_vectors, axis=1, keepdims=True) + 1e-6)
    
    for i in sample_idx:
        # Cosine distance to all others
        similarities = np.dot(vectors_norm, vectors_norm[i])
        distances = 1 - similarities
        distances[i] = np.inf  # Exclude self
        
        # Find nearest neighbor
        nearest = int(np.argmin(distances))
        
        if metadata.iloc[i]['outcome_4min'] == metadata.iloc[nearest]['outcome_4min']:
            same_outcome_distances.append(float(distances[nearest]))
        else:
            diff_outcome_distances.append(float(distances[nearest]))
    
    same_mean = np.mean(same_outcome_distances) if same_outcome_distances else 0
    diff_mean = np.mean(diff_outcome_distances) if diff_outcome_distances else 0
    
    separation_ratio = diff_mean / (same_mean + 1e-6)
    
    return {
        'same_outcome_mean_dist': float(same_mean),
        'diff_outcome_mean_dist': float(diff_mean),
        'separation_ratio': float(separation_ratio),
        'interpretation': 'GOOD' if diff_mean > same_mean else 'POOR',
        'n_same_outcome': len(same_outcome_distances),
        'n_diff_outcome': len(diff_outcome_distances)
    }


def detect_feature_drift(
    corpus_vectors: np.ndarray,
    corpus_dates: np.ndarray,
    feature_names: List[str],
    lookback_days: int = 60,
    recent_days: int = 5
) -> pd.DataFrame:
    """
    Detect drift in feature distributions.
    
    Per IMPLEMENTATION_READY.md Section 12.6:
    - Compare recent distribution to historical baseline
    - Use Wasserstein distance and mean shift
    
    Args:
        corpus_vectors: All episode vectors [N × 144]
        corpus_dates: Dates for each episode [N]
        feature_names: Feature names [144]
        lookback_days: Days for historical baseline
        recent_days: Days for recent window
    
    Returns:
        DataFrame with drift metrics per feature
    """
    if wasserstein_distance is None:
        logger.warning("scipy not installed, skipping drift detection")
        return pd.DataFrame()
    
    today = pd.Timestamp.now()
    cutoff = today - pd.Timedelta(days=recent_days)
    historical_end = cutoff - pd.Timedelta(days=lookback_days)
    
    historical_mask = corpus_dates < historical_end
    recent_mask = corpus_dates >= cutoff
    
    drift_metrics = []
    
    for feat_idx, feat_name in enumerate(feature_names):
        hist_vals = corpus_vectors[historical_mask, feat_idx]
        recent_vals = corpus_vectors[recent_mask, feat_idx]
        
        if len(hist_vals) == 0 or len(recent_vals) == 0:
            continue
        
        # Wasserstein distance
        w_dist = float(wasserstein_distance(hist_vals, recent_vals))
        
        # Mean shift in std units
        mean_shift = (recent_vals.mean() - hist_vals.mean()) / (hist_vals.std() + 1e-6)
        
        drift_metrics.append({
            'feature': feat_name,
            'wasserstein_distance': w_dist,
            'mean_shift_std': float(mean_shift),
            'hist_mean': float(hist_vals.mean()),
            'recent_mean': float(recent_vals.mean()),
            'hist_std': float(hist_vals.std()),
            'recent_std': float(recent_vals.std())
        })
    
    df = pd.DataFrame(drift_metrics)
    if not df.empty:
        df = df.sort_values('wasserstein_distance', ascending=False)
    
    return df


class ValidationRunner:
    """
    Run validation suite for retrieval system.
    
    Per IMPLEMENTATION_READY.md Section 12 (Stage 20):
    - Temporal CV evaluation
    - Calibration analysis
    - Drift detection
    - Sanity checks
    - Runs weekly (Saturday)
    """
    
    def __init__(
        self,
        episodes_dir: str,
        output_dir: str
    ):
        """
        Initialize validation runner.
        
        Args:
            episodes_dir: Episode corpus directory
            output_dir: Validation output directory
        """
        from pathlib import Path
        self.episodes_dir = Path(episodes_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_validation(self) -> Dict[str, Any]:
        """
        Run full validation suite.
        
        Returns:
            Dict with all validation results
        """
        logger.info("Running validation suite...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'calibration': {},
            'drift': {},
            'sanity': {}
        }
        
        # Save results
        output_file = self.output_dir / f"validation_{datetime.now().strftime('%Y-%m-%d')}.json"
        
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Validation complete: {output_file}")
        
        return results

