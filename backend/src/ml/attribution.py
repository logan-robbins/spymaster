"""Attribution system for explainability - IMPLEMENTATION_READY.md Section 11."""
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from src.ml.episode_vector import VECTOR_SECTIONS, get_feature_names

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    logger.warning("scikit-learn not installed. Outcome attribution will not work.")
    LogisticRegression = None


# Physics buckets per IMPLEMENTATION_READY.md Section 11.4
PHYSICS_BUCKETS = {
    'kinematics': [
        'velocity_', 'acceleration_', 'jerk_', 'momentum_trend_',
        'approach_velocity', 'approach_bars', 'approach_distance_atr',
        'predicted_accel', 'accel_residual', 'force_mass_ratio'
    ],
    'order_flow': [
        'ofi_', 'tape_imbalance', 'tape_velocity', 'sweep_detected',
        'tape_log_ratio', 'tape_log_total'
    ],
    'liquidity_barrier': [
        'barrier_', 'wall_ratio'
    ],
    'dealer_gamma': [
        'gamma_exposure', 'fuel_effect', 'gex_', 'net_gex_2strike'
    ],
    'context': [
        'level_kind', 'direction', 'minutes_since_open', 'level_stacking_',
        'dist_to_', 'prior_touches', 'attempt_index', 'atr'
    ]
}


def feature_matches_pattern(feature_name: str, patterns: List[str]) -> bool:
    """Check if feature name matches any pattern."""
    for pattern in patterns:
        if pattern in feature_name:
            return True
    return False


def compute_similarity_attribution(
    query_vector: np.ndarray,
    retrieved_vectors: np.ndarray,
    similarities: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Explain which features drove similarity for each neighbor.
    
    Per IMPLEMENTATION_READY.md Section 11.1:
    - For L2-normalized vectors using inner product: similarity = sum(q_i * r_i)
    - contribution of feature i = q_i * r_i
    - Weight by similarity across all neighbors
    
    Args:
        query_vector: Query vector [D]
        retrieved_vectors: Retrieved vectors [N × D]
        similarities: Similarity scores [N]
        feature_names: Feature names [D]
    
    Returns:
        Dict with top matching features and all attributions
    """
    if len(retrieved_vectors) == 0:
        return {'top_matching_features': [], 'all_attributions': {}}
    
    n_neighbors = len(retrieved_vectors)
    n_features = len(feature_names)
    
    # Normalize vectors
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
    
    # Aggregate feature contributions across all neighbors
    feature_contributions = np.zeros(n_features)
    
    for i in range(n_neighbors):
        neighbor_norm = retrieved_vectors[i] / (np.linalg.norm(retrieved_vectors[i]) + 1e-10)
        
        # Element-wise contribution to inner product
        contributions = query_norm * neighbor_norm
        
        # Weight by similarity
        feature_contributions += similarities[i] * contributions
    
    # Normalize by total similarity
    feature_contributions = feature_contributions / (similarities.sum() + 1e-10)
    
    # Create sorted list
    attribution_list = [(feature_names[i], float(feature_contributions[i]))
                       for i in range(n_features)]
    attribution_list.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return {
        'top_matching_features': attribution_list[:10],
        'all_attributions': dict(attribution_list)
    }


def compute_outcome_attribution(
    query_vector: np.ndarray,
    retrieved_vectors: np.ndarray,
    outcomes: np.ndarray,
    similarities: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Explain which features differentiate BREAK vs REJECT in the neighborhood.
    
    Per IMPLEMENTATION_READY.md Section 11.2:
    - Use weighted logistic regression as local surrogate
    - Positive coefficients favor BREAK
    - Negative coefficients favor REJECT
    
    Args:
        query_vector: Query vector [D]
        retrieved_vectors: Retrieved vectors [N × D]
        outcomes: Outcome labels [N]
        similarities: Similarity scores [N]
        feature_names: Feature names [D]
    
    Returns:
        Dict with top BREAK/REJECT drivers and all coefficients
    """
    if LogisticRegression is None:
        return {'top_break_drivers': [], 'top_reject_drivers': [], 'model_accuracy': 0}
    
    # Filter to BREAK and REJECT only
    mask = (outcomes == 'BREAK') | (outcomes == 'REJECT')
    if mask.sum() < 10:
        return {'top_break_drivers': [], 'top_reject_drivers': [], 'model_accuracy': 0}
    
    X = retrieved_vectors[mask]
    y = (outcomes[mask] == 'BREAK').astype(int)
    weights = similarities[mask]
    
    # Fit weighted logistic regression
    try:
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X, y, sample_weight=weights)
        
        # Extract coefficients
        coefficients = model.coef_[0]
        
        # Pair with feature names
        feature_importance = [(feature_names[i], float(coefficients[i]))
                             for i in range(len(feature_names))]
        
        # Positive coefficients favor BREAK
        break_drivers = [(f, c) for f, c in feature_importance if c > 0]
        break_drivers.sort(key=lambda x: x[1], reverse=True)
        
        # Negative coefficients favor REJECT
        reject_drivers = [(f, abs(c)) for f, c in feature_importance if c < 0]
        reject_drivers.sort(key=lambda x: x[1], reverse=True)
        
        accuracy = float(model.score(X, y, sample_weight=weights))
        
        return {
            'top_break_drivers': break_drivers[:10],
            'top_reject_drivers': reject_drivers[:10],
            'model_accuracy': accuracy,
            'all_coefficients': dict(feature_importance)
        }
    
    except Exception as e:
        logger.warning(f"Outcome attribution failed: {e}")
        return {'top_break_drivers': [], 'top_reject_drivers': [], 'model_accuracy': 0}


def compute_section_attribution(
    query_vector: np.ndarray,
    retrieved_vectors: np.ndarray,
    outcomes: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Identify which vector sections differentiate BREAK vs REJECT.
    
    Per IMPLEMENTATION_READY.md Section 11.3:
    - Compute centroids for BREAK and REJECT per section
    - Measure distance from query to each centroid
    - Identify which sections lean toward which outcome
    
    Args:
        query_vector: Query vector [111]
        retrieved_vectors: Retrieved vectors [N × 111]
        outcomes: Outcome labels [N]
    
    Returns:
        Dict with analysis per section
    """
    break_mask = outcomes == 'BREAK'
    reject_mask = outcomes == 'REJECT'
    
    if break_mask.sum() == 0 or reject_mask.sum() == 0:
        return {}
    
    break_vectors = retrieved_vectors[break_mask]
    reject_vectors = retrieved_vectors[reject_mask]
    
    section_analysis = {}
    
    for section_name, (start, end) in VECTOR_SECTIONS.items():
        # Compute centroids
        break_centroid = break_vectors[:, start:end].mean(axis=0)
        reject_centroid = reject_vectors[:, start:end].mean(axis=0)
        
        # Distance from query to each centroid
        query_section = query_vector[start:end]
        dist_to_break = float(np.linalg.norm(query_section - break_centroid))
        dist_to_reject = float(np.linalg.norm(query_section - reject_centroid))
        
        total_dist = dist_to_break + dist_to_reject
        
        section_analysis[section_name] = {
            'dist_to_break': dist_to_break,
            'dist_to_reject': dist_to_reject,
            'lean': 'BREAK' if dist_to_break < dist_to_reject else 'REJECT',
            'confidence': abs(dist_to_reject - dist_to_break) / (total_dist + 1e-6)
        }
    
    return section_analysis


def compute_physics_attribution(all_coefficients: Dict[str, float]) -> Dict[str, float]:
    """
    Aggregate feature attributions into physics buckets.
    
    Per IMPLEMENTATION_READY.md Section 11.4:
    - Group features by physics concept
    - Sum absolute coefficients per bucket
    - Normalize to sum to 1
    
    Args:
        all_coefficients: Feature coefficient dict from outcome attribution
    
    Returns:
        Dict with normalized scores per physics bucket
    """
    bucket_scores = {bucket: 0.0 for bucket in PHYSICS_BUCKETS}
    
    for feature, coef in all_coefficients.items():
        for bucket, patterns in PHYSICS_BUCKETS.items():
            if feature_matches_pattern(feature, patterns):
                bucket_scores[bucket] += abs(coef)
                break
    
    # Normalize to sum to 1
    total = sum(bucket_scores.values())
    if total > 0:
        bucket_scores = {k: v / total for k, v in bucket_scores.items()}
    
    return bucket_scores


def compute_attribution(
    query_vector: np.ndarray,
    retrieved_vectors: np.ndarray,
    outcomes: np.ndarray,
    similarities: np.ndarray = None,
    feature_names: List[str] = None
) -> Dict[str, Any]:
    """
    Compute all attribution analyses.
    
    Convenience function that calls all attribution methods.
    
    Args:
        query_vector: Query vector [111]
        retrieved_vectors: Retrieved vectors [N × 111]
        outcomes: Outcome labels [N]
        similarities: Similarity scores [N] (optional)
        feature_names: Feature names [111] (optional, will use default)
    
    Returns:
        Dict with all attribution analyses
    """
    if len(retrieved_vectors) == 0:
        return {
            'similarity': {},
            'outcome': {},
            'section': {},
            'physics': {}
        }
    
    if feature_names is None:
        feature_names = get_feature_names()
    
    if similarities is None:
        # Compute similarities as inner product (assumes normalized vectors)
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        vectors_norm = retrieved_vectors / (np.linalg.norm(retrieved_vectors, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(vectors_norm, query_norm)
    
    # Similarity attribution
    similarity_attr = compute_similarity_attribution(
        query_vector=query_vector,
        retrieved_vectors=retrieved_vectors,
        similarities=similarities,
        feature_names=feature_names
    )
    
    # Outcome attribution
    outcome_attr = compute_outcome_attribution(
        query_vector=query_vector,
        retrieved_vectors=retrieved_vectors,
        outcomes=outcomes,
        similarities=similarities,
        feature_names=feature_names
    )
    
    # Section attribution
    section_attr = compute_section_attribution(
        query_vector=query_vector,
        retrieved_vectors=retrieved_vectors,
        outcomes=outcomes
    )
    
    # Physics attribution
    physics_attr = {}
    if 'all_coefficients' in outcome_attr:
        physics_attr = compute_physics_attribution(outcome_attr['all_coefficients'])
    
    return {
        'similarity': similarity_attr,
        'outcome': outcome_attr,
        'section': section_attr,
        'physics': physics_attr
    }

