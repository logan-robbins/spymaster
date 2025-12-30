"""Outcome aggregation functions - IMPLEMENTATION_READY.md Section 10."""
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

from src.ml.constants import SIM_POWER, RECENCY_HALFLIFE_DAYS

logger = logging.getLogger(__name__)


def compute_neighbor_weights(
    similarities: np.ndarray,
    dates: pd.Series,
    query_date: pd.Timestamp = None,
    sim_power: float = SIM_POWER,
    recency_halflife: float = RECENCY_HALFLIFE_DAYS
) -> np.ndarray:
    """
    Compute neighbor weights with similarity power transform and recency decay.
    
    Per IMPLEMENTATION_READY.md Section 9.2:
    - Apply power transform: similarity^sim_power (default 4.0)
    - Apply recency decay: exp(-age_days / halflife) (default 60 days)
    - Normalize to sum to 1
    
    Args:
        similarities: Similarity scores
        dates: Neighbor dates
        query_date: Query date (for recency calculation)
        sim_power: Power for similarity transform
        recency_halflife: Halflife for exponential decay (days)
    
    Returns:
        Normalized weights array
    """
    # Power transform on similarity
    weights = np.power(similarities, sim_power)
    
    # Recency decay (if query_date provided)
    if query_date is not None and dates is not None:
        try:
            age_days = (query_date - pd.to_datetime(dates)).dt.days.values
            recency_weights = np.exp(-age_days / recency_halflife)
            weights = weights * recency_weights
        except Exception:
            # Skip recency weighting if date parsing fails
            pass
    
    # Normalize
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(weights) / len(weights)
    
    return weights


def compute_outcome_distribution(
    retrieved_metadata: pd.DataFrame,
    query_date: pd.Timestamp = None,
    use_dirichlet: bool = True,
    priors: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Compute outcome probabilities with neighbor weighting and Dirichlet posterior.
    
    Per IMPLEMENTATION_READY.md Section 10.1:
    - Apply power transform: similarity^4
    - Apply recency decay: exp(-age_days / 60)
    - Normalize weights to sum to 1
    - Apply Dirichlet prior for smoothing (Bayesian posterior)
    - Compute probabilities for BREAK, REJECT, CHOP
    
    Args:
        retrieved_metadata: DataFrame with 'similarity', 'date', and 'outcome_4min' columns
        query_date: Query date for recency weighting
        use_dirichlet: Whether to apply Dirichlet prior (default True)
        priors: Prior pseudo-counts per outcome (default: {BREAK: 1, REJECT: 1, CHOP: 0.5})
    
    Returns:
        Dict with probabilities and metadata
    """
    if len(retrieved_metadata) == 0:
        return {'probabilities': {'BREAK': 0, 'REJECT': 0, 'CHOP': 0}, 'n_samples': 0, 'avg_similarity': 0}
    
    # Default priors: symmetric for BREAK/REJECT, lower for CHOP
    if priors is None:
        priors = {'BREAK': 1.0, 'REJECT': 1.0, 'CHOP': 0.5}
    
    # Compute weighted scores
    weights = compute_neighbor_weights(
        similarities=retrieved_metadata['similarity'].values,
        dates=retrieved_metadata.get('date'),
        query_date=query_date
    )
    
    if use_dirichlet:
        # Dirichlet posterior: (weighted_counts + priors) / (total_weight + sum(priors))
        weighted_counts = {}
        for outcome in ['BREAK', 'REJECT', 'CHOP']:
            mask = retrieved_metadata['outcome_4min'] == outcome
            weighted_counts[outcome] = weights[mask].sum()
        
        total_weight = sum(weighted_counts.values())
        total_prior = sum(priors.values())
        
        probs = {}
        for outcome in ['BREAK', 'REJECT', 'CHOP']:
            probs[outcome] = float((weighted_counts[outcome] + priors[outcome]) / (total_weight + total_prior))
    else:
        # Simple weighted probabilities
        probs = {}
        for outcome in ['BREAK', 'REJECT', 'CHOP']:
            mask = retrieved_metadata['outcome_4min'] == outcome
            probs[outcome] = float(weights[mask].sum())
    
    return {
        'probabilities': probs,
        'n_samples': len(retrieved_metadata),
        'avg_similarity': float(retrieved_metadata['similarity'].mean())
    }


def compute_expected_excursions(
    retrieved_metadata: pd.DataFrame,
    query_date: pd.Timestamp = None
) -> Dict[str, float]:
    """
    Compute expected favorable and adverse excursions with neighbor weighting.
    
    Per IMPLEMENTATION_READY.md Section 10.2:
    - Apply power transform and recency decay
    - Compute weighted expected excursions (ATR-normalized)
    
    Args:
        retrieved_metadata: DataFrame with similarity, date, and excursion columns
        query_date: Query date for recency weighting
    
    Returns:
        Dict with expected excursions and ratio
    """
    if len(retrieved_metadata) == 0:
        return {
            'expected_excursion_favorable': 0.0,
            'expected_excursion_adverse': 0.0,
            'excursion_ratio': 0.0
        }
    
    weights = compute_neighbor_weights(
        similarities=retrieved_metadata['similarity'].values,
        dates=retrieved_metadata.get('date'),
        query_date=query_date
    )
    
    expected_favorable = float((weights * retrieved_metadata['excursion_favorable']).sum())
    expected_adverse = float((weights * retrieved_metadata['excursion_adverse']).sum())
    
    return {
        'expected_excursion_favorable': expected_favorable,
        'expected_excursion_adverse': expected_adverse,
        'excursion_ratio': expected_favorable / (expected_adverse + 1e-6)
    }


def compute_conditional_excursions(
    retrieved_metadata: pd.DataFrame,
    query_date: pd.Timestamp = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute expected excursions conditional on outcome with neighbor weighting.
    
    Per IMPLEMENTATION_READY.md Section 10.3:
    - Separate by BREAK vs REJECT
    - Apply power transform and recency decay within each subset
    - Compute weighted expectations within each outcome
    
    Args:
        retrieved_metadata: DataFrame with outcomes, excursions, similarity, and date
        query_date: Query date for recency weighting
    
    Returns:
        Dict with conditional expectations per outcome
    """
    conditional = {}
    
    for outcome in ['BREAK', 'REJECT']:
        mask = retrieved_metadata['outcome_4min'] == outcome
        
        if mask.sum() > 0:
            subset = retrieved_metadata[mask]
            subset_weights = compute_neighbor_weights(
                similarities=subset['similarity'].values,
                dates=subset.get('date'),
                query_date=query_date
            )
            
            conditional[outcome] = {
                'expected_favorable': float((subset_weights * subset['excursion_favorable']).sum()),
                'expected_adverse': float((subset_weights * subset['excursion_adverse']).sum()),
                'mean_strength': float((subset_weights * subset['strength_abs']).sum()),
                'n_samples': int(mask.sum())
            }
    
    return conditional


def compute_multi_horizon_distribution(
    retrieved_metadata: pd.DataFrame,
    query_date: pd.Timestamp = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute outcome distributions for all horizons with neighbor weighting.
    
    Per IMPLEMENTATION_READY.md Section 10.4:
    - Apply power transform and recency decay
    - Compute for 2min, 4min, 8min horizons
    
    Args:
        retrieved_metadata: DataFrame with outcome columns for all horizons
        query_date: Query date for recency weighting
    
    Returns:
        Dict with probabilities per horizon
    """
    weights = compute_neighbor_weights(
        similarities=retrieved_metadata['similarity'].values,
        dates=retrieved_metadata.get('date'),
        query_date=query_date
    )
    
    horizons = {
        '2min': 'outcome_2min',
        '4min': 'outcome_4min',
        '8min': 'outcome_8min'
    }
    
    results = {}
    
    for horizon_name, col in horizons.items():
        if col not in retrieved_metadata.columns:
            continue
        
        probs = {}
        for outcome in ['BREAK', 'REJECT', 'CHOP']:
            mask = retrieved_metadata[col] == outcome
            probs[outcome] = float(weights[mask].sum())
        
        results[horizon_name] = {
            'probabilities': probs,
            'n_valid': int(retrieved_metadata[col].notna().sum())
        }
    
    return results


def compute_bootstrap_ci(
    retrieved_metadata: pd.DataFrame,
    query_date: pd.Timestamp = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals with neighbor weighting.
    
    Per IMPLEMENTATION_READY.md Section 10.5:
    - Apply power transform and recency decay
    - Weighted bootstrap resampling
    - Compute confidence intervals for each outcome
    
    Args:
        retrieved_metadata: DataFrame with outcomes, similarities, and dates
        query_date: Query date for recency weighting
        n_bootstrap: Number of bootstrap samples
        alpha: Confidence level (0.05 = 95% CI)
    
    Returns:
        Dict with CI per outcome
    """
    if len(retrieved_metadata) < 5:
        return {
            outcome: {'mean': 0, 'ci_low': 0, 'ci_high': 1, 'std': 0}
            for outcome in ['BREAK', 'REJECT', 'CHOP']
        }
    
    weights = compute_neighbor_weights(
        similarities=retrieved_metadata['similarity'].values,
        dates=retrieved_metadata.get('date'),
        query_date=query_date
    )
    outcomes = retrieved_metadata['outcome_4min'].values
    
    boot_probs = {'BREAK': [], 'REJECT': [], 'CHOP': []}
    
    for _ in range(n_bootstrap):
        # Weighted bootstrap sample
        sample_idx = np.random.choice(
            len(outcomes),
            size=len(outcomes),
            replace=True,
            p=weights
        )
        sample_outcomes = outcomes[sample_idx]
        
        # Compute proportions
        for outcome in boot_probs:
            prop = (sample_outcomes == outcome).mean()
            boot_probs[outcome].append(prop)
    
    ci = {}
    for outcome, probs in boot_probs.items():
        probs = np.array(probs)
        ci[outcome] = {
            'mean': float(probs.mean()),
            'ci_low': float(np.percentile(probs, 100 * alpha / 2)),
            'ci_high': float(np.percentile(probs, 100 * (1 - alpha / 2))),
            'std': float(probs.std())
        }
    
    return ci


def compute_reliability(retrieved_metadata: pd.DataFrame) -> Dict[str, float]:
    """
    Compute reliability metrics for the retrieval.
    
    Per IMPLEMENTATION_READY.md Section 10.6:
    - Effective sample size (accounts for similarity weighting)
    - Similarity statistics
    - Entropy (diversity of weights)
    
    Args:
        retrieved_metadata: DataFrame with similarity scores
    
    Returns:
        Dict with reliability metrics
    """
    if len(retrieved_metadata) == 0:
        return {
            'n_retrieved': 0,
            'effective_n': 0,
            'avg_similarity': 0,
            'min_similarity': 0,
            'max_similarity': 0,
            'similarity_std': 0,
            'entropy': 0
        }
    
    similarities = retrieved_metadata['similarity'].values
    weights = similarities / similarities.sum()
    
    # Effective sample size (inverse of sum of squared weights)
    effective_n = (weights.sum() ** 2) / (weights ** 2).sum()
    
    # Entropy (diversity of weights)
    entropy = -np.sum(weights * np.log(weights + 1e-10))
    
    return {
        'n_retrieved': int(len(retrieved_metadata)),
        'effective_n': float(effective_n),
        'avg_similarity': float(similarities.mean()),
        'min_similarity': float(similarities.min()),
        'max_similarity': float(similarities.max()),
        'similarity_std': float(similarities.std()),
        'entropy': float(entropy)
    }


def aggregate_query_results(
    retrieved_metadata: pd.DataFrame,
    query_date: pd.Timestamp = None,
    compute_ci: bool = True,
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """
    Aggregate all outcome metrics from retrieved neighbors with weighting.
    
    Convenience function that calls all aggregation functions with neighbor weighting:
    - Power transform: similarity^4
    - Recency decay: exp(-age_days / 60)
    
    Args:
        retrieved_metadata: DataFrame with retrieved neighbors
        query_date: Query date for recency weighting
        compute_ci: Whether to compute bootstrap CIs (expensive)
        n_bootstrap: Number of bootstrap samples if compute_ci=True
    
    Returns:
        Dict with all outcome metrics
    """
    if len(retrieved_metadata) == 0:
        return {
            'outcome_probabilities': {'probabilities': {}, 'n_samples': 0},
            'expected_excursions': {},
            'conditional_excursions': {},
            'multi_horizon': {},
            'confidence_intervals': {},
            'reliability': {'n_retrieved': 0}
        }
    
    results = {
        'outcome_probabilities': compute_outcome_distribution(retrieved_metadata, query_date),
        'expected_excursions': compute_expected_excursions(retrieved_metadata, query_date),
        'conditional_excursions': compute_conditional_excursions(retrieved_metadata, query_date),
        'multi_horizon': compute_multi_horizon_distribution(retrieved_metadata, query_date),
        'reliability': compute_reliability(retrieved_metadata)
    }
    
    if compute_ci:
        results['confidence_intervals'] = compute_bootstrap_ci(
            retrieved_metadata,
            query_date=query_date,
            n_bootstrap=n_bootstrap
        )
    else:
        results['confidence_intervals'] = {}
    
    return results

