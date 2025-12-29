"""Outcome aggregation functions - IMPLEMENTATION_READY.md Section 10."""
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_outcome_distribution(retrieved_metadata: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute outcome probabilities weighted by similarity.
    
    Per IMPLEMENTATION_READY.md Section 10.1:
    - Weight by similarity scores
    - Normalize weights to sum to 1
    - Compute probabilities for BREAK, REJECT, CHOP
    
    Args:
        retrieved_metadata: DataFrame with 'similarity' and 'outcome_4min' columns
    
    Returns:
        Dict with probabilities and metadata
    """
    if len(retrieved_metadata) == 0:
        return {'probabilities': {'BREAK': 0, 'REJECT': 0, 'CHOP': 0}, 'n_samples': 0, 'avg_similarity': 0}
    
    weights = retrieved_metadata['similarity'].values
    weights = weights / weights.sum()  # Normalize to sum to 1
    
    probs = {}
    for outcome in ['BREAK', 'REJECT', 'CHOP']:
        mask = retrieved_metadata['outcome_4min'] == outcome
        probs[outcome] = float(weights[mask].sum())
    
    return {
        'probabilities': probs,
        'n_samples': len(retrieved_metadata),
        'avg_similarity': float(retrieved_metadata['similarity'].mean())
    }


def compute_expected_excursions(retrieved_metadata: pd.DataFrame) -> Dict[str, float]:
    """
    Compute expected favorable and adverse excursions.
    
    Per IMPLEMENTATION_READY.md Section 10.2:
    - Weight by similarity
    - Compute expected excursions (ATR-normalized)
    
    Args:
        retrieved_metadata: DataFrame with similarity and excursion columns
    
    Returns:
        Dict with expected excursions and ratio
    """
    if len(retrieved_metadata) == 0:
        return {
            'expected_excursion_favorable': 0.0,
            'expected_excursion_adverse': 0.0,
            'excursion_ratio': 0.0
        }
    
    weights = retrieved_metadata['similarity'].values
    weights = weights / weights.sum()
    
    expected_favorable = float((weights * retrieved_metadata['excursion_favorable']).sum())
    expected_adverse = float((weights * retrieved_metadata['excursion_adverse']).sum())
    
    return {
        'expected_excursion_favorable': expected_favorable,
        'expected_excursion_adverse': expected_adverse,
        'excursion_ratio': expected_favorable / (expected_adverse + 1e-6)
    }


def compute_conditional_excursions(retrieved_metadata: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute expected excursions conditional on outcome.
    
    Per IMPLEMENTATION_READY.md Section 10.3:
    - Separate by BREAK vs REJECT
    - Compute weighted expectations within each outcome
    
    Args:
        retrieved_metadata: DataFrame with outcomes and excursions
    
    Returns:
        Dict with conditional expectations per outcome
    """
    weights = retrieved_metadata['similarity'].values
    
    conditional = {}
    
    for outcome in ['BREAK', 'REJECT']:
        mask = retrieved_metadata['outcome_4min'] == outcome
        
        if mask.sum() > 0:
            subset = retrieved_metadata[mask]
            subset_weights = weights[mask]
            subset_weights = subset_weights / subset_weights.sum()
            
            conditional[outcome] = {
                'expected_favorable': float((subset_weights * subset['excursion_favorable']).sum()),
                'expected_adverse': float((subset_weights * subset['excursion_adverse']).sum()),
                'mean_strength': float((subset_weights * subset['strength_abs']).sum()),
                'n_samples': int(mask.sum())
            }
    
    return conditional


def compute_multi_horizon_distribution(retrieved_metadata: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute outcome distributions for all horizons (2min, 4min, 8min).
    
    Per IMPLEMENTATION_READY.md Section 10.4
    
    Args:
        retrieved_metadata: DataFrame with outcome columns for all horizons
    
    Returns:
        Dict with probabilities per horizon
    """
    weights = retrieved_metadata['similarity'].values
    weights = weights / weights.sum()
    
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
    n_bootstrap: int = 1000,
    alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for outcome probabilities.
    
    Per IMPLEMENTATION_READY.md Section 10.5:
    - Weighted bootstrap resampling
    - Compute confidence intervals for each outcome
    
    Args:
        retrieved_metadata: DataFrame with outcomes and similarities
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
    
    weights = retrieved_metadata['similarity'].values
    weights = weights / weights.sum()
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
    compute_ci: bool = True,
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """
    Aggregate all outcome metrics from retrieved neighbors.
    
    Convenience function that calls all aggregation functions.
    
    Args:
        retrieved_metadata: DataFrame with retrieved neighbors
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
        'outcome_probabilities': compute_outcome_distribution(retrieved_metadata),
        'expected_excursions': compute_expected_excursions(retrieved_metadata),
        'conditional_excursions': compute_conditional_excursions(retrieved_metadata),
        'multi_horizon': compute_multi_horizon_distribution(retrieved_metadata),
        'reliability': compute_reliability(retrieved_metadata)
    }
    
    if compute_ci:
        results['confidence_intervals'] = compute_bootstrap_ci(
            retrieved_metadata,
            n_bootstrap=n_bootstrap
        )
    else:
        results['confidence_intervals'] = {}
    
    return results

