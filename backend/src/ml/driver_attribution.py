"""
Driver Attribution Module for Level Break/Reject Predictions

Decomposes  episode vector predictions into trader-understandable driver components.
Quantifies which physics domains are pushing toward BREAK vs REJECT.

Usage:
    from src.ml.driver_attribution import DriverAttributor
    
    attributor = DriverAttributor(model, scaler, feature_names)
    drivers = attributor.explain(episode_vector, p_break=0.67)
    
    # Returns:
    # {
    #   'break_drivers': [{'name': 'Clean Setup', 'contribution': 0.22, 'percentile': 85}, ...],
    #   'reject_drivers': [{'name': '4th Touch', 'contribution': -0.05, 'percentile': 15}, ...],
    #   'net_edge': 0.22,
    #   'confidence': 'HIGH'
    # }
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .constants import VECTOR_SECTIONS
from .episode_vector import get_feature_names


@dataclass
class Driver:
    """Single driver component."""
    name: str
    category: str  # 'BREAK' or 'REJECT'
    contribution: float  # Marginal probability contribution
    percentile: float  # 0-100 percentile vs historical
    raw_value: float
    trader_description: str


@dataclass
class DriverAttribution:
    """Complete driver attribution for a prediction."""
    p_break: float
    base_rate: float
    edge: float
    confidence: str
    
    break_drivers: List[Driver]
    reject_drivers: List[Driver]
    
    # Section-level contributions
    section_contributions: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'p_break': round(self.p_break, 3),
            'base_rate': round(self.base_rate, 3),
            'edge': round(self.edge, 3),
            'confidence': self.confidence,
            'break_drivers': [
                {
                    'name': d.name,
                    'contribution': round(d.contribution, 3),
                    'percentile': round(d.percentile, 1),
                    'description': d.trader_description
                }
                for d in self.break_drivers
            ],
            'reject_drivers': [
                {
                    'name': d.name,
                    'contribution': round(d.contribution, 3),
                    'percentile': round(d.percentile, 1),
                    'description': d.trader_description
                }
                for d in self.reject_drivers
            ],
            'section_contributions': {
                k: round(v, 3) for k, v in self.section_contributions.items()
            }
        }


class DriverAttributor:
    """
    Explains model predictions by decomposing feature contributions into trader-friendly drivers.
    
    Uses feature importance + local perturbation to quantify marginal contributions.
    """
    
    def __init__(
        self,
        model: RandomForestClassifier,
        scaler: Optional[StandardScaler] = None,
        feature_names: Optional[List[str]] = None,
        base_rate: float = 0.455,
        historical_stats: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Args:
            model: Trained classifier (must have feature_importances_ or predict_proba)
            scaler: StandardScaler fitted on training data (optional)
            feature_names: List of 144 feature names (will generate if None)
            base_rate: Base BREAK rate for edge calculation
            historical_stats: Dict[feature_name] -> (mean, std) for percentile calculation
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names or get_feature_names()
        self.base_rate = base_rate
        self.historical_stats = historical_stats or {}
        
        # Get feature importances if available
        self.feature_importances = None
        if hasattr(model, 'feature_importances_'):
            self.feature_importances = model.feature_importances_
        
        # Build trader-friendly feature mapping
        self.feature_map = self._build_feature_map()
        
        # Build section index map
        self.section_indices = self._build_section_indices()
    
    def _build_feature_map(self) -> Dict[str, Dict[str, str]]:
        """
        Map feature names to trader-understandable descriptions.
        
        Returns dict: feature_name -> {
            'trader_name': str,
            'description': str,
            'direction': 'BREAK' or 'REJECT' (when high)
        }
        """
        mapping = {}
        
        # High-impact features identified in PENTAVIEW_RESEARCH.md
        
        # Setup Quality (from sigma_s components)
        mapping['proximity'] = {
            'trader_name': 'Proximity',
            'description': 'How close to level (0-1)',
            'direction': 'BREAK'
        }
        mapping['level_stacking_5pt'] = {
            'trader_name': 'Confluence',
            'description': 'Multiple levels nearby',
            'direction': 'BREAK'
        }
        mapping['attempt_number'] = {
            'trader_name': 'Attempt Freshness',
            'description': 'Nth touch of this level',
            'direction': 'REJECT'
        }
        mapping['prior_touches'] = {
            'trader_name': 'Prior Touches',
            'description': 'Historical touches count',
            'direction': 'REJECT'
        }
        
        # GEX features
        mapping['gamma_exposure'] = {
            'trader_name': 'Dealer Gamma',
            'description': 'Market maker gamma position',
            'direction': 'CONTEXT'
        }
        mapping['gex_below_1strike'] = {
            'trader_name': 'Support GEX',
            'description': 'Gamma wall below (dealers buy dips)',
            'direction': 'BREAK'  # Amplifies upward moves
        }
        mapping['gex_above_1strike'] = {
            'trader_name': 'Resistance GEX',
            'description': 'Gamma wall above (dealers sell rallies)',
            'direction': 'REJECT'
        }
        
        # Barrier/Book features
        mapping['barrier_delta_liq'] = {
            'trader_name': 'Book Depth Change',
            'description': 'Liquidity added/removed',
            'direction': 'CONTEXT'
        }
        mapping['barrier_replenishment_ratio'] = {
            'trader_name': 'Book Replenishment',
            'description': 'How fast book rebuilds',
            'direction': 'REJECT'  # High replenishment = strong defense
        }
        mapping['wall_ratio'] = {
            'trader_name': 'Liquidity Wall',
            'description': 'Depth concentration at level',
            'direction': 'REJECT'
        }
        
        # Flow features
        mapping['ofi_acceleration'] = {
            'trader_name': 'Flow Acceleration',
            'description': 'Order flow speeding up',
            'direction': 'BREAK'
        }
        mapping['ofi_60s'] = {
            'trader_name': 'Order Flow Imbalance',
            'description': 'Buy/sell pressure (60s)',
            'direction': 'BREAK'
        }
        
        # Distance/momentum
        mapping['distance_signed_atr'] = {
            'trader_name': 'Distance to Level',
            'description': 'How far away (ATR)',
            'direction': 'REJECT'  # Further away = lower break prob
        }
        mapping['velocity_2min'] = {
            'trader_name': 'Price Momentum',
            'description': 'Speed toward level',
            'direction': 'BREAK'
        }
        
        return mapping
    
    def _build_section_indices(self) -> Dict[str, np.ndarray]:
        """Build index arrays for each vector section."""
        indices = {}
        for section_name, (start, end) in VECTOR_SECTIONS.items():
            indices[section_name] = np.arange(start, end)
        return indices
    
    def explain(
        self,
        episode_vector: np.ndarray,
        p_break: Optional[float] = None,
        top_k: int = 5
    ) -> DriverAttribution:
        """
        Generate driver attribution for a single episode prediction.
        
        Args:
            episode_vector:  episode vector (raw, pre-scaling)
            p_break: Pre-computed break probability (will compute if None)
            top_k: Number of top drivers to return per category
        
        Returns:
            DriverAttribution with break/reject drivers ranked by contribution
        """
        # Ensure 2D
        if episode_vector.ndim == 1:
            episode_vector = episode_vector.reshape(1, -1)
        
        # Scale if scaler provided
        if self.scaler is not None:
            X = self.scaler.transform(episode_vector)
        else:
            X = episode_vector
        
        # Get prediction if not provided
        if p_break is None:
            p_break = self.model.predict_proba(X)[0, 1]
        
        # Compute edge
        edge = p_break - self.base_rate
        
        # Confidence level
        if p_break >= 0.6:
            confidence = 'HIGH'
        elif p_break >= 0.5:
            confidence = 'MODERATE'
        elif p_break <= 0.4:
            confidence = 'HIGH'  # High confidence REJECT
        else:
            confidence = 'MODERATE'
        
        # Compute section-level contributions
        section_contribs = self._compute_section_contributions(
            episode_vector[0], X[0], p_break
        )
        
        # Compute feature-level contributions
        feature_contribs = self._compute_feature_contributions(
            episode_vector[0], X[0], p_break
        )
        
        # Convert to Driver objects
        drivers = self._build_driver_list(
            episode_vector[0], feature_contribs
        )
        
        # Split into BREAK vs REJECT
        break_drivers = sorted(
            [d for d in drivers if d.category == 'BREAK'],
            key=lambda d: abs(d.contribution),
            reverse=True
        )[:top_k]
        
        reject_drivers = sorted(
            [d for d in drivers if d.category == 'REJECT'],
            key=lambda d: abs(d.contribution),
            reverse=True
        )[:top_k]
        
        return DriverAttribution(
            p_break=p_break,
            base_rate=self.base_rate,
            edge=edge,
            confidence=confidence,
            break_drivers=break_drivers,
            reject_drivers=reject_drivers,
            section_contributions=section_contribs
        )
    
    def _compute_section_contributions(
        self,
        episode_vector: np.ndarray,
        X_scaled: np.ndarray,
        p_break: float
    ) -> Dict[str, float]:
        """
        Compute marginal contribution of each vector section via ablation.
        
        For each section, zero it out and measure probability change.
        """
        contributions = {}
        
        for section_name, indices in self.section_indices.items():
            # Create ablated vector (zero out this section)
            X_ablated = X_scaled.copy()
            X_ablated[indices] = 0.0
            
            # Predict
            p_ablated = self.model.predict_proba(X_ablated.reshape(1, -1))[0, 1]
            
            # Contribution = drop in probability when section removed
            contributions[section_name] = p_break - p_ablated
        
        return contributions
    
    def _compute_feature_contributions(
        self,
        episode_vector: np.ndarray,
        X_scaled: np.ndarray,
        p_break: float,
        use_importance: bool = True
    ) -> Dict[str, float]:
        """
        Compute feature-level contributions.
        
        Two methods:
        1. If model has feature_importances, use importance * value
        2. Otherwise use local perturbation
        """
        contributions = {}
        
        if use_importance and self.feature_importances is not None:
            # Method 1: Importance-weighted values
            for i, feat_name in enumerate(self.feature_names):
                # Contribution = importance * scaled_value
                contributions[feat_name] = (
                    self.feature_importances[i] * X_scaled[i]
                )
        else:
            # Method 2: Local perturbation (slower but more accurate)
            for i, feat_name in enumerate(self.feature_names):
                X_perturbed = X_scaled.copy()
                X_perturbed[i] = 0.0  # Zero out this feature
                
                p_perturbed = self.model.predict_proba(
                    X_perturbed.reshape(1, -1)
                )[0, 1]
                
                contributions[feat_name] = p_break - p_perturbed
        
        return contributions
    
    def _build_driver_list(
        self,
        episode_vector: np.ndarray,
        feature_contribs: Dict[str, float]
    ) -> List[Driver]:
        """Convert feature contributions to Driver objects."""
        drivers = []
        
        for feat_name, contribution in feature_contribs.items():
            # Only include mapped features
            if feat_name not in self.feature_map:
                continue
            
            feat_info = self.feature_map[feat_name]
            
            # Get feature index and raw value
            try:
                feat_idx = self.feature_names.index(feat_name)
                raw_value = episode_vector[feat_idx]
            except (ValueError, IndexError):
                continue
            
            # Compute percentile if we have historical stats
            percentile = self._compute_percentile(feat_name, raw_value)
            
            # Determine category based on contribution sign
            if contribution > 0:
                category = 'BREAK'
            else:
                category = 'REJECT'
            
            # Build trader description
            if feat_info['direction'] == 'BREAK' and contribution > 0:
                desc = f"Strong {feat_info['trader_name']} (top {100-percentile:.0f}%)"
            elif feat_info['direction'] == 'REJECT' and contribution < 0:
                desc = f"Weak {feat_info['trader_name']} (bottom {percentile:.0f}%)"
            else:
                desc = feat_info['description']
            
            drivers.append(Driver(
                name=feat_info['trader_name'],
                category=category,
                contribution=contribution,
                percentile=percentile,
                raw_value=raw_value,
                trader_description=desc
            ))
        
        return drivers
    
    def _compute_percentile(self, feat_name: str, value: float) -> float:
        """
        Compute percentile of value relative to historical distribution.
        
        Returns percentile 0-100.
        """
        if feat_name not in self.historical_stats:
            return 50.0  # Default to median if no stats
        
        mean, std = self.historical_stats[feat_name]
        
        if std == 0:
            return 50.0
        
        # Z-score
        z = (value - mean) / std
        
        # Convert to percentile (assuming normal distribution)
        from scipy.stats import norm
        percentile = norm.cdf(z) * 100
        
        return np.clip(percentile, 0, 100)
    
    def batch_explain(
        self,
        episode_vectors: np.ndarray,
        p_breaks: Optional[np.ndarray] = None,
        top_k: int = 5
    ) -> List[DriverAttribution]:
        """Batch version of explain() for multiple episodes."""
        n = len(episode_vectors)
        
        if p_breaks is None:
            # Compute all predictions
            if self.scaler is not None:
                X = self.scaler.transform(episode_vectors)
            else:
                X = episode_vectors
            p_breaks = self.model.predict_proba(X)[:, 1]
        
        results = []
        for i in range(n):
            attribution = self.explain(
                episode_vectors[i],
                p_break=p_breaks[i],
                top_k=top_k
            )
            results.append(attribution)
        
        return results


def compute_historical_stats(
    episode_vectors: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Compute mean/std for each feature across historical episodes.
    
    Used for percentile calculation in driver attribution.
    
    Args:
        episode_vectors: (N, 144) array of historical episodes
        feature_names: List of feature names
    
    Returns:
        Dict[feature_name] -> (mean, std)
    """
    if feature_names is None:
        feature_names = get_feature_names()
    
    stats = {}
    for i, feat_name in enumerate(feature_names):
        values = episode_vectors[:, i]
        stats[feat_name] = (
            float(np.nanmean(values)),
            float(np.nanstd(values))
        )
    
    return stats
