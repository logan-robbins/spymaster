"""Vectorized operations for pipeline stages.

Public API for vectorized operations used by pipeline stages.
All operations are NumPy/pandas vectorized and optimized for Apple M4 Silicon.
"""

# Import from internal module
from src.pipeline.utils._vectorized_functions import (
    # Data structures
    LevelInfo,

    # OHLCV operations
    build_ohlcv_vectorized,
    compute_atr_vectorized,

    # Data conversion
    _futures_trades_from_df as futures_trades_from_df,
    _mbp10_from_df as mbp10_from_df,

    # Level generation
    generate_level_universe_vectorized,
    compute_dynamic_level_series,

    # Touch detection
    detect_touches_vectorized,
    detect_dynamic_level_touches,

    # Physics computation
    compute_physics_batch,

    # Context features
    compute_approach_context_vectorized,
    compute_structural_distances,
    compute_mean_reversion_features,
    compute_confluence_features_dynamic,
    compute_confluence_alignment,
    compute_dealer_velocity_features,
    compute_pressure_indicators,
    add_sparse_feature_transforms,
    add_normalized_features,
    compute_attempt_features,
    compute_confluence_level_features,

    # Labeling
    label_outcomes_vectorized,
)

__all__ = [
    # Data structures
    "LevelInfo",

    # OHLCV operations
    "build_ohlcv_vectorized",
    "compute_atr_vectorized",

    # Data conversion
    "futures_trades_from_df",
    "mbp10_from_df",

    # Level generation
    "generate_level_universe_vectorized",
    "compute_dynamic_level_series",

    # Touch detection
    "detect_touches_vectorized",
    "detect_dynamic_level_touches",

    # Physics computation
    "compute_physics_batch",

    # Context features
    "compute_approach_context_vectorized",
    "compute_structural_distances",
    "compute_mean_reversion_features",
    "compute_confluence_features_dynamic",
    "compute_confluence_alignment",
    "compute_dealer_velocity_features",
    "compute_pressure_indicators",
    "add_sparse_feature_transforms",
    "add_normalized_features",
    "compute_attempt_features",
    "compute_confluence_level_features",

    # Labeling
    "label_outcomes_vectorized",
]
