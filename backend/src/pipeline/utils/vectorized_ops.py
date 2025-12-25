"""
Vectorized operations - Re-exports for backwards compatibility.

NOTE: This module is deprecated. Import directly from stage modules instead:
- build_ohlcv: build_ohlcv(), compute_atr()
- generate_levels: generate_level_universe(), LevelInfo
- detect_touches: detect_touches()
- label_outcomes: label_outcomes()
- compute_context: compute_structural_distances()
- compute_sma: compute_mean_reversion_features()
- compute_confluence: compute_confluence_features_dynamic(), etc.
- compute_approach: compute_approach_context(), add_normalized_features()
"""

# Re-export LevelInfo for backwards compatibility
from src.pipeline.stages.generate_levels import LevelInfo

__all__ = ["LevelInfo"]
