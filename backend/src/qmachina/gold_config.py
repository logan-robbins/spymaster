"""GoldFeatureConfig: spec-driven parameters for gold feature computation.

All parameters derive from RuntimeConfig, ensuring live-stream parity with
offline gold_builder computation.
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .config import RuntimeConfig


class GoldFeatureConfig(BaseModel):
    """Parameters for computing gold features from silver data.

    Captures the full parameterization of:
      1. Force block: pressure/vacuum per cell
      2. Spectrum block: composite + temporal derivatives
      3. Scoring block: flow_score / flow_state_code

    All params correspond 1:1 to RuntimeConfig fields, ensuring that offline
    gold_builder and live frontend runtime produce identical outputs.
    """

    # Force block
    # pressure = c1*v_add + c2*max(v_rest_depth, 0) + c3*max(a_add, 0)
    # vacuum   = c4*v_pull + c5*v_fill + c6*max(-v_rest_depth, 0) + c7*max(a_pull, 0)
    c1_v_add: float
    c2_v_rest_pos: float
    c3_a_add: float
    c4_v_pull: float
    c5_v_fill: float
    c6_v_rest_neg: float
    c7_a_pull: float

    # Spectrum: rolling window composite rollup
    flow_windows: list[int]
    flow_rollup_weights: list[float]

    # Scoring
    flow_derivative_weights: list[float] = Field(default=[0.55, 0.30, 0.15])
    flow_tanh_scale: float = 3.0
    flow_neutral_threshold: float = 0.15
    flow_zscore_window_bins: int = 300
    flow_zscore_min_periods: int = 75

    def config_hash(self) -> str:
        """Deterministic 8-char hash of all config params."""
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    @classmethod
    def from_runtime_config(cls, config: RuntimeConfig) -> GoldFeatureConfig:
        """Build GoldFeatureConfig from a RuntimeConfig instance."""
        return cls(
            c1_v_add=config.c1_v_add,
            c2_v_rest_pos=config.c2_v_rest_pos,
            c3_a_add=config.c3_a_add,
            c4_v_pull=config.c4_v_pull,
            c5_v_fill=config.c5_v_fill,
            c6_v_rest_neg=config.c6_v_rest_neg,
            c7_a_pull=config.c7_a_pull,
            flow_windows=list(config.flow_windows),
            flow_rollup_weights=list(config.flow_rollup_weights),
            flow_derivative_weights=list(config.flow_derivative_weights),
            flow_tanh_scale=config.flow_tanh_scale,
            flow_neutral_threshold=config.flow_neutral_threshold,
            flow_zscore_window_bins=config.flow_zscore_window_bins,
            flow_zscore_min_periods=config.flow_zscore_min_periods,
        )
