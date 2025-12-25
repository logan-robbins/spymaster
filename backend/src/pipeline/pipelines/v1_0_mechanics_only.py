"""v1.0_mechanics_only pipeline - Pure physics, no TA features.

This pipeline computes only the core physics features:
- Barrier state and liquidity metrics
- Tape imbalance and velocity
- Fuel (options flow) effects

Excludes SMA, confluence, and approach context features.
"""
from src.pipeline.core.pipeline import Pipeline
from src.pipeline.stages import (
    LoadBronzeStage,
    BuildOHLCVStage,
    InitMarketStateStage,
    GenerateLevelsStage,
    DetectTouchesStage,
    ComputePhysicsStage,
    ComputeContextFeaturesStage,
    LabelOutcomesStage,
    FilterRTHStage,
)


def build_v1_0_pipeline() -> Pipeline:
    """Build v1.0_mechanics_only pipeline.

    Pipeline stages:
    1. Load Bronze data
    2. Build 1-min OHLCV bars
    3. Build 2-min OHLCV bars (for level generation)
    4. Initialize MarketState
    5. Generate levels
    6. Detect touches
    7. Compute physics
    8. Add context features
    9. Label outcomes
    10. Filter to RTH

    Returns:
        Configured Pipeline instance
    """
    return Pipeline(
        stages=[
            LoadBronzeStage(),
            BuildOHLCVStage(freq='1min'),
            BuildOHLCVStage(freq='2min', output_key='ohlcv_2min', include_warmup=True),
            InitMarketStateStage(),
            GenerateLevelsStage(),
            DetectTouchesStage(),
            ComputePhysicsStage(),
            ComputeContextFeaturesStage(),
            LabelOutcomesStage(),
            FilterRTHStage(),
        ],
        name="mechanics_only",
        version="v1.0"
    )
