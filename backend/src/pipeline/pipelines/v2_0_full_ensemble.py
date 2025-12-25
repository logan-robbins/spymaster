"""v2.0_full_ensemble pipeline - Physics + TA features.

This pipeline computes all features including:
- Core physics (barrier, tape, fuel)
- SMA-based mean reversion
- Confluence and pressure indicators
- Approach context and normalized features
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
    ComputeSMAFeaturesStage,
    ComputeConfluenceStage,
    ComputeApproachFeaturesStage,
    LabelOutcomesStage,
    FilterRTHStage,
)


def build_v2_0_pipeline() -> Pipeline:
    """Build v2.0_full_ensemble pipeline.

    Pipeline stages:
    1. Load Bronze data
    2. Build 1-min OHLCV bars
    3. Build 2-min OHLCV bars (with SMA warmup)
    4. Initialize MarketState
    5. Generate levels
    6. Detect touches
    7. Compute physics
    8. Add context features
    9. Compute SMA features (NEW)
    10. Compute confluence features (NEW)
    11. Compute approach features (NEW)
    12. Label outcomes
    13. Filter to RTH

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
            ComputeSMAFeaturesStage(),
            ComputeConfluenceStage(),
            ComputeApproachFeaturesStage(),
            LabelOutcomesStage(),
            FilterRTHStage(),
        ],
        name="full_ensemble",
        version="v2.0"
    )
