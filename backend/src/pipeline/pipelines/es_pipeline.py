"""
ES Futures + ES Options Multi-Window Physics Pipeline.

Architecture: ES futures (spot + liquidity) + ES 0DTE options (gamma)
Inference: Continuous (every 2-min candle)
Features: ~70 physics features (multi-window 1-20min lookback) + ~40 labels
Levels: 6 kinds (PM/OR high/low + SMA_200/400)
Outcome: Triple-barrier ±75pts (3 strikes), 8min forward
RTH: 09:30-13:30 ET (first 4 hours)
"""

from src.pipeline.core.pipeline import Pipeline
from src.pipeline.core.stage import StageContext

from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.stages.build_spx_ohlcv import BuildOHLCVStage
from src.pipeline.stages.init_market_state import InitMarketStateStage
from src.pipeline.stages.generate_levels import GenerateLevelsStage
from src.pipeline.stages.detect_interaction_zones import DetectInteractionZonesStage
from src.pipeline.stages.compute_physics import ComputePhysicsStage
from src.pipeline.stages.compute_multiwindow_kinematics import ComputeMultiWindowKinematicsStage
from src.pipeline.stages.compute_multiwindow_ofi import ComputeMultiWindowOFIStage
from src.pipeline.stages.compute_barrier_evolution import ComputeBarrierEvolutionStage
from src.pipeline.stages.compute_level_distances import ComputeLevelDistancesStage
from src.pipeline.stages.compute_gex_features import ComputeGEXFeaturesStage
from src.pipeline.stages.compute_force_mass import ComputeForceMassStage
from src.pipeline.stages.compute_approach import ComputeApproachFeaturesStage
from src.pipeline.stages.label_outcomes import LabelOutcomesStage
from src.pipeline.stages.filter_rth import FilterRTHStage


def build_es_pipeline() -> Pipeline:
    """
    Build ES futures + ES options multi-window physics pipeline.
    
    Stage sequence:
    1. LoadBronze (ES futures + options, front-month filtered)
    2. BuildOHLCV (1min for ATR/volatility)
    3. BuildOHLCV (2min with warmup for SMA_200/400)
    4. InitMarketState (ES market state)
    5. GenerateLevels (6 level kinds: PM/OR high/low + SMA_200/400)
    6. DetectInteractionZones (continuous: every 2-min candle, deterministic IDs)
    7. ComputePhysics (barrier, tape, fuel from engines)
    8. ComputeMultiWindowKinematics (velocity/accel/jerk at 1,3,5,10,20min)
    9. ComputeMultiWindowOFI (integrated OFI at 30,60,120,300s)
    10. ComputeBarrierEvolution (depth changes at 1,3,5min)
    11. ComputeLevelDistances (signed distances to all structural levels)
    12. ComputeGEX (strike-banded gamma ±1/±2/±3 strikes)
    13. ComputeForceMass (F=ma validation features)
    14. ComputeApproach (approach context + session timing)
    15. LabelOutcomes (triple-barrier: ±75pts, 8min forward)
    16. FilterRTH (09:30-13:30 ET with forward spillover)
    
    Returns:
        Pipeline instance
    """
    return Pipeline(
        name="es_pipeline",
        version="2.0.0",
        stages=[
            LoadBronzeStage(),
            BuildOHLCVStage(freq='1min', output_key='ohlcv_1min'),
            BuildOHLCVStage(freq='2min', output_key='ohlcv_2min', include_warmup=True),
            InitMarketStateStage(),
            GenerateLevelsStage(),
            DetectInteractionZonesStage(),
            ComputePhysicsStage(),
            ComputeMultiWindowKinematicsStage(),
            ComputeMultiWindowOFIStage(),
            ComputeBarrierEvolutionStage(),
            ComputeLevelDistancesStage(),
            ComputeGEXFeaturesStage(),
            ComputeForceMassStage(),
            ComputeApproachFeaturesStage(),
            LabelOutcomesStage(),
            FilterRTHStage()
        ]
    )

