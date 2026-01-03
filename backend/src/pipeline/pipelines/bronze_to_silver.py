"""
Bronze → Silver Pipeline: Feature Engineering

Transforms Bronze data (normalized trades/depth) into Silver engineered features.

Stages 0-16:
- Loads Bronze data (ES futures + ES 0DTE options)
- Computes multi-window physics (barrier, tape, fuel, kinematics, OFI)
- Generates levels and detects interaction zones
- Labels outcomes (BREAK/REJECT/CHOP)
- Filters to RTH (09:30-12:30 ET)

Output: Silver feature table (~142 columns) written to:
  silver/features/es_pipeline/version={version}/date=YYYY-MM-DD/signals.parquet

Consumers: Data scientists, ML experiments, Silver → Gold pipelines
"""

from src.pipeline.core.pipeline import Pipeline
from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.stages.build_all_ohlcv import BuildAllOHLCVStage
from src.pipeline.stages.init_market_state import InitMarketStateStage
from src.pipeline.stages.generate_levels import GenerateLevelsStage
from src.pipeline.stages.detect_interaction_zones import DetectInteractionZonesStage
from src.pipeline.stages.compute_physics import ComputePhysicsStage
from src.pipeline.stages.compute_multiwindow_kinematics import ComputeMultiWindowKinematicsStage
from src.pipeline.stages.compute_multiwindow_ofi import ComputeMultiWindowOFIStage
from src.pipeline.stages.compute_microstructure import ComputeMicrostructureStage
from src.pipeline.stages.compute_barrier_evolution import ComputeBarrierEvolutionStage
from src.pipeline.stages.compute_level_distances import ComputeLevelDistancesStage
from src.pipeline.stages.compute_gex_features import ComputeGEXFeaturesStage
from src.pipeline.stages.compute_force_mass import ComputeForceMassStage
from src.pipeline.stages.compute_approach import ComputeApproachFeaturesStage
from src.pipeline.stages.label_outcomes import LabelOutcomesStage
from src.pipeline.stages.filter_rth import FilterRTHStage


def build_bronze_to_silver_pipeline() -> Pipeline:
    """
    Build Bronze → Silver pipeline (feature engineering).
    
    Stage sequence (0-indexed, stages 0-14):
    0. LoadBronze (ES futures + options, front-month filtered)
    1. BuildAllOHLCV (trades → 10s → 1min → 2min hierarchically, ATR, warmup)
    2. InitMarketState (market state + Greeks)
    3. GenerateLevels (6 level kinds: PM/OR high/low + SMA_90/EMA_20)
    4. DetectInteractionZones (event-driven zone entry)
    5. ComputePhysics (barrier/tape/fuel + Market Tide: call_tide, put_tide)
    6. ComputeMultiWindowKinematics (velocity/accel/jerk/momentum at 1,2,3,5,10,20min)
    7. ComputeMultiWindowOFI (integrated OFI at 30,60,120,300s)
    8. ComputeMicrostructure (vacuum/latency detection)
    9. ComputeBarrierEvolution (depth changes at 1,2,3,5min)
    10. ComputeLevelDistances (signed distances to all structural levels)
    11. ComputeGEXFeatures (gamma within ±1/±2/±3 strikes)
    12. ComputeForceMass (F=ma validation features)
    13. ComputeApproachFeatures (approach context + timing + normalization + clustering)
    14. LabelOutcomes (first-crossing: 1 ATR threshold, BREAK/REJECT/CHOP, 2/4/8min)
    15. FilterRTH (09:30-12:30 ET + write to Silver)
    
    Returns:
        Pipeline instance
    """
    return Pipeline(
        name="bronze_to_silver",
        version="4.7.0",  # Phase 4.7: MBP-10 w/ action/side + level-specific generation
        stages=[
            LoadBronzeStage(),
            BuildAllOHLCVStage(),
            InitMarketStateStage(),
            GenerateLevelsStage(),
            DetectInteractionZonesStage(),
            ComputePhysicsStage(),
            ComputeMultiWindowKinematicsStage(),
            ComputeMultiWindowOFIStage(),
            ComputeMicrostructureStage(),  # New Microstructure layer (Vacuum/Latency)
            ComputeBarrierEvolutionStage(),
            ComputeLevelDistancesStage(),
            ComputeGEXFeaturesStage(),
            ComputeForceMassStage(),
            ComputeApproachFeaturesStage(),
            LabelOutcomesStage(),
            FilterRTHStage(),  # Stage 16: Writes to Silver
        ]
    )

