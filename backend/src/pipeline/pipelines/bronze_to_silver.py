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
from src.pipeline.stages.build_es_ohlcv import BuildOHLCVStage
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
    
    Stage sequence (0-indexed, stages 0-16):
    0. LoadBronze (ES futures + options, front-month filtered)
    1. BuildOHLCV (1min for ATR/volatility)
    2. BuildOHLCV (10s for high-res physics validation)
    3. BuildOHLCV (2min with warmup for SMA_90/EMA_20)
    4. InitMarketState (market state + Greeks)
    5. GenerateLevels (6 level kinds: PM/OR high/low + SMA_90/EMA_20)
    6. DetectInteractionZones (event-driven zone entry)
    7. ComputePhysics (barrier/tape/fuel + Market Tide: call_tide, put_tide)
    8. ComputeMultiWindowKinematics (velocity/accel/jerk/momentum at 1,2,3,5,10,20min)
    9. ComputeMultiWindowOFI (integrated OFI at 30,60,120,300s)
    10. ComputeBarrierEvolution (depth changes at 1,2,3,5min)
    11. ComputeLevelDistances (signed distances to all structural levels)
    12. ComputeGEXFeatures (gamma within ±1/±2/±3 strikes)
    13. ComputeForceMass (F=ma validation features)
    14. ComputeApproachFeatures (approach context + timing + normalization + clustering)
    15. LabelOutcomes (first-crossing: 1 ATR threshold, BREAK/REJECT/CHOP, 2/4/8min)
    16. FilterRTH (09:30-12:30 ET + write to Silver)
    
    Returns:
        Pipeline instance
    """
    return Pipeline(
        name="bronze_to_silver",
        version="4.5.0",  # Phase 4.5: Market Tide + multi-scale completion
        stages=[
            LoadBronzeStage(),
            BuildOHLCVStage(freq='1min', output_key='ohlcv_1min', rth_only=False),
            BuildOHLCVStage(freq='10s', output_key='ohlcv_10s', rth_only=False),
            BuildOHLCVStage(freq='2min', output_key='ohlcv_2min', include_warmup=True, rth_only=False),
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

