"""
ES Futures + ES Options Multi-Window Physics Pipeline + Episode Retrieval.

Architecture: ES futures (spot + liquidity) + ES 0DTE options (gamma)
Inference: Event-driven (zone entry + adaptive cadence)
Features: 182 columns (10 identity + 108 engineered features + 64 labels)
Levels: 6 kinds (PM/OR high/low + SMA_90/EMA_20)
Outcome: First-crossing semantics (BREAK/REJECT/CHOP), 2/4/8min horizons
Episode Vectors: 144-dimensional vectors with DCT trajectory basis for similarity retrieval
Zone Threshold: 2.0 ATR (compromise between 3.0 original and 1.25 analyst)
RTH: 09:30-12:30 ET (first 3 hours)
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
from src.pipeline.stages.materialize_state_table import MaterializeStateTableStage
from src.pipeline.stages.construct_episodes import ConstructEpisodesStage


def build_es_pipeline() -> Pipeline:
    """
    Build ES futures + ES options multi-window physics pipeline.
    
    Stage sequence (0-indexed):
    0. LoadBronze (ES futures + options, front-month filtered)
    1. BuildOHLCV (1min for ATR/volatility)
    2. BuildOHLCV (2min with warmup for SMA_90/EMA_20)
    3. InitMarketState (ES market state)
    4. GenerateLevels (6 level kinds: PM/OR high/low + SMA_90/EMA_20)
    5. DetectInteractionZones (event-driven zone entry, deterministic IDs)
    6. ComputePhysics (barrier, tape, fuel from engines)
    7. ComputeMultiWindowKinematics (velocity/accel/jerk at 1,3,5,10,20min)
    8. ComputeMultiWindowOFI (integrated OFI at 30,60,120,300s)
    9. ComputeBarrierEvolution (depth changes at 1,3,5min)
    10. ComputeLevelDistances (signed distances to all structural levels)
    11. ComputeGEX (gamma within ±1/±2/±3 strikes; 5pt spacing for ES 0DTE ATM)
    12. ComputeForceMass (F=ma validation features)
    13. ComputeApproach (approach context + timing + normalization + clustering)
    14. LabelOutcomes (first-crossing: 1 ATR threshold, BREAK/REJECT/CHOP, 2/4/8min)
    15. FilterRTH (09:30-12:30 ET)
    16. MaterializeStateTable (30s cadence state for episode construction)
    17. ConstructEpisodes (144-dim vectors with DCT trajectory basis)
    
    Returns:
        Pipeline instance
    """
    return Pipeline(
        name="es_pipeline",
        version="3.1.0",  # Matches README canonical version (Dec 2025)
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
            ComputeBarrierEvolutionStage(),
            ComputeLevelDistancesStage(),
            ComputeGEXFeaturesStage(),
            ComputeForceMassStage(),
            ComputeApproachFeaturesStage(),
            LabelOutcomesStage(),
            FilterRTHStage(),
            MaterializeStateTableStage(),  # Stage 16 (index 16)
            ConstructEpisodesStage()       # Stage 17 (index 17)
        ]
    )
