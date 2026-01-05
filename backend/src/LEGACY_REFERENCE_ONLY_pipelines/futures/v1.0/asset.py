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
from src.pipeline.stages.compute_level_walls import ComputeLevelWallsStage
from src.pipeline.stages.compute_force_mass import ComputeForceMassStage
from src.pipeline.stages.compute_approach import ComputeApproachFeaturesStage
from src.pipeline.stages.label_outcomes import LabelOutcomesStage
from src.pipeline.stages.filter_rth import FilterRTHStage


def build_bronze_to_silver_pipeline() -> Pipeline:
    """
    Build Bronze → Silver pipeline (feature engineering).
    
    Stage sequence (0-indexed, stages 0-16):

    
    Returns:
        Pipeline instance
    """
    return Pipeline(
        name="bronze_to_silver",
        version="1.0",  
        stages=[
            BuildAllOHLCVStage(),
            InitMarketStateStage(),
            GenerateLevelsStage(),
            DetectInteractionZonesStage(),
            ComputePhysicsStage(),
            ComputeMultiWindowKinematicsStage(),
            ComputeMultiWindowOFIStage(),
            ComputeMicrostructureStage(),  
            ComputeBarrierEvolutionStage(),
            ComputeLevelDistancesStage(),
            ComputeGEXFeaturesStage(),
            ComputeLevelWallsStage(),
            ComputeForceMassStage(),
            ComputeApproachFeaturesStage(),
            LabelOutcomesStage(),
            FilterRTHStage(), 
        ]
    )

