"""
v1.0 Final Call Pipeline: ES/SPX Physics-Only Attribution.

Per Final Call v1 spec implementation order (Section 10):
1. ✅ ES front-month selector
2. ✅ ET sessionization + minutes_since_open
3. ✅ SPX spot series (ES futures, no conversion needed)
4. ✅ Level generator (PM/OR/SMA200/SMA400 only)
5. ✅ Interaction-zone event extractor
6. ✅ ES→SPX basis tracking
7. ✅ Physics features (kinematics + barrier + tape + OFI + GEX)
8. ✅ Triple-barrier labels with Policy B
9. ✅ Drop confluence/pressure features
10. ⏳ QA gates

Scope:
- Instruments: ES futures + SPX 0DTE options (not SPY!)
- Time: 09:30-13:30 ET (first 4 hours)
- Levels: PM_HIGH/PM_LOW, OR_HIGH/OR_LOW, SMA_200/SMA_400
- Features: Raw physics observables only (no TA indicators)
"""

from src.pipeline.core.pipeline import Pipeline
from src.pipeline.core.stage import StageContext

from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.stages.build_spx_ohlcv import BuildOHLCVStage
from src.pipeline.stages.init_market_state import InitMarketStateStage
from src.pipeline.stages.generate_levels import GenerateLevelsStage
from src.pipeline.stages.detect_interaction_zones import DetectInteractionZonesStage
from src.pipeline.stages.compute_physics import ComputePhysicsStage
from src.pipeline.stages.compute_kinematics import ComputeKinematicsStage
from src.pipeline.stages.compute_level_distances import ComputeLevelDistancesStage
from src.pipeline.stages.compute_ofi import ComputeOFIStage
from src.pipeline.stages.compute_gex_features import ComputeGEXFeaturesStage
from src.pipeline.stages.compute_force_mass import ComputeForceMassStage
from src.pipeline.stages.compute_approach import ComputeApproachFeaturesStage
from src.pipeline.stages.label_outcomes import LabelOutcomesStage
from src.pipeline.stages.filter_rth import FilterRTHStage


def build_v1_0_spx_final_call_pipeline() -> Pipeline:
    """
    Build v1.0 Final Call pipeline (ES/SPX physics-only).
    
    Stage sequence:
    1. LoadBronze (with front-month ES filtering)
    2. BuildOHLCV (1min, RTH-only for ATR/vol)
    3. BuildOHLCV (2min, with warmup for SMA)
    4. InitMarketState (load ES + SPX options)
    5. GenerateLevels (PM/OR/SMA only)
    6. DetectInteractionZones (zone-based events with deterministic IDs)
    7. ComputePhysics (barrier, tape, fuel from existing engines)
    8. ComputeKinematics (velocity, acceleration, jerk in level frame)
    9. ComputeLevelDistances (signed distances to all v1 levels)
    10. ComputeOFI (integrated order flow imbalance)
    11. ComputeGEX (strike-banded gamma exposure)
    12. ComputeForceMass (F=ma consistency checks)
    13. ComputeApproach (approach context + session timing)
    14. LabelOutcomes (triple-barrier with Policy B)
    15. FilterRTH (09:30-13:30 ET with forward spillover)
    
    Returns:
        Pipeline instance
    """
    return Pipeline(
        name="spx_final_call",
        version="v1.0_spx_final_call",
        stages=[
            LoadBronzeStage(),
            BuildOHLCVStage(freq='1min', output_key='ohlcv_1min'),
            BuildOHLCVStage(freq='2min', output_key='ohlcv_2min', include_warmup=True),
            InitMarketStateStage(),
            GenerateLevelsStage(),
            DetectInteractionZonesStage(),
            ComputePhysicsStage(),
            ComputeKinematicsStage(),
            ComputeLevelDistancesStage(),
            ComputeOFIStage(),
            ComputeGEXFeaturesStage(),
            ComputeForceMassStage(),
            ComputeApproachFeaturesStage(),
            LabelOutcomesStage(),
            FilterRTHStage()
        ]
    )

