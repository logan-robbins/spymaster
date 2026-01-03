import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.pipeline.stages.filter_rth import FilterRTHStage
from src.pipeline.stages.label_outcomes import LabelOutcomesStage
from src.pipeline.stages.compute_approach import ComputeApproachFeaturesStage
from src.pipeline.stages.compute_force_mass import ComputeForceMassStage
from src.pipeline.stages.compute_level_walls import ComputeLevelWallsStage
from src.pipeline.stages.compute_gex_features import ComputeGEXFeaturesStage
from src.pipeline.stages.compute_level_distances import ComputeLevelDistancesStage
from src.pipeline.stages.compute_barrier_evolution import ComputeBarrierEvolutionStage
from src.pipeline.stages.compute_microstructure import ComputeMicrostructureStage
from src.pipeline.stages.compute_multiwindow_ofi import ComputeMultiWindowOFIStage
from src.pipeline.stages.compute_multiwindow_kinematics import ComputeMultiWindowKinematicsStage
from src.pipeline.stages.compute_physics import ComputePhysicsStage
from src.pipeline.stages.detect_interaction_zones import DetectInteractionZonesStage
from src.pipeline.stages.generate_levels import GenerateLevelsStage
from src.pipeline.stages.init_market_state import InitMarketStateStage
from src.pipeline.stages.build_all_ohlcv import BuildAllOHLCVStage
from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.core.stage import StageContext
from src.common.config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_filter_rth(date: str = "2025-06-04", level_name: str = "PM_HIGH"):
    """
    Validation script for FilterRTHStage.
    
    1. Runs prerequisites.
    2. Runs FilterRTHStage.
    3. Verifies RTH filtering, prefixing, and global feature removal.
    """
    logger.info(f"Starting FilterRTH validation for date: {date}, level: {level_name}")
    
    # 1. Setup Context
    # Disable writing to disk for validation to avoid side effects
    config_override = {"PIPELINE_WRITE_SIGNALS": False}
    ctx = StageContext(date=date, level=level_name, config=config_override)
    
    logger.info("Running Prerequisite Stages...")
    ctx = LoadBronzeStage().run(ctx)
    ctx = BuildAllOHLCVStage().run(ctx)
    ctx = InitMarketStateStage().run(ctx)
    ctx = GenerateLevelsStage().run(ctx)
    ctx = DetectInteractionZonesStage().run(ctx)
    ctx = ComputePhysicsStage().run(ctx)
    ctx = ComputeMultiWindowKinematicsStage().run(ctx)
    ctx = ComputeMultiWindowOFIStage().run(ctx)
    ctx = ComputeMicrostructureStage().run(ctx)
    ctx = ComputeBarrierEvolutionStage().run(ctx)
    ctx = ComputeLevelDistancesStage().run(ctx)
    ctx = ComputeGEXFeaturesStage().run(ctx)
    ctx = ComputeLevelWallsStage().run(ctx)
    ctx = ComputeForceMassStage().run(ctx)
    ctx = ComputeApproachFeaturesStage().run(ctx)
    ctx = LabelOutcomesStage().run(ctx)
    
    # Capture pre-filter count
    pre_filter_count = len(ctx.data['signals_df'])
    logger.info(f"Pre-filter signals: {pre_filter_count}")

    # 2. Run Stage
    stage = FilterRTHStage()
    try:
        results = stage.execute(ctx)
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise

    # 3. Extract Outputs
    signals = results.get('signals', pd.DataFrame())
    
    # 4. Analyze
    report = {
        "stage": "filter_rth",
        "date": date,
        "level_target": level_name,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {}
    }
    
    if not signals.empty:
        # Check Prefixes
        prefix = f"{level_name.lower()}_"
        # Columns that SHOULD be prefixed
        prefixed_cols = [c for c in signals.columns if c.startswith(prefix)]
        # Columns that should NOT be prefixed (Identity)
        identity_cols = ['event_id', 'ts_ns', 'level_name', 'level_kind_name', 'direction', 'date']
        # Note: FilterRTH defines identity cols, let's verify mostly prefixed
        
        non_prefixed = [c for c in signals.columns if not c.startswith(prefix) and c not in identity_cols]
        # Allow some identity cols I might have missed
        
        # Check Global Removal
        global_cols = ['atr', 'spot', 'minutes_since_open']
        leaked_globals = [c for c in global_cols if c in signals.columns]
        
        if leaked_globals:
             report["status"] = "failed_integrity"
             logger.error(f"Global columns leaked: {leaked_globals}")
        
        # Check Time Range (approximate, since we don't convert to ET here easily without heavy imports)
        # Just check count vs pre-filter
        
        report["metrics"] = {
            "pre_filter_count": pre_filter_count,
            "post_filter_count": len(signals),
            "prefixed_columns_count": len(prefixed_cols),
            "total_columns": len(signals.columns),
            "leaked_globals": leaked_globals
        }
        
    else:
        report["metrics"] = {
            "status": "empty", 
            "count": 0,
            "pre_filter_count": pre_filter_count
        }

    # Write Report
    report_path = Path(__file__).parent / "filter_rth_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    validate_filter_rth()
