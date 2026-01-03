import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_compute_force_mass(date: str = "2025-06-04", level_name: str = "PM_HIGH"):
    """
    Validation script for ComputeForceMassStage.
    
    1. Runs prerequisites.
    2. Runs ComputeForceMassStage.
    3. Verifies F=ma columns.
    """
    logger.info(f"Starting ComputeForceMass validation for date: {date}, level: {level_name}")
    
    # 1. Setup Context
    ctx = StageContext(date=date, level=level_name)
    
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
    
    # 2. Run Stage
    stage = ComputeForceMassStage()
    try:
        results = stage.execute(ctx)
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise

    # 3. Extract Outputs
    signals_df = results.get('signals_df', pd.DataFrame())
    
    # 4. Analyze
    report = {
        "stage": "compute_force_mass",
        "date": date,
        "level_target": level_name,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {}
    }
    
    if not signals_df.empty:
        # Check Columns
        fm_cols = [
            'predicted_accel', 'accel_residual', 'force_mass_ratio', 
            'force_proxy', 'mass_proxy', 'flow_alignment'
        ]
        
        missing_cols = [c for c in fm_cols if c not in signals_df.columns]
        
        if missing_cols:
             report["status"] = "failed_integrity"
             logger.error(f"Missing F=ma columns: {missing_cols}")
        
        # Stats
        stats = {}
        for col in fm_cols:
            if col in signals_df.columns:
                stats[col] = {
                    "mean": float(signals_df[col].mean()),
                    "min": float(signals_df[col].min()),
                    "max": float(signals_df[col].max()),
                    "std": float(signals_df[col].std()) if len(signals_df) > 1 else 0,
                    "zeros": int((signals_df[col] == 0).sum())
                }
        
        report["metrics"] = {
            "count": len(signals_df),
            "force_mass_stats": stats
        }
        
    else:
        report["metrics"] = {"status": "empty", "count": 0}

    # Write Report
    report_path = Path(__file__).parent / "compute_force_mass_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    validate_compute_force_mass()
