import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

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

def validate_compute_physics(date: str = "2025-06-04", level_name: str = "PM_HIGH"):
    """
    Validation script for ComputePhysicsStage.
    
    1. Runs prerequisites (Load -> ... -> DetectZones).
    2. Runs ComputePhysicsStage.
    3. Verifies physics metrics (Barrier, Tape, Fuel).
    """
    logger.info(f"Starting ComputePhysics validation for date: {date}, level: {level_name}")
    
    # 1. Setup Context
    ctx = StageContext(date=date, level=level_name)
    
    logger.info("Running Prerequisite Stages...")
    ctx = LoadBronzeStage().run(ctx)
    ctx = BuildAllOHLCVStage().run(ctx)
    ctx = InitMarketStateStage().run(ctx)
    ctx = GenerateLevelsStage().run(ctx)
    ctx = DetectInteractionZonesStage().run(ctx)
    
    # 2. Run Stage
    stage = ComputePhysicsStage()
    try:
        results = stage.execute(ctx)
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise

    # 3. Extract Outputs
    signals_df = results.get('signals_df', pd.DataFrame())
    
    # 4. Analyze
    report = {
        "stage": "compute_physics",
        "date": date,
        "level_target": level_name,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {}
    }
    
    if not signals_df.empty:
        # Check Columns
        physics_cols = [
            'barrier_state', 'tape_velocity', 'fuel_effect', 
            'call_tide', 'put_tide', 'gamma_exposure'
        ]
        missing_cols = [c for c in physics_cols if c not in signals_df.columns]
        
        if missing_cols:
             report["status"] = "failed_integrity"
             logger.error(f"Missing physics columns: {missing_cols}")
        
        # Categorical Distributions
        cat_stats = {}
        for col in ['barrier_state', 'fuel_effect', 'sweep_detected']:
            if col in signals_df.columns:
                cat_stats[col] = signals_df[col].astype(str).value_counts().to_dict()
        
        # Continuous Stats
        cont_stats = {}
        for col in ['tape_velocity', 'gamma_exposure', 'call_tide', 'put_tide']:
            if col in signals_df.columns:
                cont_stats[col] = {
                    "mean": float(signals_df[col].mean()),
                    "min": float(signals_df[col].min()),
                    "max": float(signals_df[col].max()),
                    "std": float(signals_df[col].std()) if len(signals_df) > 1 else 0
                }
        
        report["metrics"] = {
            "count": len(signals_df),
            "columns_present": len(signals_df.columns),
            "categorical_dist": cat_stats,
            "continuous_stats": cont_stats
        }
        
    else:
        report["metrics"] = {"status": "empty", "count": 0}
        # Possibly expected if no zones detected

    # Write Report
    report_path = Path(__file__).parent / "compute_physics_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    validate_compute_physics()
