import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

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

def validate_compute_ofi(date: str = "2025-06-04", level_name: str = "PM_HIGH"):
    """
    Validation script for ComputeMultiWindowOFIStage.
    
    1. Runs prerequisites.
    2. Runs ComputeMultiWindowOFIStage.
    3. Verifies OFI columns (spatial bands, multi-window).
    """
    logger.info(f"Starting ComputeOFI validation for date: {date}, level: {level_name}")
    
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
    
    # 2. Run Stage
    stage = ComputeMultiWindowOFIStage()
    try:
        results = stage.execute(ctx)
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise

    # 3. Extract Outputs
    signals_df = results.get('signals_df', pd.DataFrame())
    
    # 4. Analyze
    report = {
        "stage": "compute_multiwindow_ofi",
        "date": date,
        "level_target": level_name,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {}
    }
    
    if not signals_df.empty:
        # Check Columns
        windows = [30, 60, 120, 300]
        ofi_cols = []
        for w in windows:
            suffix = f"_{w}s"
            ofi_cols.extend([
                f"ofi{suffix}",
                f"ofi_near_level{suffix}", 
                f"ofi_above_5pt{suffix}", 
                f"ofi_below_5pt{suffix}"
            ])
        
        missing_cols = [c for c in ofi_cols if c not in signals_df.columns]
        
        if missing_cols:
             report["status"] = "failed_integrity"
             logger.error(f"Missing OFI columns: {missing_cols}")
        
        # Stats
        stats = {}
        target_cols = [f"ofi_30s", f"ofi_near_level_30s", f"ofi_acceleration"]
        for col in target_cols:
            if col in signals_df.columns:
                stats[col] = {
                    "mean": float(signals_df[col].mean()),
                    "min": float(signals_df[col].min()),
                    "max": float(signals_df[col].max()),
                    "nans": int(signals_df[col].isna().sum()),
                    "zeros": int((signals_df[col] == 0).sum())
                }
        
        report["metrics"] = {
            "count": len(signals_df),
            "key_ofi_stats": stats
        }
        
    else:
        report["metrics"] = {"status": "empty", "count": 0}

    # Write Report
    report_path = Path(__file__).parent / "compute_multiwindow_ofi_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    validate_compute_ofi()
