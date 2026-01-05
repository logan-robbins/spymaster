import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.pipeline.global_stages.compute_market_kinematics import ComputeMarketKinematicsStage
from src.pipeline.global_stages.compute_market_micro import ComputeMarketMicroStage
from src.pipeline.global_stages.compute_market_ofi import ComputeMarketOFIStage
from src.pipeline.global_stages.compute_market_options import ComputeMarketOptionsStage
from src.pipeline.global_stages.generate_time_grid import GenerateTimeGridStage
from src.pipeline.stages.init_market_state import InitMarketStateStage
from src.pipeline.stages.build_all_ohlcv import BuildAllOHLCVStage
from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.core.stage import StageContext
from src.common.config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_compute_market_kinematics(date: str = "2025-06-04"):
    """
    Validation script for ComputeMarketKinematicsStage.
    
    1. Runs prerequisites.
    2. Runs ComputeMarketKinematicsStage.
    3. Verifies global kinematic columns.
    """
    logger.info(f"Starting ComputeMarketKinematics validation for date: {date}")
    
    # 1. Setup Context
    ctx = StageContext(date=date, level="GLOBAL")
    
    logger.info("Running Prerequisite Stages...")
    ctx = LoadBronzeStage().run(ctx)
    ctx = BuildAllOHLCVStage().run(ctx)
    ctx = InitMarketStateStage().run(ctx)
    ctx = GenerateTimeGridStage(interval_seconds=30.0).run(ctx)
    # Optional stages for context but not strict dependencies for Kinematics (which needs OHLCV)
    # Running them to simulate full pipeline flow
    ctx = ComputeMarketOptionsStage().run(ctx)
    ctx = ComputeMarketOFIStage().run(ctx)
    ctx = ComputeMarketMicroStage().run(ctx)
    
    # 2. Run Stage
    stage = ComputeMarketKinematicsStage()
    try:
        results = stage.execute(ctx)
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise

    # 3. Extract Outputs
    signals_df = results.get('signals_df', pd.DataFrame())
    
    # 4. Analyze
    report = {
        "stage": "compute_market_kinematics",
        "date": date,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {}
    }
    
    if not signals_df.empty:
        # Check Columns
        windows = ['1min', '5min']
        kin_cols = []
        for w in windows:
            kin_cols.extend([f'velocity_{w}', f'velocity_{w}_abs', f'acceleration_{w}', f'jerk_{w}', f'momentum_{w}'])
        
        missing_cols = [c for c in kin_cols if c not in signals_df.columns]
        
        if missing_cols:
             report["status"] = "failed_integrity"
             logger.error(f"Missing columns: {missing_cols}")
        
        # Stats
        stats = {}
        for col in kin_cols:
            if col in signals_df.columns:
                stats[col] = {
                    "mean": float(signals_df[col].mean()),
                    "min": float(signals_df[col].min()),
                    "max": float(signals_df[col].max()),
                    "std": float(signals_df[col].std()) if len(signals_df) > 1 else 0
                }
        
        # Sanity Check: velocity vs momentum
        # momentum_1min should be approx velocity_1min * 60 (check correlation or ratio)
        
        report["metrics"] = {
            "count": len(signals_df),
            "stats": stats
        }
        
    else:
        report["metrics"] = {"status": "empty", "count": 0}

    # Write Report
    report_path = Path(__file__).parent / "compute_market_kinematics_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    validate_compute_market_kinematics()
