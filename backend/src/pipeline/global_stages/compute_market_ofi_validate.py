import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.pipeline.global_stages.compute_market_ofi import ComputeMarketOFIStage
from src.pipeline.global_stages.generate_time_grid import GenerateTimeGridStage
from src.pipeline.stages.init_market_state import InitMarketStateStage
from src.pipeline.stages.build_all_ohlcv import BuildAllOHLCVStage
from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.core.stage import StageContext
from src.common.config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_compute_market_ofi(date: str = "2025-06-04"):
    """
    Validation script for ComputeMarketOFIStage.
    
    1. Runs prerequisites.
    2. Runs ComputeMarketOFIStage.
    3. Verifies global OFI columns.
    """
    logger.info(f"Starting ComputeMarketOFI validation for date: {date}")
    
    # 1. Setup Context
    ctx = StageContext(date=date, level="GLOBAL")
    
    logger.info("Running Prerequisite Stages...")
    ctx = LoadBronzeStage().run(ctx)
    ctx = BuildAllOHLCVStage().run(ctx)
    ctx = InitMarketStateStage().run(ctx)
    ctx = GenerateTimeGridStage(interval_seconds=30.0).run(ctx)
    
    # 2. Run Stage
    stage = ComputeMarketOFIStage()
    try:
        results = stage.execute(ctx)
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise

    # 3. Extract Outputs
    signals_df = results.get('signals_df', pd.DataFrame())
    
    # 4. Analyze
    report = {
        "stage": "compute_market_ofi",
        "date": date,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {}
    }
    
    if not signals_df.empty:
        # Check Columns
        ofi_cols = ['ofi_30s', 'ofi_60s', 'ofi_120s', 'ofi_300s', 'ofi_acceleration']
        
        missing_cols = [c for c in ofi_cols if c not in signals_df.columns]
        
        if missing_cols:
             report["status"] = "failed_integrity"
             logger.error(f"Missing columns: {missing_cols}")
        
        # Stats
        stats = {}
        for col in ofi_cols:
            if col in signals_df.columns:
                stats[col] = {
                    "mean": float(signals_df[col].mean()),
                    "min": float(signals_df[col].min()),
                    "max": float(signals_df[col].max()),
                    "std": float(signals_df[col].std()) if len(signals_df) > 1 else 0
                }
        
        report["metrics"] = {
            "count": len(signals_df),
            "stats": stats
        }
        
    else:
        report["metrics"] = {"status": "empty", "count": 0}

    # Write Report
    report_path = Path(__file__).parent / "compute_market_ofi_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    validate_compute_market_ofi()
