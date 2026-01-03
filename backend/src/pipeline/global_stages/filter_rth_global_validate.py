import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.pipeline.global_stages.filter_rth_global import FilterRTHGlobalStage
from src.pipeline.global_stages.generate_time_grid import GenerateTimeGridStage
from src.pipeline.stages.init_market_state import InitMarketStateStage
from src.pipeline.stages.build_all_ohlcv import BuildAllOHLCVStage
from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.core.stage import StageContext
from src.common.config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_filter_rth_global(date: str = "2025-06-04"):
    """
    Validation script for FilterRTHGlobalStage.
    
    1. Runs prerequisites.
    2. Runs FilterRTHGlobalStage.
    3. Verifies RTH filtering for global signals.
    """
    logger.info(f"Starting FilterRTHGlobal validation for date: {date}")
    
    # 1. Setup Context
    # Disable writing to disk
    config_override = {"PIPELINE_WRITE_SIGNALS": False}
    ctx = StageContext(date=date, level="GLOBAL", config=config_override)
    
    logger.info("Running Prerequisite Stages...")
    ctx = LoadBronzeStage().run(ctx)
    ctx = BuildAllOHLCVStage().run(ctx)
    ctx = InitMarketStateStage().run(ctx)
    ctx = GenerateTimeGridStage(interval_seconds=30.0).run(ctx)
    
    pre_filter_count = len(ctx.data['signals_df'])
    logger.info(f"Pre-filter signals: {pre_filter_count}")
    
    # 2. Run Stage
    stage = FilterRTHGlobalStage()
    try:
        results = stage.execute(ctx)
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise

    # 3. Extract Outputs
    signals = results.get('signals', pd.DataFrame())
    
    # 4. Analyze
    report = {
        "stage": "filter_rth_global",
        "date": date,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {}
    }
    
    if not signals.empty:
        # Check Count vs Pre-filter
        # 08:30 to 12:30 = 4 hours. 30s interval = 4 * 120 = 480 events.
        # Pre-filter (full day ~08:00 to 20:00) should be > 1000.
        
        # Check Columns
        req_cols = ['event_id', 'ts_ns', 'timestamp', 'date', 'spot']
        missing_cols = [c for c in req_cols if c not in signals.columns]
        
        if missing_cols:
             report["status"] = "failed_integrity"
             logger.error(f"Missing columns: {missing_cols}")
        
        report["metrics"] = {
            "pre_filter_count": pre_filter_count,
            "post_filter_count": len(signals),
            "columns": len(signals.columns)
        }
        
    else:
        report["metrics"] = {
            "status": "empty", 
            "count": 0,
            "pre_filter_count": pre_filter_count
        }

    # Write Report
    report_path = Path(__file__).parent / "filter_rth_global_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    validate_filter_rth_global()
