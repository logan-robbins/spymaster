import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.pipeline.global_stages.generate_time_grid import GenerateTimeGridStage
from src.pipeline.stages.init_market_state import InitMarketStateStage
from src.pipeline.stages.build_all_ohlcv import BuildAllOHLCVStage
from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.core.stage import StageContext
from src.common.config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_generate_time_grid(date: str = "2025-06-04"):
    """
    Validation script for GenerateTimeGridStage.
    
    1. Runs prerequisites.
    2. Runs GenerateTimeGridStage.
    3. Verifies time grid regularity and context columns.
    """
    logger.info(f"Starting GenerateTimeGrid validation for date: {date}")
    
    # 1. Setup Context
    ctx = StageContext(date=date, level="GLOBAL")
    
    logger.info("Running Prerequisite Stages...")
    ctx = LoadBronzeStage().run(ctx)
    ctx = BuildAllOHLCVStage().run(ctx)
    ctx = InitMarketStateStage().run(ctx)
    
    # 2. Run Stage
    stage = GenerateTimeGridStage(interval_seconds=30.0)
    try:
        results = stage.execute(ctx)
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise

    # 3. Extract Outputs
    signals_df = results.get('signals_df', pd.DataFrame())
    
    # 4. Analyze
    report = {
        "stage": "generate_time_grid",
        "date": date,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {}
    }
    
    if not signals_df.empty:
        # Check Columns
        req_cols = ['event_id', 'ts_ns', 'timestamp', 'date', 'minutes_since_open', 'or_active']
        missing_cols = [c for c in req_cols if c not in signals_df.columns]
        
        if missing_cols:
             report["status"] = "failed_integrity"
             logger.error(f"Missing columns: {missing_cols}")
        
        # Check Interval Regularity
        ts = signals_df['ts_ns'].sort_values().values
        diffs = np.diff(ts)
        # Expected interval = 30s * 1e9 = 30,000,000,000 ns
        expected_ns = 30_000_000_000
        # Check if most diffs are close to expected (allow some jitter if logic varies, but grid should be exact)
        is_regular = np.allclose(diffs, expected_ns)
        
        if not is_regular:
            unique_diffs = np.unique(diffs)
            logger.warning(f"Time grid irregular. Unique intervals: {unique_diffs}")
            # Identify where irregularity happens (maybe gap in session?)
            # But the code generates np.arange, so it SHOULD be regular unless filtered?
        
        # Stats
        stats = {
            "count": len(signals_df),
            "start_time": signals_df['timestamp'].min().isoformat(),
            "end_time": signals_df['timestamp'].max().isoformat(),
            "interval_seconds": 30.0,
            "is_regular": bool(is_regular)
        }
        
        report["metrics"] = stats
        
    else:
        report["metrics"] = {"status": "empty", "count": 0}

    # Write Report
    report_path = Path(__file__).parent / "generate_time_grid_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    validate_generate_time_grid()
