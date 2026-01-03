import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.pipeline.stages.detect_interaction_zones import DetectInteractionZonesStage
from src.pipeline.stages.generate_levels import GenerateLevelsStage
from src.pipeline.stages.init_market_state import InitMarketStateStage
from src.pipeline.stages.build_all_ohlcv import BuildAllOHLCVStage
from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.core.stage import StageContext

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_detect_zones(date: str = "2025-06-04", level_name: str = "PM_HIGH"):
    """
    Validation script for DetectInteractionZonesStage.
    
    1. Runs prerequisites (Load -> OHLCV -> State -> Levels).
    2. Runs DetectInteractionZonesStage.
    3. Verifies event detection.
    """
    logger.info(f"Starting DetectInteractionZones validation for date: {date}, level: {level_name}")
    
    # 1. Setup Context
    ctx = StageContext(date=date, level=level_name)
    
    logger.info("Running Prerequisite Stages...")
    ctx = LoadBronzeStage().run(ctx)
    ctx = BuildAllOHLCVStage().run(ctx)
    ctx = InitMarketStateStage().run(ctx)
    ctx = GenerateLevelsStage().run(ctx)
    
    # 2. Run Stage
    stage = DetectInteractionZonesStage()
    try:
        results = stage.execute(ctx)
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise

    # 3. Extract Inputs
    touches_df = results.get('touches_df', pd.DataFrame())
    
    # 4. Analyze
    report = {
        "stage": "detect_interaction_zones",
        "date": date,
        "level_target": level_name,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {}
    }
    
    if not touches_df.empty:
        report["metrics"]["events"] = {
            "count": len(touches_df),
            "columns": touches_df.columns.tolist(),
            "directions": touches_df['direction'].value_counts().to_dict(),
            "level_prices": touches_df['level_price'].unique().tolist(),
            "first_event": str(touches_df.iloc[0]['timestamp']),
            "last_event": str(touches_df.iloc[-1]['timestamp'])
        }
        
        # Verify IDs are deterministic (check format)
        sample_id = touches_df.iloc[0]['event_id']
        parts = sample_id.split('_')
        # Expect: date_level_price_ts_dir => e.g. 20250604_PM_HIGH_604000_..._UP
        if len(parts) < 5:
             logger.warning(f"Event ID format seems unexpected: {sample_id}")
             
    else:
        report["metrics"]["events"] = {"status": "empty", "count": 0}
        # Not necessarily a failure if price didn't touch level, but useful to know.

    # Write Report
    report_path = Path(__file__).parent / "detect_interaction_zones_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    validate_detect_zones()
