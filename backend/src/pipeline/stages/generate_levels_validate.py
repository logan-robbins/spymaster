import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.pipeline.stages.generate_levels import GenerateLevelsStage
from src.pipeline.stages.init_market_state import InitMarketStateStage
from src.pipeline.stages.build_all_ohlcv import BuildAllOHLCVStage
from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.core.stage import StageContext
from src.common.config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_generate_levels(date: str = "2025-06-04", level_name: str = "PM_HIGH"):
    """
    Validation script for GenerateLevelsStage.
    
    1. Runs Setup (LoadBronze -> BuildAllOHLCV -> InitMarketState).
    2. Runs GenerateLevelsStage (for PM_HIGH).
    3. Verifies output structure and level correctness.
    """
    logger.info(f"Starting GenerateLevels validation for date: {date}, level: {level_name}")
    
    # 1. Setup Context with Dependencies
    ctx = StageContext(date=date, level=level_name)
    
    logger.info("Running Prerequisite Stages...")
    ctx = LoadBronzeStage().run(ctx)
    ctx = BuildAllOHLCVStage().run(ctx)
    ctx = InitMarketStateStage().run(ctx)
    
    # 2. Run Stage
    stage = GenerateLevelsStage()
    try:
        results = stage.execute(ctx)
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise

    # 3. Extract Outputs
    level_info = results.get('level_info')
    dynamic_levels = results.get('dynamic_levels', {})
    
    # 4. Analyze
    report = {
        "stage": "generate_levels",
        "date": date,
        "level_target": level_name,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {}
    }

    # Check Level Info
    if level_info:
        report["metrics"]["level_info"] = {
            "kind_names": level_info.kind_names,
            "level_prices": level_info.prices.tolist(),
            "count": len(level_info.prices)
        }
        
        # Verify only target level returned
        if len(level_info.kind_names) != 1 or level_info.kind_names[0] != level_name:
            report["status"] = "failed_integrity"
            logger.error(f"Expected only {level_name}, got {level_info.kind_names}")
    else:
        report["status"] = "failed_integrity"
        logger.error("LevelInfo is missing/empty")

    # Check Dynamic Levels
    if dynamic_levels:
        report["metrics"]["dynamic_levels_count"] = len(dynamic_levels)
        if level_name in dynamic_levels:
            series = dynamic_levels[level_name]
            report["metrics"][level_name] = {
                "count": len(series),
                "nans": int(series.isna().sum()),
                "min": float(series.min()),
                "max": float(series.max())
            }
            if series.isna().all():
                 logger.warning(f"Dynamic level series for {level_name} is all NaN (might be expected if not reached yet)")
        else:
            report["status"] = "failed_integrity"
            logger.error(f"Dynamic series for {level_name} missing")
    else:
        report["status"] = "failed_integrity"
        logger.error("Dynamic levels dict missing/empty")

    # Write Report
    report_path = Path(__file__).parent / "generate_levels_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    validate_generate_levels()
