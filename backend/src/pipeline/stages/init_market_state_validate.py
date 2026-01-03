import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.pipeline.stages.init_market_state import InitMarketStateStage
from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.core.stage import StageContext
from src.common.config import CONFIG
from src.core.market_state import MarketState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_init_market_state(date: str = "2025-06-04"):
    """
    Validation script for InitMarketStateStage.
    
    1. Runs LoadBronze.
    2. Runs InitMarketState.
    3. Verifies MarketState integrity and Greeks.
    """
    logger.info(f"Starting InitMarketState validation for date: {date}")
    
    # 1. Prerequisite: Load Data
    ctx = StageContext(date=date, level="VALIDATION")
    load_stage = LoadBronzeStage()
    ctx = load_stage.run(ctx)
    
    # 2. Run Stage
    stage = InitMarketStateStage()
    try:
        results = stage.execute(ctx)
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise

    # 3. Extract Outputs
    market_state = results.get('market_state')
    option_trades_df = results.get('option_trades_df', pd.DataFrame())
    spot_price = results.get('spot_price')
    
    # 4. Analyze
    report = {
        "stage": "init_market_state",
        "date": date,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {}
    }
    
    # Spot Price Check
    if spot_price is None or spot_price < 3000 or spot_price > 10000:
        report["status"] = "failed_integrity"
        logger.error(f"Invalid Spot Price: {spot_price}")
    
    report["metrics"]["spot_price"] = spot_price
    
    # Options Analysis
    if not option_trades_df.empty:
        # Check Greeks
        greeks_stats = {}
        for col in ['delta', 'gamma']:
            if col in option_trades_df.columns:
                greeks_stats[col] = {
                    "mean": float(option_trades_df[col].mean()),
                    "nans": int(option_trades_df[col].isna().sum()),
                    "zeros": int((option_trades_df[col] == 0).sum())
                }
            else:
                 greeks_stats[col] = "missing"
                 report["status"] = "failed_integrity"
                 logger.error(f"Missing Greek column: {col}")
        
        report["metrics"]["greeks"] = greeks_stats
        
        # Check Aggressor Inference
        if 'aggressor' in option_trades_df.columns:
             counts = option_trades_df['aggressor'].value_counts().to_dict()
             report["metrics"]["aggressor_inference"] = {str(k): int(v) for k, v in counts.items()}
    else:
        report["metrics"]["options"] = {"status": "empty"}

    # MarketState Object Check
    if isinstance(market_state, MarketState):
        report["metrics"]["market_state"] = {
            "initialized": True,
            "option_flow_count": len(market_state.option_flows)
        }
        if len(market_state.option_flows) != len(option_trades_df):
             logger.warning(f"MarketState flow count ({len(market_state.option_flows)}) != DataFrame count ({len(option_trades_df)})")
    else:
        report["metrics"]["market_state"] = {"initialized": False}
        report["status"] = "failed_integrity"
        logger.error("MarketState object not returned or invalid")

    # Write Report
    report_path = Path(__file__).parent / "init_market_state_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    validate_init_market_state()
