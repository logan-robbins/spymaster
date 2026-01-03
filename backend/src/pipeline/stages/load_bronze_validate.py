import json
import logging
import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.core.stage import StageContext
from src.common.config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_series_stats(series: pd.Series, name: str) -> Dict[str, Any]:
    """Calculate basic stats for a series, safe for JSON serialization."""
    if series.empty:
        return {"count": 0, "status": "empty"}
    
    return {
        "count": int(len(series)),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std()) if len(series) > 1 else 0.0,
        "zeros": int((series == 0).sum()),
        "nans": int(series.isna().sum())
    }

def validate_load_bronze(date: str = "2025-06-04"):
    """
    Validation script for LoadBronzeStage.
    
    1. Runs the stage.
    2. Analyzes output (Futures Trades, MBP-10, Options).
    3. Writes report to load_bronze_report.json.
    """
    logger.info(f"Starting LoadBronze validation for date: {date}")
    
    # 1. Setup Context and Run Stage
    stage = LoadBronzeStage()
    ctx = StageContext(date=date, level="VALIDATION")
    
    try:
        results = stage.execute(ctx)
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise

    # 2. Extract Data
    trades = results.get('trades', [])
    trades_df = results.get('trades_df', pd.DataFrame())
    mbp10 = results.get('mbp10_snapshots', [])
    option_trades = results.get('option_trades_df', pd.DataFrame())
    
    logger.info(f"Data Loaded: Trades={len(trades)}, MBP10={len(mbp10)}, Options={len(option_trades)}")

    # 3. Analyze Data
    report = {
        "stage": "load_bronze",
        "date": date,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {}
    }

    # -- Futures Trades Analysis --
    if not trades_df.empty:
        # Check schema and monotonicity
        expected_cols = ['ts_event_ns', 'price', 'size', 'symbol']
        missing = [c for c in expected_cols if c not in trades_df.columns]
        if missing:
             logger.error(f"Trades missing columns: {missing}")
        
        # Check front-month dominance
        if 'symbol' in trades_df.columns:
            symbol_counts = trades_df['symbol'].value_counts(normalize=True)
            dominant = symbol_counts.index[0]
            pct = symbol_counts.iloc[0]
            logger.info(f"Futures Dominant Contract: {dominant} ({pct:.1%})")
            if pct < 0.95:
                logger.warning(f"Low front-month dominance: {pct:.1%}")

        report["metrics"]["futures_trades"] = {
            "count": len(trades_df),
            "price_stats": analyze_series_stats(trades_df["price"], "trade_price"),
            "size_stats": analyze_series_stats(trades_df["size"], "trade_size"),
            "time_range": {
                "start": int(trades_df["ts_event_ns"].min()),
                "end": int(trades_df["ts_event_ns"].max()),
                "duration_sec": (trades_df["ts_event_ns"].max() - trades_df["ts_event_ns"].min()) / 1e9
            }
        }
    else:
        report["metrics"]["futures_trades"] = {"status": "empty"}

    # -- MBP-10 Analysis --
    if mbp10:
        ts_start = mbp10[0].ts_event_ns
        ts_end = mbp10[-1].ts_event_ns
        
        # Check levels
        first_levels = len(mbp10[0].levels) if mbp10[0].levels else 0
        if first_levels != 10:
             logger.warning(f"MBP-10 First snapshot has {first_levels} levels (expected 10)")

        report["metrics"]["mbp10_snapshots"] = {
            "count": len(mbp10),
            "time_range": {
                "start": ts_start,
                "end": ts_end,
                "duration_sec": (ts_end - ts_start) / 1e9
            },
            "snapshot_ratio": sum(1 for m in mbp10 if m.is_snapshot) / len(mbp10),
            "levels_per_snapshot": first_levels
        }
    else:
        report["metrics"]["mbp10_snapshots"] = {"status": "empty"}

    # -- Options Trades Analysis --
    if not option_trades.empty:
        # Schema check
        opt_cols = ['ts_event_ns', 'underlying', 'strike', 'price', 'size']
        missing_opt = [c for c in opt_cols if c not in option_trades.columns]
        if missing_opt:
            logger.error(f"Options missing columns: {missing_opt}")

        # Check for 0DTE
        if 'exp_date' in option_trades.columns:
            # Simple check: exp_date should match session date (roughly) or be single day
            unique_exps = option_trades['exp_date'].unique()
            logger.info(f"Options Expirations: {unique_exps}")
        
        # Use ts_event_ns if available, else look for sip_timestamp (legacy?)
        ts_col = 'ts_event_ns' if 'ts_event_ns' in option_trades.columns else 'sip_timestamp'
        
        report["metrics"]["option_trades"] = {
            "count": len(option_trades),
            "price_stats": analyze_series_stats(option_trades["price"], "opt_price"),
            "size_stats": analyze_series_stats(option_trades["size"], "opt_size"),
             "time_range": {
                "start": int(option_trades[ts_col].min()) if ts_col in option_trades.columns else 0,
                "end": int(option_trades[ts_col].max()) if ts_col in option_trades.columns else 0,
                "duration_sec": ((option_trades[ts_col].max() - option_trades[ts_col].min()) / 1e9) if ts_col in option_trades.columns else 0
            }
        }
    else:
         report["metrics"]["option_trades"] = {"status": "empty"}

    # 4. Write Report
    report_path = Path(__file__).parent / "load_bronze_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report written to {report_path}")
    print(json.dumps(report, indent=2)) # Print for caller

if __name__ == "__main__":
    validate_load_bronze()
