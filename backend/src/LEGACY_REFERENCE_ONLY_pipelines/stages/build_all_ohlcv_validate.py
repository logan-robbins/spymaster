import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.pipeline.stages.build_all_ohlcv import BuildAllOHLCVStage
from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.core.stage import StageContext
from src.common.config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_ohlcv_stats(df: pd.DataFrame, freq: str) -> Dict[str, Any]:
    """Analyze OHLCV dataframe integrity."""
    if df.empty:
        return {"status": "empty", "freq": freq}
    
    # Check Price Integrity
    invalid_high_low = (df['high'] < df['low']).sum()
    invalid_close_open = (df['close'] < 0) | (df['open'] < 0) # Just sanity
    zeros_vol = (df['volume'] == 0).sum()
    
    # Check Frequency
    if 'ts_ns' in df.columns:
        diffs = np.diff(df['ts_ns']) / 1e9
        freq_stats = {
            "mean_diff_sec": float(diffs.mean()) if len(diffs) > 0 else 0,
            "min_diff_sec": float(diffs.min()) if len(diffs) > 0 else 0,
            "max_diff_sec": float(diffs.max()) if len(diffs) > 0 else 0,
        }
    else:
        freq_stats = {}

    return {
        "count": len(df),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "integrity": {
            "invalid_high_low": int(invalid_high_low),
            "negative_prices": int(invalid_close_open.sum()),
            "zero_volume_bars": int(zeros_vol)
        },
        "freq_stats": freq_stats,
        "price_stats": {
            "min": float(df['low'].min()),
            "max": float(df['high'].max()),
            "mean": float(df['close'].mean())
        }
    }

def validate_build_ohlcv(date: str = "2025-06-04"):
    """
    Validation script for BuildAllOHLCVStage.
    
    1. Runs LoadBronze (prereq).
    2. Runs BuildAllOHLCV.
    3. Analyzes 10s, 1m, 2m bars and ATR/Vol.
    4. Writes report.
    """
    logger.info(f"Starting BuildAllOHLCV validation for date: {date}")
    
    # Prerequisite: Load Data
    ctx = StageContext(date=date, level="VALIDATION")
    load_stage = LoadBronzeStage()
    ctx = load_stage.run(ctx)
    
    # Run Stage
    stage = BuildAllOHLCVStage()
    try:
        results = stage.execute(ctx)
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise

    # Extract Data
    ohlcv_10s = results.get('ohlcv_10s', pd.DataFrame())
    ohlcv_1min = results.get('ohlcv_1min', pd.DataFrame())
    ohlcv_2min = results.get('ohlcv_2min', pd.DataFrame())
    atr = results.get('atr', pd.Series(dtype=float))
    vol = results.get('volatility', pd.Series(dtype=float))
    warmup_dates = results.get('warmup_dates', [])
    
    logger.info(f"Generated: 10s={len(ohlcv_10s)}, 1min={len(ohlcv_1min)}, 2min={len(ohlcv_2min)}")

    # Analyze
    report = {
        "stage": "build_all_ohlcv",
        "date": date,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "metrics": {}
    }
    
    report["metrics"]["ohlcv_10s"] = analyze_ohlcv_stats(ohlcv_10s, "10s")
    report["metrics"]["ohlcv_1min"] = analyze_ohlcv_stats(ohlcv_1min, "1min")
    report["metrics"]["ohlcv_2min"] = analyze_ohlcv_stats(ohlcv_2min, "2min")
    
    # Check Warmup in 2min
    if not ohlcv_2min.empty:
        # Check if start time is before session start (assuming 4am session start for date)
        session_start = pd.Timestamp(date, tz="America/New_York").replace(hour=4, minute=0, second=0)
        actual_start = ohlcv_2min.index.min().tz_convert("America/New_York")
        
        has_warmup = actual_start < session_start
        report["metrics"]["warmup"] = {
            "has_warmup_data": bool(has_warmup),
            "warmup_days_loaded": len(warmup_dates),
            "start_time": str(actual_start)
        }
        if not has_warmup and CONFIG.SMA_WARMUP_DAYS > 0:
             logger.warning("No warmup data detected in 2min bars despite configuration.")
    
    # Check ATR/Vol
    report["metrics"]["indicators"] = {
        "atr": {
            "count": len(atr),
            "nans": int(atr.isna().sum()),
            "mean": float(atr.mean())
        },
        "volatility": {
            "count": len(vol),
            "nans": int(vol.isna().sum()),
            "mean": float(vol.mean())
        }
    }
    
    # Final Validation Status
    if report["metrics"]["ohlcv_1min"]["integrity"]["invalid_high_low"] > 0:
        report["status"] = "failed_integrity"
        logger.error("Data Integrity Check Failed (High < Low)")

    # Write Report
    report_path = Path(__file__).parent / "build_all_ohlcv_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report written to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    validate_build_ohlcv()
