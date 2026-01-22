
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Setup environment
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from src.data_eng.config import AppConfig, load_config
from src.data_eng.contracts import enforce_contract, load_avro_contract
from src.data_eng.io import write_partition, partition_ref

# Default robust scaling bounds (approximate from experience)
# These allow the pipeline to run; they will be refined by real data later.
DEFAULTS = {
    "pull_add_log": {"q05": -2.0, "q95": 2.0},
    "log1p_pull_intensity_rest": {"q05": 0.0, "q95": 0.5},
    "log1p_erosion_norm": {"q05": 0.0, "q95": 0.5},
    "d2_pull_add_log": {"q05": -1.0, "q95": 1.0},
    "wall_strength_log": {"q05": 0.0, "q95": 10.0},
    "gex_abs": {"q05": 0.0, "q95": 1000000.0} # Placeholder
}

def bootstrap_calibration(symbol: str, dt: str):
    """
    Writes a synthetic calibration file for a specific date to unblock the pipeline
    when insufficient history exists for the real calibration stage.
    """
    print(f"Bootstrapping calibration for {symbol} {dt}...")
    
    cfg = load_config(backend_dir, backend_dir / "src/data_eng/config/datasets.yaml") # Load config
    
    dataset_key = "gold.hud.physics_norm_calibration"
    
    rows: List[Dict[str, Any]] = []
    
    for metric, bounds in DEFAULTS.items():
        rows.append({
            "metric_name": metric,
            "q05": float(bounds["q05"]),
            "q95": float(bounds["q95"]),
            "lookback_sessions": 0, # logical indicator of bootstrap
            "session_window": "BOOTSTRAP",
            "asof_dt": dt
        })
        
    df_out = pd.DataFrame(rows)
    
    # Load contract
    contract_path = backend_dir / cfg.dataset(dataset_key).contract
    contract = load_avro_contract(contract_path)
    
    df_out = enforce_contract(df_out, contract)
    
    # Write
    write_partition(
        cfg=cfg,
        dataset_key=dataset_key,
        symbol=symbol,
        dt=dt,
        df=df_out,
        contract_path=contract_path,
        inputs=[], # No lineage inputs
        stage="bootstrap_calibration"
    )
    print(f"Success. Wrote {len(rows)} calibration metrics for {dt}.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Default for the task at hand
        bootstrap_calibration("ES", "2026-01-06")
    else:
        bootstrap_calibration(sys.argv[1], sys.argv[2])
