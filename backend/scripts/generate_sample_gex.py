"""
Generate sample GEX Surface data for development/testing.
Matches the schema in gex_surface_1s.avsc.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from src.data_eng.config import load_config
from src.data_eng.contracts import enforce_contract, load_avro_contract
from src.data_eng.io import write_partition

# Constants from IMPLEMENT.md
GEX_STRIKE_STEP_POINTS = 5
GEX_MAX_STRIKE_OFFSETS = 12  # Updated from 30 per user feedback
PRICE_SCALE = 1e-9
WINDOW_NS = 1_000_000_000
SESSION_START_HOUR = 6
SESSION_END_HOUR = 16

def generate_sample_gex(symbol: str, dt: str):
    cfg = load_config(backend_dir, backend_dir / "src/data_eng/config/datasets.yaml")
    
    # Base parameters
    base_spot = 6000.0  # ES spot around 6000
    
    # Generate 1-second windows for 10 hours (06:00 - 16:00)
    session_start = pd.Timestamp(f"{dt} 06:00:00", tz="Etc/GMT+5")
    session_end = pd.Timestamp(f"{dt} 16:00:00", tz="Etc/GMT+5")
    
    start_ns = int(session_start.tz_convert("UTC").value)
    end_ns = int(session_end.tz_convert("UTC").value)
    
    # Sample every 60 seconds for 10 hours = 600 windows (faster generation)
    sample_interval_ns = 60 * WINDOW_NS
    
    rows = []
    rng = np.random.default_rng(42)
    
    # Simulate a price path with drift
    spot_path = [base_spot]
    for i in range(600):
        drift = rng.normal(0, 0.5)
        spot_path.append(max(5800, min(6200, spot_path[-1] + drift)))
    
    window_idx = 0
    for window_end_ns in range(start_ns + sample_interval_ns, end_ns + 1, sample_interval_ns):
        window_start_ns = window_end_ns - sample_interval_ns
        spot = spot_path[window_idx]
        
        # Strike grid
        strike_ref = round(spot / GEX_STRIKE_STEP_POINTS) * GEX_STRIKE_STEP_POINTS
        for offset in range(-GEX_MAX_STRIKE_OFFSETS, GEX_MAX_STRIKE_OFFSETS + 1):
            strike_points = strike_ref + offset * GEX_STRIKE_STEP_POINTS
            strike_price_int = int(round(strike_points / PRICE_SCALE))
            
            # Simulate GEX: higher near ATM, decays away
            distance = abs(strike_points - spot)
            base_gex = max(0, 1000000 - distance * 10000)  # Peak near ATM
            
            # Add noise
            gex_call_abs = base_gex * (0.5 + 0.5 * rng.random())
            gex_put_abs = base_gex * (0.5 + 0.5 * rng.random())
            gex_abs = gex_call_abs + gex_put_abs
            gex = gex_call_abs - gex_put_abs
            gex_imbalance = gex / (gex_abs + 1e-6)
            
            rows.append({
                "window_start_ts_ns": window_start_ns,
                "window_end_ts_ns": window_end_ns,
                "underlying": symbol,
                "strike_price_int": strike_price_int,
                "underlying_spot_ref": spot,
                "strike_points": strike_points,
                "gex_call_abs": gex_call_abs,
                "gex_put_abs": gex_put_abs,
                "gex_abs": gex_abs,
                "gex": gex,
                "gex_imbalance_ratio": gex_imbalance,
                "d1_gex_abs": 0.0,
                "d2_gex_abs": 0.0,
                "d3_gex_abs": 0.0,
                "d1_gex": 0.0,
                "d2_gex": 0.0,
                "d3_gex": 0.0,
                "d1_gex_imbalance_ratio": 0.0,
                "d2_gex_imbalance_ratio": 0.0,
                "d3_gex_imbalance_ratio": 0.0,
            })
        
        window_idx += 1
    
    df = pd.DataFrame(rows)
    
    # Compute derivatives
    df = df.sort_values(["strike_price_int", "window_end_ts_ns"])
    for col in ["gex_abs", "gex", "gex_imbalance_ratio"]:
        df[f"d1_{col}"] = df.groupby("strike_price_int")[col].diff().fillna(0.0)
        df[f"d2_{col}"] = df.groupby("strike_price_int")[f"d1_{col}"].diff().fillna(0.0)
        df[f"d3_{col}"] = df.groupby("strike_price_int")[f"d2_{col}"].diff().fillna(0.0)
    
    # Contract enforcement
    contract_path = backend_dir / cfg.dataset("silver.future_option_mbo.gex_surface_1s").contract
    contract = load_avro_contract(contract_path)
    df = enforce_contract(df, contract)
    
    # Write
    write_partition(
        cfg=cfg,
        dataset_key="silver.future_option_mbo.gex_surface_1s",
        symbol=symbol,
        dt=dt,
        df=df,
        contract_path=contract_path,
        inputs=[],
        stage="generate_sample_gex"
    )
    
    print(f"Generated {len(df)} sample GEX rows for {symbol} {dt}")

if __name__ == "__main__":
    generate_sample_gex("ESH6", "2026-01-06")
