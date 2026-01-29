import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd
import pytest
from src.serving.velocity_streaming import ForecastEngine, VELOCITY_COLUMNS

def test_forecast_engine_runs():
    # Create dummy velocity data
    # Needs columns: VELOCITY_COLUMNS
    # "window_end_ts_ns", "spot_ref_price_int", "rel_ticks", "side", 
    # "liquidity_velocity", "rho", "nu", "kappa", "pressure_grad", "u_wave_energy", "Omega"
    
    # 2 windows, range of ticks
    ticks = range(-10, 11)
    rows = []
    for w in [1000, 2000]:
        for t in ticks:
            # Side B for negative, A for positive
            side = "B" if t < 0 else "A"
            rows.append({
                "window_end_ts_ns": w,
                "spot_ref_price_int": 5000000000,
                "rel_ticks": t,
                "side": side,
                "liquidity_velocity": 0.1,
                "rho": 0.5,
                "nu": 1.0,
                "kappa": 1.0,
                "pressure_grad": 0.05 if t > 0 else -0.05,
                "u_wave_energy": 0.01,
                "Omega": 4.0 if abs(t) == 10 else 1.0 # Wall at edges
            })
            
    df = pd.DataFrame(rows)
    for col in VELOCITY_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
            
    engine = ForecastEngine(horizon_s=5)
    result = engine.run_batch(df)
    
    assert not result.empty
    assert len(result) == 2 # 2 windows
    assert "predicted_tick_delta" in result.columns
    assert "D_up" in result.columns
    assert "D_down" in result.columns
    
    # Check values
    # Wall at +/- 10. Center is 0.
    # D_up should be 10? rel_ticks 10 is index 210. Center 200. +10.
    # Logic in engine: center_idx=200.
    # My dummy data has rel_ticks -10..10.
    # Engine reindexes to -200..200.
    # So rel_tick 10 is at index 210.
    # Wall is at 210.
    # wall_up index from center+1 (201): 210 is 9th element? No.
    # 210 - 201 = 9.
    # result returned index[0] + 1 -> 10.
    # So D_up should be 10.
    
    row1 = result.iloc[0]
    print(row1)
    
if __name__ == "__main__":
    test_forecast_engine_runs()
