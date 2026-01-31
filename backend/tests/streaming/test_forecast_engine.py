import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.serving.velocity_streaming import ForecastEngine, VELOCITY_COLUMNS

def test_forecast_engine_runs():
    # Create dummy velocity data
    # Needs columns: VELOCITY_COLUMNS + u_near + u_p_slow
    
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
                "Omega": 4.0 if abs(t) == 10 else 1.0,
                "u_near": -0.2 if abs(t) < 5 else 0.1,
                "u_p_slow": 0.15 if abs(t) == 10 else 0.0,
            })
            
    df = pd.DataFrame(rows)
    for col in VELOCITY_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
            
    df_options = pd.DataFrame(
        {
            "window_end_ts_ns": [1000, 2000],
            "rel_ticks": [0, 0],
            "Omega_opt": [0.0, 0.0],
        }
    )

    engine = ForecastEngine(beta=0.5, gamma=0.1, horizon_s=5)
    result = engine.run_batch(df, df_options)
    
    assert not result.empty
    assert len(result) == 12  # 2 windows * (horizon 0..5)
    assert "predicted_tick_delta" in result.columns
    assert "D_up" in result.columns
    assert "D_down" in result.columns

    for window_id in [1000, 2000]:
        rows_window = result.loc[result["window_end_ts_ns"] == window_id]
        assert set(rows_window["horizon_s"].tolist()) == set(range(0, 6))
        diag = rows_window.loc[rows_window["horizon_s"] == 0].iloc[0]
        assert diag["D_up"] is not None
        assert diag["D_down"] is not None
    
if __name__ == "__main__":
    test_forecast_engine_runs()
