
import numpy as np
import pandas as pd
from src.core.batch_engines import build_vectorized_market_data, compute_fuel_metrics_batch
from src.common.event_types import FuturesTrade, MBP10, Aggressor

def test_replay_logic():
    print("Testing Market Tide Replay Logic...")
    
    # 1. Create Mock Option Trades
    # 3 trades:
    # 1. Call, Strike 4000, Price 10.0, Size 1, Aggressor BUY (+1) -> Time 100
    # 2. Put, Strike 4000, Price 10.0, Size 1, Aggressor BUY (+1) -> Time 200
    # 3. Call, Strike 4000, Price 20.0, Size 1, Aggressor SELL (-1) -> Time 300
    
    df = pd.DataFrame({
        'ts_event_ns': [100, 200, 300],
        'strike': [4000.0, 4000.0, 4000.0],
        'right': ['C', 'P', 'C'],
        'price': [10.0, 10.0, 20.0],
        'size': [1, 1, 1],
        'aggressor': [1, 1, -1], # BUY, BUY, SELL
        'gamma': [0.01, 0.01, 0.01],
        'delta': [0.5, -0.5, 0.5]
    })
    
    # 2. Build Vectorized Data
    vmd = build_vectorized_market_data(
        trades=[], 
        mbp10_snapshots=[], 
        option_flows={}, 
        option_trades_df=df
    )
    
    print(f"VMD Arrays: {len(vmd.opt_ts_ns)} trades loaded.")
    print(f"VMD opt_ts: {vmd.opt_ts_ns}")
    print(f"VMD opt_premium: {vmd.opt_premium}")
    print(f"VMD opt_strikes: {vmd.opt_strikes}")
    
    # 3. Compute Metrics for Touches
    # Touch 1 at Time 50 (Before all) -> Should be 0
    # Touch 2 at Time 150 (After T1) -> Call +1000, Put 0
    # Touch 3 at Time 250 (After T2) -> Call +1000, Put +1000
    # Touch 4 at Time 350 (After T3) -> Call -1000 (+1000 - 2000), Put +1000
    
    touch_ts = np.array([50, 150, 250, 350], dtype=np.int64)
    level_prices = np.array([4000.0, 4000.0, 4000.0, 4000.0]) # At correct strike
    
    metrics = compute_fuel_metrics_batch(
        touch_ts_ns=touch_ts,
        level_prices=level_prices,
        market_data=vmd,
        strike_range=10.0
    )
    
    print("Results:")
    print("Call Tide:", metrics['call_tide'])
    print("Put Tide:", metrics['put_tide'])
    
    expected_call = np.array([0.0, 1000.0, 1000.0, -1000.0])
    expected_put = np.array([0.0, 0.0, 1000.0, 1000.0])
    
    np.testing.assert_allclose(metrics['call_tide'], expected_call, err_msg="Call Tide Mismatch")
    np.testing.assert_allclose(metrics['put_tide'], expected_put, err_msg="Put Tide Mismatch")
    
    print("SUCCESS: Logic is sound.")

if __name__ == "__main__":
    test_replay_logic()
