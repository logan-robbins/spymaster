
import pytest
import pandas as pd
import numpy as np
from src.data_eng.stages.silver.future_option_mbo.options_book_engine import OptionsBookEngine

ACTION_ADD = "A"
ACTION_CANCEL = "C"
ACTION_MODIFY = "M"
ACTION_FILL = "F"

def test_options_book_engine():
    # Setup
    engine = OptionsBookEngine(window_ns=1_000_000_000, rest_ns=500_000_000)
    
    # Create Synthetic Data
    # 1. Add order at T=100ms
    # 2. Add order at T=200ms
    # 3. Pull order 1 at T=700ms (Resting > 500ms)
    # 4. Pull order 2 at T=800ms (Resting > 500ms? No, 800-200=600 > 500. Yes.)
    
    # 1. Add Order 1: side A, price 100, qty 10
    row1 = {
        "ts_event": 100_000_000, "instrument_id": 1, "action": ACTION_ADD, "side": "A", 
        "price": 100000000000, "size": 10, "order_id": 1, "flags": 0, "sequence": 1
    }
    
    # 2. Add Order 2: side A, price 100, qty 5
    row2 = {
        "ts_event": 200_000_000, "instrument_id": 1, "action": ACTION_ADD, "side": "A",
        "price": 100000000000, "size": 5, "order_id": 2, "flags": 0, "sequence": 2
    }
    
    # 3. Pull Order 1 (Resting) at 800ms
    row3 = {
        "ts_event": 800_000_000, "instrument_id": 1, "action": ACTION_CANCEL, "side": "A",
        "price": 100000000000, "size": 10, "order_id": 1, "flags": 0, "sequence": 3
    }
    
    # Window 2: Add Fill
    row4 = {
        "ts_event": 1_100_000_000, "instrument_id": 1, "action": ACTION_FILL, "side": "A",
        "price": 100000000000, "size": 2, "order_id": 2, "flags": 0, "sequence": 4
    }
    
    df = pd.DataFrame([row1, row2, row3, row4])
    
    # Run
    res_flow, res_bbo = engine.process_batch(df)
    
    print("Flow:\n", res_flow)
    print("BBO:\n", res_bbo)
    
    # Validation Window 1 (0-1s)
    # Total Adds: 10 + 5 = 15
    # Total Pulls: 10
    # Total Pull Rest: 10 (800-100 = 700 > 500)
    # End Depth: 5
    
    w1 = res_flow[res_flow["window_end_ts_ns"] == 1_000_000_000].iloc[0]
    assert w1["add_qty"] == 15
    assert w1["pull_qty"] == 10
    assert w1["pull_rest_qty"] == 10
    assert w1["depth_total"] == 5
    
    # Validation Window 2 (1-2s)
    # Fill: 2
    # End Depth: 3
    w2 = res_flow[res_flow["window_end_ts_ns"] == 2_000_000_000].iloc[0]
    assert w2["fill_qty"] == 2
    assert w2["depth_total"] == 3
    
    # Validation BBO
    # Window 1: Best Bid = 100, Best Ask = Empty? No orders were Ask?
    # Wait, all orders were Side A (Ask).
    # So Best Ask should be 100. Best Bid should be 0 (or empty).
    # mid = (0 + 100) * 0.5? Logic check: if bb>0 and ba>0.
    # If using _numba_mbo_to_mids logic: requires BOTH bb > 0 and ba > 0.
    # My synthetic data only added Asks. So NO BBO should be emitted.
    
    assert res_bbo.empty
    
    print("Test Passed!")

if __name__ == "__main__":
    test_options_book_engine()
