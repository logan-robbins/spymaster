
import pytest
import math
import pandas as pd
from dataclasses import dataclass
from .compute_radar_vacuum_1s import (
    _snapshot,
    _bucket_for,
    _compute_base_features,
    OrderState,
    ASK_ABOVE_AT,
    ASK_ABOVE_NEAR,
    ASK_ABOVE_FAR,
    ASK_ABOVE_MID,
    BID_BELOW_AT,
    BID_BELOW_NEAR,
    BID_BELOW_FAR,
    TICK_INT,
    PRICE_SCALE,
    DELTA_TICKS,
    BUCKET_OUT
)

# Mock OrderState for testing - needs to match the class in compute_radar_vacuum_1s
# @dataclass
# class OrderState:
#     side: str
#     price_int: int
#     qty: int
#     ts_enter_price: int

def test_bucket_logic():
    spot = 5000000000 # 5.0
    # Tick = 0.25 => 250,000,000 int
    tick_int = 250000000
    
    # AT: 0-2 ticks (0 to 0.50 away)
    # Price 5.0 -> 0 ticks -> AT
    assert _bucket_for("A", spot, spot) == ASK_ABOVE_AT
    
    # Price 5.25 -> 1 tick -> AT
    assert _bucket_for("A", spot + tick_int, spot) == ASK_ABOVE_AT
    
    # Price 5.75 -> 3 ticks -> NEAR (3-5)
    assert _bucket_for("A", spot + 3*tick_int, spot) == ASK_ABOVE_NEAR
    
    # Price 7.50 -> 10 ticks -> MID (6-14)
    assert _bucket_for("A", spot + 10*tick_int, spot) == ASK_ABOVE_MID
    
    # Price 8.75 -> 15 ticks -> FAR (15-20)
    assert _bucket_for("A", spot + 15*tick_int, spot) == ASK_ABOVE_FAR
    
    # Price 10.0 -> 20 ticks -> FAR
    assert _bucket_for("A", spot + 20*tick_int, spot) == ASK_ABOVE_FAR
    
    # Price 10.25 -> 21 ticks -> OUT
    assert _bucket_for("A", spot + 21*tick_int, spot) == BUCKET_OUT

def test_snapshot_empty_book_fix_behavior():
    orders = {}
    p_ref = 5000.0
    p_ref_int = int(round(p_ref / PRICE_SCALE))

    snapshot = _snapshot(orders, p_ref_int)

    # Improved logic should return DELTA_TICKS (20.0) or similar for empty book
    # In compute_radar_vacuum_1s.py:
    # if d["ask_depth_total"] < TINY_TOL: d["d_ask_ticks"] = float(DELTA_TICKS)
    assert snapshot["d_ask_ticks"] >= 20.0, f"Empty ask book should have max distance, got {snapshot['d_ask_ticks']}"
    assert snapshot["d_bid_ticks"] >= 20.0, f"Empty bid book should have max distance, got {snapshot['d_bid_ticks']}"
    
    # Check bbo
    assert snapshot["bbo_ask_ticks"] == float(DELTA_TICKS)


def test_snapshot_bbo_tracking():
    # Setup: P_ref = 100.00
    TICK_SIZE = 0.25
    p_ref_val = 100.0
    p_ref_int = int(round(p_ref_val / PRICE_SCALE))
    
    ask_at_price = int(round((p_ref_val + 1 * TICK_SIZE) / PRICE_SCALE))
    ask_far_price = int(round((p_ref_val + 10 * TICK_SIZE) / PRICE_SCALE))
    
    # OrderState(side, price_int, qty, ts_enter_price)
    orders = {
        1: OrderState("A", ask_at_price, 1, 0),
        2: OrderState("A", ask_far_price, 100, 0)
    }
    
    snapshot = _snapshot(orders, p_ref_int)
    
    # COM should be pulled heavily towards the far order (qty 100 vs 1)
    # COM price ~= (1*100.25 + 100*102.50)/101 ~= 102.47
    # Dist ticks ~= (102.47 - 100.0) / 0.25 ~= 9.9 ticks
    assert snapshot["d_ask_ticks"] > 8.0
    
    # BBO should be at 1 tick
    # In logic: if min_ask is None: ... else: max((min_ask - spot_ref)/TICK_INT, 0.0)
    # min_ask is 100.25. spot is 100.0. Diff 0.25. Ticks = 1.
    assert abs(snapshot["bbo_ask_ticks"] - 1.0) < 0.1, f"BBO should be 1.0, got {snapshot.get('bbo_ask_ticks')}"
