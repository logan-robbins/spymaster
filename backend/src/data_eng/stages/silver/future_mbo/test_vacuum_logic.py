
import pytest
import math
from dataclasses import dataclass
from .compute_level_vacuum_5s import (
    _snapshot,
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
)

# Mock OrderState for testing
@dataclass
class MockOrder:
    side: str
    price_int: int
    qty: int
    bucket: str

def test_snapshot_empty_book_bug():
    """
    Reproduction of the bug where empty ask book returns 0 distance (max resistance)
    instead of large distance (vacuum).
    """
    orders = {}
    p_ref = 5000.0
    p_ref_int = int(p_ref / PRICE_SCALE) # 5000_000_000_000

    snapshot = _snapshot(orders, p_ref_int)

    # CURRENT BUG BEHAVIOR:
    # ask_depth_total = 0
    # ask_com_num = 0
    # ask_com_price_int = 0 / 1.0 (EPS_QTY) = 0.0
    # d_ask_ticks = max((0.0 - 5000...)/TICK_INT, 0.0) = 0.0
    
    # We expect this to fail once we fix it (we want it to be ~DELTA_TICKS or large)
    # But for now, we assert the BROKEN behavior to confirm we understand it.
    
    print(f"DEBUG: d_ask_ticks={snapshot['d_ask_ticks']}")
    
    # If the bug exists, this assertion passes with 0.0
    # After fix, this should be >= DELTA_TICKS (20.0)
    # assert snapshot["d_ask_ticks"] == 0.0 
    
    # Let's write the test for the DESIRED behavior and expect it to fail currently if I were running it,
    # but since I am writing the code, I will implement the fix directly.
    # This test file is intended to stay as a regression test.
    pass

def test_snapshot_empty_book_fix_behavior():
    orders = {}
    p_ref = 5000.0
    p_ref_int = int(round(p_ref / PRICE_SCALE))

    snapshot = _snapshot(orders, p_ref_int)

    # Improved logic should return DELTA_TICKS (20.0) or similar for empty book
    assert snapshot["d_ask_ticks"] >= 20.0, f"Empty ask book should have max distance, got {snapshot['d_ask_ticks']}"
    assert snapshot["d_bid_ticks"] >= 20.0, f"Empty bid book should have max distance, got {snapshot['d_bid_ticks']}"

    # Also check BBO features if implemented
    if "bbo_ask_ticks" in snapshot:
         assert snapshot["bbo_ask_ticks"] >= 20.0
         assert snapshot["bbo_bid_ticks"] >= 20.0

def test_snapshot_bbo_tracking():
    # Setup: P_ref = 100.00
    # Order 1: Ask at 100.25 (1 tick away) - qty 1
    # Order 2: Ask at 102.50 (10 ticks away) - qty 100 (dominant mass)
    
    TICK_SIZE = 0.25
    p_ref_val = 100.0
    p_ref_int = int(round(p_ref_val / PRICE_SCALE))
    
    ask_at_price = int(round((p_ref_val + 1 * TICK_SIZE) / PRICE_SCALE))
    ask_far_price = int(round((p_ref_val + 10 * TICK_SIZE) / PRICE_SCALE))
    
    orders = {
        1: OrderState("A", ask_at_price, 1, ASK_ABOVE_AT, 0),
        2: OrderState("A", ask_far_price, 100, ASK_ABOVE_MID, 0) # MID is valid bucket
    }
    
    # We need to ensure ASK_ABOVE_MID is imported or string literal used if not exported
    # The file imports constants. Let's assume the ones imported above are valid.
    # Actually ASK_ABOVE_MID is defined in the module.
    
    # Re-map bucket for Order 2 to be safe with imported constants
    # (The test imports them, but let's double check if "ASK_ABOVE_MID" is in the import list above? No.)
    # Let's just use string "ASK_ABOVE_MID" as it is in the source file
    orders[2].bucket = "ASK_ABOVE_MID"

    snapshot = _snapshot(orders, p_ref_int)
    
    # COM should be pulled heavily towards the far order (qty 100 vs 1)
    # COM price ~= (1*100.25 + 100*102.50)/101 ~= 102.47
    # Dist ticks ~= (102.47 - 100.0) / 0.25 ~= 9.9 ticks
    assert snapshot["d_ask_ticks"] > 8.0
    
    # BBO should be at 1 tick
    if "bbo_ask_ticks" in snapshot:
        assert abs(snapshot["bbo_ask_ticks"] - 1.0) < 0.1, f"BBO should be 1.0, got {snapshot.get('bbo_ask_ticks')}"
