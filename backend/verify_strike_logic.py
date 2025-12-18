from src.strike_manager import StrikeManager

def test_strike_manager():
    print("🧪 Testing StrikeManager Dynamic Update Logic...")
    sm = StrikeManager()
    
    # Initial state
    assert sm.last_update_price == 0.0, "Initial last_update_price should be 0.0"
    
    # FIRST UPDATE
    print("1. Initial Update at $500.00")
    add, remove = sm.get_target_strikes(500.0)
    assert sm.last_update_price == 500.0, f"last_update_price should be 500.0, got {sm.last_update_price}"
    assert len(sm.current_subscriptions) > 0, "Should have subscriptions"
    
    # SMALL MOVE
    print("2. Small move to $500.30 (Should NOT update)")
    should = sm.should_update(500.30)
    assert not should, "Should not update for $0.30 move"
    
    # BIG MOVE
    print("3. Big move to $500.60 (Should UPDATE)")
    should = sm.should_update(500.60)
    assert should, "Should update for $0.60 move (diff > 0.50)"
    
    # TRIGGER UPDATE
    print("4. Triggering update at $500.60")
    add, remove = sm.get_target_strikes(500.60)
    assert sm.last_update_price == 500.60, "last_update_price should be updated to 500.60"
    
    # CHECK RESET
    print("5. Checking reset (500.70 should NOT update vs 500.60)")
    should = sm.should_update(500.70)
    assert not should, "Should not update for $0.10 move from new baseline"
    
    print("✅ StrikeManager verification PASSED")

if __name__ == "__main__":
    test_strike_manager()
