import pytest
import json
from dataclasses import asdict
import pandas as pd


@pytest.fixture
def orderbook_state_class():
    from dataclasses import dataclass
    
    @dataclass
    class OrderbookState:
        best_bid: float = 0.0
        best_ask: float = 0.0
        bid_size: float = 0.0
        ask_size: float = 0.0
        last_update_ns: int = 0
        
        def to_dict(self):
            return asdict(self)
        
        @classmethod
        def from_dict(cls, d: dict):
            return cls(**d)
    
    return OrderbookState


def test_orderbook_state_initialization(orderbook_state_class):
    state = orderbook_state_class()
    assert state.best_bid == 0.0
    assert state.best_ask == 0.0
    assert state.bid_size == 0.0
    assert state.ask_size == 0.0


def test_orderbook_state_serialization(orderbook_state_class):
    state = orderbook_state_class(
        best_bid=6050.0,
        best_ask=6050.25,
        bid_size=100.0,
        ask_size=50.0,
        last_update_ns=1700000000000000000
    )
    
    state_dict = state.to_dict()
    state_json = json.dumps(state_dict)
    
    restored_state = orderbook_state_class.from_dict(json.loads(state_json))
    
    assert restored_state.best_bid == 6050.0
    assert restored_state.best_ask == 6050.25
    assert restored_state.bid_size == 100.0
    assert restored_state.ask_size == 50.0


def test_orderbook_bid_update(orderbook_state_class):
    state = orderbook_state_class()
    
    state.best_bid = 6050.0
    state.bid_size = 100.0
    
    new_price = 6051.0
    new_size = 150.0
    
    if new_price > state.best_bid:
        state.best_bid = new_price
        state.bid_size = new_size
    
    assert state.best_bid == 6051.0
    assert state.bid_size == 150.0


def test_orderbook_ask_update(orderbook_state_class):
    state = orderbook_state_class()
    
    state.best_ask = 6052.0
    state.ask_size = 100.0
    
    new_price = 6051.0
    new_size = 150.0
    
    if state.best_ask == 0.0 or new_price < state.best_ask:
        state.best_ask = new_price
        state.ask_size = new_size
    
    assert state.best_ask == 6051.0
    assert state.ask_size == 150.0


def test_orderbook_cancel_bid(orderbook_state_class):
    state = orderbook_state_class(best_bid=6050.0, bid_size=100.0)
    
    cancel_size = 30.0
    
    if state.bid_size >= cancel_size:
        state.bid_size -= cancel_size
    
    assert state.bid_size == 70.0


def test_pandas_processing(orderbook_state_class):
    data = {
        "action": ["A", "A", "M", "C"],
        "price": [6050.0, 6050.25, 6050.0, 6050.0],
        "size": [100, 50, 120, 20],
        "side": ["B", "A", "B", "B"],
        "event_time": [1000, 1001, 1002, 1003]
    }
    
    pdf = pd.DataFrame(data)
    
    state = orderbook_state_class()
    
    for idx in range(len(pdf)):
        row = pdf.iloc[idx]
        action = row["action"]
        price = row["price"]
        size = row["size"]
        side = row["side"]
        event_time = row["event_time"]
        
        if side == "B":
            if action == "A":
                if price > state.best_bid:
                    state.best_bid = price
                    state.bid_size = size
            elif action == "M":
                if price == state.best_bid:
                    state.bid_size = size
            elif action == "C":
                if price == state.best_bid:
                    state.bid_size = max(0, state.bid_size - size)
        
        elif side == "A":
            if action == "A":
                if state.best_ask == 0.0 or price < state.best_ask:
                    state.best_ask = price
                    state.ask_size = size
        
        state.last_update_ns = event_time
    
    assert state.best_bid == 6050.0
    assert state.bid_size == 100.0
    assert state.best_ask == 6050.25
    assert state.ask_size == 50.0
    assert state.last_update_ns == 1003


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
