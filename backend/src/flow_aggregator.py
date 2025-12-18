from collections import defaultdict
from typing import Dict, Any, List
from .strike_manager import StrikeManager
from .greek_enricher import GreekEnricher
from .persistence_engine import PersistenceEngine
from datetime import datetime
import asyncio

class FlowAggregator:
    """
    Process raw trade messages into consumable metrics and update state.
    """
    def __init__(self, greek_enricher: GreekEnricher, persistence_engine: PersistenceEngine):
        self.greek_enricher = greek_enricher
        self.persistence = persistence_engine
        
        # State Store: Dict[Ticker, ContractMetrics]
        # We keep it simple: Dict[Ticker, Dict]
        self.state: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "cumulative_volume": 0,
            "cumulative_premium": 0.0,
            "last_price": 0.0,
            "net_delta_flow": 0.0,
            "net_gamma_flow": 0.0,
            "delta": 0.0,
            "gamma": 0.0,
            "strike_price": 0.0,
            "type": "",  # 'C' or 'P'
            "expiration": "",
            "last_timestamp": 0  # ms - for frontend time sync
        })
        
        # We need to know which tickers are active to filter?
        # Or we just aggregate everything we receive? 
        # We receive what we subscribe to.

    async def process_message(self, msg: Any):
        # Parse Polygon Trade Message
        # Expected 'T' message (Trade)
        # msg is a WebSocketMessage object or dict? Check `test_stream.py` output.
        # It's an object from `polygon.websocket.models`.
        # Standard attributes: symbol, price, size, timestamp, conditions, etc.
        
        # Map attributes
        ticker = getattr(msg, "symbol", "")
        price = getattr(msg, "price", 0.0)
        size = getattr(msg, "size", 0)
        timestamp = getattr(msg, "timestamp", 0) # ms
        conditions = getattr(msg, "conditions", [])
        
        if not ticker:
            return

        # Basic filtering (Optional: check conditions for delayed trades)
        # For now, process all.
        
        # Enrich with Greeks
        # Check if message already carries Greeks (Replay/Simulation case)
        msg_greeks = getattr(msg, "greeks", None)
        if msg_greeks:
            greeks = msg_greeks
        else:
            greeks = self.greek_enricher.get_greeks(ticker)
        
        # Calculate Flows
        premium = price * size * 100
        delta_notional = size * greeks.delta * 100 # Contract size is 100
        gamma_notional = size * greeks.gamma * 100 
        
        # Aggressor Logic (Heuristic)
        # Polygon doesn't give 'aggressor_side' directly in standard T? 
        # We infer from price vs prior quote? We don't have quotes here.
        # APP.md says: `net_aggressor: Int`.
        # Without BBO, simple heuristic: 
        # If price > last_price -> Buy?
        # This is flaky. 
        # For v1, maybe we just accumulate Volume. 
        # Or we treat all Volume as "Activity".
        # Let's track `net_aggressor` if we can, else just volume.
        # We can update `last_price` in the state.
        
        prev_price = self.state[ticker]["last_price"]
        aggressor = 0
        if prev_price > 0:
            if price > prev_price:
                aggressor = 1 # Buy
            elif price < prev_price:
                aggressor = -1 # Sell
        
        # Update State
        stats = self.state[ticker]
        stats["cumulative_volume"] += size
        stats["cumulative_premium"] += premium
        stats["last_price"] = price
        stats["net_delta_flow"] += delta_notional
        stats["net_gamma_flow"] += gamma_notional
        stats["delta"] = greeks.delta # Update current greeks
        stats["gamma"] = greeks.gamma
        stats["last_timestamp"] = timestamp  # Track last update time
        
        # Parse extra info from ticker if needed (Strike, Type)
        # Ex: O:SPY251216C00572000
        # This parsing ideally happens once.
        if not stats["strike_price"]:
            try:
                # O:SPY 251216 C 00572000
                # Split logic
                # 0-5: O:SPY (variable len if not SPY)
                # Let's assume SPY -> len 5
                # Date: 6 chars
                # Type: 1 char
                # Strike: 8 chars
                # Total suffix len: 6+1+8 = 15.
                suffix = ticker[-15:]
                # date = suffix[:6]
                type_ = suffix[6]
                strike_str = suffix[7:]
                stats["type"] = type_
                stats["strike_price"] = float(strike_str) / 1000.0
                if stats["type"] == 'P':
                    print(f"🐻 PUT: {ticker} | Size: {size} | Delta: {greeks.delta} | Flow: {delta_notional}")
            except Exception as e:
                print(f"FAILED PARSING TICKER: {ticker} | Error: {e}")
                pass

        # Persist
        record = {
            "timestamp": datetime.fromtimestamp(timestamp / 1000.0) if timestamp else datetime.utcnow(),
            "ticker": ticker,
            "price": price,
            "size": size,
            "premium": premium,
            "aggressor_side": aggressor,
            "delta": greeks.delta,
            "gamma": greeks.gamma,
            "net_delta_impact": delta_notional
        }
        await self.persistence.process_trade(record)

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Return current state for broadcast.
        """
        # Convert defaultdict to regular dict for JSON serialization
        return dict(self.state)
