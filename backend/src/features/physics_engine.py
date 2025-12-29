"""
Physics Engine - Agent A Implementation

Quantifies the microstructure of the order book during level touches.
Calculates wall ratios, replenishment speeds, and other physics metrics.

Works with real MBP-10 (Market By Price) data from ES futures and 
FuturesTrade data as defined in src/common/event_types.py
"""

from typing import Optional, List, Any
import time

# Import actual data structures from the system
from src.common.event_types import MBP10, BidAskLevel, FuturesTrade, EventSource, Aggressor


class PhysicsEngine:
    """
    Agent A: The Physics Engineer
    
    Quantifies microstructure behavior at price levels:
    - Wall Ratio: Size at level vs average volume
    - Replenishment Speed: How fast liquidity reloads after sweep
    - Tape Velocity: Trade frequency in recent window
    """
    
    # Default constants
    DEFAULT_AVG_VOLUME = 5000  # shares per minute (hardcoded for now)
    REPLENISHMENT_WINDOW_MS = 100  # milliseconds to check for reload
    TAPE_VELOCITY_WINDOW_S = 5  # seconds for velocity calculation
    
    def __init__(self, market_data: Optional[Any] = None):
        """
        Initialize Physics Engine.
        
        Args:
            market_data: Reference to market data feed (mocked for now)
        """
        self.market_data = market_data
    
    def calculate_wall_ratio(
        self, 
        mbp10: MBP10, 
        level_price: float,
        tolerance: float = 0.01
    ) -> float:
        """
        Calculate the Wall Ratio at a specific price level using MBP-10 data.
        
        Logic: Get quantity resting at level_price (within tolerance) and 
        normalize by average volume.
        
        Args:
            mbp10: Current MBP-10 snapshot with 10 levels
            level_price: The price level to analyze
            tolerance: Price matching tolerance (default 0.01 = 1 cent)
            
        Returns:
            Wall ratio (e.g., 2.0 means 2x average volume at this level)
            
        Example:
            10,000 shares on bid / 5000 avg vol = Ratio of 2.0
        """
        if not mbp10 or not mbp10.levels:
            return 0.0
        
        total_size = 0
        
        # Check all 10 levels for liquidity near this price
        for level in mbp10.levels:
            # Check bid side
            if abs(level.bid_px - level_price) <= tolerance and level.bid_sz > 0:
                total_size += level.bid_sz
            
            # Check ask side
            if abs(level.ask_px - level_price) <= tolerance and level.ask_sz > 0:
                total_size += level.ask_sz
        
        # Normalize by average volume
        wall_ratio = total_size / self.DEFAULT_AVG_VOLUME
        
        return wall_ratio
    
    def detect_replenishment(
        self,
        trade_tape: List[FuturesTrade],
        mbp10_snapshots: List[MBP10],
        level_price: float,
        window_ms: int = REPLENISHMENT_WINDOW_MS,
        tolerance: float = 0.01
    ) -> Optional[float]:
        """
        Detect liquidity replenishment after a sweep using real MBP-10 data.
        
        Logic:
        1. Scan trade_tape for "Sweep" - aggressive orders consuming 
           liquidity at level
        2. Check if limit orders at that price increased AFTER the sweep 
           within window_ms
        3. Return the latency in milliseconds
        
        Args:
            trade_tape: List of FuturesTrade objects (chronological)
            mbp10_snapshots: List of MBP10 snapshots (chronological)
            level_price: The price level to analyze
            window_ms: Time window to check for replenishment (milliseconds)
            tolerance: Price matching tolerance (default 0.01)
            
        Returns:
            Replenishment latency in milliseconds, or None if no reload detected
            
        Example:
            Returns 35.0 if liquidity was reloaded 35ms after sweep
        """
        if not trade_tape or not mbp10_snapshots:
            return None
        
        # Helper to get total size at level
        def get_size_at_level(mbp: MBP10) -> int:
            total = 0
            for level in mbp.levels:
                if abs(level.bid_px - level_price) <= tolerance:
                    total += level.bid_sz
                if abs(level.ask_px - level_price) <= tolerance:
                    total += level.ask_sz
            return total
        
        # Step 1: Find sweep event (trade at or near level_price)
        sweep_timestamp = None
        pre_sweep_size = 0
        
        # Get initial book state
        if mbp10_snapshots:
            pre_sweep_size = get_size_at_level(mbp10_snapshots[0])
        
        # Look for sweep (significant trade at level)
        for trade in trade_tape:
            if abs(trade.price - level_price) <= tolerance:
                sweep_timestamp = trade.ts_event_ns
                break
        
        if sweep_timestamp is None:
            return None
        
        # Step 2: Check MBP-10 snapshots after sweep for replenishment
        window_ns = window_ms * 1_000_000  # Convert ms to ns
        
        for mbp in mbp10_snapshots:
            # Skip snapshots before sweep
            if mbp.ts_event_ns <= sweep_timestamp:
                continue
            
            # Check if outside window
            if mbp.ts_event_ns > sweep_timestamp + window_ns:
                break
            
            # Check if liquidity increased
            current_size = get_size_at_level(mbp)
            
            if current_size > pre_sweep_size:
                # Replenishment detected!
                latency_ns = mbp.ts_event_ns - sweep_timestamp
                latency_ms = latency_ns / 1_000_000
                return latency_ms
        
        return None
    
    def calculate_tape_velocity(
        self,
        trade_tape: List[FuturesTrade],
        current_time_ns: int,
        window_s: float = TAPE_VELOCITY_WINDOW_S
    ) -> float:
        """
        Calculate tape velocity (trades per second) using real FuturesTrade data.
        
        Args:
            trade_tape: List of FuturesTrade objects (chronological)
            current_time_ns: Current timestamp in nanoseconds
            window_s: Time window in seconds (default 5s)
            
        Returns:
            Trades per second in the lookback window
        """
        window_ns = int(window_s * 1_000_000_000)
        cutoff_time = current_time_ns - window_ns
        
        # Count trades in window
        recent_trades = [
            t for t in trade_tape 
            if t.ts_event_ns >= cutoff_time
        ]
        
        if not recent_trades:
            return 0.0
        
        velocity = len(recent_trades) / window_s
        return velocity
    
    # --- Mock Data Generators (for testing) ---
    
    @staticmethod
    def generate_mock_mbp10(
        timestamp_ns: int = None,
        level_price: float = 6870.0,  # ES price (index points)
        wall_size: int = 10000,
        symbol: str = "ES"
    ) -> MBP10:
        """
        Generate a mock MBP-10 snapshot for testing.
        
        Args:
            timestamp_ns: Timestamp (defaults to current time)
            level_price: Price to place the wall (ES terms)
            wall_size: Size of liquidity at level
            symbol: Symbol (default "ES")
            
        Returns:
            MBP10 with mock data
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        
        # Create 10 levels with wall at level_price
        levels = []
        for i in range(10):
            bid_px = level_price - (i * 0.25)  # ES tick size is 0.25
            ask_px = level_price + ((i + 1) * 0.25)
            
            # Put wall at first level
            bid_sz = wall_size if i == 0 else 2000 - (i * 100)
            ask_sz = 2000 - (i * 100)
            
            levels.append(BidAskLevel(
                bid_px=bid_px,
                bid_sz=bid_sz,
                ask_px=ask_px,
                ask_sz=ask_sz
            ))
        
        return MBP10(
            ts_event_ns=timestamp_ns,
            ts_recv_ns=timestamp_ns,
            source=EventSource.SIM,
            symbol=symbol,
            levels=levels,
            is_snapshot=True
        )
    
    @staticmethod
    def generate_mock_trades(
        start_time_ns: int = None,
        num_trades: int = 100,
        price_level: float = 6870.0,  # ES price
        symbol: str = "ES"
    ) -> List[FuturesTrade]:
        """
        Generate mock trade tape for testing.
        
        Args:
            start_time_ns: Starting timestamp (defaults to current time)
            num_trades: Number of trades to generate
            price_level: Center price for trades (ES terms)
            symbol: Symbol (default "ES")
            
        Returns:
            List of FuturesTrade objects
        """
        if start_time_ns is None:
            start_time_ns = time.time_ns()
        
        trades = []
        for i in range(num_trades):
            trade = FuturesTrade(
                ts_event_ns=start_time_ns + (i * 10_000_000),  # 10ms apart
                ts_recv_ns=start_time_ns + (i * 10_000_000),
                source=EventSource.SIM,
                symbol=symbol,
                price=price_level + (i % 3 - 1) * 0.25,  # ES tick size
                size=1 + (i % 5),  # 1-5 contracts
                aggressor=Aggressor.BUY if (i % 2 == 0) else Aggressor.SELL
            )
            trades.append(trade)
        
        return trades


# --- Example Usage ---
if __name__ == "__main__":
    print("Physics Engine - Agent A")
    print("=" * 60)
    print("Using real MBP-10 and FuturesTrade schemas")
    print("=" * 60)
    
    # Initialize engine
    engine = PhysicsEngine()
    
    # Test 1: Wall Ratio with real MBP-10
    print("\n[TEST 1] Wall Ratio Calculation (MBP-10)")
    mock_mbp10 = PhysicsEngine.generate_mock_mbp10(
        level_price=6870.0,  # ES price
        wall_size=10000
    )
    wall_ratio = engine.calculate_wall_ratio(mock_mbp10, 6870.0)
    print(f"Wall at ES $6870.00: {10000} contracts")
    print(f"Average Volume: {engine.DEFAULT_AVG_VOLUME} contracts")
    print(f"Wall Ratio: {wall_ratio:.2f}x")
    print(f"MBP-10 Levels: {len(mock_mbp10.levels)}")
    print(f"Best Bid: ${mock_mbp10.levels[0].bid_px} x {mock_mbp10.levels[0].bid_sz}")
    print(f"Best Ask: ${mock_mbp10.levels[0].ask_px} x {mock_mbp10.levels[0].ask_sz}")
    
    # Test 2: Tape Velocity with real FuturesTrade
    print("\n[TEST 2] Tape Velocity (FuturesTrade)")
    current_time = time.time_ns()
    mock_trades = PhysicsEngine.generate_mock_trades(
        start_time_ns=current_time - 5_000_000_000,  # 5 seconds ago
        num_trades=50,
        price_level=6870.0
    )
    velocity = engine.calculate_tape_velocity(mock_trades, current_time)
    print(f"Trades in last 5s: {len(mock_trades)}")
    print(f"Tape Velocity: {velocity:.2f} trades/sec")
    print(f"Sample trade: {mock_trades[0].symbol} @ ${mock_trades[0].price}, aggressor={mock_trades[0].aggressor.name}")
    
    # Test 3: Replenishment Detection with real schemas
    print("\n[TEST 3] Replenishment Detection (MBP-10 + FuturesTrade)")
    
    # Create scenario: sweep then replenishment
    base_time = time.time_ns()
    
    # Before sweep: 10k contracts
    mbp_before = PhysicsEngine.generate_mock_mbp10(
        timestamp_ns=base_time,
        level_price=6870.0,
        wall_size=10000
    )
    
    # Sweep event
    sweep_trade = FuturesTrade(
        ts_event_ns=base_time + 10_000_000,  # 10ms later
        ts_recv_ns=base_time + 10_000_000,
        source=EventSource.SIM,
        symbol="ES",
        price=6870.0,
        size=100,
        aggressor=Aggressor.BUY
    )
    
    # After sweep: replenished to 8k after 35ms
    mbp_after = PhysicsEngine.generate_mock_mbp10(
        timestamp_ns=base_time + 45_000_000,  # 45ms later
        level_price=6870.0,
        wall_size=8000
    )
    
    replenishment = engine.detect_replenishment(
        trade_tape=[sweep_trade],
        mbp10_snapshots=[mbp_before, mbp_after],
        level_price=6870.0
    )
    
    if replenishment:
        print(f"Replenishment detected: {replenishment:.2f}ms")
    else:
        print("No replenishment detected")
    
    print("\n" + "=" * 60)
    print("✓ Physics Engine tests complete")
    print("✓ All methods use real MBP-10 and FuturesTrade schemas")
    print("=" * 60)
