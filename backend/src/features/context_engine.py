"""
Context Engine - Agent B Implementation

Determines the "macro" state: Time-of-day and Structural Price Levels.
Identifies critical price levels (Pre-Market High/Low, SMA-200) and 
market timing context (first 15 minutes of trading).

Works with OHLCV data and provides level identification for the research pipeline.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, time, timezone, timedelta
import pandas as pd
from enum import Enum

# Import the shared schema
from src.common.schemas.levels_signals import LevelKind


# Eastern Time offset (UTC-5 during standard time, UTC-4 during DST)
# For simplicity, we'll use a fixed offset approach
# Production systems should use pytz or zoneinfo for proper DST handling
ET_OFFSET_HOURS = -5  # Standard time; adjust for DST as needed


class ContextEngine:
    """
    Agent B: The Context Engineer
    
    Determines macro state:
    - Time Context: Is it the first 15 minutes of trading?
    - Level Context: What structural levels are nearby (PM High/Low, SMA-200)?
    
    Uses OHLCV data (1-minute candles) to identify levels.
    """
    
    # Market hours (ET)
    PREMARKET_START = time(4, 0, 0)     # 04:00 ET
    MARKET_OPEN = time(9, 30, 0)         # 09:30 ET
    FIRST_15M_END = time(9, 45, 0)       # 09:45 ET
    
    # Level detection tolerance
    LEVEL_TOLERANCE_USD = 0.10  # $0.10
    
    # SMA parameters
    SMA_PERIOD = 200  # 200 periods
    SMA_400_PERIOD = 400  # 400 periods
    SMA_TIMEFRAME_MINUTES = 2  # 2-minute bars for SMA calculation
    
    def __init__(self, ohlcv_df: Optional[pd.DataFrame] = None):
        """
        Initialize Context Engine.
        
        Args:
            ohlcv_df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                     timestamp should be Unix nanoseconds (UTC) or datetime
        """
        self.ohlcv_df = ohlcv_df
        
        # Cached values (computed once when data is loaded)
        self._premarket_high: Optional[float] = None
        self._premarket_low: Optional[float] = None
        self._sma_200: Optional[pd.Series] = None
        self._sma_400: Optional[pd.Series] = None
        
        # Process data if provided
        if ohlcv_df is not None:
            self._process_data()
    
    def _process_data(self):
        """
        Pre-process OHLCV data:
        - Calculate pre-market high/low
        - Calculate SMA-200 on 2-minute timeframe
        """
        if self.ohlcv_df is None or len(self.ohlcv_df) == 0:
            return
        
        df = self.ohlcv_df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            if pd.api.types.is_integer_dtype(df['timestamp']):
                # Convert from Unix nanoseconds to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True)
            elif not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        else:
            raise ValueError("OHLCV DataFrame must have 'timestamp' column")
        
        # Convert to ET for time-of-day logic
        df['time_et'] = df['timestamp'].dt.tz_convert('America/New_York').dt.time
        df['date_et'] = df['timestamp'].dt.tz_convert('America/New_York').dt.date
        
        # Calculate pre-market high/low (04:00 - 09:30 ET)
        premarket_mask = (df['time_et'] >= self.PREMARKET_START) & (df['time_et'] < self.MARKET_OPEN)
        if premarket_mask.any():
            pm_data = df[premarket_mask]
            self._premarket_high = pm_data['high'].max()
            self._premarket_low = pm_data['low'].min()
        
        # Calculate SMA-200 and SMA-400 on 2-minute timeframe
        # Resample 1-minute data to 2-minute bars
        df_2min = df.set_index('timestamp').resample('2min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        sma_200_2min = None
        sma_400_2min = None

        if len(df_2min) >= self.SMA_PERIOD:
            sma_200_2min = df_2min['close'].rolling(window=self.SMA_PERIOD).mean()

        if len(df_2min) >= self.SMA_400_PERIOD:
            sma_400_2min = df_2min['close'].rolling(window=self.SMA_400_PERIOD).mean()

        if sma_200_2min is not None:
            df['sma_200'] = df['timestamp'].apply(
                lambda ts: self._get_sma_at_time(ts, df_2min.index, sma_200_2min)
            )
            self._sma_200 = df.set_index('timestamp')['sma_200']

        if sma_400_2min is not None:
            df['sma_400'] = df['timestamp'].apply(
                lambda ts: self._get_sma_at_time(ts, df_2min.index, sma_400_2min)
            )
            self._sma_400 = df.set_index('timestamp')['sma_400']
        
        # Store processed dataframe
        self.ohlcv_df = df
    
    def _get_sma_at_time(
        self, 
        target_time: pd.Timestamp, 
        sma_index: pd.DatetimeIndex,
        sma_values: pd.Series
    ) -> Optional[float]:
        """
        Get SMA value at a specific time (forward fill from 2-min bars).
        
        Args:
            target_time: Target timestamp
            sma_index: DatetimeIndex of 2-min bars
            sma_values: SMA values for 2-min bars
            
        Returns:
            SMA value or None if not available
        """
        # Find the most recent 2-min bar at or before target_time
        valid_indices = sma_index[sma_index <= target_time]
        if len(valid_indices) == 0:
            return None
        
        latest_index = valid_indices[-1]
        sma_value = sma_values.loc[latest_index]
        
        return sma_value if pd.notna(sma_value) else None
    
    def is_first_15m(self, ts: int) -> bool:
        """
        Check if timestamp is in the first 15 minutes of trading (09:30-09:45 ET).
        
        Args:
            ts: Unix timestamp in nanoseconds (UTC)
            
        Returns:
            True if timestamp is between 09:30:00 and 09:45:00 ET
            
        Example:
            >>> engine = ContextEngine()
            >>> ts = 1640094600000000000  # Some timestamp
            >>> is_opening = engine.is_first_15m(ts)
        """
        # Convert nanoseconds to datetime (UTC)
        dt_utc = pd.to_datetime(ts, unit='ns', utc=True)
        
        # Convert to ET
        dt_et = dt_utc.tz_convert('America/New_York')
        time_et = dt_et.time()
        
        # Check if between 09:30:00 and 09:45:00 ET
        return self.MARKET_OPEN <= time_et < self.FIRST_15M_END
    
    def get_active_levels(
        self, 
        current_price: float, 
        current_time: int
    ) -> List[Dict[str, Any]]:
        """
        Get all structural levels within tolerance of current price.
        
        Logic:
        - Pre-Market High/Low (calculated from 04:00-09:30 ET data)
        - SMA-200 (calculated on 2-minute timeframe)
        - Returns levels within $0.10 of current_price
        
        Args:
            current_price: Current price (e.g., 687.50)
            current_time: Current timestamp in nanoseconds (UTC)
            
        Returns:
            List of level dictionaries with keys:
                - 'level_kind': LevelKind enum value
                - 'level_price': float price of the level
                - 'distance': float distance from current price
                
        Example:
            If price is 687.55 and SMA-200 is 687.50:
            Returns [{'level_kind': LevelKind.SMA_200, 'level_price': 687.50, 'distance': 0.05}]
        """
        levels = []
        
        # Check Pre-Market High
        if self._premarket_high is not None:
            distance = abs(current_price - self._premarket_high)
            if distance <= self.LEVEL_TOLERANCE_USD:
                levels.append({
                    'level_kind': LevelKind.PM_HIGH,
                    'level_price': self._premarket_high,
                    'distance': distance
                })
        
        # Check Pre-Market Low
        if self._premarket_low is not None:
            distance = abs(current_price - self._premarket_low)
            if distance <= self.LEVEL_TOLERANCE_USD:
                levels.append({
                    'level_kind': LevelKind.PM_LOW,
                    'level_price': self._premarket_low,
                    'distance': distance
                })
        
        # Check SMA-200
        if self._sma_200 is not None:
            # Get SMA value at current_time
            current_dt = pd.to_datetime(current_time, unit='ns', utc=True)
            
            # Find closest timestamp in SMA series
            if current_dt in self._sma_200.index:
                sma_value = self._sma_200.loc[current_dt]
            else:
                # Use forward fill logic
                valid_times = self._sma_200.index[self._sma_200.index <= current_dt]
                if len(valid_times) > 0:
                    sma_value = self._sma_200.loc[valid_times[-1]]
                else:
                    sma_value = None
            
            if sma_value is not None and pd.notna(sma_value):
                distance = abs(current_price - sma_value)
                if distance <= self.LEVEL_TOLERANCE_USD:
                    levels.append({
                        'level_kind': LevelKind.SMA_200,
                        'level_price': float(sma_value),
                        'distance': distance
                    })

        # Check SMA-400
        if self._sma_400 is not None:
            current_dt = pd.to_datetime(current_time, unit='ns', utc=True)
            if current_dt in self._sma_400.index:
                sma_value = self._sma_400.loc[current_dt]
            else:
                valid_times = self._sma_400.index[self._sma_400.index <= current_dt]
                if len(valid_times) > 0:
                    sma_value = self._sma_400.loc[valid_times[-1]]
                else:
                    sma_value = None

            if sma_value is not None and pd.notna(sma_value):
                distance = abs(current_price - sma_value)
                if distance <= self.LEVEL_TOLERANCE_USD:
                    levels.append({
                        'level_kind': LevelKind.SMA_400,
                        'level_price': float(sma_value),
                        'distance': distance
                    })
        
        return levels
    
    def get_premarket_high(self) -> Optional[float]:
        """Get cached pre-market high."""
        return self._premarket_high
    
    def get_premarket_low(self) -> Optional[float]:
        """Get cached pre-market low."""
        return self._premarket_low
    
    def get_sma_200_at_time(self, ts: int) -> Optional[float]:
        """
        Get SMA-200 value at a specific time.
        
        Args:
            ts: Unix timestamp in nanoseconds (UTC)
            
        Returns:
            SMA-200 value or None if not available
        """
        if self._sma_200 is None:
            return None
        
        current_dt = pd.to_datetime(ts, unit='ns', utc=True)
        
        # Find closest timestamp in SMA series
        if current_dt in self._sma_200.index:
            return self._sma_200.loc[current_dt]
        else:
            # Use forward fill logic
            valid_times = self._sma_200.index[self._sma_200.index <= current_dt]
            if len(valid_times) > 0:
                sma_val = self._sma_200.loc[valid_times[-1]]
                return float(sma_val) if pd.notna(sma_val) else None
        
        return None

    def get_sma_400_at_time(self, ts: int) -> Optional[float]:
        """
        Get SMA-400 value at a specific time.
        
        Args:
            ts: Unix timestamp in nanoseconds (UTC)
            
        Returns:
            SMA-400 value or None if not available
        """
        if self._sma_400 is None:
            return None
        
        current_dt = pd.to_datetime(ts, unit='ns', utc=True)
        
        if current_dt in self._sma_400.index:
            return self._sma_400.loc[current_dt]
        else:
            valid_times = self._sma_400.index[self._sma_400.index <= current_dt]
            if len(valid_times) > 0:
                sma_val = self._sma_400.loc[valid_times[-1]]
                return float(sma_val) if pd.notna(sma_val) else None
        
        return None
    
    # --- Mock Data Generators (for testing) ---
    
    @staticmethod
    def generate_mock_ohlcv(
        start_time: Optional[datetime] = None,
        num_minutes: int = 480,  # 8 hours (4am - 12pm)
        base_price: float = 6870.0,
        volatility: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate mock 1-minute OHLCV data for testing.
        
        Creates a full trading day with pre-market and regular session data.
        
        Args:
            start_time: Start timestamp (defaults to 04:00 ET today)
            num_minutes: Number of 1-minute bars to generate
            base_price: Base price level (ES price, e.g., 6870.00)
            volatility: Price volatility (dollar amount)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        import numpy as np
        
        # Default to 04:00 ET (pre-market start)
        if start_time is None:
            # Get today's date at 04:00 ET
            et_tz = pd.Timestamp.now(tz='America/New_York').normalize() + pd.Timedelta(hours=4)
            start_time = et_tz.tz_convert('UTC')
        
        # Generate timestamps (1-minute intervals)
        timestamps = pd.date_range(start=start_time, periods=num_minutes, freq='1min', tz='UTC')
        
        # Generate synthetic price data with trend
        np.random.seed(42)
        
        # Create a slight upward trend
        trend = np.linspace(0, 2.0, num_minutes)  # +$2 over the day
        
        # Random walk
        returns = np.random.randn(num_minutes) * volatility
        price_path = base_price + trend + np.cumsum(returns)
        
        # Generate OHLC from close prices
        data = []
        for i, (ts, close_price) in enumerate(zip(timestamps, price_path)):
            # Add some intrabar volatility
            high = close_price + abs(np.random.randn() * 0.1)
            low = close_price - abs(np.random.randn() * 0.1)
            open_price = price_path[i - 1] if i > 0 else close_price
            
            volume = int(np.random.randint(1000, 10000))
            
            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        
        return df


# --- Example Usage ---
if __name__ == "__main__":
    print("Context Engine - Agent B")
    print("=" * 60)
    print("Identifies structural levels and time-of-day context")
    print("=" * 60)
    
    # Generate mock OHLCV data
    print("\n[TEST 1] Generate Mock OHLCV Data")
    mock_ohlcv = ContextEngine.generate_mock_ohlcv(
        num_minutes=480,  # 8 hours
        base_price=687.0
    )
    print(f"Generated {len(mock_ohlcv)} 1-minute bars")
    print(f"Time range: {mock_ohlcv['timestamp'].min()} to {mock_ohlcv['timestamp'].max()}")
    print(f"Price range: ${mock_ohlcv['close'].min():.2f} - ${mock_ohlcv['close'].max():.2f}")
    
    # Initialize engine with data
    print("\n[TEST 2] Initialize Context Engine")
    engine = ContextEngine(ohlcv_df=mock_ohlcv)
    print(f"Pre-Market High: ${engine.get_premarket_high():.2f}" if engine.get_premarket_high() else "Pre-Market High: Not available")
    print(f"Pre-Market Low: ${engine.get_premarket_low():.2f}" if engine.get_premarket_low() else "Pre-Market Low: Not available")
    
    # Test is_first_15m
    print("\n[TEST 3] Check First 15 Minutes")
    # Get a timestamp at 09:35 ET (should be True)
    sample_time = mock_ohlcv[mock_ohlcv['timestamp'].dt.tz_convert('America/New_York').dt.time >= time(9, 35, 0)].iloc[0] if len(mock_ohlcv) > 100 else mock_ohlcv.iloc[100]
    ts_ns = int(sample_time['timestamp'].value)  # Convert to nanoseconds
    is_opening = engine.is_first_15m(ts_ns)
    sample_dt = pd.to_datetime(ts_ns, unit='ns', utc=True).tz_convert('America/New_York')
    print(f"Timestamp: {sample_dt.strftime('%H:%M:%S')} ET")
    print(f"Is First 15m: {is_opening}")
    
    # Test get_active_levels
    print("\n[TEST 4] Get Active Levels Near Price")
    current_price = sample_time['close']
    current_time_ns = ts_ns
    active_levels = engine.get_active_levels(current_price, current_time_ns)
    print(f"Current Price: ${current_price:.2f}")
    print(f"Active Levels within ${engine.LEVEL_TOLERANCE_USD}:")
    if active_levels:
        for level in active_levels:
            print(f"  - {level['level_kind'].value}: ${level['level_price']:.2f} (distance: ${level['distance']:.3f})")
    else:
        print("  No levels detected")
    
    # Test SMA-200
    print("\n[TEST 5] SMA-200 Calculation")
    sma_value = engine.get_sma_200_at_time(current_time_ns)
    if sma_value:
        print(f"SMA-200 at {sample_dt.strftime('%H:%M')} ET: ${sma_value:.2f}")
        print(f"Distance to SMA-200: ${abs(current_price - sma_value):.2f}")
    else:
        print("SMA-200: Not enough data")
    
    print("\n" + "=" * 60)
    print("✓ Context Engine tests complete")
    print("✓ Ready for integration with Physics Engine (Agent A)")
    print("=" * 60)
