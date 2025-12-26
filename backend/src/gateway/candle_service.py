"""
Candle Service: Fetches historical OHLCV data for charts.
"""

import os
import httpx
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from src.common.config import CONFIG

class CandleService:
    """
    Fetches historical candles from Polygon.io.
    """
    
    BASE_URL = "https://api.polygon.io/v2/aggs/ticker"
    
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            # We don't crash init here to allow Gateway to start for non-chart usages,
            # but we will fail hard on request.
            pass
            
    async def get_candles(self, symbol: str, timeframe_minutes: int = 2, lookback_days: int = 1) -> List[Dict[str, Any]]:
        """
        Fetch candles from Polygon.
        
        Args:
            symbol: Ticker (e.g., 'SPY')
            timeframe_minutes: Candle size in minutes
            lookback_days: How far back to query
            
        Returns:
            List of dicts suitable for Lightweight Charts:
            [{ time: unix_sec, open: float, high: float, ... }]
        """
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY is missing in environment variables.")

        # Calculate time range
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=lookback_days)
        
        # Format: YYYY-MM-DD
        from_str = start_dt.strftime("%Y-%m-%d")
        to_str = end_dt.strftime("%Y-%m-%d")
        
        # Polygon API format: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
        url = f"{self.BASE_URL}/{symbol}/range/{timeframe_minutes}/minute/{from_str}/{to_str}"
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            
            if response.status_code != 200:
                raise RuntimeError(f"Polygon API Error {response.status_code}: {response.text}")
                
            data = response.json()
            
            if "results" not in data:
                return []
                
            # Convert to Lightweight Charts format (time in seconds)
            candles = []
            for item in data["results"]:
                # Polygon 't' is ms timestamp
                ts_sec = int(item["t"] / 1000)
                candles.append({
                    "time": ts_sec,
                    "open": item["o"],
                    "high": item["h"],
                    "low": item["l"],
                    "close": item["c"],
                    "volume": item["v"]
                })
                
            return candles
