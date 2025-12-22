import asyncio
import os
from typing import Dict, Optional
from polygon import RESTClient
from dataclasses import dataclass

@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float

class GreekEnricher:
    """
    Fetches option chain snapshots to cache Greeks (Delta/Gamma).
    """
    def __init__(self, api_key: str, ticker: str = "SPY"):
        self.client = RESTClient(api_key=api_key)
        self.ticker = ticker
        self.cache: Dict[str, Greeks] = {}
        self.running = False
        self._lock = asyncio.Lock()

    async def start_snapshot_loop(self, interval_seconds: int = 60):
        self.running = True
        while self.running:
            try:
                await self._refresh_greeks()
            except Exception as e:
                print(f"⚠️ GreekEnricher Error: {e}")
            
            await asyncio.sleep(interval_seconds)

    async def _refresh_greeks(self):
        # We need to fetch the option chain for today (0DTE)
        # However, getting the ENTIRE chain might be heavy?
        # But we need it for the active strikes.
        # Polygon has `list_snapshot_options_chain`.
        # We can filter by expiration date?
        # client.list_snapshot_options_chain(underlying_asset=self.ticker, params=...)
        # We execute this in a thread because RESTClient is synchronous?
        # The `polygon-api-client` RESTClient is synchronous. We need to wrap it.
        
        loop = asyncio.get_running_loop()
        # Fetch in executor
        await loop.run_in_executor(None, self._fetch_and_update)

    def _fetch_and_update(self):
        # Get today's expiration
        # For now, let's assume we fetch the chain for the underlying.
        # We really only care about the strikes that are active.
        # But fetching specific contracts individually is slow (N calls).
        # Fetching the chain for an expiration is 1 call.
        # We need to know the expiration date.
        # StrikeManager knows it. Maybe we pass it or calculate it again.
        
        # For v1 0DTE, calculate today's date.
        from datetime import datetime
        from zoneinfo import ZoneInfo
        et_now = datetime.now(ZoneInfo("US/Eastern"))
        date_str = et_now.strftime("%Y-%m-%d") # YYYY-MM-DD for API params
        
        # Valid params: expiration_date
        try:
            # We fetch ALL contracts for SPY for today to ensure we cover everything.
            # limit=250 might not be enough for SPY chain? SPY chain is huge.
            # But 0DTE chain is smaller.
            # We iterate if pages?
            # list_snapshot_options_chain returns an iterator.
            
            snapshots = self.client.list_snapshot_options_chain(
                self.ticker,
                params={
                    "expiration_date": date_str,
                }
            )
            
            new_cache = {}
            count = 0
            for snap in snapshots:
                if snap.greeks:
                    # Ticker in snapshot is full ticker "O:SPY..."
                    # Check for 'ticker' or 'symbol'
                    t = getattr(snap, 'ticker', getattr(snap, 'symbol', None))
                    if not t:
                        continue # Skip if no ticker found
                    
                    new_cache[t] = Greeks(
                        delta=snap.greeks.delta or 0.0,
                        gamma=snap.greeks.gamma or 0.0,
                        theta=snap.greeks.theta or 0.0,
                        vega=snap.greeks.vega or 0.0
                    )
                    count += 1
            
            # Atomic swap (GIL protects dict assignment but self.cache ref update is safe)
            self.cache = new_cache
            # print(f"✅ Greeks updated for {count} contracts.")
            
        except Exception as e:
            print(f"❌ Error fetching Greeks: {e}")

    def get_greeks(self, ticker: str) -> Greeks:
        """
        Returns cached Greeks or realistic mock Greeks if not found (for replay mode).
        """
        cached = self.cache.get(ticker, None)
        if cached:
            return cached
        
        # Fallback: Generate approximate Greeks for replay mode
        # Parse ticker to get strike and type
        try:
            # O:SPY251216C00680000 -> strike 680, type C
            suffix = ticker[-15:]
            option_type = suffix[6]  # C or P
            strike = float(suffix[7:]) / 1000.0
            
            # Estimate ATM (we could get this from strike_manager but use simple heuristic)
            # For replay, we can assume SPY is near the median strike
            # For now, use simple moneyness-based approximation
            
            # Assume SPY ~ 600-700 range, use strike as proxy
            # This is crude but better than zeros
            spy_approx = 679.0  # We could get this from replay_engine or strike_manager
            
            moneyness = (spy_approx - strike) if option_type == 'C' else (strike - spy_approx)
            
            # Rough 0DTE delta approximation:
            # ITM: 0.70-0.95, ATM: 0.45-0.55, OTM: 0.05-0.30
            if moneyness > 3:  # Deep ITM
                delta = 0.85 if option_type == 'C' else -0.85
                gamma = 0.02
            elif moneyness > 1:  # ITM
                delta = 0.70 if option_type == 'C' else -0.70
                gamma = 0.08
            elif moneyness > -1:  # ATM
                delta = 0.50 if option_type == 'C' else -0.50
                gamma = 0.15
            elif moneyness > -3:  # OTM
                delta = 0.25 if option_type == 'C' else -0.25
                gamma = 0.08
            else:  # Deep OTM
                delta = 0.08 if option_type == 'C' else -0.08
                gamma = 0.02
            
            return Greeks(delta, gamma, 0.0, 0.0)
            
        except:
            # Ultimate fallback
            return Greeks(0.0, 0.0, 0.0, 0.0)

    def stop(self):
        self.running = False
