from datetime import datetime
from zoneinfo import ZoneInfo
import math

class StrikeManager:
    """
    Determines which contracts are 'Active' based on the underlying price.
    Targets ATM +/- 3 strikes.
    """
    def __init__(self, ticker_symbol: str = "SPY"):
        self.ticker = ticker_symbol
        self.current_subscriptions = set()

    def get_target_strikes(self, current_price: float) -> tuple[list[str], list[str]]:
        """
        Calculates target strikes based on current price.
        Returns (subscribe_list, unsubscribe_list).
        """
        center_strike = int(round(current_price))
        target_strikes_int = range(center_strike - 3, center_strike + 4) # +/- 3 inclusive thus +4 for range
        
        # Format tickers
        # Polygon Option Ticker Format: O:SPY{YYMMDD}{C/P}{STRIKE_8_DIGIT}
        # Example: O:SPY251216C00572000 for 572.00 Call
        
        # Get today's date in Eastern Time (0DTE)
        et_now = datetime.now(ZoneInfo("US/Eastern"))
        # Format YYMMDD
        date_str = et_now.strftime("%y%m%d")
        
        new_target_tickers = set()
        
        for strike in target_strikes_int:
            # Format strike to 8 digits (xxxxx.xxx) -> implies * 1000 and zero pad to 8 chars?
            # Polygon format: 
            # Strike 572.00 -> 00572000
            # Strike 572.50 -> 00572500
            # Logic: strike * 1000, pad to 8 chars with leading zeros.
            
            strike_val = strike * 1000
            strike_str = f"{strike_val:08d}"
            
            # Add Call and Put
            call_ticker = f"O:{self.ticker}{date_str}C{strike_str}"
            put_ticker = f"O:{self.ticker}{date_str}P{strike_str}"
            
            new_target_tickers.add(call_ticker)
            new_target_tickers.add(put_ticker)
            
        # Specific T subscribe format: "T." + ticker 
        # But here we just return the raw ticker, the caller handles "T." prefix or the client does?
        # Polygon subscribe takes "T.O:..."
        # Let's return the raw tickers first, or the full channel string?
        # The APP.md says "Format Tickers: O:SPY..."
        # But StreamIngestor calls `ws.subscribe(add)`.
        # Usually one subscribes to `T.O:...`. 
        # I'll return the full subscription string to be safe and easy.
        
        target_subs = {f"T.{t}" for t in new_target_tickers}
        
        # Diff
        to_add = list(target_subs - self.current_subscriptions)
        to_remove = list(self.current_subscriptions - target_subs)
        
        # Update active state
        self.current_subscriptions = target_subs
        
        return to_add, to_remove
