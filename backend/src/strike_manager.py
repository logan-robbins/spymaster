from datetime import datetime
from zoneinfo import ZoneInfo
import math

class StrikeManager:
    """
    Determines which contracts are 'Active' based on the underlying price.
    Targets ATM +/- 3 strikes, plus optional 'Hotzone' target.
    """
    def __init__(self, ticker_symbol: str = "SPY"):
        self.ticker = ticker_symbol
        self.current_subscriptions = set()
        self.focus_strike = None # Optional: User-defined "Hotzone" strike

    def set_focus_strike(self, strike: float | None):
        """
        Sets a specific strike to always monitor (Hotzone).
        """
        self.focus_strike = strike
        print(f"🎯 StrikeManager: Focus strike set to {self.focus_strike}")

    def get_target_strikes(self, current_price: float) -> tuple[list[str], list[str]]:
        """
        Calculates target strikes based on current price AND focus strike.
        Returns (subscribe_list, unsubscribe_list).
        """
        center_strike = int(round(current_price))
        
        # Base Range: ATM +/- 3
        target_strikes_int = set(range(center_strike - 3, center_strike + 4))
        
        # Hotzone Range: Focus Strike -1 to +2 (to capture acceleration around it)
        if self.focus_strike is not None:
             focus_center = int(round(self.focus_strike))
             # We want to see flow slightly above/below the hotzone
             hotzone_range = range(focus_center - 1, focus_center + 3) # -1, 0, +1, +2
             target_strikes_int.update(hotzone_range)
        
        # Format tickers
        # Polygon Option Ticker Format: O:SPY{YYMMDD}{C/P}{STRIKE_8_DIGIT}
        # Example: O:SPY251216C00572000 for 572.00 Call
        
        # Get today's date in Eastern Time (0DTE)
        et_now = datetime.now(ZoneInfo("US/Eastern"))
        # Format YYMMDD
        date_str = et_now.strftime("%y%m%d")
        
        new_target_tickers = set()
        
        for strike in target_strikes_int:
            strike_val = strike * 1000
            strike_str = f"{strike_val:08d}"
            
            # Add Call and Put
            call_ticker = f"O:{self.ticker}{date_str}C{strike_str}"
            put_ticker = f"O:{self.ticker}{date_str}P{strike_str}"
            
            new_target_tickers.add(call_ticker)
            new_target_tickers.add(put_ticker)
            
        target_subs = {f"T.{t}" for t in new_target_tickers}
        
        # Diff
        to_add = list(target_subs - self.current_subscriptions)
        to_remove = list(self.current_subscriptions - target_subs)
        
        # Update active state
        self.current_subscriptions = target_subs
        
        return to_add, to_remove
