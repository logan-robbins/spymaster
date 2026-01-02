
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, "/Users/loganrobbins/research/qmachina/spymaster/backend")

from src.core.market_state import MarketState
from src.common.config import CONFIG

def debug():
    date = "2025-10-29"
    print(f"Checking Data for {date}...")
    
    # 1. Load Bronze (simplified, using reader directly if possible or stage)
    # We'll use LoadBronzeStage because it encapsulates the logic
    
    # We need to mock the context data retrieval or let loader do it?
    # LoadBronzeStage.execute calls BronzeReader.
    # Let's use BronzeReader directly to avoid setting up full pipeline context
    from src.io.bronze import BronzeReader
    reader = BronzeReader()
    trades_df = reader.read_futures_trades(symbol="ES", date=date, front_month_only=False)
    options_df = reader.read_option_trades(underlying="ES", date=date)
    
    print(f"Loaded {len(trades_df)} trades")
    print(f"Loaded {len(options_df)} options")
    
    if len(trades_df) == 0:
        print("No trades!")
        return

    # 2. Determine Spot Price
    # InitMarketState logic: filter reasonable range
    # 3000 < trade.price < 10000
    valid_trades = trades_df[(trades_df['price'] > 3000) & (trades_df['price'] < 10000)]
    if len(valid_trades) > 0:
        mean_price = valid_trades['price'].mean()
        min_price = valid_trades['price'].min()
        max_price = valid_trades['price'].max()
        print(f"Spot Price Stats: Mean={mean_price:.2f}, Min={min_price}, Max={max_price}")
    else:
        print("No valid trades in [3000, 10000] range!")
        return
        
    # 3. Check Option Strikes
    if len(options_df) == 0:
        print("No options!")
        return
    
    # Parse strikes if missing (mimic InitMarketState logic)
    if 'strike' not in options_df.columns:
        print("Parsing strikes...")
        parts = options_df['option_symbol'].astype(str).str.split(' ', expand=True)
        # parts[1] is e.g. "C6000"
        if parts.shape[1] > 1:
            options_df['strike'] = pd.to_numeric(parts[1].str[1:], errors='coerce')
        else:
            options_df['strike'] = 0.0
            
    strikes = options_df['strike']
    print(f"Option Strikes: Min={strikes.min()}, Max={strikes.max()}, Count={len(strikes)}")
    
    # 4. Check Overlap
    # CONFIG.FUEL_STRIKE_RANGE = 15.0
    # Check how many options are within range of mean spot
    
    lower = mean_price - CONFIG.FUEL_STRIKE_RANGE
    upper = mean_price + CONFIG.FUEL_STRIKE_RANGE
    
    in_range = options_df[(options_df['strike'] >= lower) & (options_df['strike'] <= upper)]
    print(f"Options in range [{lower:.2f}, {upper:.2f}]: {len(in_range)}")
    
    if len(in_range) == 0:
        print("CRITICAL: No options near spot price! Metrics will be zero.")
        # Find nearest strike
        dist = np.abs(strikes - mean_price)
        idx_nearest = dist.idxmin()
        nearest_strike = strikes.iloc[idx_nearest]
        nearest_sym = options_df.iloc[idx_nearest]['option_symbol']
        print(f"Nearest strike: {nearest_strike} (Symbol: {nearest_sym}), Dist: {dist.min():.2f}")

if __name__ == "__main__":
    debug()
