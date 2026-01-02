
"""
Debug Bronze Loading & Market State
Verify if interaction between LoadBronze and InitMarketState is dropping options data.
"""
import sys
from pathlib import Path
import logging
import pandas as pd
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.core.stage import StageContext
from src.pipeline.stages.load_bronze import LoadBronzeStage
from src.pipeline.stages.init_market_state import InitMarketStateStage
from src.common.config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    date = "2025-10-29" # Late October sample
    
    # 1. Load Bronze
    print(f"Loading Bronze for {date}...")
    load_stage = LoadBronzeStage()
    ctx = StageContext(date=date, config=vars(CONFIG))
    
    # Mock config to ensure we load enough data
    ctx.config['PIPELINE_START_HOUR'] = 9
    ctx.config['PIPELINE_END_HOUR'] = 10
    
    data = load_stage.execute(ctx)
    trades = data['trades']
    options = data['option_trades_df']
    
    print(f"Loaded {len(trades)} futures trades")
    print(f"Loaded {len(options)} option trades")
    
    
    if len(options) > 0:
        print("\nOption Columns:", options.columns.tolist())
        print("\nSample Option Symbols:")
        print(options['option_symbol'].head())
        
        # Check active contract logic
        first_trade_symbol = trades[0].symbol if trades else "UNKNOWN"
        print(f"\nActive Futures Contract from trades: {first_trade_symbol}")
        # return # EARLY EXIT TO SEE COLUMNS
        
    # 2. Init Market State
    print("\nInitializing Market State...")
    try:
        ctx.data = data
        init_stage = InitMarketStateStage()
        res = init_stage.execute(ctx)
        
        ms = res['market_state']
        
        # Check if options made it into MarketState
        # MarketState stores options in .option_flows which is a Dict[expiry, Dict[strike, ...]]
        # or we can check simple counts if exposed
        
        print("\nChecking MarketState internals:")
        # We can inspect the _option_flows structure if accessible, or check if greeks were computed
        
        # Check updated options df
        updated_options = res['option_trades_df']
        print(f"Options after Init: {len(updated_options)}")
        if 'delta' in updated_options.columns:
            print(f"Delta stats: Mean={updated_options['delta'].mean():.4f}, NaNs={updated_options['delta'].isna().sum()}")
        else:
            print("Delta column MISSING")

        # Inspect Aggressor
        if 'aggressor' in updated_options.columns:
            print(f"\nAggressor Stats:\n{updated_options['aggressor'].value_counts(dropna=False)}")
        else:
            print("\nAggressor column MISSING")
            
        # 3. Compute Physics (Tide)
        print("\nComputing Physics (feature generation)...")
        from src.core.fuel_engine import FuelEngine
        
        fuel_engine = FuelEngine()
        
        # Check if fuel engine has options (via market_state flows)
        # InitMarketState populated ms.option_flows.
        print(f"MarketState Option Flows: {len(ms.option_flows)} entries")
        
        # Compute sample tide at a dummy level (e.g. spot price)
        # We need a sample price.
        # Let's peek at flows to find a strike.
        if ms.option_flows:
            first_key = next(iter(ms.option_flows))
            dummy_level = first_key[0] # Strike
            print(f"Testing Tide at level {dummy_level}...")
            
            metrics = fuel_engine.compute_fuel_state(
                level_price=dummy_level,
                market_state=ms
            )
            print(f"Computed Tide Metrics: CallTide={metrics.call_tide:.2f}, PutTide={metrics.put_tide:.2f}")
            print(f"Net Dealer Gamma: {metrics.net_dealer_gamma:.2f}")
            print(f"Effect: {metrics.effect}")
        else:
            print("NO OPTION FLOWS found in MarketState!")
            
    except Exception as e:
        print(f"\nERROR in InitMarketState/Physics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
