"""
Stage: Init Market State
Type: Initialization
Input: Futures Trades, MBP-10 Snapshots, Options Trades
Output: MarketState Object, Enrichment Options DataFrame

Transformation:
1. Initializes the `MarketState` container (Greeks, Liquidity, Sentiment).
2. Identifies the "Active Contract" based on volume dominance.
3. Computes Delta and Gamma for all Options Trades using Black-76 (vectorized).
4. Infers Aggressor side (Buy/Sell/Mid) for options using Tick Test if missing.
5. Aggregates option flows into MarketState.

Note: This stage prepares the core state object that tracks market context. Computing Greeks here allows downstream stages to just sum them up for GEX/Tide.
"""
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.core.market_state import MarketState
from src.core.black_scholes import compute_greeks_for_dataframe
from src.common.event_types import OptionTrade, Aggressor
from src.common.config import CONFIG


class InitMarketStateStage(BaseStage):
    """Initialize MarketState with trades, MBP-10, and options.

    Computes Greeks for options using vectorized Black-76 (futures options).

    Outputs:
        market_state: MarketState instance
        option_trades_df: Updated with delta/gamma columns
    """

    @property
    def name(self) -> str:
        return "init_market_state"

    @property
    def required_inputs(self) -> List[str]:
        return ['trades', 'mbp10_snapshots', 'option_trades_df']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        trades = ctx.data['trades']
        mbp10_snapshots = ctx.data['mbp10_snapshots']
        option_trades_df = ctx.data['option_trades_df']

        # Initialize MarketState with appropriate buffer
        buffer_seconds = max(CONFIG.W_b, CONFIG.CONFIRMATION_WINDOW_SECONDS)
        market_state = MarketState(max_buffer_window_seconds=buffer_seconds * 2)

        # Load trades and find spot price (ES points)
        spot_price = None
        active_contract = None
        
        for trade in trades:
            if active_contract is None:
                active_contract = trade.symbol
                
            market_state.update_es_trade(trade)
            if 3000 < trade.price < 10000:
                spot_price = trade.price

        if spot_price is None:
            spot_price = 6000.0

        # Load MBP-10 snapshots
        for mbp in mbp10_snapshots:
            market_state.update_es_mbp10(mbp)

        # Load options with vectorized Greeks
        if not option_trades_df.empty:
            import logging
            logger = logging.getLogger(__name__)
            
            # Log contract distribution
            opt_contracts_all = option_trades_df['option_symbol'].astype(str).str.split(' ').str[0]
            dist = opt_contracts_all.value_counts().to_dict()
            logger.info(f"DEBUG: Found {len(option_trades_df)} options. Active contract: {active_contract}")
            logger.info(f"DEBUG: Option contract distribution: {dist}")
            
            # Filter options to match active futures contract (avoids basis mismatch during rollover)
            if active_contract and active_contract != "ES":
                # e.g. "ESM5 C6000" -> "ESM5"
                opt_contracts = option_trades_df['option_symbol'].astype(str).str.split(' ').str[0]
                total_opts = len(option_trades_df)
                option_trades_df = option_trades_df[opt_contracts == active_contract].copy()
                logger.info(f"DEBUG: Filtered options from {total_opts} to {len(option_trades_df)} using contract {active_contract}")

            # Parse strike and right if missing (fallback for Polygon/Bronze irregularities)
            if 'strike' not in option_trades_df.columns or 'right' not in option_trades_df.columns:
                 try:
                    # Expect format "SYMBOL RightStrike" e.g. "ESH6 C6000" or "ESZ5 P5900"
                    # Split by space
                    parts = option_trades_df['option_symbol'].astype(str).str.split(' ', expand=True)
                    if parts.shape[1] >= 2:
                        # part 1 is "C6000" or "P5900"
                        option_trades_df['right'] = parts[1].str[0]  # 'C' or 'P'
                        option_trades_df['strike'] = pd.to_numeric(parts[1].str[1:], errors='coerce')
                        
                        # Log strike stats
                        valid_strikes = option_trades_df['strike'].dropna()
                    else:
                        # Fallback/Fail safe
                        option_trades_df['strike'] = 0.0
                        option_trades_df['right'] = 'C'
                 except Exception as e:
                    # Log error if possible, or skip
                    option_trades_df['strike'] = 0.0
                    option_trades_df['right'] = 'C'

            delta_arr, gamma_arr = compute_greeks_for_dataframe(
                df=option_trades_df,
                spot=spot_price,
                exp_date=ctx.date
            )

            option_trades_df = option_trades_df.copy()
            option_trades_df['delta'] = delta_arr
            option_trades_df['gamma'] = gamma_arr
            
            logger.info(f"DEBUG: Loading {len(option_trades_df)} options into MarketState...")
            option_trades_df = self._load_options_to_market_state(
                market_state, option_trades_df
            )
            logger.info(f"DEBUG: MarketState now has {len(market_state.option_flows)} aggregated flows.")
        else:
            # Empty DF with correct columns if no trades
            option_trades_df = pd.DataFrame(columns=[
                'ts_event_ns', 'strike', 'right', 'price', 
                'size', 'aggressor', 'delta', 'gamma'
            ])

        return {
            'market_state': market_state,
            'option_trades_df': option_trades_df,
            'spot_price': spot_price,
        }

    def _load_options_to_market_state(
        self,
        market_state: MarketState,
        option_trades_df: pd.DataFrame
    ):
        """Load options into MarketState. Returns list of OptionTrade objects."""
        last_price_map = {}
        last_aggr_map = {}
        
        processed_data = []

        # Sort by timestamp to ensure correct tick test order
        if 'ts_event_ns' in option_trades_df.columns:
            option_trades_df = option_trades_df.sort_values('ts_event_ns')

        for idx in range(len(option_trades_df)):
            try:
                row = option_trades_df.iloc[idx]
                aggressor_val = row.get('aggressor', 0)
                
                # Check directly if column exists or is NaN
                has_aggressor_col = 'aggressor' in option_trades_df.columns
                
                aggressor_enum = Aggressor.MID
                
                if has_aggressor_col and not (pd.isna(aggressor_val) or aggressor_val == '<NA>'):
                    try:
                        val = int(aggressor_val)
                    except:
                        val = 0
                    if val in (1, -1, 0):
                        aggressor_enum = Aggressor(val)

                # Fallback: Tick Test if Aggressor is MID/Missing
                if aggressor_enum == Aggressor.MID:
                    sym = row['option_symbol']
                    price = float(row['price'])
                    prev_price = last_price_map.get(sym)
                    
                    if prev_price is not None:
                        if price > prev_price:
                            aggressor_enum = Aggressor.BUY
                        elif price < prev_price:
                            aggressor_enum = Aggressor.SELL
                        else:
                            # Equal price: Use previous aggressor (or MID if none)
                            aggressor_enum = last_aggr_map.get(sym, Aggressor.MID)
                    else:
                        # First trade: Assume MID or infer? 
                        # Without bid/ask, we can't tell. Leave as MID.
                        pass
                    
                    last_price_map[sym] = price
                    last_aggr_map[sym] = aggressor_enum
                else:
                    # Update maps strictly
                    sym = row['option_symbol']
                    last_price_map[sym] = float(row['price'])
                    last_aggr_map[sym] = aggressor_enum

                trade = OptionTrade(
                    ts_event_ns=int(row['ts_event_ns']),
                    ts_recv_ns=int(row.get('ts_recv_ns', row['ts_event_ns'])),
                    source=row.get('source', 'polygon_rest'),
                    underlying=row.get('underlying', 'ES'),
                    option_symbol=row['option_symbol'],
                    exp_date=str(row.get('exp_date', row.get('date'))), # Fallback usage
                    strike=float(row['strike']),
                    right=row['right'],
                    price=float(row['price']),
                    size=int(row['size']),
                    opt_bid=row.get('opt_bid'),
                    opt_ask=row.get('opt_ask'),
                    aggressor=aggressor_enum,
                    conditions=None,
                    seq=row.get('seq')
                )

                processed_data.append({
                    'ts_event_ns': trade.ts_event_ns,
                    'strike': trade.strike,
                    'right': trade.right,
                    'price': trade.price,
                    'size': trade.size,
                    'aggressor': aggressor_enum.value,
                    'delta': row['delta'],
                    'gamma': row['gamma']
                })
                
                market_state.update_option_trade(
                    trade,
                    delta=row['delta'],
                    gamma=row['gamma']
                )
            except Exception as e:
                # Log first error only to avoid spam
                if idx == 0:
                     print(f"Error loading option trade: {e}")
                continue
        
        return pd.DataFrame(processed_data)
