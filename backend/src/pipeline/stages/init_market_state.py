"""Initialize MarketState stage."""
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

    Computes Greeks for options using vectorized Black-Scholes.

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

        # Load trades and find spot price
        spot_price = None
        for trade in trades:
            market_state.update_es_trade(trade)
            if 3000 < trade.price < 10000:
                spot_price = trade.price / 10.0

        if spot_price is None:
            spot_price = 600.0

        # Load MBP-10 snapshots
        for mbp in mbp10_snapshots:
            market_state.update_es_mbp10(mbp)

        # Load options with vectorized Greeks
        if not option_trades_df.empty:
            delta_arr, gamma_arr = compute_greeks_for_dataframe(
                df=option_trades_df,
                spot=spot_price,
                exp_date=ctx.date
            )

            option_trades_df = option_trades_df.copy()
            option_trades_df['delta'] = delta_arr
            option_trades_df['gamma'] = gamma_arr

            self._load_options_to_market_state(
                market_state, option_trades_df
            )

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
        """Load options into MarketState."""
        for idx in range(len(option_trades_df)):
            try:
                row = option_trades_df.iloc[idx]
                aggressor_val = row.get('aggressor', 0)

                if hasattr(aggressor_val, 'value'):
                    aggressor_enum = aggressor_val
                else:
                    aggressor_enum = Aggressor(
                        int(aggressor_val) if aggressor_val and aggressor_val != '<NA>' else 0
                    )

                trade = OptionTrade(
                    ts_event_ns=int(row['ts_event_ns']),
                    ts_recv_ns=int(row.get('ts_recv_ns', row['ts_event_ns'])),
                    source=row.get('source', 'polygon_rest'),
                    underlying=row.get('underlying', 'SPY'),
                    option_symbol=row['option_symbol'],
                    exp_date=str(row['exp_date']),
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

                market_state.update_option_trade(
                    trade,
                    delta=row['delta'],
                    gamma=row['gamma']
                )
            except Exception:
                continue
