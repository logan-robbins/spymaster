"""
Stage: Compute Physics
Type: Feature Engineering (Instantaneous)
Input: Interaction Events (touches_df), Trades, MBP-10, Options Flow
Output: Signals DataFrame (augmented touches_df)

Transformation:
1. For each interaction event (touch), computes "Instantaneous Physics" metrics at that exact timestamp:
   - Barrier State: Vacuum/Wall/Neutral based on limit order book replenishment.
   - Tape Metrics: Velocity, Buy/Sell Volume imbalance, Sweep detection.
   - Fuel Metrics: Gamma Exposure (GEX) and Market Tide (Options Flow) relative to price.
2. Uses fully vectorized "Batch Engines" to compute these metrics efficiently across all events.
3. Encodes categorical states (e.g. VACUUM -> -3) for ML consumption.

Note: This stage captures the *micro-structure* context at the precise moment of interaction, answering "What was the texture of the market when price hit this level?"
"""
import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.core.barrier_engine import BarrierEngine, Direction as BarrierDirection
from src.core.tape_engine import TapeEngine
from src.core.fuel_engine import FuelEngine
from src.core.market_state import MarketState
from src.common.event_types import FuturesTrade, MBP10, OptionTrade
from src.common.config import CONFIG

logger = logging.getLogger(__name__)


def compute_physics_batch(
    touches_df: pd.DataFrame,
    market_state: MarketState,
    barrier_engine: BarrierEngine,
    tape_engine: TapeEngine,
    fuel_engine: FuelEngine,
    exp_date: str,
    trades: List[FuturesTrade] = None,
    mbp10_snapshots: List[MBP10] = None,
    option_trades_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Compute physics metrics for all touches in batch.

    Tries to use vectorized engines when raw data is available for proper
    time-windowed queries, falls back to per-signal processing otherwise.

    Args:
        touches_df: DataFrame from detect_interaction_zones
        market_state: Initialized MarketState
        barrier_engine: BarrierEngine instance
        tape_engine: TapeEngine instance
        fuel_engine: FuelEngine instance
        exp_date: Expiration date for options
        trades: Optional raw trades for vectorized processing
        mbp10_snapshots: Optional MBP-10 snapshots for vectorized processing
        option_trades_df: Optional raw option trades DF for vectorized processing

    Returns:
        DataFrame with physics columns added
    """
    if touches_df.empty:
        return touches_df

    n = len(touches_df)

    # Try to use fully vectorized engines if we have raw data
    if trades is not None and len(trades) > 0:
        try:
            from src.core.batch_engines import (
                build_vectorized_market_data,
                compute_tape_metrics_batch,
                compute_barrier_metrics_batch,
                compute_fuel_metrics_batch
            )

            # Build vectorized market data
            vmd = build_vectorized_market_data(
                trades=trades,
                mbp10_snapshots=mbp10_snapshots or [],
                option_flows=market_state.option_flows,
                option_trades_df=option_trades_df,
                date=exp_date
            )

            # Extract arrays from touches_df
            touch_ts_ns = touches_df['ts_ns'].values.astype(np.int64)
            level_prices = touches_df['level_price'].values.astype(np.float64)
            directions = np.where(touches_df['direction'].values == 'UP', 1, -1)

            # Compute tape metrics (vectorized)
            tape_metrics = compute_tape_metrics_batch(
                touch_ts_ns, level_prices, vmd,
                window_seconds=CONFIG.W_t,
                band_dollars=CONFIG.TAPE_BAND
            )

            # Compute barrier metrics (vectorized)
            barrier_metrics = compute_barrier_metrics_batch(
                touch_ts_ns, level_prices, directions, vmd,
                window_seconds=CONFIG.W_b,
                zone_es_ticks=CONFIG.BARRIER_ZONE_ES_TICKS
            )

            # Compute fuel metrics (vectorized) with timestamp filtering
            fuel_metrics = compute_fuel_metrics_batch(
                touch_ts_ns, level_prices, vmd,
                strike_range=CONFIG.FUEL_STRIKE_RANGE,
                split_range=CONFIG.TIDE_SPLIT_RANGE
            )

            # Build result
            result = touches_df.copy()
            result['barrier_state'] = barrier_metrics['barrier_state']
            result['barrier_delta_liq'] = barrier_metrics['barrier_delta_liq']
            result['barrier_replenishment_ratio'] = barrier_metrics['barrier_replenishment_ratio']
            result['wall_ratio'] = barrier_metrics['wall_ratio']
            # Barrier / Spatial Depth
            result['barrier_size'] = barrier_metrics['depth_in_zone']
            result['barrier_dist'] = np.zeros(n, dtype=np.float64)
            result['barrier_dist_atr'] = np.zeros(n, dtype=np.float64)
            
            # Spatial Limit Liquidity (Depth)
            result['limit_bid_size'] = barrier_metrics.get('limit_bid_size', np.zeros(n))
            result['limit_ask_size'] = barrier_metrics.get('limit_ask_size', np.zeros(n))
            result['limit_bid_size_above'] = barrier_metrics.get('limit_bid_size_above', np.zeros(n))
            result['limit_ask_size_above'] = barrier_metrics.get('limit_ask_size_above', np.zeros(n))
            result['limit_bid_size_below'] = barrier_metrics.get('limit_bid_size_below', np.zeros(n))
            result['limit_ask_size_below'] = barrier_metrics.get('limit_ask_size_below', np.zeros(n))

            # Tape Metrics (Spatial)
            result['tape_imbalance'] = tape_metrics['tape_imbalance']
            result['tape_buy_vol'] = tape_metrics['tape_buy_vol']
            result['tape_sell_vol'] = tape_metrics['tape_sell_vol']
            result['tape_velocity'] = tape_metrics['tape_velocity']
            
            result['tape_buy_vol_above'] = tape_metrics.get('tape_buy_vol_above', np.zeros(n))
            result['tape_sell_vol_above'] = tape_metrics.get('tape_sell_vol_above', np.zeros(n))
            result['tape_imbalance_above'] = tape_metrics.get('tape_imbalance_above', np.zeros(n))
            
            result['tape_buy_vol_below'] = tape_metrics.get('tape_buy_vol_below', np.zeros(n))
            result['tape_sell_vol_below'] = tape_metrics.get('tape_sell_vol_below', np.zeros(n))
            result['tape_imbalance_below'] = tape_metrics.get('tape_imbalance_below', np.zeros(n))

            # Fuel / Tide Metrics (Spatial)
            result['gamma_exposure'] = fuel_metrics['gamma_exposure']
            result['fuel_effect'] = fuel_metrics['fuel_effect']
            result['call_tide'] = fuel_metrics['call_tide']
            result['put_tide'] = fuel_metrics['put_tide']
            
            result['call_tide_above'] = fuel_metrics.get('call_tide_above', np.zeros(n))
            result['put_tide_above'] = fuel_metrics.get('put_tide_above', np.zeros(n))
            result['call_tide_below'] = fuel_metrics.get('call_tide_below', np.zeros(n))
            result['put_tide_below'] = fuel_metrics.get('put_tide_below', np.zeros(n))
            
            # Split Tide Features (Phase 4.6)
            for feat in ['call_tide_above_5pt', 'call_tide_below_5pt', 'put_tide_above_5pt', 'put_tide_below_5pt']:
                result[feat] = fuel_metrics.get(feat, np.zeros(n, dtype=np.float64))
            
            # Encode barrier_state and fuel_effect to numeric
            # Per EPISODE_VECTOR_SCHEMA.md Section D
            barrier_state_map = {
                'VACUUM': -3,
                'CONSUMED': -2,
                'ABSORPTION': -1,
                'NEUTRAL': 0,
                'WEAK': 1,
                'WALL': 2
            }
            fuel_effect_map = {
                'AMPLIFY': -1,  # Dealer short gamma amplifies moves
                'NEUTRAL': 0,
                'DAMPEN': 1     # Dealer long gamma dampens moves
            }
            
            result['barrier_state_encoded'] = result['barrier_state'].map(barrier_state_map).fillna(0).astype(np.int32)
            result['fuel_effect_encoded'] = result['fuel_effect'].map(fuel_effect_map).fillna(0).astype(np.int32)

            return result

        except Exception as e:
            # No fallback - vectorized engines required
            raise RuntimeError(f"Vectorized batch engines failed: {e}")


class ComputePhysicsStage(BaseStage):
    """Compute physics metrics (barrier, tape, fuel) for all touches.

    Uses vectorized engines when historical data is available,
    falls back to per-signal processing otherwise.

    Outputs:
        signals_df: DataFrame with physics columns added:
            - barrier_state, barrier_delta_liq, barrier_replenishment_ratio, wall_ratio
            - tape_imbalance, tape_buy_vol, tape_sell_vol, tape_velocity, sweep_detected
            - fuel_effect, gamma_exposure
    """

    def __init__(self):
        self.barrier_engine = BarrierEngine()
        self.tape_engine = TapeEngine()
        self.fuel_engine = FuelEngine()

    @property
    def name(self) -> str:
        return "compute_physics"

    @property
    def required_inputs(self) -> List[str]:
        return ['touches_df', 'market_state', 'trades', 'mbp10_snapshots']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        touches_df = ctx.data['touches_df']
        market_state = ctx.data['market_state']
        trades = ctx.data['trades']
        mbp10_snapshots = ctx.data['mbp10_snapshots']
        option_trades_df = ctx.data.get('option_trades_df')
        
        n_touches = len(touches_df)
        logger.info(f"  Computing physics for {n_touches:,} touches...")
        # Check if df is valid
        num_opts = len(option_trades_df) if option_trades_df is not None else 0
        logger.debug(f"    Using {len(trades):,} trades, {len(mbp10_snapshots):,} MBP-10 snapshots, {num_opts:,} option trades")

        signals_df = compute_physics_batch(
            touches_df=touches_df,
            market_state=market_state,
            barrier_engine=self.barrier_engine,
            tape_engine=self.tape_engine,
            fuel_engine=self.fuel_engine,
            exp_date=None, # Disable 0DTE filter to allow all loaded options (e.g. ESZ5 in Oct)
            trades=trades,
            mbp10_snapshots=mbp10_snapshots,
            option_trades_df=option_trades_df
        )

        # Log barrier state distribution
        if 'barrier_state' in signals_df.columns:
            barrier_dist = signals_df['barrier_state'].value_counts().to_dict()
            logger.info(f"    Barrier states: {barrier_dist}")

        # Log fuel effect distribution
        if 'fuel_effect' in signals_df.columns:
            fuel_dist = signals_df['fuel_effect'].value_counts().to_dict()
            logger.info(f"    Fuel effects: {fuel_dist}")

        return {'signals_df': signals_df}
