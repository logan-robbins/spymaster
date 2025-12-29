"""Compute physics metrics stage."""
import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.core.barrier_engine import BarrierEngine, Direction as BarrierDirection
from src.core.tape_engine import TapeEngine
from src.core.fuel_engine import FuelEngine
from src.core.market_state import MarketState
from src.common.event_types import FuturesTrade, MBP10
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
    mbp10_snapshots: List[MBP10] = None
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

            # Compute fuel metrics (vectorized)
            fuel_metrics = compute_fuel_metrics_batch(
                level_prices, vmd,
                strike_range=CONFIG.FUEL_STRIKE_RANGE
            )

            # Build result
            result = touches_df.copy()
            result['barrier_state'] = barrier_metrics['barrier_state']
            result['barrier_delta_liq'] = barrier_metrics['barrier_delta_liq']
            result['barrier_replenishment_ratio'] = np.zeros(n)
            result['wall_ratio'] = barrier_metrics['wall_ratio']
            result['tape_imbalance'] = tape_metrics['tape_imbalance']
            result['tape_buy_vol'] = tape_metrics['tape_buy_vol']
            result['tape_sell_vol'] = tape_metrics['tape_sell_vol']
            result['tape_velocity'] = tape_metrics['tape_velocity']
            result['sweep_detected'] = np.zeros(n, dtype=bool)
            result['fuel_effect'] = fuel_metrics['fuel_effect']
            result['gamma_exposure'] = fuel_metrics['gamma_exposure']

            return result

        except Exception as e:
            import logging
            logging.warning(f"Vectorized engines failed, falling back to per-signal: {e}")

    # Fallback: per-signal processing using MarketState
    barrier_states = np.empty(n, dtype=object)
    barrier_delta_liq = np.zeros(n, dtype=np.float64)
    barrier_replen = np.zeros(n, dtype=np.float64)
    wall_ratios = np.zeros(n, dtype=np.float64)
    tape_imbalance = np.zeros(n, dtype=np.float64)
    tape_buy_vol = np.zeros(n, dtype=np.int64)
    tape_sell_vol = np.zeros(n, dtype=np.int64)
    tape_velocity = np.zeros(n, dtype=np.float64)
    sweep_detected = np.zeros(n, dtype=bool)
    fuel_effects = np.empty(n, dtype=object)
    gamma_exposure = np.zeros(n, dtype=np.float64)

    for i in range(n):
        row = touches_df.iloc[i]
        level_price = row['level_price']
        direction_str = row['direction']

        barrier_dir = BarrierDirection.RESISTANCE if direction_str == 'UP' else BarrierDirection.SUPPORT

        try:
            barrier_metrics_result = barrier_engine.compute_barrier_state(
                level_price=level_price,
                direction=barrier_dir,
                market_state=market_state
            )
            barrier_states[i] = barrier_metrics_result.state.value
            barrier_delta_liq[i] = barrier_metrics_result.delta_liq
            barrier_replen[i] = barrier_metrics_result.replenishment_ratio
            wall_ratios[i] = barrier_metrics_result.depth_in_zone / 5000.0 if barrier_metrics_result.depth_in_zone else 0.0
        except:
            barrier_states[i] = 'NEUTRAL'

        try:
            tape_metrics_result = tape_engine.compute_tape_state(
                level_price=level_price,
                market_state=market_state
            )
            tape_imbalance[i] = tape_metrics_result.imbalance
            tape_buy_vol[i] = tape_metrics_result.buy_vol
            tape_sell_vol[i] = tape_metrics_result.sell_vol
            tape_velocity[i] = tape_metrics_result.velocity
            sweep_detected[i] = tape_metrics_result.sweep.detected
        except:
            pass

        try:
            fuel_metrics_result = fuel_engine.compute_fuel_state(
                level_price=level_price,
                market_state=market_state,
                exp_date_filter=exp_date
            )
            fuel_effects[i] = fuel_metrics_result.effect.value
            gamma_exposure[i] = fuel_metrics_result.net_dealer_gamma
        except:
            fuel_effects[i] = 'NEUTRAL'

    result = touches_df.copy()
    result['barrier_state'] = barrier_states
    result['barrier_delta_liq'] = barrier_delta_liq
    result['barrier_replenishment_ratio'] = barrier_replen
    result['wall_ratio'] = wall_ratios
    result['tape_imbalance'] = tape_imbalance
    result['tape_buy_vol'] = tape_buy_vol
    result['tape_sell_vol'] = tape_sell_vol
    result['tape_velocity'] = tape_velocity
    result['sweep_detected'] = sweep_detected
    result['fuel_effect'] = fuel_effects
    result['gamma_exposure'] = gamma_exposure

    return result


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

        n_touches = len(touches_df)
        logger.info(f"  Computing physics for {n_touches:,} touches...")
        logger.debug(f"    Using {len(trades):,} trades, {len(mbp10_snapshots):,} MBP-10 snapshots")

        signals_df = compute_physics_batch(
            touches_df=touches_df,
            market_state=market_state,
            barrier_engine=self.barrier_engine,
            tape_engine=self.tape_engine,
            fuel_engine=self.fuel_engine,
            exp_date=ctx.date,
            trades=trades,
            mbp10_snapshots=mbp10_snapshots
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
