"""Compute physics metrics stage."""
from typing import Any, Dict, List

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.utils.vectorized_ops import compute_physics_batch
from src.core.barrier_engine import BarrierEngine
from src.core.tape_engine import TapeEngine
from src.core.fuel_engine import FuelEngine


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

        return {'signals_df': signals_df}
