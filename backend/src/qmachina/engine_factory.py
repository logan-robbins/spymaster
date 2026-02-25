"""Factory for creating AbsoluteTickEngine instances from RuntimeConfig."""
from __future__ import annotations

from .config import RuntimeConfig
from .stream_time_utils import resolve_tick_int


def create_absolute_tick_engine(config: RuntimeConfig):
    """Create an AbsoluteTickEngine configured for the given RuntimeConfig."""
    from qm_engine import AbsoluteTickEngine

    tick_int = resolve_tick_int(config)
    return AbsoluteTickEngine(
        n_ticks=config.n_absolute_ticks,
        tick_int=tick_int,
        bucket_size_dollars=config.bucket_size_dollars,
        tau_velocity=config.tau_velocity,
        tau_acceleration=config.tau_acceleration,
        tau_jerk=config.tau_jerk,
        tau_rest_decay=config.tau_rest_decay,
        c1_v_add=config.c1_v_add,
        c2_v_rest_pos=config.c2_v_rest_pos,
        c3_a_add=config.c3_a_add,
        c4_v_pull=config.c4_v_pull,
        c5_v_fill=config.c5_v_fill,
        c6_v_rest_neg=config.c6_v_rest_neg,
        c7_a_pull=config.c7_a_pull,
    )
