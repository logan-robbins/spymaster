"""Setup configuration for level approach extraction.

All configuration knobs required by COMPLIANCE_GPT.md are defined here.
These parameters are persisted per-vector for full auditability.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


@dataclass(frozen=True)
class SetupConfig:
    """Configuration for level-approach setup extraction.

    All parameters are persisted per-vector for reproducibility.
    """

    bar_interval_sec: int = 120
    lookback_bars_infer: int = 6
    confirm_bars: int = 1
    lookfwd_bars_label: int = 6

    cross_epsilon_pts: float = 0.01
    approach_side_min_bars: int = 2
    reset_min_bars: int = 2
    reset_distance_pts: float = 1.0

    break_threshold_pts: float = 2.0
    reject_threshold_pts: float = 2.0
    outcome_price_basis: Literal["close", "hl"] = "close"

    confirm_close_rule: Literal["hold_near_level", "no_retrace_trigger"] = "hold_near_level"
    confirm_close_buffer_pts: float = 1.0

    chop_retrace_to_trigger_open_pts: float = 0.0
    chop_override_next_close_delta_pts: float = 10.0
    failed_break_confirm_close_below_level_pts: float = 0.5

    stop_buffer_pts: float = 1.0
    max_adverse_excursion_horizon_bars: int = 6

    min_bars_between_touches: int = 3
    max_touches_per_level_per_session: int = 10

    def to_dict(self) -> dict:
        """Convert to dictionary for persistence."""
        return asdict(self)


DEFAULT_CONFIG = SetupConfig()
