from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from src.common.config import CONFIG


@dataclass
class FeasibilityMask:
    allow_break: bool
    allow_bounce: bool


class FeasibilityGate:
    """
    Deterministic feasibility mask using barrier/tape/gamma regimes.
    """

    def __init__(self, config=None):
        self.config = config or CONFIG

    def compute_mask(
        self,
        direction: str,
        barrier_state: str,
        tape_imbalance: float,
        tape_velocity: float,
        fuel_effect: str,
        gamma_exposure: float
    ) -> FeasibilityMask:
        direction_sign = 1.0 if direction == "UP" else -1.0
        tape_signal = tape_imbalance * direction_sign

        allow_break = True
        allow_bounce = True

        if barrier_state in {"WALL", "ABSORPTION"} and fuel_effect == "DAMPEN":
            if tape_signal < -self.config.FEASIBILITY_TAPE_IMBALANCE:
                allow_break = False

        if barrier_state in {"VACUUM", "CONSUMED"} and fuel_effect == "AMPLIFY":
            if tape_signal > self.config.FEASIBILITY_TAPE_IMBALANCE:
                allow_bounce = False

        if abs(gamma_exposure) > self.config.FEASIBILITY_GAMMA_EXPOSURE:
            if gamma_exposure < 0 and fuel_effect == "AMPLIFY":
                allow_bounce = False
            if gamma_exposure > 0 and fuel_effect == "DAMPEN":
                allow_break = False

        return FeasibilityMask(allow_break=allow_break, allow_bounce=allow_bounce)

    @staticmethod
    def apply_mask(prob_break: float, mask: FeasibilityMask) -> float:
        if not mask.allow_break and mask.allow_bounce:
            return 0.0
        if not mask.allow_bounce and mask.allow_break:
            return 1.0
        return prob_break
