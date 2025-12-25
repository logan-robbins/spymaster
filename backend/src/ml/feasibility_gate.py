from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

from src.common.config import CONFIG


@dataclass
class FeasibilityMask:
    allow_break: bool
    allow_bounce: bool
    break_logit_bias: float


class FeasibilityGate:
    """
    Physics prior using barrier/tape/gamma regimes (soft logit bias).
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
        break_bias = 0.0

        if barrier_state in {"WALL", "ABSORPTION"} and fuel_effect == "DAMPEN":
            if tape_signal < -self.config.FEASIBILITY_TAPE_IMBALANCE:
                break_bias -= self.config.FEASIBILITY_LOGIT_STEP

        if barrier_state in {"VACUUM", "CONSUMED"} and fuel_effect == "AMPLIFY":
            if tape_signal > self.config.FEASIBILITY_TAPE_IMBALANCE:
                break_bias += self.config.FEASIBILITY_LOGIT_STEP

        if abs(gamma_exposure) > self.config.FEASIBILITY_GAMMA_EXPOSURE:
            if gamma_exposure < 0 and fuel_effect == "AMPLIFY":
                break_bias += self.config.FEASIBILITY_LOGIT_STEP
            if gamma_exposure > 0 and fuel_effect == "DAMPEN":
                break_bias -= self.config.FEASIBILITY_LOGIT_STEP

        break_bias = max(-self.config.FEASIBILITY_LOGIT_CAP, min(self.config.FEASIBILITY_LOGIT_CAP, break_bias))

        return FeasibilityMask(
            allow_break=allow_break,
            allow_bounce=allow_bounce,
            break_logit_bias=break_bias
        )

    @staticmethod
    def apply_mask(prob_break: float, mask: FeasibilityMask) -> float:
        if not mask.allow_break and mask.allow_bounce:
            return 0.0
        if not mask.allow_bounce and mask.allow_break:
            return 1.0

        if not 0.0 < prob_break < 1.0:
            return prob_break

        logit = math.log(prob_break / (1.0 - prob_break))
        logit += mask.break_logit_bias
        return 1.0 / (1.0 + math.exp(-logit))
