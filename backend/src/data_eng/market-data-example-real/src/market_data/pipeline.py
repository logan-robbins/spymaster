from __future__ import annotations

from typing import List

from .stages.base import Stage
from .stages.silver_convert_utc_to_est import SilverConvertUtcToEst
from .stages.gold_filter_first3h import GoldFilterFirst3Hours


def build_pipeline() -> List[Stage]:
    """Return the ordered list of stages for this demo pipeline."""

    return [
        SilverConvertUtcToEst(),
        GoldFilterFirst3Hours(),
    ]
