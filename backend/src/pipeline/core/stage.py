"""Base stage interface for pipeline stages."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class StageContext:
    """Context passed between pipeline stages.

    Attributes:
        date: Date being processed (YYYY-MM-DD)
        level: Level type (PM_HIGH, PM_LOW, OR_HIGH, OR_LOW, SMA_90)
        data: Outputs from previous stages, keyed by name
        config: Stage-specific configuration from CONFIG
    """
    date: str
    level: str
    data: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


class BaseStage(ABC):
    """Base class for all pipeline stages.

    All operations must be vectorized (NumPy/pandas, no Python loops over data).
    Optimized for Apple M4 Silicon with 128GB RAM.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name for logging."""
        pass

    @property
    def required_inputs(self) -> List[str]:
        """Keys from context.data this stage needs. Empty = no dependencies."""
        return []

    @abstractmethod
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        """Execute stage using vectorized operations.

        Args:
            ctx: Context with date, previous stage outputs, config

        Returns:
            Dict to merge into context.data for downstream stages
        """
        pass

    def run(self, ctx: StageContext) -> StageContext:
        """Run stage and update context.

        Validates required inputs are present, then executes and merges outputs.
        """
        # Validate required inputs
        missing = [key for key in self.required_inputs if key not in ctx.data]
        if missing:
            raise ValueError(
                f"Stage '{self.name}' missing required inputs: {missing}. "
                f"Available: {list(ctx.data.keys())}"
            )

        outputs = self.execute(ctx)
        ctx.data.update(outputs)
        return ctx
