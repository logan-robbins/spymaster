from __future__ import annotations

from typing import List

from .stages.base import Stage


def build_pipeline(product_type: str, layer: str = "all") -> List[Stage]:
    """Return the ordered list of stages for the given product_type and layer.

    Args:
        product_type: One of 'future_mbo', 'future_option_mbo'
        layer: One of 'bronze', 'silver', 'gold', 'all'

    Returns:
        List of Stage instances to execute in order
    """

    if product_type == "future_option_mbo":
        from .stages.bronze.future_option_mbo.ingest import BronzeIngestFutureOptionMbo
        from .stages.silver.future_option_mbo.compute_book_states_1s import SilverComputeOptionBookStates1s
        from .stages.gold.future_option_mbo.compute_physics_surface_1s import GoldComputeOptionPhysicsSurface1s

        if layer == "bronze":
            return [BronzeIngestFutureOptionMbo()]
        elif layer == "silver":
            return [SilverComputeOptionBookStates1s()]
        elif layer == "gold":
            return [GoldComputeOptionPhysicsSurface1s()]
        elif layer == "all":
            return [
                BronzeIngestFutureOptionMbo(),
                SilverComputeOptionBookStates1s(),
                GoldComputeOptionPhysicsSurface1s(),
            ]

    elif product_type == "future_mbo":
        from .stages.bronze.future_mbo.ingest import BronzeIngestFutureMbo
        from .stages.silver.future_mbo.compute_book_states_1s import SilverComputeBookStates1s
        from .stages.gold.future_mbo.compute_physics_surface_1s import GoldComputePhysicsSurface1s
        if layer == "bronze":
            return [BronzeIngestFutureMbo()]
        elif layer == "silver":
            return [
                SilverComputeBookStates1s(),
            ]
        elif layer == "gold":
            return [
                GoldComputePhysicsSurface1s(),
            ]
        elif layer == "all":
            return [
                BronzeIngestFutureMbo(),
                SilverComputeBookStates1s(),
                GoldComputePhysicsSurface1s(),
            ]

    raise ValueError(
        f"Unknown product_type: {product_type}. "
        f"Must be one of: future_mbo, future_option_mbo"
    )
