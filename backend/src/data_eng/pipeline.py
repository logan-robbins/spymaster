from __future__ import annotations

from typing import List

from .stages.base import Stage


def build_pipeline(product_type: str, layer: str = "all") -> List[Stage]:
    """Return the ordered list of stages for the given product_type and layer.

    Args:
        product_type: One of 'future_mbo', 'future_option_mbo', 'equity_mbo', 'equity_option_mbo'
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

    elif product_type == "equity_option_mbo":
        from .stages.bronze.equity_option_mbo.ingest import BronzeIngestEquityOptionMbo
        from .stages.silver.equity_option_mbo.compute_book_states_1s import SilverComputeEquityOptionBookStates1s
        from .stages.gold.equity_option_mbo.compute_physics_surface_1s import GoldComputeEquityOptionPhysicsSurface1s

        if layer == "bronze":
            return [BronzeIngestEquityOptionMbo()]
        elif layer == "silver":
            return [SilverComputeEquityOptionBookStates1s()]
        elif layer == "gold":
            return [GoldComputeEquityOptionPhysicsSurface1s()]
        elif layer == "all":
            return [
                BronzeIngestEquityOptionMbo(),
                SilverComputeEquityOptionBookStates1s(),
                GoldComputeEquityOptionPhysicsSurface1s(),
            ]

    elif product_type == "equity_mbo":
        from .stages.bronze.equity_mbo.ingest import BronzeIngestEquityMbo
        from .stages.silver.equity_mbo.compute_book_states_1s import SilverComputeEquityBookStates1s
        from .stages.gold.equity_mbo.compute_physics_surface_1s import GoldComputeEquityPhysicsSurface1s
        if layer == "bronze":
            return [BronzeIngestEquityMbo()]
        elif layer == "silver":
            return [
                SilverComputeEquityBookStates1s(),
            ]
        elif layer == "gold":
            return [
                GoldComputeEquityPhysicsSurface1s(),
            ]
        elif layer == "all":
            return [
                BronzeIngestEquityMbo(),
                SilverComputeEquityBookStates1s(),
                GoldComputeEquityPhysicsSurface1s(),
            ]

    raise ValueError(
        f"Unknown product_type: {product_type}. "
        f"Must be one of: future_mbo, future_option_mbo, equity_mbo, equity_option_mbo"
    )
