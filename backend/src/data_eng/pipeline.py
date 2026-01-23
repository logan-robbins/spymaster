from __future__ import annotations

from typing import List

from .stages.base import Stage


def build_pipeline(product_type: str, layer: str = "all") -> List[Stage]:
    """Return the ordered list of stages for the given product_type and layer.

    Args:
        product_type: One of 'future_mbo', 'future_option_mbo', 'equity_mbo', 'equity_option_cmbp_1', 'hud'
        layer: One of 'bronze', 'silver', 'gold', 'all'

    Returns:
        List of Stage instances to execute in order
    """

    if product_type == "future_option_mbo":
        from .stages.bronze.shared.ingest_instrument_definitions import BronzeIngestInstrumentDefinitions
        from .stages.bronze.future_option_mbo.ingest import BronzeIngestFutureOptionMbo
        from .stages.bronze.future_option.ingest_statistics import BronzeIngestFutureOptionStatistics
        from .stages.silver.future_option.compute_statistics_clean import SilverComputeStatisticsClean
        from .stages.silver.future_option_mbo.compute_gex_surface_1s import SilverComputeGexSurface1s

        if layer == "bronze":
            return [
                BronzeIngestInstrumentDefinitions(),
                BronzeIngestFutureOptionStatistics(),
                BronzeIngestFutureOptionMbo(),
            ]
        elif layer == "silver":
            return [
                SilverComputeStatisticsClean(),
                SilverComputeGexSurface1s(),
            ]
        elif layer == "gold":
            return []
        elif layer == "all":
            return [
                BronzeIngestInstrumentDefinitions(),
                BronzeIngestFutureOptionStatistics(),
                BronzeIngestFutureOptionMbo(),
                SilverComputeStatisticsClean(),
                SilverComputeGexSurface1s(),
            ]

    elif product_type == "future_mbo":
        from .stages.bronze.future_mbo.ingest_preview import BronzeIngestMboPreview
        from .stages.silver.future_mbo.compute_snapshot_and_wall_1s import SilverComputeSnapshotAndWall1s
        from .stages.silver.future_mbo.compute_vacuum_surface_1s import SilverComputeVacuumSurface1s
        from .stages.silver.future_mbo.compute_physics_bands_1s import SilverComputePhysicsBands1s

        if layer == "bronze":
            return [BronzeIngestMboPreview()]
        elif layer == "silver":
            return [
                SilverComputeSnapshotAndWall1s(),
                SilverComputeVacuumSurface1s(),
                SilverComputePhysicsBands1s(),
            ]
        elif layer == "gold":
            return []

    elif product_type == "equity_mbo":
        from .stages.bronze.equity_mbo.ingest import BronzeIngestEquityMbo
        from .stages.silver.equity_mbo.compute_snapshot_and_wall_1s import SilverComputeEquitySnapshotAndWall1s
        from .stages.silver.equity_mbo.compute_vacuum_surface_1s import SilverComputeEquityVacuumSurface1s
        from .stages.silver.equity_mbo.compute_radar_vacuum_1s import SilverComputeEquityRadarVacuum1s
        from .stages.silver.equity_mbo.compute_physics_bands_1s import SilverComputeEquityPhysicsBands1s
        from .stages.gold.equity_mbo.build_physics_norm_calibration import GoldBuildEquityPhysicsNormCalibration

        if layer == "bronze":
            return [BronzeIngestEquityMbo()]
        elif layer == "silver":
            return [
                SilverComputeEquitySnapshotAndWall1s(),
                SilverComputeEquityVacuumSurface1s(),
                SilverComputeEquityRadarVacuum1s(),
                SilverComputeEquityPhysicsBands1s(),
            ]
        elif layer == "gold":
            return [GoldBuildEquityPhysicsNormCalibration()]
        elif layer == "all":
            return [
                BronzeIngestEquityMbo(),
                SilverComputeEquitySnapshotAndWall1s(),
                GoldBuildEquityPhysicsNormCalibration(),
                SilverComputeEquityVacuumSurface1s(),
                SilverComputeEquityRadarVacuum1s(),
                SilverComputeEquityPhysicsBands1s(),
            ]

    elif product_type == "equity_option_cmbp_1":
        from .stages.bronze.equity_option_cmbp_1.ingest import BronzeIngestEquityOptionCmbp1

        if layer == "bronze":
            return [BronzeIngestEquityOptionCmbp1()]
        elif layer == "silver":
            return []
        elif layer == "gold":
            return []
        elif layer == "all":
            return [BronzeIngestEquityOptionCmbp1()]

    elif product_type == "hud":
        from .stages.gold.hud.build_physics_norm_calibration import GoldBuildHudPhysicsNormCalibration

        if layer == "gold":
            return [GoldBuildHudPhysicsNormCalibration()]
        elif layer == "all":
            return [GoldBuildHudPhysicsNormCalibration()]

    raise ValueError(
        f"Unknown product_type: {product_type}. "
        f"Must be one of: future_mbo, future_option_mbo, equity_mbo, equity_option_cmbp_1, hud"
    )
