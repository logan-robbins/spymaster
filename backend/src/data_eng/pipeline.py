from __future__ import annotations

from typing import List

from .stages.base import Stage


def build_pipeline(product_type: str, layer: str = "all") -> List[Stage]:
    """Return the ordered list of stages for the given product_type and layer.

    Args:
        product_type: One of 'future_mbo', 'future_option_mbo', 'hud'
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
        from .stages.gold.future_option_mbo.build_gex_enriched_trigger_vectors import (
            GoldBuildGexEnrichedTriggerVectors,
        )

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
            return [GoldBuildGexEnrichedTriggerVectors()]
        elif layer == "all":
            return [
                BronzeIngestInstrumentDefinitions(),
                BronzeIngestFutureOptionStatistics(),
                BronzeIngestFutureOptionMbo(),
                SilverComputeStatisticsClean(),
                SilverComputeGexSurface1s(),
                GoldBuildGexEnrichedTriggerVectors(),
            ]

    elif product_type == "future_mbo":
        from .stages.bronze.future_mbo.ingest_preview import BronzeIngestMboPreview
        from .stages.silver.future_mbo.compute_snapshot_and_wall_1s import SilverComputeSnapshotAndWall1s
        from .stages.silver.future_mbo.compute_vacuum_surface_1s import SilverComputeVacuumSurface1s
        from .stages.silver.future_mbo.compute_radar_vacuum_1s import SilverComputeRadarVacuum1s
        from .stages.silver.future_mbo.compute_physics_bands_1s import SilverComputePhysicsBands1s
        from .stages.gold.future_mbo.build_trigger_vectors import (
            GoldBuildMboTriggerVectors,
        )
        from .stages.gold.future_mbo.build_trigger_signals import (
            GoldBuildMboTriggerSignals,
        )
        from .stages.gold.future_mbo.build_pressure_stream import (
            GoldBuildMboPressureStream,
        )

        if layer == "bronze":
            return [BronzeIngestMboPreview()]
        elif layer == "silver":
            return [
                SilverComputeSnapshotAndWall1s(),
                SilverComputeVacuumSurface1s(),
                SilverComputeRadarVacuum1s(),
                SilverComputePhysicsBands1s(),
            ]
        elif layer == "gold":
            return [
                GoldBuildMboTriggerVectors(),
                GoldBuildMboTriggerSignals(),
                GoldBuildMboPressureStream(),
            ]
        elif layer == "all":
            return [
                BronzeIngestMboPreview(),
                SilverComputeSnapshotAndWall1s(),
                SilverComputeVacuumSurface1s(),
                SilverComputeRadarVacuum1s(),
                SilverComputePhysicsBands1s(),
                GoldBuildMboTriggerVectors(),
                GoldBuildMboTriggerSignals(),
                GoldBuildMboPressureStream(),
            ]

    elif product_type == "hud":
        from .stages.gold.hud.build_physics_norm_calibration import GoldBuildHudPhysicsNormCalibration

        if layer == "gold":
            return [GoldBuildHudPhysicsNormCalibration()]
        elif layer == "all":
            return [GoldBuildHudPhysicsNormCalibration()]

    raise ValueError(
        f"Unknown product_type: {product_type}. "
        f"Must be one of: future_mbo, future_option_mbo, hud"
    )
