from __future__ import annotations

from typing import List

from .stages.base import Stage


def build_pipeline(product_type: str, layer: str = "all") -> List[Stage]:
    """Return the ordered list of stages for the given product_type and layer.

    Args:
        product_type: One of 'future', 'future_option', 'equity', 'equity_option', 'shared'
        layer: One of 'bronze', 'silver', 'gold', 'all'

    Returns:
        List of Stage instances to execute in order
    """

    if product_type == "shared":
        from .stages.bronze.shared.ingest_instrument_definitions import BronzeIngestInstrumentDefinitions
        
        if layer == "bronze":
            return [BronzeIngestInstrumentDefinitions()]
        elif layer == "all":
            return [BronzeIngestInstrumentDefinitions()]

    elif product_type == "future":
        from .stages.bronze.future.process_dbn import BronzeProcessDBN
        from .stages.silver.future.add_session_levels import SilverAddSessionLevels
        from .stages.silver.future.filter_first4h import SilverFilterFirst4Hours
        from .stages.silver.future.compute_bar5s_features import SilverComputeBar5sFeatures
        from .stages.silver.future.extract_level_approach2m import SilverExtractLevelApproach2m
        from .stages.gold.future.extract_setup_vectors import GoldExtractSetupVectors

        if layer == "bronze":
            return [BronzeProcessDBN()]
        elif layer == "silver":
            return [
                SilverAddSessionLevels(),
                SilverFilterFirst4Hours(),
                SilverComputeBar5sFeatures(),
                SilverExtractLevelApproach2m(),
            ]
        elif layer == "gold":
            return [GoldExtractSetupVectors()]
        elif layer == "all":
            return [
                BronzeProcessDBN(),
                SilverAddSessionLevels(),
                SilverFilterFirst4Hours(),
                SilverComputeBar5sFeatures(),
                SilverExtractLevelApproach2m(),
                GoldExtractSetupVectors(),
            ]

    elif product_type == "future_option_mbo":
        from .stages.bronze.future_option_mbo.ingest import BronzeIngestFutureOptionMbo
        from .stages.silver.future_option_mbo.compute_gex_surface_1s import SilverComputeGexSurface1s
        from .stages.gold.future_option_mbo.build_gex_enriched_trigger_vectors import (
            GoldBuildGexEnrichedTriggerVectors,
        )

        if layer == "bronze":
            return [BronzeIngestFutureOptionMbo()]
        elif layer == "silver":
            # Replaced GEX 5s with GEX Surface 1s
            return [SilverComputeGexSurface1s()]
        elif layer == "gold":
            return [GoldBuildGexEnrichedTriggerVectors()]
        elif layer == "all":
            return [
                BronzeIngestFutureOptionMbo(),
                SilverComputeGexSurface1s(),
                GoldBuildGexEnrichedTriggerVectors(),
            ]

    elif product_type == "future_option":
        from .stages.bronze.future_option.ingest_statistics import BronzeIngestFutureOptionStatistics
        from .stages.silver.future_option.compute_statistics_clean import SilverComputeStatisticsClean
        
        if layer == "bronze":
            return [BronzeIngestFutureOptionStatistics()]
        elif layer == "silver":
            return [SilverComputeStatisticsClean()]
        elif layer == "all":
            return [BronzeIngestFutureOptionStatistics(), SilverComputeStatisticsClean()]

    elif product_type == "future_mbo":
        from .stages.bronze.future_mbo.ingest_preview import BronzeIngestMboPreview
        from .stages.silver.future_mbo.compute_level_vacuum_5s import (
            SilverComputeMboLevelVacuum5s,
        )
        from .stages.silver.future_mbo.compute_snapshot_and_wall_1s import SilverComputeSnapshotAndWall1s
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
            # Keeping legacy 5s vacuum for now, adding 1s snapshot/wall
            return [
                SilverComputeMboLevelVacuum5s(),
                SilverComputeSnapshotAndWall1s()
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
                SilverComputeMboLevelVacuum5s(),
                SilverComputeSnapshotAndWall1s(),
                GoldBuildMboTriggerVectors(),
                GoldBuildMboTriggerSignals(),
                GoldBuildMboPressureStream(),
            ]

    raise ValueError(
        f"Unknown product_type: {product_type}. "
        f"Must be one of: future, future_option, future_mbo, equity, equity_option, shared"
    )
