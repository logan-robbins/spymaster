"""Pipeline orchestrator for executing stages sequentially."""
import logging
import time
from typing import List, Optional
import pandas as pd

from src.common.config import CONFIG
from src.pipeline.core.stage import BaseStage, StageContext

logger = logging.getLogger(__name__)


class Pipeline:
    """Pipeline orchestrator. Executes stages sequentially.

    Each pipeline is defined by a sequence of stages that transform
    Bronze data into Silver features. Different versions can use
    different stage compositions.

    Example:
        pipeline = Pipeline([
            LoadBronzeStage(),
            BuildOHLCVStage(freq='1min'),
            ComputeBarrierFeaturesStage(),
            LabelOutcomesStage(),
        ], name="mechanics_only", version="v1.0")

        signals_df = pipeline.run("2025-12-16")
    """

    def __init__(
        self,
        stages: List[BaseStage],
        name: str,
        version: str
    ):
        """Initialize pipeline.

        Args:
            stages: Ordered list of stages to execute
            name: Pipeline name (e.g., "mechanics_only")
            version: Version string (e.g., "v1.0")
        """
        self.stages = stages
        self.name = name
        self.version = version

    def run(
        self,
        date: str,
        log_level: Optional[int] = None
    ) -> pd.DataFrame:
        """Execute all stages in sequence.

        Args:
            date: YYYY-MM-DD date string
            log_level: Optional logging level (default: INFO)

        Returns:
            Final signals DataFrame
        """
        if log_level is None:
            log_level = logging.INFO

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(message)s',
            force=True
        )

        total_start = time.time()
        logger.info(f"[{self.version}] Starting pipeline: {self.name}")
        logger.info(f"[{self.version}] Date: {date}")
        logger.info(f"[{self.version}] Stages: {len(self.stages)}")

        # Initialize context with CONFIG
        ctx = StageContext(
            date=date,
            data={},
            config=vars(CONFIG).copy()
        )

        # Execute stages
        for i, stage in enumerate(self.stages, 1):
            stage_start = time.time()
            logger.info(f"[{self.version}] ({i}/{len(self.stages)}) Running: {stage.name}")

            try:
                ctx = stage.run(ctx)
            except Exception as e:
                logger.error(f"[{self.version}] Stage '{stage.name}' failed: {e}")
                raise

            elapsed = time.time() - stage_start
            logger.info(f"[{self.version}]   Completed in {elapsed:.2f}s")

        # Extract final signals
        if 'signals' not in ctx.data:
            logger.warning(f"[{self.version}] No 'signals' in context, returning empty DataFrame")
            return pd.DataFrame()

        signals_df = ctx.data['signals']

        total_time = time.time() - total_start
        logger.info(f"[{self.version}] {'='*50}")
        logger.info(f"[{self.version}] PIPELINE COMPLETE in {total_time:.2f}s")
        logger.info(f"[{self.version}] Total signals: {len(signals_df):,}")
        logger.info(f"[{self.version}] Throughput: {len(signals_df)/total_time:.0f} signals/sec")
        logger.info(f"[{self.version}] {'='*50}")

        return signals_df
