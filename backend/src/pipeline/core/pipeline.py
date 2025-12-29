"""Pipeline orchestrator for executing stages sequentially."""
import logging
import time
from typing import List, Optional
import pandas as pd

from src.common.config import CONFIG
from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.core.checkpoint import CheckpointManager

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
        log_level: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        resume_from_stage: Optional[int] = None,
        stop_at_stage: Optional[int] = None
    ) -> pd.DataFrame:
        """Execute all stages in sequence with optional checkpointing.

        Args:
            date: YYYY-MM-DD date string
            log_level: Optional logging level (default: INFO)
            checkpoint_dir: Directory for checkpoints (enables checkpointing if provided)
            resume_from_stage: Resume from stage N (0-based, loads stage N-1 checkpoint)
            stop_at_stage: Stop after stage N (0-based, for debugging)

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
        
        # Initialize checkpoint manager if requested
        checkpoint_manager = None
        if checkpoint_dir:
            checkpoint_manager = CheckpointManager(checkpoint_dir)
            logger.info(f"[{self.version}] Checkpointing enabled: {checkpoint_dir}")

        # Determine start stage
        start_stage_idx = 0
        ctx = None
        
        if resume_from_stage is not None:
            if not checkpoint_manager:
                raise ValueError("Cannot resume without checkpoint_dir")
            
            if resume_from_stage < 0 or resume_from_stage >= len(self.stages):
                raise ValueError(f"Invalid resume_from_stage={resume_from_stage}, must be 0-{len(self.stages)-1}")
            
            # Load checkpoint from previous stage
            if resume_from_stage > 0:
                logger.info(f"[{self.version}] Resuming from stage {resume_from_stage}")
                ctx = checkpoint_manager.load_checkpoint(
                    pipeline_name=self.name,
                    date=date,
                    stage_idx=resume_from_stage - 1
                )
                
                if ctx is None:
                    raise ValueError(f"Checkpoint not found for stage {resume_from_stage-1}")
                
                # Update config
                ctx.config = vars(CONFIG).copy()
                start_stage_idx = resume_from_stage
            else:
                logger.info(f"[{self.version}] Starting from beginning (resume_from_stage=0)")
        
        # Initialize context if not resumed
        if ctx is None:
            ctx = StageContext(
                date=date,
                data={},
                config=vars(CONFIG).copy()
            )

        # Determine end stage
        end_stage_idx = len(self.stages)
        if stop_at_stage is not None:
            if stop_at_stage < 0 or stop_at_stage >= len(self.stages):
                raise ValueError(f"Invalid stop_at_stage={stop_at_stage}, must be 0-{len(self.stages)-1}")
            end_stage_idx = stop_at_stage + 1
            logger.info(f"[{self.version}] Will stop after stage {stop_at_stage}")

        # Execute stages
        for idx in range(start_stage_idx, end_stage_idx):
            stage = self.stages[idx]
            stage_start = time.time()
            logger.info(f"[{self.version}] ({idx+1}/{len(self.stages)}) Running: {stage.name}")

            try:
                ctx = stage.run(ctx)
            except Exception as e:
                logger.error(f"[{self.version}] Stage '{stage.name}' failed: {e}")
                raise

            elapsed = time.time() - stage_start
            logger.info(f"[{self.version}]   Completed in {elapsed:.2f}s")
            
            # Save checkpoint if enabled
            if checkpoint_manager:
                checkpoint_manager.save_checkpoint(
                    pipeline_name=self.name,
                    date=date,
                    stage_idx=idx,
                    stage_name=stage.name,
                    ctx=ctx,
                    elapsed_time=elapsed
                )

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
