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
        level: str,
        log_level: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        resume_from_stage: Optional[int] = None,
        stop_at_stage: Optional[int] = None,
        *,
        canonical_version: Optional[str] = None,
        data_root: Optional[str] = None,
        write_outputs: bool = False,
        overwrite_partitions: bool = True
    ) -> pd.DataFrame:
        """Execute all stages in sequence.

        Args:
            date: YYYY-MM-DD date string
            level: Level type (PM_HIGH, PM_LOW, OR_HIGH, OR_LOW, SMA_90)
            log_level: Logging level (default: INFO)
            checkpoint_dir: Directory for checkpoints
            resume_from_stage: Resume from stage N (0-based)
            stop_at_stage: Stop after stage N (0-based)

        Returns:
            Final signals DataFrame for this level
        """
        if log_level is None:
            log_level = logging.INFO

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(message)s',
            force=True
        )

        # Build per-run config snapshot (used by stages and checkpoint hash).
        run_config = vars(CONFIG).copy()
        if data_root:
            run_config["DATA_ROOT"] = data_root
        run_config["PIPELINE_CANONICAL_VERSION"] = canonical_version or self.version
        run_config["PIPELINE_WRITE_SIGNALS"] = write_outputs
        run_config["PIPELINE_WRITE_STATE_TABLE"] = write_outputs
        run_config["PIPELINE_WRITE_EPISODES"] = write_outputs
        run_config["PIPELINE_OVERWRITE_PARTITIONS"] = overwrite_partitions

        total_start = time.time()
        logger.info(f"[{self.version}] Starting pipeline: {self.name}")
        logger.info(f"[{self.version}] Date: {date}")
        logger.info(f"[{self.version}] Level: {level}")
        logger.info(f"[{self.version}] Stages: {len(self.stages)}")
        logger.info(f"[{self.version}] Canonical version: {run_config['PIPELINE_CANONICAL_VERSION']}")
        if run_config.get("DATA_ROOT"):
            logger.info(f"[{self.version}] DATA_ROOT: {run_config['DATA_ROOT']}")
        
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
                raise ValueError(f"Invalid resume_from_stage(stage_idx)={resume_from_stage}, must be 0-{len(self.stages)-1}")
            
            # Load checkpoint from previous stage
            if resume_from_stage > 0:
                logger.info(f"[{self.version}] Resuming from stage_idx {resume_from_stage} (step {resume_from_stage + 1}/{len(self.stages)})")
                ctx = checkpoint_manager.load_checkpoint(
                    pipeline_name=self.name,
                    date=date,
                    stage_idx=resume_from_stage - 1,
                    level=level
                )
                
                if ctx is None:
                    raise ValueError(f"Checkpoint not found for stage {resume_from_stage-1}")
                
                # Update config
                ctx.config = run_config.copy()
                start_stage_idx = resume_from_stage
            else:
                logger.info(f"[{self.version}] Starting from beginning (resume_from_stage(stage_idx)=0)")
        
        # Initialize context if not resumed
        if ctx is None:
            ctx = StageContext(
                date=date,
                level=level,
                data={'date': date},
                config=run_config.copy()
            )
        else:
            ctx.data.setdefault('date', date)
            ctx.level = level

        # Determine end stage
        end_stage_idx = len(self.stages)
        if stop_at_stage is not None:
            if stop_at_stage < 0 or stop_at_stage >= len(self.stages):
                raise ValueError(f"Invalid stop_at_stage(stage_idx)={stop_at_stage}, must be 0-{len(self.stages)-1}")
            end_stage_idx = stop_at_stage + 1
            logger.info(f"[{self.version}] Will stop after stage_idx {stop_at_stage} (step {stop_at_stage + 1}/{len(self.stages)})")

        # Execute stages
        for idx in range(start_stage_idx, end_stage_idx):
            stage = self.stages[idx]
            stage_start = time.time()
            logger.info(f"[{self.version}] (stage_idx={idx}, step {idx+1}/{len(self.stages)}) Running: {stage.name}")

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
