"""Async wrapper around the synchronous ExperimentRunner.

Manages the full lifecycle of an experiment job: status transitions,
event logging, cancellation checks, and artifact persistence.
The heavy compute (``ExperimentRunner.run()``) executes in a thread
pool via ``asyncio.to_thread`` so it never blocks the event loop.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Any
from uuid import UUID

logger: logging.Logger = logging.getLogger(__name__)

# Module-level cancellation registry.  The worker sets a flag here;
# the runner checks it between major steps.
cancellation_flags: dict[UUID, bool] = {}


async def run_experiment_job(
    job_id: UUID,
    spec_ref: str,
    workspace_id: UUID,
    lake_root: Path,
) -> None:
    """Execute an experiment job end-to-end with DB status tracking.

    Args:
        job_id: Primary key of the ``ExperimentJob`` row.
        spec_ref: Filename (not full path) of the experiment spec YAML
            inside ``lake/research/harness/configs/experiments/``.
        workspace_id: Owning workspace UUID (for audit context).
        lake_root: Absolute path to the data lake root.

    Raises:
        asyncio.CancelledError: Propagated if the task is cancelled.
        Exception: Any unhandled error from the runner (after recording
            the failure in the database).
    """
    from ..qmachina.db.engine import get_db_session
    from ..qmachina.db.repositories import ExperimentJobRepository

    # ---- mark running ----
    async with get_db_session() as session:
        await ExperimentJobRepository.update_status(
            session, job_id=job_id, status="running",
        )
        await ExperimentJobRepository.append_event(
            session,
            job_id=job_id,
            event_type="start",
            payload={"spec_ref": spec_ref},
        )

    try:
        _check_cancelled(job_id)

        # ---- load spec ----
        from ..qmachina.experiment_config import ExperimentSpec

        spec_path: Path = (
            lake_root / "research" / "harness" / "configs" / "experiments" / spec_ref
        )
        spec: ExperimentSpec = ExperimentSpec.from_yaml(spec_path)

        _check_cancelled(job_id)

        # ---- build runner config ----
        harness_dict: dict[str, Any] = spec.to_runner_config(lake_root)
        from ..experiment_harness.config_schema import ExperimentConfig

        config: ExperimentConfig = ExperimentConfig.model_validate(harness_dict)

        _check_cancelled(job_id)

        # ---- execute in thread pool ----
        from ..experiment_harness.runner import ExperimentRunner

        runner = ExperimentRunner(
            lake_root=lake_root,
            feature_store_config=(
                spec.feature_store if spec.feature_store.enabled else None
            ),
        )

        # Start a background progress reporter.
        progress_task: asyncio.Task[None] = asyncio.create_task(
            _progress_reporter(job_id, interval_seconds=5.0),
        )

        try:
            run_ids: list[str] = await asyncio.to_thread(runner.run, config)
        finally:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

        _check_cancelled(job_id)

        # ---- persist artifacts ----
        await _persist_artifacts(job_id, run_ids, lake_root)

        # ---- mark completed ----
        async with get_db_session() as session:
            await ExperimentJobRepository.update_status(
                session, job_id=job_id, status="completed",
            )
            await ExperimentJobRepository.append_event(
                session,
                job_id=job_id,
                event_type="complete",
                payload={
                    "run_ids": run_ids,
                    "n_runs": len(run_ids),
                },
            )

        logger.info(
            "Job %s completed with %d runs", job_id, len(run_ids),
        )

    except _CancelledByFlag:
        async with get_db_session() as session:
            await ExperimentJobRepository.update_status(
                session, job_id=job_id, status="canceled",
            )
            await ExperimentJobRepository.append_event(
                session,
                job_id=job_id,
                event_type="cancel",
                payload={"reason": "user_requested"},
            )
        logger.info("Job %s canceled by user", job_id)

    except asyncio.CancelledError:
        async with get_db_session() as session:
            await ExperimentJobRepository.update_status(
                session, job_id=job_id, status="canceled",
            )
            await ExperimentJobRepository.append_event(
                session,
                job_id=job_id,
                event_type="cancel",
                payload={"reason": "task_cancelled"},
            )
        logger.info("Job %s task was cancelled", job_id)
        raise

    except Exception as exc:
        error_message: str = f"{type(exc).__name__}: {exc}"
        async with get_db_session() as session:
            await ExperimentJobRepository.update_status(
                session,
                job_id=job_id,
                status="failed",
                error_message=error_message,
            )
            await ExperimentJobRepository.append_event(
                session,
                job_id=job_id,
                event_type="error",
                payload={"error": error_message},
            )
        logger.exception("Job %s failed: %s", job_id, exc)
        raise

    finally:
        cancellation_flags.pop(job_id, None)


# ---------------------------------------------------------------------------
# Cancellation helpers
# ---------------------------------------------------------------------------

class _CancelledByFlag(Exception):
    """Internal sentinel raised when the cancellation flag is set."""


def _check_cancelled(job_id: UUID) -> None:
    """Raise ``_CancelledByFlag`` if the job has been flagged for cancellation.

    Args:
        job_id: Job to check.
    """
    if cancellation_flags.get(job_id, False):
        raise _CancelledByFlag(f"Job {job_id} cancelled via flag")


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------

async def _progress_reporter(
    job_id: UUID,
    interval_seconds: float = 5.0,
) -> None:
    """Emit periodic progress events while a job is running.

    Runs as a background ``asyncio.Task`` and is cancelled by the
    caller when the job finishes.

    Args:
        job_id: Job to emit progress for.
        interval_seconds: Seconds between progress events.
    """
    from ..qmachina.db.engine import get_db_session
    from ..qmachina.db.repositories import ExperimentJobRepository

    tick: int = 0
    while True:
        await asyncio.sleep(interval_seconds)
        tick += 1
        try:
            async with get_db_session() as session:
                await ExperimentJobRepository.append_event(
                    session,
                    job_id=job_id,
                    event_type="progress",
                    payload={"tick": tick, "message": f"Running ({tick * interval_seconds:.0f}s elapsed)"},
                )
        except Exception:
            logger.debug("Progress event write failed for job %s", job_id, exc_info=True)


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: Absolute path to the file.

    Returns:
        64-character hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


async def _persist_artifacts(
    job_id: UUID,
    run_ids: list[str],
    lake_root: Path,
) -> None:
    """Scan result files and write artifact metadata to the database.

    Looks for ``runs.parquet`` and ``runs_meta.parquet`` in the
    results directory and records each as a ``JobArtifact``.

    Args:
        job_id: Job that produced the artifacts.
        run_ids: List of run IDs from the experiment.
        lake_root: Data lake root path.
    """
    from ..qmachina.db.engine import get_db_session
    from ..qmachina.db.repositories import JobArtifactRepository

    results_dir: Path = lake_root / "research" / "harness" / "results"
    artifact_files: list[tuple[str, Path]] = []

    for name in ("runs.parquet", "runs_meta.parquet"):
        path = results_dir / name
        if path.exists():
            artifact_files.append((name, path))

    if not artifact_files:
        logger.debug("No artifact files found for job %s", job_id)
        return

    async with get_db_session() as session:
        for name, path in artifact_files:
            checksum: str = await asyncio.to_thread(_sha256_file, path)
            await JobArtifactRepository.create(
                session,
                job_id=job_id,
                artifact_type="run_results",
                uri=str(path),
                checksum=checksum,
                metadata={"filename": name, "run_ids": run_ids},
            )

    logger.info(
        "Persisted %d artifacts for job %s", len(artifact_files), job_id,
    )
