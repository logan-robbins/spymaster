"""Background worker loop that polls the job queue and dispatches jobs.

Designed to run as a long-lived ``asyncio.Task`` started from the
FastAPI lifespan. Gracefully shuts down when ``cancel_event`` is set.

Usage::

    cancel = asyncio.Event()
    queue = await get_job_queue()
    task = asyncio.create_task(job_worker_loop(queue, cancel))
    # ... on shutdown ...
    cancel.set()
    await task
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from .experiment_job_runner import cancellation_flags, run_experiment_job
from .queue import JobQueue

logger: logging.Logger = logging.getLogger(__name__)

# Active tasks keyed by job_id -- used for external cancellation.
active_jobs: dict[UUID, asyncio.Task[None]] = {}

# Default poll interval when the queue is empty (seconds).
_POLL_INTERVAL: float = 1.0


async def job_worker_loop(
    queue: JobQueue,
    cancel_event: asyncio.Event,
    lake_root: Path | None = None,
) -> None:
    """Poll the job queue and dispatch experiment jobs.

    Runs indefinitely until ``cancel_event`` is set. Each dequeued job
    is dispatched as a new ``asyncio.Task`` so multiple jobs can run
    concurrently (bounded by system resources).

    Args:
        queue: Job queue to poll (Redis-backed or in-process).
        cancel_event: Set this event to signal the worker to stop.
        lake_root: Data lake root. Defaults to ``backend/lake``
            relative to this file.
    """
    if lake_root is None:
        lake_root = Path(__file__).resolve().parents[2] / "lake"

    logger.info("Job worker loop started (lake_root=%s)", lake_root)

    while not cancel_event.is_set():
        try:
            item = await queue.dequeue()
        except Exception:
            logger.exception("Error dequeueing from job queue")
            await asyncio.sleep(_POLL_INTERVAL)
            continue

        if item is None:
            # Nothing in the queue -- wait before polling again.
            try:
                await asyncio.wait_for(
                    cancel_event.wait(), timeout=_POLL_INTERVAL,
                )
                break  # cancel_event was set
            except asyncio.TimeoutError:
                continue

        raw_job_id, payload = item
        try:
            job_id = UUID(raw_job_id)
        except ValueError:
            logger.error("Invalid job_id from queue: %s", raw_job_id)
            continue

        spec_ref: str = payload.get("spec_ref", "")
        workspace_id_str: str = payload.get("workspace_id", "")

        try:
            workspace_id = UUID(workspace_id_str)
        except ValueError:
            logger.error(
                "Invalid workspace_id for job %s: %s",
                job_id, workspace_id_str,
            )
            continue

        logger.info(
            "Dispatching job %s (spec_ref=%s)", job_id, spec_ref,
        )

        task: asyncio.Task[None] = asyncio.create_task(
            run_experiment_job(
                job_id=job_id,
                spec_ref=spec_ref,
                workspace_id=workspace_id,
                lake_root=lake_root,
            ),
            name=f"job-{job_id}",
        )
        active_jobs[job_id] = task
        task.add_done_callback(lambda t, jid=job_id: _on_task_done(jid, t))

    # --- shutdown: cancel all active tasks ---
    if active_jobs:
        logger.info("Cancelling %d active jobs on shutdown", len(active_jobs))
        for jid, task in list(active_jobs.items()):
            task.cancel()
        await asyncio.gather(*active_jobs.values(), return_exceptions=True)
        active_jobs.clear()

    logger.info("Job worker loop stopped")


def _on_task_done(job_id: UUID, task: asyncio.Task[None]) -> None:
    """Callback invoked when a job task finishes.

    Removes the task from the active registry and logs any unexpected
    errors.

    Args:
        job_id: Job identifier.
        task: The completed asyncio task.
    """
    active_jobs.pop(job_id, None)
    cancellation_flags.pop(job_id, None)

    if task.cancelled():
        logger.info("Job task %s was cancelled", job_id)
        return

    exc = task.exception()
    if exc is not None:
        logger.error("Job task %s raised: %s", job_id, exc)
