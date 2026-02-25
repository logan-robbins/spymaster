"""REST + SSE routes for experiment job orchestration.

Endpoints:
    POST   /v1/jobs/experiments           Submit a new experiment job
    GET    /v1/jobs/experiments            List jobs (filterable)
    GET    /v1/jobs/experiments/specs      List available experiment spec files
    GET    /v1/jobs/experiments/{job_id}   Job status and progress
    POST   /v1/jobs/experiments/{job_id}/cancel   Cancel a running job
    GET    /v1/jobs/experiments/{job_id}/events   SSE stream of job events
    GET    /v1/jobs/experiments/{job_id}/artifacts  List job artifacts
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

logger: logging.Logger = logging.getLogger(__name__)

# Terminal statuses -- SSE streams close when a job reaches one of these.
_TERMINAL_STATUSES: frozenset[str] = frozenset({
    "completed", "failed", "canceled",
})


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SubmitJobRequest(BaseModel):
    """Payload for submitting a new experiment job."""

    spec_ref: str = Field(
        ..., description="Filename of the experiment spec YAML.",
    )
    workspace_id: str = Field(
        ..., description="Workspace UUID as a string.",
    )


class SubmitJobResponse(BaseModel):
    """Response after successfully submitting a job."""

    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """Full status of an experiment job."""

    job_id: str
    workspace_id: str
    spec_ref: str
    status: str
    progress: dict[str, Any] | None = None
    error_message: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    created_at: str


class JobEventResponse(BaseModel):
    """A single job event."""

    sequence: int
    event_type: str
    payload: dict[str, Any] | None = None
    created_at: str


class ArtifactResponse(BaseModel):
    """A single job artifact."""

    artifact_id: str
    artifact_type: str
    uri: str
    checksum: str
    metadata: dict[str, Any] | None = None
    created_at: str


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register_job_routes(app: FastAPI, *, lake_root: Path) -> None:
    """Register job orchestration API routes on the FastAPI app.

    Args:
        app: FastAPI application instance.
        lake_root: Absolute path to the data lake root.
    """

    @app.post("/v1/jobs/experiments", response_model=SubmitJobResponse)
    async def submit_experiment_job(body: SubmitJobRequest) -> SubmitJobResponse:
        """Submit a new experiment job.

        Creates a database row in ``pending`` status and enqueues the
        job for background execution.
        """
        from .db.engine import get_db_session
        from .db.repositories import ExperimentJobRepository
        from ..jobs.queue import get_job_queue

        workspace_id: UUID
        try:
            workspace_id = UUID(body.workspace_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid workspace_id: {exc}",
            ) from exc

        async with get_db_session() as session:
            job = await ExperimentJobRepository.create(
                session,
                workspace_id=workspace_id,
                spec_ref=body.spec_ref,
                created_by="api",
            )
            job_id: UUID = job.job_id

        queue = await get_job_queue()
        await queue.enqueue(
            str(job_id),
            {"spec_ref": body.spec_ref, "workspace_id": str(workspace_id)},
        )

        return SubmitJobResponse(job_id=str(job_id), status="pending")

    @app.get("/v1/jobs/experiments/specs")
    async def list_experiment_specs() -> dict[str, list[str]]:
        """List available experiment spec YAML files."""
        from ..qmachina.experiment_config import ExperimentSpec

        specs_dir: Path = ExperimentSpec.configs_dir(lake_root)
        if not specs_dir.is_dir():
            return {"specs": []}

        specs: list[str] = sorted(
            f.name for f in specs_dir.iterdir()
            if f.suffix in (".yaml", ".yml") and f.is_file()
        )
        return {"specs": specs}

    @app.get("/v1/jobs/experiments/{job_id}", response_model=JobStatusResponse)
    async def get_experiment_job(job_id: str) -> JobStatusResponse:
        """Retrieve the status of a single experiment job."""
        from .db.engine import get_db_session
        from .db.repositories import ExperimentJobRepository

        uid: UUID
        try:
            uid = UUID(job_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid job_id: {exc}",
            ) from exc

        async with get_db_session() as session:
            job = await ExperimentJobRepository.get(session, job_id=uid)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")

            return JobStatusResponse(
                job_id=str(job.job_id),
                workspace_id=str(job.workspace_id),
                spec_ref=job.spec_ref,
                status=job.status,
                progress=job.progress_json,
                error_message=job.error_message,
                started_at=job.started_at.isoformat() if job.started_at else None,
                completed_at=job.completed_at.isoformat() if job.completed_at else None,
                created_at=job.created_at.isoformat(),
            )

    @app.post("/v1/jobs/experiments/{job_id}/cancel")
    async def cancel_experiment_job(job_id: str) -> dict[str, str]:
        """Cancel a running experiment job."""
        from .db.engine import get_db_session
        from .db.repositories import ExperimentJobRepository
        from ..jobs.experiment_job_runner import cancellation_flags
        from ..jobs.worker import active_jobs

        uid: UUID
        try:
            uid = UUID(job_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid job_id: {exc}",
            ) from exc

        async with get_db_session() as session:
            job = await ExperimentJobRepository.get(session, job_id=uid)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            if job.status in _TERMINAL_STATUSES:
                return {"status": job.status, "detail": "Job already in terminal state"}

        # Set the cancellation flag and cancel the asyncio task if active.
        cancellation_flags[uid] = True
        task = active_jobs.get(uid)
        if task is not None and not task.done():
            task.cancel()

        async with get_db_session() as session:
            await ExperimentJobRepository.update_status(
                session, job_id=uid, status="canceled",
            )

        return {"status": "canceled", "detail": f"Job {job_id} cancellation requested"}

    @app.get("/v1/jobs/experiments/{job_id}/events")
    async def stream_job_events(job_id: str) -> EventSourceResponse:
        """SSE stream of job events.

        Polls the database every second for new events and sends them
        as SSE messages. The stream closes when the job reaches a
        terminal status.
        """
        import json

        from .db.engine import get_db_session
        from .db.repositories import ExperimentJobRepository

        uid: UUID
        try:
            uid = UUID(job_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid job_id: {exc}",
            ) from exc

        async def event_generator():
            last_seq: int = 0
            while True:
                async with get_db_session() as session:
                    events = await ExperimentJobRepository.list_events(
                        session, job_id=uid, after_sequence=last_seq,
                    )
                    for evt in events:
                        last_seq = evt.sequence
                        yield {
                            "event": evt.event_type,
                            "data": json.dumps({
                                "sequence": evt.sequence,
                                "event_type": evt.event_type,
                                "payload": evt.payload_json,
                                "created_at": evt.created_at.isoformat(),
                            }),
                        }

                    # Check if the job is in a terminal state.
                    job = await ExperimentJobRepository.get(session, job_id=uid)
                    if job is not None and job.status in _TERMINAL_STATUSES:
                        # Send a final status event then close.
                        yield {
                            "event": "done",
                            "data": json.dumps({
                                "status": job.status,
                                "message": f"Job reached terminal state: {job.status}",
                            }),
                        }
                        return

                await asyncio.sleep(1.0)

        return EventSourceResponse(event_generator())

    @app.get("/v1/jobs/experiments/{job_id}/artifacts")
    async def list_job_artifacts(job_id: str) -> dict[str, list[dict[str, Any]]]:
        """List artifacts produced by an experiment job."""
        from .db.engine import get_db_session
        from .db.repositories import JobArtifactRepository

        uid: UUID
        try:
            uid = UUID(job_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid job_id: {exc}",
            ) from exc

        async with get_db_session() as session:
            artifacts = await JobArtifactRepository.list_by_job(
                session, job_id=uid,
            )

        return {
            "artifacts": [
                {
                    "artifact_id": str(a.artifact_id),
                    "artifact_type": a.artifact_type,
                    "uri": a.uri,
                    "checksum": a.checksum,
                    "metadata": a.metadata_json,
                    "created_at": a.created_at.isoformat(),
                }
                for a in artifacts
            ],
        }

    @app.get("/v1/jobs/experiments")
    async def list_experiment_jobs(
        workspace_id: str | None = Query(None),
        status: str | None = Query(None),
    ) -> dict[str, list[dict[str, Any]]]:
        """List experiment jobs with optional filters."""
        from .db.engine import get_db_session
        from .db.repositories import ExperimentJobRepository

        ws_uuid: UUID | None = None
        if workspace_id is not None:
            try:
                ws_uuid = UUID(workspace_id)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail=f"Invalid workspace_id: {exc}",
                ) from exc

        async with get_db_session() as session:
            jobs = await ExperimentJobRepository.list_all(
                session, workspace_id=ws_uuid, status=status,
            )

        return {
            "jobs": [
                {
                    "job_id": str(j.job_id),
                    "workspace_id": str(j.workspace_id),
                    "spec_ref": j.spec_ref,
                    "status": j.status,
                    "progress": j.progress_json,
                    "error_message": j.error_message,
                    "started_at": j.started_at.isoformat() if j.started_at else None,
                    "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                    "created_at": j.created_at.isoformat(),
                }
                for j in jobs
            ],
        }
