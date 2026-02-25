"""Async repository classes for control-plane database operations.

Each repository is stateless and operates on an ``AsyncSession`` passed
by the caller (typically via ``get_db_session()``). This keeps transaction
boundaries explicit and supports both real Postgres and test SQLite backends.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    AuditEvent,
    ExperimentJob,
    IngestionLiveSession,
    JobArtifact,
    JobEvent,
    ModelingSession,
    ModelingStepState,
    Workspace,
    WorkspaceMember,
)


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

class WorkspaceRepository:
    """CRUD operations for workspaces."""

    @staticmethod
    async def create(session: AsyncSession, *, name: str) -> Workspace:
        """Create a new workspace.

        Args:
            session: Active async session.
            name: Unique workspace name.

        Returns:
            The newly created ``Workspace``.
        """
        ws = Workspace(name=name)
        session.add(ws)
        await session.flush()
        return ws

    @staticmethod
    async def get(
        session: AsyncSession, *, workspace_id: uuid.UUID,
    ) -> Workspace | None:
        """Fetch a workspace by ID.

        Args:
            session: Active async session.
            workspace_id: UUID of the workspace.

        Returns:
            The ``Workspace`` if found, otherwise ``None``.
        """
        return await session.get(Workspace, workspace_id)

    @staticmethod
    async def list_all(session: AsyncSession) -> list[Workspace]:
        """List all workspaces ordered by creation time.

        Args:
            session: Active async session.

        Returns:
            List of all ``Workspace`` instances.
        """
        result = await session.execute(
            select(Workspace).order_by(Workspace.created_at),
        )
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Experiment Job
# ---------------------------------------------------------------------------

class ExperimentJobRepository:
    """CRUD and event-append operations for experiment jobs."""

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        workspace_id: uuid.UUID,
        spec_ref: str,
        created_by: str,
    ) -> ExperimentJob:
        """Submit a new experiment job in ``pending`` status.

        Args:
            session: Active async session.
            workspace_id: Owning workspace UUID.
            spec_ref: Reference to the experiment/serving spec.
            created_by: Actor who submitted the job.

        Returns:
            The newly created ``ExperimentJob``.
        """
        job = ExperimentJob(
            workspace_id=workspace_id,
            spec_ref=spec_ref,
            created_by=created_by,
            status="pending",
        )
        session.add(job)
        await session.flush()
        return job

    @staticmethod
    async def get(
        session: AsyncSession, *, job_id: uuid.UUID,
    ) -> ExperimentJob | None:
        """Fetch an experiment job by ID.

        Args:
            session: Active async session.
            job_id: UUID of the job.

        Returns:
            The ``ExperimentJob`` if found, otherwise ``None``.
        """
        return await session.get(ExperimentJob, job_id)

    @staticmethod
    async def list_by_workspace(
        session: AsyncSession,
        *,
        workspace_id: uuid.UUID,
        status: str | None = None,
    ) -> list[ExperimentJob]:
        """List jobs for a workspace, optionally filtered by status.

        Args:
            session: Active async session.
            workspace_id: Owning workspace UUID.
            status: If provided, only return jobs with this status.

        Returns:
            List of matching ``ExperimentJob`` instances.
        """
        stmt = select(ExperimentJob).where(
            ExperimentJob.workspace_id == workspace_id,
        )
        if status is not None:
            stmt = stmt.where(ExperimentJob.status == status)
        stmt = stmt.order_by(ExperimentJob.created_at.desc())
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def update_status(
        session: AsyncSession,
        *,
        job_id: uuid.UUID,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Transition a job to a new status.

        Automatically sets ``started_at`` when moving to ``running`` and
        ``completed_at`` when moving to a terminal state.

        Args:
            session: Active async session.
            job_id: UUID of the job.
            status: New status value.
            error_message: Error text (only for ``failed`` status).

        Raises:
            ValueError: If the job does not exist.
        """
        job: ExperimentJob | None = await session.get(ExperimentJob, job_id)
        if job is None:
            raise ValueError(f"ExperimentJob not found: {job_id}")

        now = _utcnow()
        job.status = status
        if status == "running" and job.started_at is None:
            job.started_at = now
        if status in ("completed", "failed", "canceled"):
            job.completed_at = now
        if error_message is not None:
            job.error_message = error_message
        await session.flush()

    @staticmethod
    async def append_event(
        session: AsyncSession,
        *,
        job_id: uuid.UUID,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> JobEvent:
        """Append an event to a job's event log.

        The sequence number is computed as ``max(existing) + 1``, starting
        at 1 for the first event.

        Args:
            session: Active async session.
            job_id: UUID of the owning job.
            event_type: Event type string.
            payload: Arbitrary JSON-serializable payload.

        Returns:
            The newly created ``JobEvent``.
        """
        result = await session.execute(
            select(func.coalesce(func.max(JobEvent.sequence), 0)).where(
                JobEvent.job_id == job_id,
            ),
        )
        max_seq: int = result.scalar_one()
        event = JobEvent(
            job_id=job_id,
            sequence=max_seq + 1,
            event_type=event_type,
            payload_json=payload,
        )
        session.add(event)
        await session.flush()
        return event

    @staticmethod
    async def list_events(
        session: AsyncSession,
        *,
        job_id: uuid.UUID,
        after_sequence: int = 0,
    ) -> list[JobEvent]:
        """List events for a job, optionally after a given sequence number.

        Args:
            session: Active async session.
            job_id: UUID of the owning job.
            after_sequence: Only return events with ``sequence > after_sequence``.

        Returns:
            List of ``JobEvent`` instances ordered by sequence ascending.
        """
        stmt = (
            select(JobEvent)
            .where(JobEvent.job_id == job_id)
            .where(JobEvent.sequence > after_sequence)
            .order_by(JobEvent.sequence)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def list_all(
        session: AsyncSession,
        *,
        workspace_id: uuid.UUID | None = None,
        status: str | None = None,
    ) -> list[ExperimentJob]:
        """List experiment jobs with optional filters.

        Args:
            session: Active async session.
            workspace_id: If provided, filter by workspace.
            status: If provided, filter by status.

        Returns:
            List of ``ExperimentJob`` instances, newest first.
        """
        stmt = select(ExperimentJob)
        if workspace_id is not None:
            stmt = stmt.where(ExperimentJob.workspace_id == workspace_id)
        if status is not None:
            stmt = stmt.where(ExperimentJob.status == status)
        stmt = stmt.order_by(ExperimentJob.created_at.desc())
        result = await session.execute(stmt)
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Job Artifact
# ---------------------------------------------------------------------------

class JobArtifactRepository:
    """CRUD operations for experiment job artifacts."""

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        job_id: uuid.UUID,
        artifact_type: str,
        uri: str,
        checksum: str,
        metadata: dict[str, Any] | None = None,
    ) -> JobArtifact:
        """Create a new artifact record for an experiment job.

        Args:
            session: Active async session.
            job_id: UUID of the owning job.
            artifact_type: Type classifier (e.g. ``"run_results"``).
            uri: Storage URI or file path.
            checksum: SHA-256 hex digest.
            metadata: Optional JSON-serializable metadata.

        Returns:
            The newly created ``JobArtifact``.
        """
        artifact = JobArtifact(
            job_id=job_id,
            artifact_type=artifact_type,
            uri=uri,
            checksum=checksum,
            metadata_json=metadata,
        )
        session.add(artifact)
        await session.flush()
        return artifact

    @staticmethod
    async def list_by_job(
        session: AsyncSession,
        *,
        job_id: uuid.UUID,
    ) -> list[JobArtifact]:
        """List all artifacts for a job.

        Args:
            session: Active async session.
            job_id: UUID of the owning job.

        Returns:
            List of ``JobArtifact`` instances.
        """
        result = await session.execute(
            select(JobArtifact)
            .where(JobArtifact.job_id == job_id)
            .order_by(JobArtifact.created_at),
        )
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Modeling Session
# ---------------------------------------------------------------------------

class ModelingSessionRepository:
    """CRUD and step-commit operations for modeling sessions."""

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        workspace_id: uuid.UUID,
        created_by: str,
    ) -> ModelingSession:
        """Create a new modeling session in ``draft`` status.

        Args:
            session: Active async session.
            workspace_id: Owning workspace UUID.
            created_by: Actor who initiated the session.

        Returns:
            The newly created ``ModelingSession``.
        """
        ms = ModelingSession(
            workspace_id=workspace_id,
            created_by=created_by,
            status="draft",
        )
        session.add(ms)
        await session.flush()
        return ms

    @staticmethod
    async def get(
        session: AsyncSession, *, session_id: uuid.UUID,
    ) -> ModelingSession | None:
        """Fetch a modeling session by ID.

        Args:
            session: Active async session.
            session_id: UUID of the modeling session.

        Returns:
            The ``ModelingSession`` if found, otherwise ``None``.
        """
        return await session.get(ModelingSession, session_id)

    @staticmethod
    async def update_status(
        session: AsyncSession,
        *,
        session_id: uuid.UUID,
        status: str,
    ) -> None:
        """Transition a modeling session to a new status.

        Args:
            session: Active async session.
            session_id: UUID of the session.
            status: New status value.

        Raises:
            ValueError: If the session does not exist.
        """
        ms: ModelingSession | None = await session.get(ModelingSession, session_id)
        if ms is None:
            raise ValueError(f"ModelingSession not found: {session_id}")
        ms.status = status
        await session.flush()

    @staticmethod
    async def commit_step(
        session: AsyncSession,
        *,
        session_id: uuid.UUID,
        step_name: str,
        payload: dict[str, Any],
    ) -> ModelingStepState:
        """Commit (upsert) a modeling step with its payload.

        Idempotent: re-committing the same step replaces the payload
        and updates ``committed_at``.

        Args:
            session: Active async session.
            session_id: UUID of the modeling session.
            step_name: Step identifier string.
            payload: Arbitrary JSON-serializable step data.

        Returns:
            The committed ``ModelingStepState``.
        """
        existing = await session.get(
            ModelingStepState, (session_id, step_name),
        )
        now = _utcnow()
        if existing is not None:
            existing.status = "committed"
            existing.payload_json = payload
            existing.committed_at = now
            await session.flush()
            return existing

        step = ModelingStepState(
            session_id=session_id,
            step_name=step_name,
            status="committed",
            payload_json=payload,
            committed_at=now,
        )
        session.add(step)
        await session.flush()
        return step

    @staticmethod
    async def get_steps(
        session: AsyncSession, *, session_id: uuid.UUID,
    ) -> list[ModelingStepState]:
        """Fetch all steps for a modeling session.

        Args:
            session: Active async session.
            session_id: UUID of the modeling session.

        Returns:
            List of ``ModelingStepState`` instances for the session.
        """
        result = await session.execute(
            select(ModelingStepState)
            .where(ModelingStepState.session_id == session_id)
            .order_by(ModelingStepState.step_name),
        )
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Ingestion Session
# ---------------------------------------------------------------------------

class IngestionSessionRepository:
    """CRUD and checkpoint operations for live ingestion sessions."""

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        workspace_id: uuid.UUID,
        symbol: str,
        config: dict[str, Any],
    ) -> IngestionLiveSession:
        """Create a new ingestion session in ``starting`` status.

        Args:
            session: Active async session.
            workspace_id: Owning workspace UUID.
            symbol: Market symbol being ingested.
            config: Ingestion configuration payload.

        Returns:
            The newly created ``IngestionLiveSession``.
        """
        ils = IngestionLiveSession(
            workspace_id=workspace_id,
            symbol=symbol,
            status="starting",
            config_json=config,
        )
        session.add(ils)
        await session.flush()
        return ils

    @staticmethod
    async def get(
        session: AsyncSession, *, session_id: uuid.UUID,
    ) -> IngestionLiveSession | None:
        """Fetch an ingestion session by ID.

        Args:
            session: Active async session.
            session_id: UUID of the ingestion session.

        Returns:
            The ``IngestionLiveSession`` if found, otherwise ``None``.
        """
        return await session.get(IngestionLiveSession, session_id)

    @staticmethod
    async def update_status(
        session: AsyncSession,
        *,
        session_id: uuid.UUID,
        status: str,
    ) -> None:
        """Transition an ingestion session to a new status.

        Automatically sets ``stopped_at`` for terminal states.

        Args:
            session: Active async session.
            session_id: UUID of the session.
            status: New status value.

        Raises:
            ValueError: If the session does not exist.
        """
        ils: IngestionLiveSession | None = await session.get(
            IngestionLiveSession, session_id,
        )
        if ils is None:
            raise ValueError(f"IngestionLiveSession not found: {session_id}")
        ils.status = status
        if status in ("stopped", "failed"):
            ils.stopped_at = _utcnow()
        await session.flush()

    @staticmethod
    async def update_checkpoint(
        session: AsyncSession,
        *,
        session_id: uuid.UUID,
        checkpoint: dict[str, Any],
    ) -> None:
        """Update the checkpoint for an active ingestion session.

        Args:
            session: Active async session.
            session_id: UUID of the session.
            checkpoint: Checkpoint data (sequence_id, ts_event_ns, etc.).

        Raises:
            ValueError: If the session does not exist.
        """
        ils: IngestionLiveSession | None = await session.get(
            IngestionLiveSession, session_id,
        )
        if ils is None:
            raise ValueError(f"IngestionLiveSession not found: {session_id}")
        ils.checkpoint = checkpoint
        await session.flush()


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class AuditRepository:
    """Write and query operations for the immutable audit log."""

    @staticmethod
    async def write(
        session: AsyncSession,
        *,
        workspace_id: uuid.UUID | None,
        actor_id: str,
        action: str,
        target: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Append an audit event.

        Args:
            session: Active async session.
            workspace_id: Owning workspace UUID, or ``None`` for system events.
            actor_id: Identity of the actor performing the action.
            action: Action descriptor (e.g. ``"workspace.create"``).
            target: Optional resource identifier.
            payload: Optional JSON-serializable details.

        Returns:
            The newly created ``AuditEvent``.
        """
        event = AuditEvent(
            workspace_id=workspace_id,
            actor_id=actor_id,
            action=action,
            target=target,
            payload_json=payload,
        )
        session.add(event)
        await session.flush()
        return event

    @staticmethod
    async def list_by_workspace(
        session: AsyncSession,
        *,
        workspace_id: uuid.UUID,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """List recent audit events for a workspace.

        Args:
            session: Active async session.
            workspace_id: UUID of the workspace.
            limit: Maximum number of events to return.

        Returns:
            List of ``AuditEvent`` instances, newest first.
        """
        result = await session.execute(
            select(AuditEvent)
            .where(AuditEvent.workspace_id == workspace_id)
            .order_by(AuditEvent.created_at.desc())
            .limit(limit),
        )
        return list(result.scalars().all())
