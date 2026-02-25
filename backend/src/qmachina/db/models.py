"""SQLAlchemy 2.x ORM models for the qMachina control-plane database.

Nine tables covering workspace management, experiment jobs, modeling
sessions, ingestion sessions, serving activation, and audit logging.
All primary keys use UUID4; all timestamps are timezone-aware UTC.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSON as PG_JSON
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def _new_uuid() -> uuid.UUID:
    """Generate a new UUID4."""
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# 1. workspace
# ---------------------------------------------------------------------------

class Workspace(Base):
    """A workspace is the top-level organizational unit.

    Each workspace owns experiment jobs, modeling sessions, ingestion
    sessions, serving activations, and audit events.
    """

    __tablename__ = "workspace"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=_new_uuid,
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow,
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=_utcnow, nullable=True,
    )

    # relationships
    members: Mapped[list[WorkspaceMember]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan",
    )
    experiment_jobs: Mapped[list[ExperimentJob]] = relationship(
        back_populates="workspace",
    )
    modeling_sessions: Mapped[list[ModelingSession]] = relationship(
        back_populates="workspace",
    )
    ingestion_sessions: Mapped[list[IngestionLiveSession]] = relationship(
        back_populates="workspace",
    )
    serving_activations: Mapped[list[ServingActivation]] = relationship(
        back_populates="workspace",
    )


# ---------------------------------------------------------------------------
# 2. workspace_member
# ---------------------------------------------------------------------------

class WorkspaceMember(Base):
    """Maps a user to a workspace with a role-based access level."""

    __tablename__ = "workspace_member"

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("workspace.workspace_id", ondelete="CASCADE"),
        primary_key=True,
    )
    role: Mapped[str] = mapped_column(
        String(32), nullable=False,
    )  # "owner" | "admin" | "editor" | "viewer"
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow,
    )

    workspace: Mapped[Workspace] = relationship(back_populates="members")


# ---------------------------------------------------------------------------
# 3. experiment_job
# ---------------------------------------------------------------------------

class ExperimentJob(Base):
    """Tracks the lifecycle of a submitted experiment or generation job."""

    __tablename__ = "experiment_job"

    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=_new_uuid,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.workspace_id"), nullable=False,
    )
    spec_ref: Mapped[str] = mapped_column(
        String(512), nullable=False,
    )  # serving spec or experiment spec reference
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending",
    )  # "pending" | "running" | "completed" | "failed" | "canceled"
    progress_json: Mapped[dict | None] = mapped_column(
        JSON, nullable=True,
    )  # {current_step, total_steps, pct, message}
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow,
    )
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)

    workspace: Mapped[Workspace] = relationship(back_populates="experiment_jobs")
    events: Mapped[list[JobEvent]] = relationship(
        back_populates="job", cascade="all, delete-orphan",
    )
    artifacts: Mapped[list[JobArtifact]] = relationship(
        back_populates="job", cascade="all, delete-orphan",
    )


# ---------------------------------------------------------------------------
# 4. job_event
# ---------------------------------------------------------------------------

class JobEvent(Base):
    """Append-only event log for an experiment job.

    Composite primary key ``(job_id, sequence)`` ensures ordering within
    a single job. ``sequence`` is auto-assigned by the repository layer.
    """

    __tablename__ = "job_event"

    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("experiment_job.job_id", ondelete="CASCADE"),
        primary_key=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(
        String(32), nullable=False,
    )  # "progress" | "log" | "error" | "complete" | "cancel"
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow,
    )

    job: Mapped[ExperimentJob] = relationship(back_populates="events")


# ---------------------------------------------------------------------------
# 4b. job_artifact
# ---------------------------------------------------------------------------

class JobArtifact(Base):
    """Artifact metadata produced by an experiment job.

    Records the type, storage URI, and integrity checksum for each
    output artifact (e.g. results parquet files).
    """

    __tablename__ = "job_artifact"

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=_new_uuid,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("experiment_job.job_id", ondelete="CASCADE"),
        nullable=False,
    )
    artifact_type: Mapped[str] = mapped_column(
        String(64), nullable=False,
    )  # "run_results" | "model_checkpoint" | ...
    uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    checksum: Mapped[str] = mapped_column(
        String(128), nullable=False,
    )  # SHA-256 hex digest
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow,
    )

    job: Mapped[ExperimentJob] = relationship(back_populates="artifacts")


# ---------------------------------------------------------------------------
# 5. modeling_session
# ---------------------------------------------------------------------------

class ModelingSession(Base):
    """Tracks a multi-step modeling workflow (wizard/pipeline).

    Steps are recorded in ``ModelingStepState``.
    """

    __tablename__ = "modeling_session"

    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=_new_uuid,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.workspace_id"), nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="draft",
    )  # "draft" | "in_progress" | "completed" | "promoted" | "abandoned"
    selected_silver_id: Mapped[str | None] = mapped_column(
        String(512), nullable=True,
    )
    model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow,
    )
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=_utcnow, nullable=True,
    )

    workspace: Mapped[Workspace] = relationship(back_populates="modeling_sessions")
    steps: Mapped[list[ModelingStepState]] = relationship(
        back_populates="session", cascade="all, delete-orphan",
    )


# ---------------------------------------------------------------------------
# 6. modeling_step_state
# ---------------------------------------------------------------------------

class ModelingStepState(Base):
    """Records the status and payload of a single modeling step."""

    __tablename__ = "modeling_step_state"

    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("modeling_session.session_id", ondelete="CASCADE"),
        primary_key=True,
    )
    step_name: Mapped[str] = mapped_column(
        String(64), primary_key=True,
    )  # "dataset_select" | "gold_config" | "signal_select" | "eval_params" | "promote_review" | "promotion"
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending",
    )  # "pending" | "committed"
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    committed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    session: Mapped[ModelingSession] = relationship(back_populates="steps")


# ---------------------------------------------------------------------------
# 7. ingestion_live_session
# ---------------------------------------------------------------------------

class IngestionLiveSession(Base):
    """Tracks a live market data ingestion session."""

    __tablename__ = "ingestion_live_session"

    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=_new_uuid,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.workspace_id"), nullable=False,
    )
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="starting",
    )  # "starting" | "active" | "stopping" | "stopped" | "failed"
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow,
    )
    stopped_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    checkpoint: Mapped[dict | None] = mapped_column(
        JSON, nullable=True,
    )  # {sequence_id, ts_event_ns, updated_at}
    config_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    workspace: Mapped[Workspace] = relationship(back_populates="ingestion_sessions")


# ---------------------------------------------------------------------------
# 8. serving_activation
# ---------------------------------------------------------------------------

class ServingActivation(Base):
    """Records the activation of a serving version under an alias."""

    __tablename__ = "serving_activation"

    activation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=_new_uuid,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.workspace_id"), nullable=False,
    )
    alias: Mapped[str] = mapped_column(String(128), nullable=False)
    serving_id: Mapped[str] = mapped_column(String(512), nullable=False)
    activated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow,
    )
    activated_by: Mapped[str] = mapped_column(String(255), nullable=False)
    deactivated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    workspace: Mapped[Workspace] = relationship(back_populates="serving_activations")


# ---------------------------------------------------------------------------
# 9. audit_event
# ---------------------------------------------------------------------------

class AuditEvent(Base):
    """Immutable audit log entry for control-plane actions.

    ``workspace_id`` is nullable because some events are system-level.
    """

    __tablename__ = "audit_event"
    __table_args__ = (
        Index("ix_audit_event_created_at", "created_at"),
        Index("ix_audit_event_workspace_id", "workspace_id"),
    )

    event_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=_new_uuid,
    )
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("workspace.workspace_id"), nullable=True,
    )
    actor_id: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(
        String(128), nullable=False,
    )  # "workspace.create" | "member.add" | "job.submit" | "serving.activate" | etc.
    target: Mapped[str | None] = mapped_column(String(512), nullable=True)
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow,
    )
