"""Tests for the control-plane database repository layer.

Uses an async SQLite in-memory backend via aiosqlite so no real
Postgres instance is required. All tests are isolated via per-test
transactions that are rolled back after each test.
"""
from __future__ import annotations

import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.qmachina.db.base import Base
from src.qmachina.db.models import (
    AuditEvent,
    ExperimentJob,
    IngestionLiveSession,
    JobEvent,
    ModelingSession,
    ModelingStepState,
    Workspace,
)
from src.qmachina.db.repositories import (
    AuditRepository,
    ExperimentJobRepository,
    IngestionSessionRepository,
    ModelingSessionRepository,
    WorkspaceRepository,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create an async SQLite in-memory engine and provision all tables."""
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async session scoped to a single test."""
    factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False,
    )
    async with factory() as sess:
        yield sess


@pytest_asyncio.fixture
async def workspace(session: AsyncSession) -> Workspace:
    """Create and return a default workspace for tests that need one."""
    ws = await WorkspaceRepository.create(session, name="test-workspace")
    await session.commit()
    return ws


# ---------------------------------------------------------------------------
# Workspace tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workspace_create_and_get(session: AsyncSession) -> None:
    """Creating a workspace persists it and is retrievable by ID."""
    ws = await WorkspaceRepository.create(session, name="alpha")
    await session.commit()
    assert ws.workspace_id is not None
    assert ws.name == "alpha"

    fetched = await WorkspaceRepository.get(session, workspace_id=ws.workspace_id)
    assert fetched is not None
    assert fetched.name == "alpha"


@pytest.mark.asyncio
async def test_workspace_list(session: AsyncSession) -> None:
    """Listing workspaces returns all created workspaces."""
    await WorkspaceRepository.create(session, name="ws-a")
    await WorkspaceRepository.create(session, name="ws-b")
    await session.commit()

    all_ws = await WorkspaceRepository.list_all(session)
    names = {ws.name for ws in all_ws}
    assert "ws-a" in names
    assert "ws-b" in names


@pytest.mark.asyncio
async def test_workspace_get_nonexistent(session: AsyncSession) -> None:
    """Getting a workspace that does not exist returns None."""
    result = await WorkspaceRepository.get(session, workspace_id=uuid.uuid4())
    assert result is None


# ---------------------------------------------------------------------------
# Experiment Job tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_job_create_defaults(
    session: AsyncSession, workspace: Workspace,
) -> None:
    """A new job starts in pending status with no timestamps set."""
    job = await ExperimentJobRepository.create(
        session,
        workspace_id=workspace.workspace_id,
        spec_ref="baseline_v1",
        created_by="test-user",
    )
    await session.commit()

    assert job.status == "pending"
    assert job.started_at is None
    assert job.completed_at is None
    assert job.error_message is None


@pytest.mark.asyncio
async def test_job_status_update_running(
    session: AsyncSession, workspace: Workspace,
) -> None:
    """Updating status to running sets started_at."""
    job = await ExperimentJobRepository.create(
        session,
        workspace_id=workspace.workspace_id,
        spec_ref="exp-1",
        created_by="test-user",
    )
    await session.commit()

    await ExperimentJobRepository.update_status(
        session, job_id=job.job_id, status="running",
    )
    await session.commit()

    fetched = await ExperimentJobRepository.get(session, job_id=job.job_id)
    assert fetched is not None
    assert fetched.status == "running"
    assert fetched.started_at is not None


@pytest.mark.asyncio
async def test_job_status_update_failed(
    session: AsyncSession, workspace: Workspace,
) -> None:
    """Updating status to failed sets completed_at and error_message."""
    job = await ExperimentJobRepository.create(
        session,
        workspace_id=workspace.workspace_id,
        spec_ref="exp-2",
        created_by="test-user",
    )
    await session.commit()

    await ExperimentJobRepository.update_status(
        session,
        job_id=job.job_id,
        status="failed",
        error_message="OOM in gold builder",
    )
    await session.commit()

    fetched = await ExperimentJobRepository.get(session, job_id=job.job_id)
    assert fetched is not None
    assert fetched.status == "failed"
    assert fetched.completed_at is not None
    assert fetched.error_message == "OOM in gold builder"


@pytest.mark.asyncio
async def test_job_status_update_nonexistent(session: AsyncSession) -> None:
    """Updating status of a nonexistent job raises ValueError."""
    with pytest.raises(ValueError, match="ExperimentJob not found"):
        await ExperimentJobRepository.update_status(
            session, job_id=uuid.uuid4(), status="running",
        )


@pytest.mark.asyncio
async def test_job_event_append(
    session: AsyncSession, workspace: Workspace,
) -> None:
    """Events are appended with auto-incrementing sequence numbers."""
    job = await ExperimentJobRepository.create(
        session,
        workspace_id=workspace.workspace_id,
        spec_ref="exp-3",
        created_by="test-user",
    )
    await session.commit()

    e1 = await ExperimentJobRepository.append_event(
        session,
        job_id=job.job_id,
        event_type="progress",
        payload={"pct": 25},
    )
    await session.commit()
    assert e1.sequence == 1

    e2 = await ExperimentJobRepository.append_event(
        session,
        job_id=job.job_id,
        event_type="log",
        payload={"message": "halfway"},
    )
    await session.commit()
    assert e2.sequence == 2

    e3 = await ExperimentJobRepository.append_event(
        session,
        job_id=job.job_id,
        event_type="complete",
        payload=None,
    )
    await session.commit()
    assert e3.sequence == 3


@pytest.mark.asyncio
async def test_job_list_by_workspace(
    session: AsyncSession, workspace: Workspace,
) -> None:
    """List jobs returns jobs for the workspace, optionally filtered by status."""
    await ExperimentJobRepository.create(
        session,
        workspace_id=workspace.workspace_id,
        spec_ref="exp-a",
        created_by="user-a",
    )
    job_b = await ExperimentJobRepository.create(
        session,
        workspace_id=workspace.workspace_id,
        spec_ref="exp-b",
        created_by="user-b",
    )
    await session.commit()

    await ExperimentJobRepository.update_status(
        session, job_id=job_b.job_id, status="running",
    )
    await session.commit()

    all_jobs = await ExperimentJobRepository.list_by_workspace(
        session, workspace_id=workspace.workspace_id,
    )
    assert len(all_jobs) == 2

    running_jobs = await ExperimentJobRepository.list_by_workspace(
        session, workspace_id=workspace.workspace_id, status="running",
    )
    assert len(running_jobs) == 1
    assert running_jobs[0].spec_ref == "exp-b"


# ---------------------------------------------------------------------------
# Modeling Session tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_modeling_session_create(
    session: AsyncSession, workspace: Workspace,
) -> None:
    """New modeling session starts in draft status."""
    ms = await ModelingSessionRepository.create(
        session,
        workspace_id=workspace.workspace_id,
        created_by="researcher",
    )
    await session.commit()

    assert ms.status == "draft"
    assert ms.created_by == "researcher"


@pytest.mark.asyncio
async def test_modeling_step_commit_ordered(
    session: AsyncSession, workspace: Workspace,
) -> None:
    """Steps can be committed in order and are retrievable."""
    ms = await ModelingSessionRepository.create(
        session,
        workspace_id=workspace.workspace_id,
        created_by="researcher",
    )
    await session.commit()

    s1 = await ModelingSessionRepository.commit_step(
        session,
        session_id=ms.session_id,
        step_name="dataset_select",
        payload={"dataset_id": "ds_abc123"},
    )
    await session.commit()
    assert s1.status == "committed"
    assert s1.committed_at is not None

    s2 = await ModelingSessionRepository.commit_step(
        session,
        session_id=ms.session_id,
        step_name="gold_config",
        payload={"c1": 0.5, "c2": 0.3},
    )
    await session.commit()
    assert s2.status == "committed"

    steps = await ModelingSessionRepository.get_steps(
        session, session_id=ms.session_id,
    )
    step_names = [s.step_name for s in steps]
    assert "dataset_select" in step_names
    assert "gold_config" in step_names


@pytest.mark.asyncio
async def test_modeling_step_commit_idempotent(
    session: AsyncSession, workspace: Workspace,
) -> None:
    """Re-committing the same step replaces payload (idempotent upsert)."""
    ms = await ModelingSessionRepository.create(
        session,
        workspace_id=workspace.workspace_id,
        created_by="researcher",
    )
    await session.commit()

    await ModelingSessionRepository.commit_step(
        session,
        session_id=ms.session_id,
        step_name="dataset_select",
        payload={"dataset_id": "old_id"},
    )
    await session.commit()

    s2 = await ModelingSessionRepository.commit_step(
        session,
        session_id=ms.session_id,
        step_name="dataset_select",
        payload={"dataset_id": "new_id"},
    )
    await session.commit()
    assert s2.payload_json == {"dataset_id": "new_id"}

    steps = await ModelingSessionRepository.get_steps(
        session, session_id=ms.session_id,
    )
    assert len(steps) == 1
    assert steps[0].payload_json == {"dataset_id": "new_id"}


@pytest.mark.asyncio
async def test_modeling_session_update_status(
    session: AsyncSession, workspace: Workspace,
) -> None:
    """Modeling session status can be updated."""
    ms = await ModelingSessionRepository.create(
        session,
        workspace_id=workspace.workspace_id,
        created_by="researcher",
    )
    await session.commit()

    await ModelingSessionRepository.update_status(
        session, session_id=ms.session_id, status="in_progress",
    )
    await session.commit()

    fetched = await ModelingSessionRepository.get(
        session, session_id=ms.session_id,
    )
    assert fetched is not None
    assert fetched.status == "in_progress"


# ---------------------------------------------------------------------------
# Ingestion Session tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingestion_create_and_update(
    session: AsyncSession, workspace: Workspace,
) -> None:
    """Ingestion session lifecycle: create, update status, update checkpoint."""
    ils = await IngestionSessionRepository.create(
        session,
        workspace_id=workspace.workspace_id,
        symbol="MNQH6",
        config={"feed": "databento", "schema": "mbo"},
    )
    await session.commit()
    assert ils.status == "starting"

    await IngestionSessionRepository.update_status(
        session, session_id=ils.session_id, status="active",
    )
    await session.commit()

    fetched = await IngestionSessionRepository.get(
        session, session_id=ils.session_id,
    )
    assert fetched is not None
    assert fetched.status == "active"

    await IngestionSessionRepository.update_checkpoint(
        session,
        session_id=ils.session_id,
        checkpoint={"sequence_id": 42, "ts_event_ns": 1000000},
    )
    await session.commit()

    fetched2 = await IngestionSessionRepository.get(
        session, session_id=ils.session_id,
    )
    assert fetched2 is not None
    assert fetched2.checkpoint == {"sequence_id": 42, "ts_event_ns": 1000000}


@pytest.mark.asyncio
async def test_ingestion_stopped_sets_timestamp(
    session: AsyncSession, workspace: Workspace,
) -> None:
    """Transitioning to stopped sets stopped_at."""
    ils = await IngestionSessionRepository.create(
        session,
        workspace_id=workspace.workspace_id,
        symbol="ESH6",
        config={},
    )
    await session.commit()

    await IngestionSessionRepository.update_status(
        session, session_id=ils.session_id, status="stopped",
    )
    await session.commit()

    fetched = await IngestionSessionRepository.get(
        session, session_id=ils.session_id,
    )
    assert fetched is not None
    assert fetched.stopped_at is not None


# ---------------------------------------------------------------------------
# Audit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_audit_write_and_list(
    session: AsyncSession, workspace: Workspace,
) -> None:
    """Audit events are written and listed newest-first."""
    await AuditRepository.write(
        session,
        workspace_id=workspace.workspace_id,
        actor_id="admin",
        action="workspace.create",
        target=str(workspace.workspace_id),
        payload={"name": workspace.name},
    )
    await session.commit()

    await AuditRepository.write(
        session,
        workspace_id=workspace.workspace_id,
        actor_id="admin",
        action="member.add",
        target="user@example.com",
    )
    await session.commit()

    events = await AuditRepository.list_by_workspace(
        session, workspace_id=workspace.workspace_id,
    )
    assert len(events) == 2
    # Most recent first
    assert events[0].action == "member.add"
    assert events[1].action == "workspace.create"


@pytest.mark.asyncio
async def test_audit_system_level_event(session: AsyncSession) -> None:
    """System-level audit events have workspace_id=None."""
    event = await AuditRepository.write(
        session,
        workspace_id=None,
        actor_id="system",
        action="db.migrate",
        payload={"version": "ddfe5d09bcb6"},
    )
    await session.commit()
    assert event.workspace_id is None
    assert event.action == "db.migrate"


@pytest.mark.asyncio
async def test_audit_list_respects_limit(
    session: AsyncSession, workspace: Workspace,
) -> None:
    """List respects the limit parameter."""
    for i in range(5):
        await AuditRepository.write(
            session,
            workspace_id=workspace.workspace_id,
            actor_id="bot",
            action=f"action.{i}",
        )
    await session.commit()

    events = await AuditRepository.list_by_workspace(
        session, workspace_id=workspace.workspace_id, limit=3,
    )
    assert len(events) == 3
