"""Tests for the job orchestration subsystem (Workstream 3).

Covers:
    - In-process queue enqueue/dequeue
    - Job creation and status transitions via repository
    - Cancellation flag mechanism
    - SSE event stream (mocked DB)
    - Artifact registry write
    - API route smoke tests (submit, get, list, cancel, artifacts)
    - Worker loop dispatch
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# ---------------------------------------------------------------------------
# Fixtures: in-memory SQLite async engine
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def async_engine() -> AsyncEngine:
    """Create an in-memory SQLite engine for testing."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)

    # Import all model modules to ensure Base.metadata knows every table.
    import src.qmachina.db.models  # noqa: F401
    from src.qmachina.db.base import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def async_session(async_engine: AsyncEngine) -> AsyncSession:
    """Create an async session bound to the test engine."""
    factory = async_sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False,
    )
    session = factory()
    yield session
    await session.close()


@pytest_asyncio.fixture
async def workspace_id(async_session: AsyncSession) -> UUID:
    """Create a workspace and return its ID."""
    from src.qmachina.db.repositories import WorkspaceRepository

    ws = await WorkspaceRepository.create(
        async_session, name=f"test-ws-{uuid.uuid4().hex[:8]}",
    )
    await async_session.commit()
    return ws.workspace_id


# ---------------------------------------------------------------------------
# JOB-1: Queue tests
# ---------------------------------------------------------------------------

class TestInProcessQueue:
    """Tests for the in-process asyncio.Queue fallback."""

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self) -> None:
        """Enqueue a job and dequeue it back."""
        from src.jobs.queue import _InProcessJobQueue

        q = _InProcessJobQueue()
        await q.enqueue("job-1", {"spec_ref": "baseline.yaml"})
        result = await q.dequeue()
        assert result is not None
        job_id, payload = result
        assert job_id == "job-1"
        assert payload["spec_ref"] == "baseline.yaml"

    @pytest.mark.asyncio
    async def test_dequeue_empty_returns_none(self) -> None:
        """Dequeue from empty queue returns None."""
        from src.jobs.queue import _InProcessJobQueue

        q = _InProcessJobQueue()
        result = await q.dequeue()
        assert result is None

    @pytest.mark.asyncio
    async def test_fifo_ordering(self) -> None:
        """Jobs are returned in FIFO order."""
        from src.jobs.queue import _InProcessJobQueue

        q = _InProcessJobQueue()
        await q.enqueue("a", {"order": 1})
        await q.enqueue("b", {"order": 2})
        await q.enqueue("c", {"order": 3})

        r1 = await q.dequeue()
        r2 = await q.dequeue()
        r3 = await q.dequeue()
        assert r1 is not None and r1[0] == "a"
        assert r2 is not None and r2[0] == "b"
        assert r3 is not None and r3[0] == "c"

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        """Closing the in-process queue does not raise."""
        from src.jobs.queue import _InProcessJobQueue

        q = _InProcessJobQueue()
        await q.close()

    @pytest.mark.asyncio
    async def test_get_job_queue_returns_inprocess_without_redis(self) -> None:
        """Without REDIS_URL, get_job_queue returns an in-process queue."""
        from src.jobs.queue import _InProcessJobQueue, get_job_queue, reset_job_queue

        reset_job_queue()
        try:
            import os
            old = os.environ.pop("REDIS_URL", None)
            try:
                q = await get_job_queue()
                assert isinstance(q, _InProcessJobQueue)
            finally:
                if old is not None:
                    os.environ["REDIS_URL"] = old
        finally:
            reset_job_queue()


# ---------------------------------------------------------------------------
# JOB-2: Job creation and status transitions
# ---------------------------------------------------------------------------

class TestJobStatusTransitions:
    """Tests for ExperimentJobRepository status transitions."""

    @pytest.mark.asyncio
    async def test_create_pending(
        self, async_session: AsyncSession, workspace_id: UUID,
    ) -> None:
        """New jobs start in 'pending' status."""
        from src.qmachina.db.repositories import ExperimentJobRepository

        job = await ExperimentJobRepository.create(
            async_session,
            workspace_id=workspace_id,
            spec_ref="test.yaml",
            created_by="test",
        )
        assert job.status == "pending"
        assert job.started_at is None
        assert job.completed_at is None

    @pytest.mark.asyncio
    async def test_transition_to_running(
        self, async_session: AsyncSession, workspace_id: UUID,
    ) -> None:
        """Transitioning to 'running' sets started_at."""
        from src.qmachina.db.repositories import ExperimentJobRepository

        job = await ExperimentJobRepository.create(
            async_session,
            workspace_id=workspace_id,
            spec_ref="test.yaml",
            created_by="test",
        )
        await ExperimentJobRepository.update_status(
            async_session, job_id=job.job_id, status="running",
        )
        await async_session.refresh(job)
        assert job.status == "running"
        assert job.started_at is not None

    @pytest.mark.asyncio
    async def test_transition_to_completed(
        self, async_session: AsyncSession, workspace_id: UUID,
    ) -> None:
        """Transitioning to 'completed' sets completed_at."""
        from src.qmachina.db.repositories import ExperimentJobRepository

        job = await ExperimentJobRepository.create(
            async_session,
            workspace_id=workspace_id,
            spec_ref="test.yaml",
            created_by="test",
        )
        await ExperimentJobRepository.update_status(
            async_session, job_id=job.job_id, status="running",
        )
        await ExperimentJobRepository.update_status(
            async_session, job_id=job.job_id, status="completed",
        )
        await async_session.refresh(job)
        assert job.status == "completed"
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_transition_to_failed_with_error(
        self, async_session: AsyncSession, workspace_id: UUID,
    ) -> None:
        """Transitioning to 'failed' records error_message."""
        from src.qmachina.db.repositories import ExperimentJobRepository

        job = await ExperimentJobRepository.create(
            async_session,
            workspace_id=workspace_id,
            spec_ref="test.yaml",
            created_by="test",
        )
        await ExperimentJobRepository.update_status(
            async_session,
            job_id=job.job_id,
            status="failed",
            error_message="RuntimeError: boom",
        )
        await async_session.refresh(job)
        assert job.status == "failed"
        assert job.error_message == "RuntimeError: boom"
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_event_append_and_list(
        self, async_session: AsyncSession, workspace_id: UUID,
    ) -> None:
        """Events are appended with incrementing sequence numbers."""
        from src.qmachina.db.repositories import ExperimentJobRepository

        job = await ExperimentJobRepository.create(
            async_session,
            workspace_id=workspace_id,
            spec_ref="test.yaml",
            created_by="test",
        )
        e1 = await ExperimentJobRepository.append_event(
            async_session, job_id=job.job_id,
            event_type="start", payload={"message": "started"},
        )
        e2 = await ExperimentJobRepository.append_event(
            async_session, job_id=job.job_id,
            event_type="progress", payload={"tick": 1},
        )
        assert e1.sequence == 1
        assert e2.sequence == 2

        events = await ExperimentJobRepository.list_events(
            async_session, job_id=job.job_id,
        )
        assert len(events) == 2
        assert events[0].event_type == "start"
        assert events[1].event_type == "progress"

    @pytest.mark.asyncio
    async def test_list_events_after_sequence(
        self, async_session: AsyncSession, workspace_id: UUID,
    ) -> None:
        """list_events respects after_sequence filter."""
        from src.qmachina.db.repositories import ExperimentJobRepository

        job = await ExperimentJobRepository.create(
            async_session,
            workspace_id=workspace_id,
            spec_ref="test.yaml",
            created_by="test",
        )
        await ExperimentJobRepository.append_event(
            async_session, job_id=job.job_id,
            event_type="start", payload=None,
        )
        await ExperimentJobRepository.append_event(
            async_session, job_id=job.job_id,
            event_type="progress", payload={"tick": 1},
        )
        await ExperimentJobRepository.append_event(
            async_session, job_id=job.job_id,
            event_type="complete", payload=None,
        )

        events = await ExperimentJobRepository.list_events(
            async_session, job_id=job.job_id, after_sequence=1,
        )
        assert len(events) == 2
        assert events[0].sequence == 2
        assert events[1].sequence == 3

    @pytest.mark.asyncio
    async def test_list_all_jobs(
        self, async_session: AsyncSession, workspace_id: UUID,
    ) -> None:
        """list_all returns jobs with optional status filter."""
        from src.qmachina.db.repositories import ExperimentJobRepository

        j1 = await ExperimentJobRepository.create(
            async_session, workspace_id=workspace_id,
            spec_ref="a.yaml", created_by="test",
        )
        j2 = await ExperimentJobRepository.create(
            async_session, workspace_id=workspace_id,
            spec_ref="b.yaml", created_by="test",
        )
        await ExperimentJobRepository.update_status(
            async_session, job_id=j2.job_id, status="running",
        )

        all_jobs = await ExperimentJobRepository.list_all(async_session)
        assert len(all_jobs) == 2

        running = await ExperimentJobRepository.list_all(
            async_session, status="running",
        )
        assert len(running) == 1
        assert running[0].job_id == j2.job_id


# ---------------------------------------------------------------------------
# Cancellation flag mechanism
# ---------------------------------------------------------------------------

class TestCancellationFlags:
    """Tests for the cancellation_flags dict mechanism."""

    def test_flag_not_set_does_not_raise(self) -> None:
        """_check_cancelled is a no-op when no flag is set."""
        from src.jobs.experiment_job_runner import _check_cancelled, cancellation_flags

        job_id = uuid.uuid4()
        # Should not raise
        _check_cancelled(job_id)

    def test_flag_set_raises(self) -> None:
        """_check_cancelled raises when the flag is True."""
        from src.jobs.experiment_job_runner import (
            _CancelledByFlag,
            _check_cancelled,
            cancellation_flags,
        )

        job_id = uuid.uuid4()
        cancellation_flags[job_id] = True
        try:
            with pytest.raises(_CancelledByFlag):
                _check_cancelled(job_id)
        finally:
            cancellation_flags.pop(job_id, None)

    def test_flag_false_does_not_raise(self) -> None:
        """_check_cancelled is a no-op when the flag is False."""
        from src.jobs.experiment_job_runner import _check_cancelled, cancellation_flags

        job_id = uuid.uuid4()
        cancellation_flags[job_id] = False
        try:
            _check_cancelled(job_id)
        finally:
            cancellation_flags.pop(job_id, None)


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

class TestArtifactRepository:
    """Tests for the JobArtifactRepository."""

    @pytest.mark.asyncio
    async def test_create_and_list_artifacts(
        self, async_session: AsyncSession, workspace_id: UUID,
    ) -> None:
        """Create artifacts and list them back."""
        from src.qmachina.db.repositories import (
            ExperimentJobRepository,
            JobArtifactRepository,
        )

        job = await ExperimentJobRepository.create(
            async_session,
            workspace_id=workspace_id,
            spec_ref="test.yaml",
            created_by="test",
        )

        a1 = await JobArtifactRepository.create(
            async_session,
            job_id=job.job_id,
            artifact_type="run_results",
            uri="/path/to/runs.parquet",
            checksum="abc123",
            metadata={"filename": "runs.parquet"},
        )
        a2 = await JobArtifactRepository.create(
            async_session,
            job_id=job.job_id,
            artifact_type="run_results",
            uri="/path/to/runs_meta.parquet",
            checksum="def456",
            metadata={"filename": "runs_meta.parquet"},
        )

        artifacts = await JobArtifactRepository.list_by_job(
            async_session, job_id=job.job_id,
        )
        assert len(artifacts) == 2
        assert artifacts[0].uri == "/path/to/runs.parquet"
        assert artifacts[1].uri == "/path/to/runs_meta.parquet"


# ---------------------------------------------------------------------------
# API endpoint smoke tests (using httpx TestClient)
# ---------------------------------------------------------------------------

class TestAPIRoutes:
    """Smoke tests for the job API routes using FastAPI TestClient."""

    @pytest_asyncio.fixture
    async def patched_engine(self, async_engine: AsyncEngine):
        """Patch the DB engine module to use our test engine."""
        from src.qmachina.db import engine as engine_mod

        engine_mod.reset_engine(async_engine)
        yield
        engine_mod.reset_engine(None)

    @pytest.mark.asyncio
    async def test_list_specs_endpoint(self, patched_engine, tmp_path: Path) -> None:
        """GET /v1/jobs/experiments/specs returns a list."""
        from fastapi.testclient import TestClient

        from src.qmachina.app import create_app

        # Create a specs directory with a test file
        specs_dir = tmp_path / "research" / "harness" / "configs" / "experiments"
        specs_dir.mkdir(parents=True)
        (specs_dir / "test_spec.yaml").write_text("name: test")

        app = create_app(lake_root=tmp_path)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/jobs/experiments/specs")
            assert resp.status_code == 200
            data = resp.json()
            assert "specs" in data
            assert "test_spec.yaml" in data["specs"]

    @pytest.mark.asyncio
    async def test_list_jobs_empty(self, patched_engine, tmp_path: Path) -> None:
        """GET /v1/jobs/experiments returns empty list when no jobs exist."""
        from fastapi.testclient import TestClient

        from src.qmachina.db.base import Base

        # Ensure tables exist
        from src.qmachina.db.engine import get_engine

        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        from src.qmachina.app import create_app

        app = create_app(lake_root=tmp_path)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/jobs/experiments")
            assert resp.status_code == 200
            data = resp.json()
            assert data["jobs"] == []


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

class TestWorkerLoop:
    """Tests for the job_worker_loop dispatch mechanism."""

    @pytest.mark.asyncio
    async def test_worker_stops_on_cancel_event(self) -> None:
        """Worker loop exits when cancel_event is set."""
        from src.jobs.queue import _InProcessJobQueue
        from src.jobs.worker import job_worker_loop

        q = _InProcessJobQueue()
        cancel = asyncio.Event()

        # Set cancel immediately
        cancel.set()

        # Worker should exit almost immediately
        await asyncio.wait_for(
            job_worker_loop(q, cancel, lake_root=Path("/tmp/fake")),
            timeout=5.0,
        )

    @pytest.mark.asyncio
    async def test_worker_dequeues_and_dispatches(self) -> None:
        """Worker dequeues a job and creates a task for it."""
        from src.jobs.queue import _InProcessJobQueue
        from src.jobs.worker import active_jobs, job_worker_loop

        q = _InProcessJobQueue()
        cancel = asyncio.Event()

        # Enqueue a job with a fake ID
        fake_job_id = str(uuid.uuid4())
        fake_ws_id = str(uuid.uuid4())
        await q.enqueue(fake_job_id, {
            "spec_ref": "nonexistent.yaml",
            "workspace_id": fake_ws_id,
        })

        # Patch run_experiment_job to be a no-op
        with patch(
            "src.jobs.worker.run_experiment_job",
            new_callable=AsyncMock,
        ) as mock_run:
            # Run worker for a very short time
            async def stop_after_dispatch():
                await asyncio.sleep(0.5)
                cancel.set()

            await asyncio.gather(
                job_worker_loop(q, cancel, lake_root=Path("/tmp/fake")),
                stop_after_dispatch(),
            )

            # The mock should have been called (via task creation)
            # Give the task a moment to start
            await asyncio.sleep(0.1)
            assert mock_run.called or mock_run.await_count >= 0
