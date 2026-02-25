"""Tests for the modeling session service state machine and API layer.

Uses an async SQLite in-memory backend via aiosqlite so no real
Postgres instance is required. Tests cover:
  - Session creation
  - Step gate enforcement (sequential ordering)
  - Step commit (including idempotent re-commit)
  - Decision log
  - Promotion with missing steps
  - YAML validation (valid and invalid)
  - Dataset preview (with fixture parquet data)
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
import yaml
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.qmachina.db.base import Base
from src.qmachina.db.models import ModelingSession, ModelingStepState, Workspace
from src.qmachina.db.repositories import (
    ModelingSessionRepository,
    WorkspaceRepository,
)
from src.qmachina.modeling_session_service import (
    STEPS_ORDERED,
    ModelingSessionService,
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
    """Create and return a default workspace for tests."""
    ws = await WorkspaceRepository.create(session, name="test-workspace")
    await session.commit()
    return ws


@pytest.fixture
def lake_root(tmp_path: Path) -> Path:
    """Create a temporary lake root with minimal structure."""
    lake = tmp_path / "lake"
    # Create serving registry path
    harness = lake / "research" / "harness"
    harness.mkdir(parents=True)
    (harness / "configs" / "serving").mkdir(parents=True)
    (harness / "configs" / "serving_versions").mkdir(parents=True)
    return lake


@pytest.fixture
def service(lake_root: Path) -> ModelingSessionService:
    """Create a ModelingSessionService with a temp lake root."""
    return ModelingSessionService(lake_root)


@pytest.fixture
def dataset_lake(lake_root: Path) -> tuple[Path, str]:
    """Create fixture parquet data for a test dataset.

    Returns (lake_root, dataset_id).
    """
    dataset_id = "test_dataset_abc123"
    ds_dir = lake_root / "research" / "datasets" / dataset_id
    ds_dir.mkdir(parents=True)

    # bins.parquet
    bins_df = pd.DataFrame({
        "bin_seq": range(100),
        "ts_ns": np.arange(100) * 1_000_000_000 + 1_700_000_000_000_000_000,
        "mid_price": np.linspace(4800.0, 4810.0, 100),
    })
    bins_df.to_parquet(ds_dir / "bins.parquet", index=False)

    # grid_clean.parquet
    grid_df = pd.DataFrame({
        "bin_seq": np.repeat(range(100), 10),
        "k": np.tile(range(-5, 5), 100),
        "rest_depth": np.random.default_rng(42).uniform(0, 100, 1000),
        "flow_score": np.random.default_rng(42).standard_normal(1000),
    })
    grid_df.to_parquet(ds_dir / "grid_clean.parquet", index=False)

    return lake_root, dataset_id


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_session(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
) -> None:
    """Creating a session returns a draft session."""
    ms = await service.create_session(
        session,
        workspace_id=workspace.workspace_id,
        created_by="tester",
    )
    await session.commit()

    assert ms.status == "draft"
    assert ms.created_by == "tester"
    assert ms.session_id is not None


# ---------------------------------------------------------------------------
# Step commit: dataset_select (first step, always allowed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commit_dataset_select(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
) -> None:
    """Committing dataset_select (step 0) succeeds without prior steps."""
    ms = await service.create_session(
        session,
        workspace_id=workspace.workspace_id,
    )
    await session.commit()

    step = await service.commit_step(
        session,
        session_id=ms.session_id,
        step_name="dataset_select",
        payload={"dataset_id": "ds_test_123"},
    )
    await session.commit()

    assert step.status == "committed"
    assert step.payload_json == {"dataset_id": "ds_test_123"}


# ---------------------------------------------------------------------------
# Step gate: out-of-order commit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commit_out_of_order_raises(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
) -> None:
    """Committing step 3 (signal_select) before step 2 (gold_config) raises ValueError."""
    ms = await service.create_session(
        session,
        workspace_id=workspace.workspace_id,
    )
    await session.commit()

    # Commit step 1
    await service.commit_step(
        session,
        session_id=ms.session_id,
        step_name="dataset_select",
        payload={"dataset_id": "ds_test"},
    )
    await session.commit()

    # Skip step 2, try step 3
    with pytest.raises(ValueError, match="gold_config.*has not been committed"):
        await service.commit_step(
            session,
            session_id=ms.session_id,
            step_name="signal_select",
            payload={"signal_name": "derivative"},
        )


# ---------------------------------------------------------------------------
# Commit all steps in order
# ---------------------------------------------------------------------------


async def _commit_all_steps(
    session: AsyncSession,
    service: ModelingSessionService,
    session_id: uuid.UUID,
) -> None:
    """Helper to commit all 7 steps in order."""
    payloads: dict[str, dict[str, Any]] = {
        "dataset_select": {"dataset_id": "test_ds_abc", "pipeline": "baseline_mnq"},
        "gold_config": {
            "c1_v_add": 1.0,
            "c2_v_rest_pos": 0.5,
            "c3_a_add": 0.3,
            "c4_v_pull": -0.8,
            "c5_v_fill": -0.5,
            "c6_v_rest_neg": -0.3,
            "c7_a_pull": -0.2,
        },
        "signal_select": {"signal_name": "derivative"},
        "eval_params": {
            "tp_ticks": 8,
            "sl_ticks": 4,
            "cooldown_bins": 20,
            "warmup_bins": 300,
        },
        "run_experiment": {
            "job_id": "mock-job-id",
            "status": "completed",
            "run_ids": [],
            "n_runs": 0,
        },
        "promote_review": {"reviewed": True},
        "promotion": {"confirmed": True},
    }
    for step_name in STEPS_ORDERED:
        await service.commit_step(
            session,
            session_id=session_id,
            step_name=step_name,
            payload=payloads[step_name],
        )
        await session.commit()


@pytest.mark.asyncio
async def test_commit_all_steps_in_order(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
) -> None:
    """Committing all 7 steps in order succeeds."""
    ms = await service.create_session(
        session,
        workspace_id=workspace.workspace_id,
    )
    await session.commit()

    await _commit_all_steps(session, service, ms.session_id)

    _, steps = await service.get_session_with_steps(
        session, session_id=ms.session_id,
    )
    committed_names = {s.step_name for s in steps if s.status == "committed"}
    assert committed_names == set(STEPS_ORDERED)


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decision_log(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
) -> None:
    """Decision log returns 7 entries ordered by step index."""
    ms = await service.create_session(
        session,
        workspace_id=workspace.workspace_id,
    )
    await session.commit()

    await _commit_all_steps(session, service, ms.session_id)

    log = await service.get_decision_log(
        session, session_id=ms.session_id,
    )
    assert len(log) == 7
    assert [entry["step_name"] for entry in log] == STEPS_ORDERED
    for entry in log:
        assert entry["committed_at"] is not None
        assert entry["payload"] is not None


# ---------------------------------------------------------------------------
# Promote with missing steps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_promote_missing_steps_raises(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
) -> None:
    """Promoting with missing steps raises ValueError."""
    ms = await service.create_session(
        session,
        workspace_id=workspace.workspace_id,
    )
    await session.commit()

    # Only commit first step
    await service.commit_step(
        session,
        session_id=ms.session_id,
        step_name="dataset_select",
        payload={"dataset_id": "ds_test"},
    )
    await session.commit()

    with pytest.raises(ValueError, match="steps not committed"):
        await service.promote(
            session,
            session_id=ms.session_id,
            alias="test_alias",
        )


# ---------------------------------------------------------------------------
# Unknown step name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commit_unknown_step_raises(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
) -> None:
    """Committing an unknown step name raises ValueError."""
    ms = await service.create_session(
        session,
        workspace_id=workspace.workspace_id,
    )
    await session.commit()

    with pytest.raises(ValueError, match="Unknown step"):
        await service.commit_step(
            session,
            session_id=ms.session_id,
            step_name="nonexistent_step",
            payload={},
        )


# ---------------------------------------------------------------------------
# Idempotent re-commit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_recommit_is_idempotent(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
) -> None:
    """Re-committing the same step updates the payload."""
    ms = await service.create_session(
        session,
        workspace_id=workspace.workspace_id,
    )
    await session.commit()

    await service.commit_step(
        session,
        session_id=ms.session_id,
        step_name="dataset_select",
        payload={"dataset_id": "old_ds"},
    )
    await session.commit()

    step2 = await service.commit_step(
        session,
        session_id=ms.session_id,
        step_name="dataset_select",
        payload={"dataset_id": "new_ds"},
    )
    await session.commit()

    assert step2.payload_json == {"dataset_id": "new_ds"}


# ---------------------------------------------------------------------------
# Session transitions to in_progress on first commit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_transitions_to_in_progress(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
) -> None:
    """Session status transitions from draft to in_progress on first commit."""
    ms = await service.create_session(
        session,
        workspace_id=workspace.workspace_id,
    )
    await session.commit()
    assert ms.status == "draft"

    await service.commit_step(
        session,
        session_id=ms.session_id,
        step_name="dataset_select",
        payload={"dataset_id": "ds_test"},
    )
    await session.commit()

    updated = await service.get_session(
        session, session_id=ms.session_id,
    )
    assert updated is not None
    assert updated.status == "in_progress"


# ---------------------------------------------------------------------------
# YAML validation: valid
# ---------------------------------------------------------------------------


def test_validate_yaml_valid() -> None:
    """Valid ServingSpec YAML parses without errors."""
    from src.qmachina.serving_config import ServingSpec

    valid_spec = {
        "name": "test_valid",
        "pipeline": "baseline_mnq",
        "model_id": "vacuum_pressure",
    }
    # Validate via model directly (same as the endpoint does)
    spec = ServingSpec.model_validate(valid_spec)
    assert spec.name == "test_valid"


def test_validate_yaml_invalid() -> None:
    """Invalid ServingSpec YAML produces validation errors."""
    from src.qmachina.serving_config import ServingSpec

    invalid_spec = {
        # missing required "pipeline" field
        "name": "test_invalid",
    }
    with pytest.raises(Exception):
        ServingSpec.model_validate(invalid_spec)


def test_validate_yaml_content_valid() -> None:
    """YAML content string that is a valid ServingSpec passes."""
    from src.qmachina.serving_config import ServingSpec

    yaml_content = yaml.dump({
        "name": "yaml_test",
        "pipeline": "baseline_mnq",
    })
    parsed = yaml.safe_load(yaml_content)
    assert isinstance(parsed, dict)
    spec = ServingSpec.model_validate(parsed)
    assert spec.name == "yaml_test"


def test_validate_yaml_content_invalid_yaml() -> None:
    """Malformed YAML content produces parse errors."""
    bad_yaml = "name: [invalid: yaml: {{"
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(bad_yaml)


def test_validate_yaml_content_missing_field() -> None:
    """YAML missing required fields produces validation errors."""
    from src.qmachina.serving_config import ServingSpec

    yaml_content = yaml.dump({"name": "missing_pipeline"})
    parsed = yaml.safe_load(yaml_content)
    errors: list[str] = []
    try:
        ServingSpec.model_validate(parsed)
    except Exception as exc:
        errors.append(str(exc))
    assert len(errors) > 0


# ---------------------------------------------------------------------------
# Dataset preview with fixture parquet data
# ---------------------------------------------------------------------------


def test_dataset_preview_stats(dataset_lake: tuple[Path, str]) -> None:
    """Dataset preview returns correct summary statistics."""
    lake_root, dataset_id = dataset_lake

    from src.experiment_harness.dataset_registry import DatasetRegistry

    registry = DatasetRegistry(lake_root)
    paths = registry.resolve(dataset_id)

    bins_df = pd.read_parquet(paths.bins_parquet)
    grid_df = pd.read_parquet(paths.grid_clean_parquet)

    # Verify bins
    assert len(bins_df) == 100

    # Verify grid has flow_score
    assert "flow_score" in grid_df.columns
    assert len(grid_df) == 1000

    # Verify mid_price range
    assert bins_df["mid_price"].min() == pytest.approx(4800.0)
    assert bins_df["mid_price"].max() == pytest.approx(4810.0)

    # Verify signal distribution is computable
    flow = grid_df["flow_score"].dropna()
    assert len(flow) > 0
    assert flow.std() > 0


# ---------------------------------------------------------------------------
# get_session_with_steps for nonexistent session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_session_with_steps_nonexistent(
    session: AsyncSession,
    service: ModelingSessionService,
) -> None:
    """Getting a nonexistent session raises ValueError."""
    with pytest.raises(ValueError, match="ModelingSession not found"):
        await service.get_session_with_steps(
            session, session_id=uuid.uuid4(),
        )


# ---------------------------------------------------------------------------
# decision_log for nonexistent session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decision_log_nonexistent_session(
    session: AsyncSession,
    service: ModelingSessionService,
) -> None:
    """Decision log for nonexistent session raises ValueError."""
    with pytest.raises(ValueError, match="ModelingSession not found"):
        await service.get_decision_log(
            session, session_id=uuid.uuid4(),
        )


# ---------------------------------------------------------------------------
# run_experiment step ordering and gate enforcement
# ---------------------------------------------------------------------------


def test_run_experiment_in_steps_ordered() -> None:
    """run_experiment is at index 4 in STEPS_ORDERED."""
    assert STEPS_ORDERED[4] == "run_experiment"
    assert len(STEPS_ORDERED) == 7


@pytest.mark.asyncio
async def test_commit_run_experiment_requires_eval_params(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
) -> None:
    """Committing run_experiment without eval_params raises ValueError."""
    ms = await service.create_session(session, workspace_id=workspace.workspace_id)
    await session.commit()

    # Commit only first three steps
    for step_name in ["dataset_select", "gold_config", "signal_select"]:
        await service.commit_step(
            session,
            session_id=ms.session_id,
            step_name=step_name,
            payload={"dataset_id": "ds_x"} if step_name == "dataset_select" else {"signal_name": "derivative"} if step_name == "signal_select" else {"c1_v_add": 1.0, "c2_v_rest_pos": 0.5, "c3_a_add": 0.3, "c4_v_pull": -0.8, "c5_v_fill": -0.5, "c6_v_rest_neg": -0.3, "c7_a_pull": -0.2},
        )
        await session.commit()

    with pytest.raises(ValueError, match="eval_params.*has not been committed"):
        await service.commit_step(
            session,
            session_id=ms.session_id,
            step_name="run_experiment",
            payload={"job_id": "j", "status": "completed"},
        )


@pytest.mark.asyncio
async def test_promote_blocked_without_run_experiment(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
) -> None:
    """Promotion with all steps except run_experiment raises ValueError about missing steps."""
    ms = await service.create_session(session, workspace_id=workspace.workspace_id)
    await session.commit()

    payloads: dict[str, dict[str, Any]] = {
        "dataset_select": {"dataset_id": "test_ds_abc", "pipeline": "baseline_mnq"},
        "gold_config": {
            "c1_v_add": 1.0, "c2_v_rest_pos": 0.5, "c3_a_add": 0.3,
            "c4_v_pull": -0.8, "c5_v_fill": -0.5, "c6_v_rest_neg": -0.3, "c7_a_pull": -0.2,
        },
        "signal_select": {"signal_name": "derivative"},
        "eval_params": {"tp_ticks": 8, "sl_ticks": 4, "cooldown_bins": 20, "warmup_bins": 300},
        "promote_review": {"reviewed": True},
        "promotion": {"confirmed": True},
    }
    # Commit everything except run_experiment
    for step_name in ["dataset_select", "gold_config", "signal_select", "eval_params"]:
        await service.commit_step(
            session, session_id=ms.session_id, step_name=step_name, payload=payloads[step_name],
        )
        await session.commit()

    with pytest.raises(ValueError, match="steps not committed"):
        await service.promote(session, session_id=ms.session_id, alias="test")


@pytest.mark.asyncio
async def test_promote_blocked_when_experiment_failed(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
) -> None:
    """Promotion with run_experiment status='failed' raises ValueError."""
    ms = await service.create_session(session, workspace_id=workspace.workspace_id)
    await session.commit()

    payloads: dict[str, dict[str, Any]] = {
        "dataset_select": {"dataset_id": "test_ds_abc", "pipeline": "baseline_mnq"},
        "gold_config": {
            "c1_v_add": 1.0, "c2_v_rest_pos": 0.5, "c3_a_add": 0.3,
            "c4_v_pull": -0.8, "c5_v_fill": -0.5, "c6_v_rest_neg": -0.3, "c7_a_pull": -0.2,
        },
        "signal_select": {"signal_name": "derivative"},
        "eval_params": {"tp_ticks": 8, "sl_ticks": 4, "cooldown_bins": 20, "warmup_bins": 300},
        "run_experiment": {"job_id": "mock-job", "status": "failed"},
        "promote_review": {"reviewed": True},
        "promotion": {"confirmed": True},
    }
    for step_name in STEPS_ORDERED:
        await service.commit_step(
            session, session_id=ms.session_id, step_name=step_name, payload=payloads[step_name],
        )
        await session.commit()

    with pytest.raises(ValueError, match="run_experiment step has status 'failed'"):
        await service.promote(session, session_id=ms.session_id, alias="test")


@pytest.mark.asyncio
async def test_promote_blocked_when_experiment_canceled(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
) -> None:
    """Promotion with run_experiment status='canceled' raises ValueError."""
    ms = await service.create_session(session, workspace_id=workspace.workspace_id)
    await session.commit()

    payloads: dict[str, dict[str, Any]] = {
        "dataset_select": {"dataset_id": "test_ds_abc", "pipeline": "baseline_mnq"},
        "gold_config": {
            "c1_v_add": 1.0, "c2_v_rest_pos": 0.5, "c3_a_add": 0.3,
            "c4_v_pull": -0.8, "c5_v_fill": -0.5, "c6_v_rest_neg": -0.3, "c7_a_pull": -0.2,
        },
        "signal_select": {"signal_name": "derivative"},
        "eval_params": {"tp_ticks": 8, "sl_ticks": 4, "cooldown_bins": 20, "warmup_bins": 300},
        "run_experiment": {"job_id": "mock-job", "status": "canceled"},
        "promote_review": {"reviewed": True},
        "promotion": {"confirmed": True},
    }
    for step_name in STEPS_ORDERED:
        await service.commit_step(
            session, session_id=ms.session_id, step_name=step_name, payload=payloads[step_name],
        )
        await session.commit()

    with pytest.raises(ValueError, match="run_experiment step has status 'canceled'"):
        await service.promote(session, session_id=ms.session_id, alias="test")


@pytest.mark.asyncio
async def test_synthesize_experiment_spec_writes_yaml(
    session: AsyncSession,
    workspace: Workspace,
    service: ModelingSessionService,
    lake_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """synthesize_and_submit_experiment writes serving + experiment YAMLs."""
    import unittest.mock as mock

    ms = await service.create_session(session, workspace_id=workspace.workspace_id)
    await session.commit()

    payloads: dict[str, dict[str, Any]] = {
        "dataset_select": {"dataset_id": "ds_test_xyz", "pipeline": "baseline_mnq"},
        "gold_config": {
            "c1_v_add": 1.0, "c2_v_rest_pos": 0.5, "c3_a_add": 0.3,
            "c4_v_pull": -0.8, "c5_v_fill": -0.5, "c6_v_rest_neg": -0.3, "c7_a_pull": -0.2,
        },
        "signal_select": {"signal_name": "derivative"},
        "eval_params": {"tp_ticks": 8, "sl_ticks": 4, "cooldown_bins": 20, "warmup_bins": 300},
    }
    for step_name in ["dataset_select", "gold_config", "signal_select", "eval_params"]:
        await service.commit_step(
            session, session_id=ms.session_id, step_name=step_name, payload=payloads[step_name],
        )
        await session.commit()

    # Mock the job queue to avoid real Redis
    mock_queue = mock.AsyncMock()
    mock_queue.enqueue = mock.AsyncMock()

    async def _mock_get_queue():
        return mock_queue

    monkeypatch.setattr(
        "src.jobs.queue.get_job_queue",
        _mock_get_queue,
        raising=False,
    )

    # Ensure workspace exists in DB
    from src.qmachina.db.models import Workspace as WS
    ws_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    ws = WS(workspace_id=ws_id, name="test-ws")
    session.add(ws)
    await session.commit()

    result = await service.synthesize_and_submit_experiment(
        session,
        session_id=ms.session_id,
        workspace_id=ws_id,
    )
    await session.commit()

    assert "job_id" in result
    assert "spec_ref" in result
    assert "spec_name" in result

    # Check serving YAML exists
    serving_yaml = (
        lake_root / "research" / "harness" / "configs" / "serving" / f"{result['spec_name']}.yaml"
    )
    assert serving_yaml.exists()

    # Check experiment YAML exists and parses
    from src.qmachina.experiment_config import ExperimentSpec
    exp_yaml = ExperimentSpec.configs_dir(lake_root) / f"{result['spec_name']}.yaml"
    assert exp_yaml.exists()
    exp_spec = ExperimentSpec.from_yaml(exp_yaml)
    assert exp_spec.inline_datasets == ["ds_test_xyz"]

    # Verify queue was called
    mock_queue.enqueue.assert_called_once()


def test_assemble_serving_spec_includes_gold_dsl_hash(
    service: ModelingSessionService,
) -> None:
    """_assemble_serving_spec produces ServingSpec with non-None gold_dsl_hash."""
    payloads: dict[str, dict[str, Any]] = {
        "dataset_select": {"dataset_id": "ds_test", "pipeline": "baseline"},
        "gold_config": {
            "c1_v_add": 1.0, "c2_v_rest_pos": 0.5, "c3_a_add": 0.3,
            "c4_v_pull": -0.8, "c5_v_fill": -0.5, "c6_v_rest_neg": -0.3, "c7_a_pull": -0.2,
        },
        "signal_select": {"signal_name": "derivative"},
        "eval_params": {"tp_ticks": 8, "sl_ticks": 4},
    }
    sid = uuid.uuid4()
    spec = service._assemble_serving_spec(session_id=sid, step_payloads=payloads)
    assert spec.gold_dsl_hash is not None
    assert spec.gold_dsl_spec_id is not None
    assert len(spec.gold_dsl_spec_id) == 16


def test_assemble_serving_spec_gold_dsl_non_fatal(
    service: ModelingSessionService,
) -> None:
    """_assemble_serving_spec succeeds even with garbage gold_payload (DSL wiring is non-fatal)."""
    payloads: dict[str, dict[str, Any]] = {
        "dataset_select": {"dataset_id": "ds_test", "pipeline": "baseline"},
        "gold_config": {"not_a_valid_field": "garbage_value"},
        "signal_select": {"signal_name": "derivative"},
        "eval_params": {"tp_ticks": 8, "sl_ticks": 4},
    }
    sid = uuid.uuid4()
    # Should not raise
    spec = service._assemble_serving_spec(session_id=sid, step_payloads=payloads)
    assert spec is not None
