"""Tests for the serving lifecycle API and diff service.

Covers:
    1. List versions endpoint — returns array (may be empty)
    2. List aliases endpoint — returns array
    3. Get version by ID — 404 for nonexistent
    4. Get alias — 404 for nonexistent
    5. diff_runtime_snapshots — identical snapshots
    6. diff_runtime_snapshots — one key changed
    7. diff_runtime_snapshots — key added in b
    8. Activate alias endpoint — with fixture serving version
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.qmachina.api_serving import create_serving_router
from src.qmachina.serving_config import PublishedServingSource, PublishedServingSpec
from src.qmachina.serving_diff import diff_runtime_snapshots
from src.qmachina.serving_registry import ServingRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_snapshot(**overrides: Any) -> dict[str, Any]:
    """Return a minimal runtime snapshot dict for testing."""
    base: dict[str, Any] = {
        "symbol": "MNQH6",
        "product_type": "future_mbo",
        "tick_size": 0.25,
        "grid_radius_ticks": 50,
        "cell_width_ms": 100,
        "stream_dt": "2026-02-06",
        "stream_start_time": "09:25",
    }
    base.update(overrides)
    return base


def _make_published_spec(
    *,
    serving_id: str,
    run_id: str = "run_test_001",
    experiment_name: str = "test_experiment",
    config_hash: str = "cfghash001",
    description: str = "test version",
    snapshot_overrides: dict[str, Any] | None = None,
) -> PublishedServingSpec:
    overrides = snapshot_overrides or {}
    return PublishedServingSpec(
        serving_id=serving_id,
        description=description,
        runtime_snapshot=_minimal_snapshot(**overrides),
        source=PublishedServingSource(
            run_id=run_id,
            experiment_name=experiment_name,
            config_hash=config_hash,
            promoted_at_utc="2026-02-20T00:00:00+00:00",
            serving_spec_name="test_serving",
            signal_name="derivative",
        ),
    )


@pytest.fixture()
def registry_env(tmp_path: Path) -> tuple[TestClient, ServingRegistry, Path]:
    """Create a test app with the serving router wired to a temp registry."""
    registry = ServingRegistry(tmp_path)
    app = FastAPI()
    serving_router = create_serving_router(
        lake_root=tmp_path,
        serving_registry=registry,
    )
    app.include_router(serving_router)
    client = TestClient(app)
    return client, registry, tmp_path


@pytest.fixture()
def seeded_env(
    registry_env: tuple[TestClient, ServingRegistry, Path],
) -> tuple[TestClient, ServingRegistry, str]:
    """Seed the registry with one serving version and alias."""
    client, registry, tmp_path = registry_env
    serving_id = "srv_test_exp_run_test_cfghash0"
    spec = _make_published_spec(serving_id=serving_id)
    registry.promote(alias="test_alias", spec=spec, actor="test_setup")
    return client, registry, serving_id


# ---------------------------------------------------------------------------
# 1. List versions endpoint
# ---------------------------------------------------------------------------


def test_list_versions_empty(
    registry_env: tuple[TestClient, ServingRegistry, Path],
) -> None:
    client, _, _ = registry_env
    resp = client.get("/v1/serving/versions")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_list_versions_with_data(
    seeded_env: tuple[TestClient, ServingRegistry, str],
) -> None:
    client, _, serving_id = seeded_env
    resp = client.get("/v1/serving/versions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    ids = [v["serving_id"] for v in data]
    assert serving_id in ids


# ---------------------------------------------------------------------------
# 2. List aliases endpoint
# ---------------------------------------------------------------------------


def test_list_aliases_empty(
    registry_env: tuple[TestClient, ServingRegistry, Path],
) -> None:
    client, _, _ = registry_env
    resp = client.get("/v1/serving/aliases")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_list_aliases_with_data(
    seeded_env: tuple[TestClient, ServingRegistry, str],
) -> None:
    client, _, serving_id = seeded_env
    resp = client.get("/v1/serving/aliases")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    aliases = [a["alias"] for a in data]
    assert "test_alias" in aliases


# ---------------------------------------------------------------------------
# 3. Get version by ID — 404 for nonexistent
# ---------------------------------------------------------------------------


def test_get_version_not_found(
    registry_env: tuple[TestClient, ServingRegistry, Path],
) -> None:
    client, _, _ = registry_env
    resp = client.get("/v1/serving/versions/srv_nonexistent_00000")
    assert resp.status_code == 404


def test_get_version_found(
    seeded_env: tuple[TestClient, ServingRegistry, str],
) -> None:
    client, _, serving_id = seeded_env
    resp = client.get(f"/v1/serving/versions/{serving_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["serving_id"] == serving_id
    assert "runtime_snapshot" in data
    assert data["runtime_snapshot"]["symbol"] == "MNQH6"


# ---------------------------------------------------------------------------
# 4. Get alias — 404 for nonexistent
# ---------------------------------------------------------------------------


def test_get_alias_not_found(
    registry_env: tuple[TestClient, ServingRegistry, Path],
) -> None:
    client, _, _ = registry_env
    resp = client.get("/v1/serving/aliases/nonexistent_alias")
    assert resp.status_code == 404


def test_get_alias_found(
    seeded_env: tuple[TestClient, ServingRegistry, str],
) -> None:
    client, _, serving_id = seeded_env
    resp = client.get("/v1/serving/aliases/test_alias")
    assert resp.status_code == 200
    data = resp.json()
    assert data["alias"] == "test_alias"
    assert data["serving_id"] == serving_id
    assert isinstance(data["history"], list)
    assert len(data["history"]) >= 1


# ---------------------------------------------------------------------------
# 5. diff_runtime_snapshots — identical snapshots
# ---------------------------------------------------------------------------


def test_diff_identical_snapshots() -> None:
    snap = _minimal_snapshot()
    result = diff_runtime_snapshots(
        snapshot_a=snap,
        snapshot_b=dict(snap),
        serving_id_a="srv_a",
        serving_id_b="srv_b",
    )
    assert result["added"] == {}
    assert result["removed"] == {}
    assert result["changed"] == {}
    assert result["unchanged_count"] == len(snap)
    assert "srv_a vs srv_b" in result["summary"]


# ---------------------------------------------------------------------------
# 6. diff_runtime_snapshots — one key changed
# ---------------------------------------------------------------------------


def test_diff_one_key_changed() -> None:
    snap_a = _minimal_snapshot()
    snap_b = _minimal_snapshot(tick_size=0.50)
    result = diff_runtime_snapshots(
        snapshot_a=snap_a,
        snapshot_b=snap_b,
        serving_id_a="srv_a",
        serving_id_b="srv_b",
    )
    assert "tick_size" in result["changed"]
    assert result["changed"]["tick_size"]["a"] == 0.25
    assert result["changed"]["tick_size"]["b"] == 0.50
    assert result["unchanged_count"] == len(snap_a) - 1


# ---------------------------------------------------------------------------
# 7. diff_runtime_snapshots — key added in b
# ---------------------------------------------------------------------------


def test_diff_key_added_in_b() -> None:
    snap_a = _minimal_snapshot()
    snap_b = _minimal_snapshot(new_field="hello")
    result = diff_runtime_snapshots(
        snapshot_a=snap_a,
        snapshot_b=snap_b,
        serving_id_a="srv_a",
        serving_id_b="srv_b",
    )
    assert "new_field" in result["added"]
    assert result["added"]["new_field"] == "hello"
    assert result["removed"] == {}


def test_diff_key_removed_in_b() -> None:
    snap_a = _minimal_snapshot(old_field="gone")
    snap_b = _minimal_snapshot()
    result = diff_runtime_snapshots(
        snapshot_a=snap_a,
        snapshot_b=snap_b,
        serving_id_a="srv_a",
        serving_id_b="srv_b",
    )
    assert "old_field" in result["removed"]
    assert result["removed"]["old_field"] == "gone"


# ---------------------------------------------------------------------------
# 8. Activate alias endpoint — with fixture serving version
# ---------------------------------------------------------------------------


def test_activate_alias_success(
    seeded_env: tuple[TestClient, ServingRegistry, str],
) -> None:
    client, _, serving_id = seeded_env
    resp = client.post(
        "/v1/serving/aliases/new_alias/activate",
        json={"serving_id": serving_id, "reason": "testing activation"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["alias"] == "new_alias"
    assert data["serving_id"] == serving_id
    assert data["from_serving_id"] is None
    assert data["reason"] == "testing activation"

    # Verify alias is now resolvable.
    resp2 = client.get("/v1/serving/aliases/new_alias")
    assert resp2.status_code == 200
    assert resp2.json()["serving_id"] == serving_id


def test_activate_alias_nonexistent_version(
    registry_env: tuple[TestClient, ServingRegistry, Path],
) -> None:
    client, _, _ = registry_env
    resp = client.post(
        "/v1/serving/aliases/any_alias/activate",
        json={"serving_id": "srv_does_not_exist"},
    )
    assert resp.status_code == 404


def test_activate_alias_repoint(
    seeded_env: tuple[TestClient, ServingRegistry, str],
) -> None:
    """Activate an existing alias to a new version, then verify from_serving_id."""
    client, registry, first_id = seeded_env

    # Create a second serving version.
    second_id = "srv_test_exp2_run2_cfg2hash"
    spec2 = _make_published_spec(
        serving_id=second_id,
        run_id="run_test_002",
        config_hash="cfghash002",
        description="second version",
    )
    registry.promote(alias="test_alias", spec=spec2, actor="test_setup")

    # Now repoint test_alias back to first_id via the API.
    resp = client.post(
        "/v1/serving/aliases/test_alias/activate",
        json={"serving_id": first_id, "reason": "rollback test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["from_serving_id"] == second_id
    assert data["serving_id"] == first_id


# ---------------------------------------------------------------------------
# Alias history endpoint
# ---------------------------------------------------------------------------


def test_alias_history(
    seeded_env: tuple[TestClient, ServingRegistry, str],
) -> None:
    client, _, serving_id = seeded_env
    resp = client.get("/v1/serving/aliases/test_alias/history")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["to_serving_id"] == serving_id


def test_alias_history_not_found(
    registry_env: tuple[TestClient, ServingRegistry, Path],
) -> None:
    client, _, _ = registry_env
    resp = client.get("/v1/serving/aliases/ghost/history")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Diff endpoint (API level)
# ---------------------------------------------------------------------------


def test_diff_endpoint(
    seeded_env: tuple[TestClient, ServingRegistry, str],
) -> None:
    """Diff a version with itself should show no changes."""
    client, _, serving_id = seeded_env
    resp = client.post(
        "/v1/serving/diff",
        json={"serving_id_a": serving_id, "serving_id_b": serving_id},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["added"] == {}
    assert data["removed"] == {}
    assert data["changed"] == {}
    assert data["unchanged_count"] > 0


def test_diff_endpoint_not_found(
    registry_env: tuple[TestClient, ServingRegistry, Path],
) -> None:
    client, _, _ = registry_env
    resp = client.post(
        "/v1/serving/diff",
        json={"serving_id_a": "srv_nope", "serving_id_b": "srv_nada"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# ExperimentSpec inline_datasets
# ---------------------------------------------------------------------------


def test_experiment_spec_inline_datasets_validates() -> None:
    """ExperimentSpec with inline_datasets validates without error."""
    from src.qmachina.experiment_config import ExperimentSpec

    spec = ExperimentSpec(
        name="test_inline",
        serving="some_serving_spec",
        inline_datasets=["ds_abc", "ds_def"],
    )
    assert spec.inline_datasets == ["ds_abc", "ds_def"]


def test_experiment_spec_to_runner_config_uses_inline_datasets(tmp_path: Path) -> None:
    """to_runner_config() with inline_datasets returns correct dataset IDs without pipeline resolution."""
    from unittest.mock import MagicMock, patch
    from src.qmachina.experiment_config import ExperimentSpec

    spec = ExperimentSpec(
        name="test_inline_runner",
        serving="some_serving_spec",
        inline_datasets=["ds_xyz"],
    )

    # Mock resolve_serving at class level to return a minimal serving spec mock
    mock_serving = MagicMock()
    mock_serving.signal = None

    with patch.object(ExperimentSpec, "resolve_serving", return_value=mock_serving):
        runner_config = spec.to_runner_config(tmp_path)

    assert runner_config["datasets"] == ["ds_xyz"]
    # resolve_pipeline must NOT have been called
    mock_serving.resolve_pipeline.assert_not_called()
