"""Golden compatibility tests for qMachina YAML config validation.

Ensures all checked-in YAML configs parse cleanly against their respective
Pydantic models, and that the validate_configs CLI produces exit code 0.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LAKE_ROOT: Path = BACKEND_ROOT / "lake"
CONFIGS_ROOT: Path = LAKE_ROOT / "research" / "harness" / "configs"


@pytest.fixture()
def serving_dir() -> Path:
    return CONFIGS_ROOT / "serving"


@pytest.fixture()
def pipelines_dir() -> Path:
    return CONFIGS_ROOT / "pipelines"


@pytest.fixture()
def experiments_dir() -> Path:
    return CONFIGS_ROOT / "experiments"


# ---------------------------------------------------------------------------
# Test 1: All YAML files parse cleanly
# ---------------------------------------------------------------------------


def test_all_yaml_files_parse_cleanly() -> None:
    """Every YAML file under configs/ must be parseable by validate_all_configs."""
    from src.qmachina.validate_configs import validate_all_configs

    summary = validate_all_configs(LAKE_ROOT)
    assert summary.total > 0, "Expected at least one config file."
    failures = [r for r in summary.results if r.status == "FAIL"]
    if failures:
        msg_lines = [f"  {r.path.name}: {r.error}" for r in failures]
        pytest.fail(
            f"{len(failures)} config(s) failed validation:\n"
            + "\n".join(msg_lines)
        )


# ---------------------------------------------------------------------------
# Test 2: derivative_baseline.yaml stream_schema roles are valid
# ---------------------------------------------------------------------------


def test_derivative_baseline_stream_roles_are_valid(serving_dir: Path) -> None:
    """derivative_baseline.yaml must have all stream_schema roles as valid StreamFieldRole values."""
    from src.qmachina.serving_config import ServingSpec, StreamFieldRole

    spec = ServingSpec.from_yaml(serving_dir / "derivative_baseline.yaml")
    valid_roles = set(StreamFieldRole)
    for field in spec.stream_schema:
        if field.role is not None:
            assert field.role in valid_roles, (
                f"Field '{field.name}' has invalid role '{field.role}'. "
                f"Valid: {[r.value for r in StreamFieldRole]}"
            )


# ---------------------------------------------------------------------------
# Test 3: ema_ensemble_baseline.yaml parses cleanly
# ---------------------------------------------------------------------------


def test_ema_ensemble_baseline_parses(serving_dir: Path) -> None:
    """ema_ensemble_baseline.yaml must load as a valid ServingSpec."""
    from src.qmachina.serving_config import ServingSpec

    spec = ServingSpec.from_yaml(serving_dir / "ema_ensemble_baseline.yaml")
    assert spec.name == "ema_ensemble_baseline"
    assert spec.model_id == "ema_ensemble"
    assert spec.ema_config is not None
    assert len(spec.stream_schema) == 4


# ---------------------------------------------------------------------------
# Test 4: mnq_60m_baseline.yaml parses as valid PipelineSpec
# ---------------------------------------------------------------------------


def test_mnq_60m_baseline_parses(pipelines_dir: Path) -> None:
    """mnq_60m_baseline.yaml must load as a valid PipelineSpec."""
    from src.qmachina.pipeline_config import PipelineSpec

    spec = PipelineSpec.from_yaml(pipelines_dir / "mnq_60m_baseline.yaml")
    assert spec.name == "mnq_60m_baseline"
    assert spec.capture.symbol == "MNQH6"
    assert spec.capture.dt == "2026-02-06"


# ---------------------------------------------------------------------------
# Test 5: sweep_derivative_rr20.yaml parses as valid ExperimentSpec
# ---------------------------------------------------------------------------


def test_sweep_derivative_rr20_parses(experiments_dir: Path) -> None:
    """sweep_derivative_rr20.yaml must load as a valid ExperimentSpec."""
    from src.qmachina.experiment_config import ExperimentSpec

    spec = ExperimentSpec.from_yaml(
        experiments_dir / "sweep_derivative_rr20.yaml"
    )
    assert spec.name == "sweep_derivative_rr20"
    assert spec.serving == "derivative_baseline"
    assert spec.eval.tp_ticks == [8]
    assert spec.eval.sl_ticks == [4]


# ---------------------------------------------------------------------------
# Test 6: validate_configs CLI exits 0 for current configs
# ---------------------------------------------------------------------------


def test_validate_configs_cli_exits_zero() -> None:
    """The validate_configs CLI must exit 0 when all current configs are valid."""
    result = subprocess.run(
        [sys.executable, "-m", "src.qmachina.validate_configs"],
        capture_output=True,
        text=True,
        cwd=str(BACKEND_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, (
        f"validate_configs exited with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 7: deliberately broken YAML causes CLI to exit 1
# ---------------------------------------------------------------------------


def test_validate_configs_cli_exits_one_on_broken_yaml() -> None:
    """A malformed serving YAML must cause the CLI to exit with code 1."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Replicate the directory structure
        serving_dir = tmp_path / "research" / "harness" / "configs" / "serving"
        serving_dir.mkdir(parents=True)

        broken_yaml = serving_dir / "broken_test.yaml"
        broken_yaml.write_text(
            "name: broken_test\n"
            "pipeline: nonexistent_pipeline\n"
            "stream_schema:\n"
            '  - { name: k, dtype: int32, role: invalid_role }\n'
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.qmachina.validate_configs",
                "--lake-root",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(BACKEND_ROOT),
            timeout=60,
        )
        assert result.returncode == 1, (
            f"Expected exit code 1, got {result.returncode}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Test 8: snapshot - derivative_baseline has expected number of stream fields
# ---------------------------------------------------------------------------


def test_derivative_baseline_stream_field_count(serving_dir: Path) -> None:
    """derivative_baseline.yaml must have exactly 40 stream fields (snapshot guard)."""
    from src.qmachina.serving_config import ServingSpec

    spec = ServingSpec.from_yaml(serving_dir / "derivative_baseline.yaml")
    assert len(spec.stream_schema) == 40, (
        f"Expected 40 stream fields in derivative_baseline, got {len(spec.stream_schema)}. "
        "If this changed intentionally, update the snapshot count."
    )
