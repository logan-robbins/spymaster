"""CLI module to discover and validate all YAML configs in the harness configs tree.

Usage (from backend/):
    uv run python -m src.qmachina.validate_configs
    uv run python -m src.qmachina.validate_configs --lake-root /custom/lake

Exit code 0 if all configs pass validation, 1 if any fail.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger: logging.Logger = logging.getLogger(__name__)

# Subdirectory -> loader function name mapping. Populated lazily to avoid
# import-time resolution of heavy modules.
_SUBDIR_LOADERS: dict[str, str] = {
    "serving": "_load_serving_spec",
    "pipelines": "_load_pipeline_spec",
    "experiments": "_load_experiment_spec",
    "gold_campaigns": "_load_yaml_only",
}


@dataclass
class ValidationResult:
    """Outcome of validating a single YAML file."""

    path: Path
    spec_type: str = ""
    status: str = ""  # "PASS" or "FAIL"
    error: str = ""


@dataclass
class ValidationSummary:
    """Aggregate results from a full config validation run."""

    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.total > 0


# ------------------------------------------------------------------
# Loader functions (each wraps a Pydantic model's from_yaml)
# ------------------------------------------------------------------


def _load_serving_spec(path: Path) -> None:
    """Validate a single ServingSpec YAML file.

    Raises on any validation failure (FileNotFoundError, yaml.YAMLError,
    pydantic.ValidationError).
    """
    from .serving_config import ServingSpec

    ServingSpec.from_yaml(path)


def _load_pipeline_spec(path: Path) -> None:
    """Validate a single PipelineSpec YAML file.

    Raises on any validation failure.
    """
    from .pipeline_config import PipelineSpec

    PipelineSpec.from_yaml(path)


def _load_experiment_spec(path: Path) -> None:
    """Validate a single ExperimentSpec YAML file.

    Raises on any validation failure.
    """
    from .experiment_config import ExperimentSpec

    ExperimentSpec.from_yaml(path)


def _load_legacy_experiment_config(path: Path) -> None:
    """Validate a legacy ExperimentConfig YAML file.

    Raises on any validation failure.
    """
    from ..experiment_harness.config_schema import ExperimentConfig

    ExperimentConfig.from_yaml(path)


def _load_yaml_only(path: Path) -> None:
    """Validate that a YAML file is loadable (no schema check).

    Used for directories without a Pydantic model (e.g. gold_campaigns).

    Raises on YAML parse failure.
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"YAML file is empty: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping, got {type(data).__name__}: {path}")


# ------------------------------------------------------------------
# Discovery and validation
# ------------------------------------------------------------------


def _get_loader(subdir: str) -> Callable[[Path], None]:
    """Return the appropriate loader function for a config subdirectory.

    Args:
        subdir: Name of the subdirectory under configs/ (e.g. "serving").

    Returns:
        Callable that takes a Path and raises on validation failure.
    """
    loader_name: str | None = _SUBDIR_LOADERS.get(subdir)
    if loader_name is None:
        return _load_legacy_experiment_config
    return globals()[loader_name]


def _spec_type_label(subdir: str) -> str:
    """Return a human-readable spec type label for reporting."""
    labels: dict[str, str] = {
        "serving": "ServingSpec",
        "pipelines": "PipelineSpec",
        "experiments": "ExperimentSpec",
        "gold_campaigns": "GoldCampaign (YAML-only)",
    }
    return labels.get(subdir, "ExperimentConfig (legacy)")


def validate_all_configs(lake_root: Path) -> ValidationSummary:
    """Discover and validate all YAML config files under the harness configs tree.

    Walks the directory tree starting at
    ``lake_root/research/harness/configs/``, identifies each YAML file,
    selects the appropriate Pydantic model based on its subdirectory,
    and attempts to load+validate it.

    Args:
        lake_root: Root of the data lake (contains ``research/harness/configs/``).

    Returns:
        ValidationSummary with per-file results.
    """
    configs_root: Path = lake_root / "research" / "harness" / "configs"
    if not configs_root.is_dir():
        logger.warning("Configs directory does not exist: %s", configs_root)
        return ValidationSummary()

    summary = ValidationSummary()

    # Walk all YAML files, skipping serving_versions (immutable published specs)
    yaml_files: list[Path] = sorted(configs_root.rglob("*.yaml"))

    for yaml_path in yaml_files:
        # Skip serving_versions directory (published specs, different schema)
        relative: Path = yaml_path.relative_to(configs_root)
        parts: tuple[str, ...] = relative.parts

        if parts[0] == "serving_versions":
            continue

        # Determine which subdirectory this file belongs to
        subdir: str = parts[0] if len(parts) > 1 else "__root__"

        loader: Callable[[Path], None] = _get_loader(subdir)
        spec_label: str = _spec_type_label(subdir)

        result = ValidationResult(
            path=yaml_path,
            spec_type=spec_label,
        )

        try:
            loader(yaml_path)
            result.status = "PASS"
        except Exception as exc:
            result.status = "FAIL"
            result.error = f"{type(exc).__name__}: {exc}"

        summary.results.append(result)

    return summary


def validate_serving_spec(path: Path) -> None:
    """Validate a single ServingSpec YAML and raise on failure.

    This is the canonical validation entry point used by the promotion
    preflight guard and the CLI.

    Args:
        path: Path to a ServingSpec YAML file.

    Raises:
        FileNotFoundError: If the path does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If the spec fails schema validation.
    """
    _load_serving_spec(path)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def _print_summary(summary: ValidationSummary) -> None:
    """Print structured validation summary to stdout."""
    if not summary.results:
        print("No config files found.")
        return

    max_path_len: int = max(len(str(r.path.name)) for r in summary.results)

    print()
    print("=" * 72)
    print("Config Validation Results")
    print("=" * 72)

    for r in summary.results:
        status_marker: str = "PASS" if r.status == "PASS" else "FAIL"
        line: str = f"  [{status_marker}] {r.path.name:<{max_path_len}}  ({r.spec_type})"
        print(line)
        if r.error:
            # Truncate long error messages for readability
            error_preview: str = r.error[:200]
            if len(r.error) > 200:
                error_preview += "..."
            print(f"         {error_preview}")

    print()
    print("-" * 72)
    print(f"  Total: {summary.total}  Passed: {summary.passed}  Failed: {summary.failed}")
    print("-" * 72)
    print()


def main() -> None:
    """CLI entry point for config validation."""
    parser = argparse.ArgumentParser(
        description="Validate all qMachina YAML configs."
    )
    parser.add_argument(
        "--lake-root",
        default="",
        help="Override lake root path (default: backend/lake).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    backend_root: Path = Path(__file__).resolve().parents[2]
    lake_root: Path = (
        Path(args.lake_root) if args.lake_root else backend_root / "lake"
    )

    summary: ValidationSummary = validate_all_configs(lake_root)
    _print_summary(summary)

    if not summary.all_passed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
