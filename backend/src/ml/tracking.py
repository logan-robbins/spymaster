"""
Shared experiment tracking helpers for MLflow + W&B.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterator, List

import mlflow
import wandb


@dataclass(frozen=True)
class TrackingRun:
    run_name: str
    wandb_run: wandb.sdk.wandb_run.Run


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_git_sha(repo_root: Path) -> str:
    head_path = repo_root / ".git" / "HEAD"
    if not head_path.exists():
        raise FileNotFoundError(f"git HEAD not found at {head_path}")
    head = head_path.read_text().strip()
    if head.startswith("ref: "):
        ref_path = repo_root / ".git" / head.split(" ", 1)[1].strip()
        if not ref_path.exists():
            raise FileNotFoundError(f"git ref not found at {ref_path}")
        return ref_path.read_text().strip()
    return head


def hash_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_params(params: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, (list, tuple, set)):
            normalized[key] = ",".join(str(v) for v in value)
        elif isinstance(value, dict):
            normalized[key] = json.dumps(value, sort_keys=True)
        else:
            normalized[key] = value
    return normalized


def ensure_wandb_config(project: str, repo_root: Path) -> None:
    if os.getenv("WANDB_MODE") == "offline":
        os.environ.setdefault("WANDB_PROJECT", project)
        return

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        key_path = repo_root / "wandb.txt"
        if not key_path.exists():
            raise EnvironmentError("WANDB_API_KEY not set and wandb.txt not found")
        api_key = key_path.read_text().strip()
        if not api_key:
            raise EnvironmentError("wandb.txt is empty")
        os.environ["WANDB_API_KEY"] = api_key

    os.environ.setdefault("WANDB_PROJECT", project)


@contextmanager
def tracking_run(
    *,
    run_name: str,
    experiment: str,
    params: Dict[str, object],
    tags: Dict[str, str],
    wandb_tags: List[str],
    project: str,
    repo_root: Path,
) -> Iterator[TrackingRun]:
    ensure_wandb_config(project, repo_root)
    mlflow.set_experiment(experiment)
    normalized_params = _normalize_params(params)

    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)
        if normalized_params:
            mlflow.log_params(normalized_params)
        entity = os.getenv("WANDB_ENTITY")
        wandb_kwargs = {
            "project": project,
            "name": run_name,
            "config": normalized_params,
            "tags": wandb_tags,
            "reinit": True,
        }
        if entity:
            wandb_kwargs["entity"] = entity
        wandb_run = wandb.init(**wandb_kwargs)
        try:
            yield TrackingRun(run_name=run_name, wandb_run=wandb_run)
        finally:
            wandb_run.finish()


def log_metrics(
    metrics: Dict[str, float],
    wandb_run: wandb.sdk.wandb_run.Run,
    *,
    step: int | None = None,
) -> None:
    filtered: Dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            filtered[key] = float(value)

    for key, value in filtered.items():
        mlflow.log_metric(key, value, step=step)

    if filtered:
        wandb_run.log(filtered, step=step)


def log_artifacts(
    paths: List[Path],
    *,
    name: str,
    artifact_type: str,
    wandb_run: wandb.sdk.wandb_run.Run,
) -> None:
    if not paths:
        return

    artifact = wandb.Artifact(name=name, type=artifact_type)
    for path in paths:
        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=name)
            artifact.add_dir(str(path))
        else:
            mlflow.log_artifact(str(path), artifact_path=name)
            artifact.add_file(str(path))

    wandb_run.log_artifact(artifact)
