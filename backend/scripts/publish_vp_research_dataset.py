"""Publish VP cache into immutable clean-grid data + agent workspaces.

This script enforces a clean data contract:
1) Immutable base data: bins + clean grid (no projection score columns)
2) Experiment metadata/workspaces: optional per-agent writable outputs

Usage examples:
    uv run scripts/publish_vp_research_dataset.py publish \
      --source-dir /tmp/vp_cache_mnqh6_20260206_0925_1025 \
      --dataset-id mnqh6_20260206_0925_1025 \
      --agents alpha,beta,gamma

    uv run scripts/publish_vp_research_dataset.py add-agents \
      --dataset-id mnqh6_20260206_0925_1025 \
      --agents delta,epsilon
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import duckdb
import pyarrow.parquet as pq

backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))

PROJ_PREFIX = "proj_score_h"
IMMUTABLE_DIRNAME = "vp_immutable"
EXPERIMENT_DIRNAME = "vp_experiments"
DEFAULT_RESEARCH_ROOT = backend_root / "lake" / "research"

SOURCE_BINS_NAME = "bins.parquet"
SOURCE_BUCKETS_NAME = "buckets.parquet"
SOURCE_MANIFEST_NAME = "manifest.json"

IMMUTABLE_BINS_NAME = "bins.parquet"
IMMUTABLE_GRID_NAME = "grid_clean.parquet"
IMMUTABLE_MANIFEST_NAME = "manifest.json"
IMMUTABLE_CHECKSUM_NAME = "checksums.json"

EXPERIMENT_MANIFEST_NAME = "manifest.json"
AGENTS_DIRNAME = "agents"


def _parse_agents(raw: str | None) -> list[str]:
    if raw is None:
        return []
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    if not tokens:
        return []
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        if "/" in token or token in {".", ".."}:
            raise ValueError(f"Invalid agent name: {token}")
        seen.add(token)
        deduped.append(token)
    return deduped


def _default_dataset_id(source_dir: Path) -> str:
    return source_dir.name


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _set_tree_read_only(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Cannot set read-only flags on missing path: {root}")

    # Ensure parent/root last to avoid blocking traversal while we apply permissions.
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            file_path.chmod(0o444)
        for dirname in dirnames:
            dir_path = Path(dirpath) / dirname
            dir_path.chmod(0o555)
        Path(dirpath).chmod(0o555)


def _quote_cols(cols: Sequence[str]) -> str:
    return ", ".join(f'"{col}"' for col in cols)


def _required_source_paths(source_dir: Path) -> tuple[Path, Path, Path]:
    bins = source_dir / SOURCE_BINS_NAME
    buckets = source_dir / SOURCE_BUCKETS_NAME
    manifest = source_dir / SOURCE_MANIFEST_NAME
    missing = [p for p in (bins, buckets, manifest) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Source cache directory is missing required files: "
            + ", ".join(str(p) for p in missing)
        )
    return bins, buckets, manifest


def _resolve_dataset_dirs(
    *,
    dataset_id: str,
    research_root: Path,
) -> tuple[Path, Path]:
    immutable_root = research_root / IMMUTABLE_DIRNAME / dataset_id
    experiment_root = research_root / EXPERIMENT_DIRNAME / dataset_id
    return immutable_root, experiment_root


def _projection_columns(buckets_path: Path) -> list[str]:
    schema = pq.read_schema(buckets_path)
    return [name for name in schema.names if name.startswith(PROJ_PREFIX)]


def _materialize_clean_grid_table(
    *,
    buckets_path: Path,
    clean_grid_path: Path,
    projection_cols: Sequence[str],
) -> None:
    schema = pq.read_schema(buckets_path)
    all_cols = list(schema.names)

    clean_cols = [col for col in all_cols if col not in set(projection_cols)]

    if not clean_cols:
        raise ValueError("Computed clean-grid column list is empty.")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=8")
    con.execute(
        f"COPY (SELECT {_quote_cols(clean_cols)} FROM read_parquet('{buckets_path}')) "
        f"TO '{clean_grid_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.close()


def _create_agent_workspaces(
    *,
    experiment_dataset_dir: Path,
    immutable_dataset_dir: Path,
    agents: Sequence[str],
) -> list[Path]:
    created: list[Path] = []
    if not agents:
        return created

    agents_root = experiment_dataset_dir / AGENTS_DIRNAME
    agents_root.mkdir(parents=True, exist_ok=True)

    for agent in agents:
        workspace = agents_root / agent
        if workspace.exists():
            raise FileExistsError(f"Agent workspace already exists: {workspace}")

        data_dir = workspace / "data"
        output_dir = workspace / "outputs"
        data_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)

        base_link = data_dir / "base_immutable"
        base_link.symlink_to(immutable_dataset_dir.resolve(), target_is_directory=True)

        created.append(workspace)

    return created


def publish_dataset(
    *,
    source_dir: Path,
    dataset_id: str,
    research_root: Path,
    agents: Sequence[str],
) -> dict:
    bins_src, buckets_src, manifest_src = _required_source_paths(source_dir)
    source_manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
    projection_cols = _projection_columns(buckets_src)

    immutable_dataset_dir, experiment_dataset_dir = _resolve_dataset_dirs(
        dataset_id=dataset_id,
        research_root=research_root,
    )
    if immutable_dataset_dir.exists():
        raise FileExistsError(f"Immutable dataset already exists: {immutable_dataset_dir}")
    if experiment_dataset_dir.exists():
        raise FileExistsError(f"Experiment dataset already exists: {experiment_dataset_dir}")

    immutable_dataset_dir.mkdir(parents=True, exist_ok=False)
    experiment_dataset_dir.mkdir(parents=True, exist_ok=False)

    immutable_bins = immutable_dataset_dir / IMMUTABLE_BINS_NAME
    immutable_grid = immutable_dataset_dir / IMMUTABLE_GRID_NAME
    immutable_manifest = immutable_dataset_dir / IMMUTABLE_MANIFEST_NAME
    immutable_checksums = immutable_dataset_dir / IMMUTABLE_CHECKSUM_NAME

    experiment_manifest = experiment_dataset_dir / EXPERIMENT_MANIFEST_NAME

    try:
        shutil.copy2(bins_src, immutable_bins)
        _materialize_clean_grid_table(
            buckets_path=buckets_src,
            clean_grid_path=immutable_grid,
            projection_cols=projection_cols,
        )

        bins_rows = int(pq.read_metadata(immutable_bins).num_rows)
        grid_rows = int(pq.read_metadata(immutable_grid).num_rows)

        checksum_payload = {
            "bins_parquet_sha256": _sha256(immutable_bins),
            "grid_clean_parquet_sha256": _sha256(immutable_grid),
        }
        _write_json(immutable_checksums, checksum_payload)

        immutable_payload = {
            "dataset_id": dataset_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(source_dir),
            "source_manifest": source_manifest,
            "rows": {
                "bins": bins_rows,
                "grid_clean": grid_rows,
            },
            "projection_columns_removed": list(projection_cols),
            "immutable_files": {
                "bins_parquet": str(immutable_bins),
                "grid_clean_parquet": str(immutable_grid),
                "checksums_json": str(immutable_checksums),
            },
        }
        _write_json(immutable_manifest, immutable_payload)

        _set_tree_read_only(immutable_dataset_dir)

        created_workspaces = _create_agent_workspaces(
            experiment_dataset_dir=experiment_dataset_dir,
            immutable_dataset_dir=immutable_dataset_dir,
            agents=agents,
        )

        experiment_payload = {
            "dataset_id": dataset_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "immutable_base_dir": str(immutable_dataset_dir),
            "agent_workspaces": [str(path) for path in created_workspaces],
        }
        _write_json(experiment_manifest, experiment_payload)

        return {
            "dataset_id": dataset_id,
            "immutable_dataset_dir": str(immutable_dataset_dir),
            "experiment_dataset_dir": str(experiment_dataset_dir),
            "rows": {
                "bins": bins_rows,
                "grid_clean": grid_rows,
            },
            "projection_columns_removed": list(projection_cols),
            "agent_workspaces": [str(path) for path in created_workspaces],
        }
    except Exception:
        if experiment_dataset_dir.exists():
            shutil.rmtree(experiment_dataset_dir, ignore_errors=True)
        if immutable_dataset_dir.exists():
            shutil.rmtree(immutable_dataset_dir, ignore_errors=True)
        raise


def add_agents(
    *,
    dataset_id: str,
    research_root: Path,
    agents: Sequence[str],
) -> dict:
    if not agents:
        raise ValueError("No agents provided for add-agents operation.")

    immutable_dataset_dir, experiment_dataset_dir = _resolve_dataset_dirs(
        dataset_id=dataset_id,
        research_root=research_root,
    )
    if not immutable_dataset_dir.exists():
        raise FileNotFoundError(f"Immutable dataset does not exist: {immutable_dataset_dir}")
    if not experiment_dataset_dir.exists():
        raise FileNotFoundError(f"Experiment dataset does not exist: {experiment_dataset_dir}")

    created = _create_agent_workspaces(
        experiment_dataset_dir=experiment_dataset_dir,
        immutable_dataset_dir=immutable_dataset_dir,
        agents=agents,
    )

    manifest_path = experiment_dataset_dir / EXPERIMENT_MANIFEST_NAME
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        payload = {"dataset_id": dataset_id}
    existing = list(payload.get("agent_workspaces", []))
    payload["agent_workspaces"] = existing + [str(path) for path in created]
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, payload)

    return {
        "dataset_id": dataset_id,
        "experiment_dataset_dir": str(experiment_dataset_dir),
        "created_agent_workspaces": [str(path) for path in created],
    }


def _cmd_publish(args: argparse.Namespace) -> None:
    dataset_id = args.dataset_id or _default_dataset_id(args.source_dir)
    agents = _parse_agents(args.agents)
    if not dataset_id:
        raise ValueError("dataset_id resolved to empty string.")

    result = publish_dataset(
        source_dir=args.source_dir,
        dataset_id=dataset_id,
        research_root=args.research_root,
        agents=agents,
    )
    print(json.dumps(result, indent=2))


def _cmd_add_agents(args: argparse.Namespace) -> None:
    agents = _parse_agents(args.agents)
    result = add_agents(
        dataset_id=args.dataset_id,
        research_root=args.research_root,
        agents=agents,
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish VP dataset into immutable clean grid + optional agent workspaces.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_publish = subparsers.add_parser(
        "publish",
        help="Publish source cache into immutable clean grid dataset and experiment workspaces.",
    )
    p_publish.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Source cache directory containing bins.parquet, buckets.parquet, manifest.json.",
    )
    p_publish.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Published dataset id (defaults to source directory name).",
    )
    p_publish.add_argument(
        "--research-root",
        type=Path,
        default=DEFAULT_RESEARCH_ROOT,
        help=f"Research root (immutable under {IMMUTABLE_DIRNAME}/, experiments under {EXPERIMENT_DIRNAME}/).",
    )
    p_publish.add_argument(
        "--agents",
        type=str,
        default=None,
        help="Optional comma-separated agent names to pre-create experiment workspaces.",
    )
    p_publish.set_defaults(func=_cmd_publish)

    p_add = subparsers.add_parser(
        "add-agents",
        help="Create additional agent workspaces for an already published dataset.",
    )
    p_add.add_argument("--dataset-id", required=True, help="Published dataset id.")
    p_add.add_argument(
        "--research-root",
        type=Path,
        default=DEFAULT_RESEARCH_ROOT,
        help=f"Research root (immutable under {IMMUTABLE_DIRNAME}/, experiments under {EXPERIMENT_DIRNAME}/).",
    )
    p_add.add_argument(
        "--agents",
        type=str,
        required=True,
        help="Comma-separated agent names.",
    )
    p_add.set_defaults(func=_cmd_add_agents)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
