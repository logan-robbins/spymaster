"""Register a ServingSpec YAML as a published serving version.

Usage:
    uv run scripts/register_serving.py <serving_spec_name> [--alias ALIAS]

Example:
    uv run scripts/register_serving.py ema_ensemble_baseline
    uv run scripts/register_serving.py ema_ensemble_baseline --alias ema_dev

The spec is loaded from lake/research/harness/configs/serving/<name>.yaml,
resolved against the pipeline config, and registered in the serving registry
so the server can stream it via ?serving=<alias_or_id>.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a ServingSpec YAML as a published serving version.")
    parser.add_argument("name", help="ServingSpec name (YAML file stem in configs/serving/)")
    parser.add_argument("--alias", default="", help="Alias to register (default: same as name)")
    parser.add_argument("--lake-root", default="", help="Override lake root path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    lake_root = Path(args.lake_root) if args.lake_root else backend_root / "lake"
    alias = args.alias.strip() or args.name.strip()

    from src.qmachina.config import build_config_with_overrides
    from src.qmachina.serving_config import (
        PublishedServingSource,
        PublishedServingSpec,
        ServingSpec,
    )
    from src.qmachina.serving_registry import ServingRegistry, validate_serving_spec_preflight

    spec = ServingSpec.load_by_name(args.name.strip(), lake_root)

    # Preflight validation: fail fast before any pipeline resolution or DB writes.
    validate_serving_spec_preflight(spec)
    pipeline_spec = spec.resolve_pipeline(lake_root)
    runtime_base = pipeline_spec.resolve_runtime_config()
    serving_runtime_fields = spec.to_runtime_fields(cell_width_ms=runtime_base.cell_width_ms)
    effective_runtime = build_config_with_overrides(runtime_base, serving_runtime_fields)

    runtime_snapshot = effective_runtime.to_dict()
    runtime_snapshot["stream_dt"] = pipeline_spec.capture.dt
    runtime_snapshot["stream_start_time"] = pipeline_spec.capture.start_time

    # Include stream_schema in snapshot (parsed by stream_session.py)
    if spec.stream_schema:
        runtime_snapshot["stream_schema"] = [
            {"name": f.name, "dtype": f.dtype, "role": f.role.value if f.role else None}
            for f in spec.stream_schema
        ]

    # Include visualization config in snapshot
    runtime_snapshot["visualization"] = spec.visualization.model_dump()

    # Include ema_config in snapshot if present
    if spec.ema_config is not None:
        runtime_snapshot["ema_config"] = spec.ema_config.model_dump()

    config_hash = hashlib.sha256(
        f"{args.name}:{effective_runtime.config_version}".encode()
    ).hexdigest()[:16]

    run_id = f"register:{args.name}"
    promoted_at_utc = datetime.now(tz=timezone.utc).isoformat()

    registry = ServingRegistry(lake_root)
    serving_id = registry.build_serving_id(
        experiment_name=args.name,
        run_id=run_id,
        config_hash=config_hash,
    )

    published = PublishedServingSpec(
        serving_id=serving_id,
        description=spec.description or f"Registered from {args.name}.yaml",
        runtime_snapshot=runtime_snapshot,
        source=PublishedServingSource(
            run_id=run_id,
            experiment_name=args.name,
            config_hash=config_hash,
            promoted_at_utc=promoted_at_utc,
            serving_spec_name=spec.name,
        ),
    )

    result = registry.promote(alias=alias, spec=published, actor="register_serving.py")

    print(f"\nServing registered:")
    print(f"  serving_id: {result.serving_id}")
    print(f"  alias:      {alias}")
    print(f"  model_id:   {spec.model_id}")
    print(f"  reused:     {result.reused_existing}")
    print(f"\nStream URL: ws://localhost:8002/v1/stream?serving={alias}")
    print(f"Frontend:   http://localhost:5174/vp-stream.html?serving={alias}")


if __name__ == "__main__":
    main()
