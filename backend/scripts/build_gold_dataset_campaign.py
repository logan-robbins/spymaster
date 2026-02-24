"""Build multiple immutable gold datasets from parameterized runtime inputs.

This script executes a campaign defined in YAML:
1. Expand cartesian sweep axes (+ optional grouped bundles).
2. Build runtime config overrides per variant.
3. Capture full VP output from raw replay.
4. Publish immutable dataset + experiment workspaces.
5. Write campaign index JSON with variant lineage.

Usage:
    uv run scripts/build_gold_dataset_campaign.py \
      --config lake/research/vp_harness/configs/gold_campaigns/mnq_grid_spectrum_force_v1.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))

from scripts.cache_output import (  # noqa: E402
    _parse_et_timestamp_ns,
    _stream_start_hhmm,
    capture_stream_output,
)
from scripts.publish_dataset import (  # noqa: E402
    DEFAULT_RESEARCH_ROOT,
    publish_dataset,
)
from src.qmachina.config import (  # noqa: E402
    RuntimeConfig,
    build_config_with_overrides,
    resolve_config,
)

logger = logging.getLogger("build_gold_dataset_campaign")

_PROJECTION_KEYS: set[str] = {
    "projection_use_cubic",
    "projection_cubic_scale",
    "projection_damping_lambda",
}

_RUNTIME_DISALLOWED_OVERRIDE_KEYS: set[str] = {
    "config_version",
    "projection_horizons_bins",
    "projection_horizons_ms",
    "symbol",
    "product_type",
    "symbol_root",
}


@dataclass(frozen=True)
class BaseCapture:
    product_type: str
    symbol: str
    dt: str
    capture_start_et: str
    capture_end_et: str


@dataclass(frozen=True)
class PublishConfig:
    agents: tuple[str, ...]


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    base_capture: BaseCapture
    publish: PublishConfig
    sweep_axes: Mapping[str, list[Any]]
    bundles: tuple[Mapping[str, Any], ...]


def _runtime_override_keys_from_config(cfg: RuntimeConfig) -> set[str]:
    keys = set(cfg.to_dict().keys())
    return keys - _RUNTIME_DISALLOWED_OVERRIDE_KEYS


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "campaign"


def _parse_agents(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return tuple()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("publish.agents must be a list of strings")

    result: list[str] = []
    seen: set[str] = set()
    for value in raw:
        agent = str(value).strip()
        if not agent:
            continue
        if agent in seen:
            continue
        if "/" in agent or agent in {".", ".."}:
            raise ValueError(f"Invalid agent name: {agent}")
        seen.add(agent)
        result.append(agent)
    return tuple(result)


def _parse_campaign(path: Path) -> CampaignConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Campaign config must be a mapping: {path}")

    if "name" not in raw:
        raise ValueError("Campaign config missing required field: name")
    if "base_capture" not in raw:
        raise ValueError("Campaign config missing required field: base_capture")

    bc_raw = raw["base_capture"]
    if not isinstance(bc_raw, dict):
        raise ValueError("base_capture must be a mapping")

    for key in (
        "product_type",
        "symbol",
        "dt",
        "capture_start_et",
        "capture_end_et",
    ):
        if key not in bc_raw:
            raise ValueError(f"base_capture missing required field: {key}")

    publish_raw = raw.get("publish", {})
    if not isinstance(publish_raw, dict):
        raise ValueError("publish must be a mapping")

    sweep_axes_raw = raw.get("sweep_axes", {})
    if not isinstance(sweep_axes_raw, dict):
        raise ValueError("sweep_axes must be a mapping of key -> non-empty list")

    sweep_axes: dict[str, list[Any]] = {}
    for key, value in sweep_axes_raw.items():
        if not isinstance(value, list) or not value:
            raise ValueError(f"sweep_axes.{key} must be a non-empty list")
        sweep_axes[str(key)] = list(value)

    bundles_raw = raw.get("bundles", [{}])
    if not isinstance(bundles_raw, list) or not bundles_raw:
        raise ValueError("bundles must be a non-empty list of mappings")

    bundles: list[Mapping[str, Any]] = []
    for idx, entry in enumerate(bundles_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"bundles[{idx}] must be a mapping")
        bundles.append(dict(entry))

    return CampaignConfig(
        name=str(raw["name"]).strip(),
        base_capture=BaseCapture(
            product_type=str(bc_raw["product_type"]).strip(),
            symbol=str(bc_raw["symbol"]).strip(),
            dt=str(bc_raw["dt"]).strip(),
            capture_start_et=str(bc_raw["capture_start_et"]).strip(),
            capture_end_et=str(bc_raw["capture_end_et"]).strip(),
        ),
        publish=PublishConfig(agents=_parse_agents(publish_raw.get("agents"))),
        sweep_axes=sweep_axes,
        bundles=tuple(bundles),
    )


def _expand_sweep_axes(sweep_axes: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    if not sweep_axes:
        return [{}]

    keys = sorted(sweep_axes.keys())
    value_lists = [sweep_axes[k] for k in keys]
    expanded: list[dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        expanded.append(dict(zip(keys, combo)))
    return expanded


def _expand_variants(campaign: CampaignConfig) -> list[dict[str, Any]]:
    axis_combos = _expand_sweep_axes(campaign.sweep_axes)
    variants: list[dict[str, Any]] = []
    for axis_combo in axis_combos:
        for bundle in campaign.bundles:
            merged = dict(axis_combo)
            merged.update(bundle)
            variants.append(merged)
    return variants


def _validate_variant_keys(
    variants: Sequence[Mapping[str, Any]],
    runtime_allowed: set[str],
) -> None:
    allowed = runtime_allowed | _PROJECTION_KEYS
    unknown: set[str] = set()
    for variant in variants:
        for key in variant.keys():
            if key not in allowed:
                unknown.add(str(key))
    if unknown:
        raise ValueError(
            f"Unknown sweep/bundle keys: {sorted(unknown)}. "
            f"Allowed keys: {sorted(allowed)}"
        )


def _extract_projection_params(variant: Mapping[str, Any]) -> tuple[bool, float, float]:
    use_cubic = bool(variant.get("projection_use_cubic", False))
    cubic_scale = float(variant.get("projection_cubic_scale", 1.0 / 6.0))
    damping_lambda = float(variant.get("projection_damping_lambda", 0.0))
    if cubic_scale < 0.0:
        raise ValueError(f"projection_cubic_scale must be >= 0, got {cubic_scale}")
    if damping_lambda < 0.0:
        raise ValueError(
            f"projection_damping_lambda must be >= 0, got {damping_lambda}"
        )
    return use_cubic, cubic_scale, damping_lambda


def _extract_runtime_overrides(
    variant: Mapping[str, Any],
    runtime_allowed: set[str],
) -> dict[str, Any]:
    return {k: v for k, v in variant.items() if k in runtime_allowed}


def _variant_hash(
    *,
    campaign: CampaignConfig,
    variant: Mapping[str, Any],
) -> str:
    payload = {
        "campaign": campaign.name,
        "base_capture": {
            "product_type": campaign.base_capture.product_type,
            "symbol": campaign.base_capture.symbol,
            "dt": campaign.base_capture.dt,
            "capture_start_et": campaign.base_capture.capture_start_et,
            "capture_end_et": campaign.base_capture.capture_end_et,
        },
        "variant": dict(sorted(variant.items())),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


def _dataset_id(campaign: CampaignConfig, variant_hash: str) -> str:
    symbol = campaign.base_capture.symbol.lower()
    dt = campaign.base_capture.dt.replace("-", "")
    start = campaign.base_capture.capture_start_et.replace(":", "")[:4]
    end = campaign.base_capture.capture_end_et.replace(":", "")[:4]
    return f"{symbol}_{dt}_{start}_{end}__{_slug(campaign.name)}__{variant_hash}"


def run_campaign(
    *,
    campaign: CampaignConfig,
    campaign_config_path: Path,
    lake_root: Path,
    research_root: Path,
    work_root: Path,
    max_variants: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    base_cfg = resolve_config(
        campaign.base_capture.product_type,
        campaign.base_capture.symbol,
    )
    runtime_allowed = _runtime_override_keys_from_config(base_cfg)

    variants = _expand_variants(campaign)
    _validate_variant_keys(variants, runtime_allowed)

    if max_variants is not None:
        if max_variants < 1:
            raise ValueError(f"max_variants must be >= 1, got {max_variants}")
        variants = variants[:max_variants]

    capture_start_et, capture_start_ns = _parse_et_timestamp_ns(
        campaign.base_capture.dt,
        campaign.base_capture.capture_start_et,
        "base_capture.capture_start_et",
    )
    _capture_end_et, capture_end_ns = _parse_et_timestamp_ns(
        campaign.base_capture.dt,
        campaign.base_capture.capture_end_et,
        "base_capture.capture_end_et",
    )
    if capture_end_ns <= capture_start_ns:
        raise ValueError("base_capture.capture_end_et must be later than capture_start_et")

    stream_start_hhmm = _stream_start_hhmm(capture_start_et)

    index_records: list[dict[str, Any]] = []
    campaign_started = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Campaign '%s': %d variant(s), dry_run=%s",
        campaign.name,
        len(variants),
        dry_run,
    )

    for idx, variant in enumerate(variants, start=1):
        variant_hash = _variant_hash(campaign=campaign, variant=variant)
        dataset_id = _dataset_id(campaign, variant_hash)
        runtime_overrides = _extract_runtime_overrides(variant, runtime_allowed)
        use_cubic, cubic_scale, damping_lambda = _extract_projection_params(variant)

        logger.info(
            "[%d/%d] variant_hash=%s dataset_id=%s runtime_overrides=%s projection=%s",
            idx,
            len(variants),
            variant_hash,
            dataset_id,
            runtime_overrides,
            {
                "projection_use_cubic": use_cubic,
                "projection_cubic_scale": cubic_scale,
                "projection_damping_lambda": damping_lambda,
            },
        )

        if dry_run:
            index_records.append(
                {
                    "variant_hash": variant_hash,
                    "dataset_id": dataset_id,
                    "variant": dict(variant),
                    "runtime_overrides": runtime_overrides,
                    "projection_model": {
                        "use_cubic": use_cubic,
                        "cubic_scale": cubic_scale,
                        "damping_lambda": damping_lambda,
                    },
                    "dry_run": True,
                }
            )
            continue

        runtime_cfg = build_config_with_overrides(base_cfg, runtime_overrides)
        capture_output_dir = (
            work_root
            / "vp_gold_campaigns"
            / _slug(campaign.name)
            / variant_hash
            / "capture"
        )

        capture_summary = capture_stream_output(
            lake_root=lake_root,
            config=runtime_cfg,
            dt=campaign.base_capture.dt,
            stream_start_time_hhmm=stream_start_hhmm,
            capture_start_ns=capture_start_ns,
            capture_end_ns=capture_end_ns,
            output_dir=capture_output_dir,
            projection_use_cubic=use_cubic,
            projection_cubic_scale=cubic_scale,
            projection_damping_lambda=damping_lambda,
        )

        publish_result = publish_dataset(
            source_dir=capture_output_dir,
            dataset_id=dataset_id,
            research_root=research_root,
            agents=campaign.publish.agents,
        )

        index_records.append(
            {
                "variant_hash": variant_hash,
                "dataset_id": dataset_id,
                "variant": dict(variant),
                "runtime_overrides": runtime_overrides,
                "projection_model": {
                    "use_cubic": use_cubic,
                    "cubic_scale": cubic_scale,
                    "damping_lambda": damping_lambda,
                },
                "runtime_config_version": runtime_cfg.config_version,
                "capture_summary": capture_summary,
                "publish_result": publish_result,
                "dry_run": False,
            }
        )

    campaign_index = {
        "campaign_name": campaign.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_started_at_utc": campaign_started,
        "campaign_config_path": str(campaign_config_path),
        "base_capture": {
            "product_type": campaign.base_capture.product_type,
            "symbol": campaign.base_capture.symbol,
            "dt": campaign.base_capture.dt,
            "capture_start_et": campaign.base_capture.capture_start_et,
            "capture_end_et": campaign.base_capture.capture_end_et,
        },
        "publish": {
            "agents": list(campaign.publish.agents),
            "research_root": str(research_root),
        },
        "variant_count": len(index_records),
        "dry_run": dry_run,
        "variants": index_records,
    }

    if not dry_run:
        index_dir = research_root / "vp_gold_campaigns" / _slug(campaign.name)
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "index.json"
        index_path.write_text(json.dumps(campaign_index, indent=2) + "\n", encoding="utf-8")
        campaign_index["index_path"] = str(index_path)

    return campaign_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build immutable gold datasets from a parameterized campaign config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML campaign config path.",
    )
    parser.add_argument(
        "--lake-root",
        type=Path,
        default=backend_root / "lake",
        help="Lake root for replay data and cache output.",
    )
    parser.add_argument(
        "--research-root",
        type=Path,
        default=DEFAULT_RESEARCH_ROOT,
        help="Research root for immutable/published datasets.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/tmp"),
        help="Scratch root used for intermediate capture outputs.",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Optional cap on number of variants to execute.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Expand variants and print campaign index without capturing/publishing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Replay can generate very high-volume out-of-range warnings; keep logs actionable.
    logging.getLogger("src.vacuum_pressure.event_engine").setLevel(logging.ERROR)

    campaign = _parse_campaign(args.config)
    result = run_campaign(
        campaign=campaign,
        campaign_config_path=args.config.resolve(),
        lake_root=args.lake_root.resolve(),
        research_root=args.research_root.resolve(),
        work_root=args.work_root.resolve(),
        max_variants=args.max_variants,
        dry_run=args.dry_run,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
