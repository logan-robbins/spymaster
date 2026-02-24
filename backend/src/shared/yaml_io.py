from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml


def load_yaml_mapping(
    path: Path,
    *,
    not_found_message: str,
    empty_message: str | None = None,
    non_mapping_message: str | None = None,
    resolve_path: bool = False,
    logger: logging.Logger | None = None,
    log_message: str | None = None,
) -> dict[str, Any]:
    """Load a YAML file and require a top-level mapping.

    Args:
        path: YAML file path.
        not_found_message: Error message template when file is missing.
            Supports `{path}` formatting.
        empty_message: Optional error template for empty YAML (`None`).
            Supports `{path}` formatting.
        non_mapping_message: Optional error template for non-dict root.
            Supports `{path}` and `{kind}` formatting.
        resolve_path: Whether to resolve path before loading.
        logger: Optional logger.
        log_message: Optional logger.info format string, called as
            `logger.info(log_message, path)`.
    """
    resolved = Path(path).resolve() if resolve_path else Path(path)
    if not resolved.exists():
        raise FileNotFoundError(not_found_message.format(path=resolved))

    if logger is not None and log_message is not None:
        logger.info(log_message, resolved)

    raw = yaml.safe_load(resolved.read_text())
    if raw is None:
        if empty_message is not None:
            raise ValueError(empty_message.format(path=resolved))
        raise ValueError(f"YAML file is empty: {resolved}")

    if not isinstance(raw, dict):
        if non_mapping_message is not None:
            raise ValueError(
                non_mapping_message.format(
                    path=resolved,
                    kind=type(raw).__name__,
                )
            )
        raise ValueError(
            f"Expected YAML mapping at top level, got {type(raw).__name__}: {resolved}"
        )

    return raw
