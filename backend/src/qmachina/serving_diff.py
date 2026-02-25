"""Structured diff between two immutable serving runtime snapshots.

Provides a flat key-level comparison suitable for display in audit UIs
and API responses. Nested dicts/lists are compared as opaque values
(JSON-serializable equality), not recursively diffed.
"""
from __future__ import annotations

import logging
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


def diff_runtime_snapshots(
    snapshot_a: dict[str, Any],
    snapshot_b: dict[str, Any],
    serving_id_a: str,
    serving_id_b: str,
) -> dict[str, Any]:
    """Compare two runtime snapshots and return a structured diff.

    Args:
        snapshot_a: Runtime snapshot dict from serving version A.
        snapshot_b: Runtime snapshot dict from serving version B.
        serving_id_a: Serving version ID for snapshot A (labelling only).
        serving_id_b: Serving version ID for snapshot B (labelling only).

    Returns:
        A dict with the following structure::

            {
                "serving_id_a": str,
                "serving_id_b": str,
                "added": {key: value_in_b},
                "removed": {key: value_in_a},
                "changed": {key: {"a": v_a, "b": v_b}},
                "unchanged_count": int,
                "summary": str,
            }
    """
    keys_a: set[str] = set(snapshot_a.keys())
    keys_b: set[str] = set(snapshot_b.keys())

    added: dict[str, Any] = {k: snapshot_b[k] for k in sorted(keys_b - keys_a)}
    removed: dict[str, Any] = {k: snapshot_a[k] for k in sorted(keys_a - keys_b)}

    changed: dict[str, dict[str, Any]] = {}
    unchanged_count: int = 0

    for key in sorted(keys_a & keys_b):
        val_a = snapshot_a[key]
        val_b = snapshot_b[key]
        if val_a == val_b:
            unchanged_count += 1
        else:
            changed[key] = {"a": val_a, "b": val_b}

    parts: list[str] = []
    if added:
        parts.append(f"{len(added)} added")
    if removed:
        parts.append(f"{len(removed)} removed")
    if changed:
        parts.append(f"{len(changed)} changed")
    parts.append(f"{unchanged_count} unchanged")
    summary: str = f"{serving_id_a} vs {serving_id_b}: {', '.join(parts)}"

    return {
        "serving_id_a": serving_id_a,
        "serving_id_b": serving_id_b,
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged_count": unchanged_count,
        "summary": summary,
    }
