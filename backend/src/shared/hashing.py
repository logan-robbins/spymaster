from __future__ import annotations

import hashlib
from typing import Any, Mapping


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, list):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_for_hash(v) for k, v in value.items()}
    return value


def stable_short_hash(fields: Mapping[str, Any], *, length: int = 12) -> str:
    """Return a deterministic short SHA256 hash over a mapping payload.

    The payload is normalized (tuples -> lists) and rendered using
    sorted key order so identical semantic configs always produce the
    same identifier across modules.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")

    stable: dict[str, Any] = {
        key: _normalize_for_hash(value)
        for key, value in fields.items()
    }
    raw = "|".join(f"{key}={value}" for key, value in sorted(stable.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:length]
