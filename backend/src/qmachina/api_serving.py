"""REST routes for serving lifecycle management.

Endpoints:
    GET   /v1/serving/versions                  List all immutable serving versions
    GET   /v1/serving/versions/{serving_id}     Get one version detail + runtime_snapshot
    GET   /v1/serving/aliases                   List all aliases with current target serving_id
    GET   /v1/serving/aliases/{alias}           Alias detail + activation history
    POST  /v1/serving/aliases/{alias}/activate  Point alias to a serving_id
    GET   /v1/serving/aliases/{alias}/history   Audit trail of alias activations
    POST  /v1/serving/diff                      Compare two serving version runtime snapshots
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .serving_diff import diff_runtime_snapshots
from .serving_registry import ServingRegistry

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ActivateAliasRequest(BaseModel):
    """Payload for activating (repointing) a serving alias."""

    serving_id: str = Field(..., description="Target immutable serving version ID.")
    reason: str | None = Field(
        default=None, description="Optional human-readable reason for the activation."
    )


class DiffRequest(BaseModel):
    """Payload for comparing two serving version runtime snapshots."""

    serving_id_a: str = Field(..., description="Serving version A identifier.")
    serving_id_b: str = Field(..., description="Serving version B identifier.")


class VersionSummary(BaseModel):
    """Abridged serving version metadata for list responses."""

    serving_id: str
    description: str
    source_experiment_name: str
    source_run_id: str
    source_config_hash: str
    created_at_utc: str
    spec_path: str


class VersionDetail(VersionSummary):
    """Full serving version detail including runtime snapshot."""

    runtime_snapshot: dict[str, Any]


class AliasSummary(BaseModel):
    """Alias with its current target serving_id."""

    alias: str
    serving_id: str
    updated_at_utc: str


class AliasDetail(AliasSummary):
    """Alias detail including recent activation history."""

    history: list[dict[str, Any]]


class ActivationEvent(BaseModel):
    """Single entry from the alias activation audit trail."""

    event_id: int
    alias: str
    from_serving_id: str | None
    to_serving_id: str
    actor: str
    timestamp_utc: str


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_serving_router(
    *,
    lake_root: Path,
    serving_registry: ServingRegistry,
) -> APIRouter:
    """Build and return the serving lifecycle router.

    Args:
        lake_root: Root of the data lake (used for spec resolution).
        serving_registry: The singleton ServingRegistry instance.

    Returns:
        A configured ``APIRouter`` with all serving lifecycle endpoints.
    """
    router: APIRouter = APIRouter(tags=["serving"])

    # ------------------------------------------------------------------
    # Versions
    # ------------------------------------------------------------------

    @router.get("/v1/serving/versions", response_model=list[VersionSummary])
    async def list_versions() -> list[dict[str, Any]]:
        """Return all immutable serving versions ordered newest-first."""
        with serving_registry._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    serving_id,
                    spec_path,
                    source_experiment_name,
                    source_run_id,
                    source_config_hash,
                    created_at_utc
                FROM serving_versions
                ORDER BY created_at_utc DESC
                """
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            spec_path = Path(str(row["spec_path"]))
            description = ""
            try:
                from .serving_config import PublishedServingSpec

                spec = PublishedServingSpec.from_yaml(spec_path)
                description = spec.description
            except Exception:
                pass
            results.append(
                {
                    "serving_id": str(row["serving_id"]),
                    "description": description,
                    "source_experiment_name": str(row["source_experiment_name"]),
                    "source_run_id": str(row["source_run_id"]),
                    "source_config_hash": str(row["source_config_hash"]),
                    "created_at_utc": str(row["created_at_utc"]),
                    "spec_path": str(row["spec_path"]),
                }
            )
        return results

    @router.get("/v1/serving/versions/{serving_id}", response_model=VersionDetail)
    async def get_version(serving_id: str) -> dict[str, Any]:
        """Return full detail for a single serving version including runtime_snapshot."""
        with serving_registry._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    serving_id,
                    spec_path,
                    source_experiment_name,
                    source_run_id,
                    source_config_hash,
                    created_at_utc
                FROM serving_versions
                WHERE serving_id = ?
                """,
                (serving_id,),
            ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail=f"Serving version '{serving_id}' not found.")

        spec_path = Path(str(row["spec_path"]))
        description = ""
        runtime_snapshot: dict[str, Any] = {}
        try:
            from .serving_config import PublishedServingSpec

            spec = PublishedServingSpec.from_yaml(spec_path)
            description = spec.description
            runtime_snapshot = spec.runtime_snapshot
        except Exception as exc:
            logger.warning("Failed to load spec YAML for %s: %s", serving_id, exc)

        return {
            "serving_id": str(row["serving_id"]),
            "description": description,
            "source_experiment_name": str(row["source_experiment_name"]),
            "source_run_id": str(row["source_run_id"]),
            "source_config_hash": str(row["source_config_hash"]),
            "created_at_utc": str(row["created_at_utc"]),
            "spec_path": str(row["spec_path"]),
            "runtime_snapshot": runtime_snapshot,
        }

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------

    @router.get("/v1/serving/aliases", response_model=list[AliasSummary])
    async def list_aliases() -> list[dict[str, Any]]:
        """Return all serving aliases with their current target."""
        with serving_registry._connect() as conn:
            rows = conn.execute(
                """
                SELECT alias, serving_id, updated_at_utc
                FROM serving_aliases
                ORDER BY alias
                """
            ).fetchall()

        return [
            {
                "alias": str(row["alias"]),
                "serving_id": str(row["serving_id"]),
                "updated_at_utc": str(row["updated_at_utc"]),
            }
            for row in rows
        ]

    @router.get("/v1/serving/aliases/{alias}", response_model=AliasDetail)
    async def get_alias(alias: str) -> dict[str, Any]:
        """Return alias detail including recent activation history."""
        with serving_registry._connect() as conn:
            alias_row = conn.execute(
                "SELECT alias, serving_id, updated_at_utc FROM serving_aliases WHERE alias = ?",
                (alias,),
            ).fetchone()

        if alias_row is None:
            raise HTTPException(status_code=404, detail=f"Alias '{alias}' not found.")

        history = _fetch_alias_history(serving_registry, alias)

        return {
            "alias": str(alias_row["alias"]),
            "serving_id": str(alias_row["serving_id"]),
            "updated_at_utc": str(alias_row["updated_at_utc"]),
            "history": history,
        }

    @router.post("/v1/serving/aliases/{alias}/activate")
    async def activate_alias(alias: str, body: ActivateAliasRequest) -> dict[str, Any]:
        """Point an alias to a new serving version.

        Creates the alias if it does not exist, or repoints if it does.
        Records an audit event with the actor set to ``api`` and the
        optional reason attached to the event.
        """
        # Verify the target serving version exists.
        with serving_registry._connect() as conn:
            version_row = conn.execute(
                "SELECT serving_id FROM serving_versions WHERE serving_id = ?",
                (body.serving_id,),
            ).fetchone()

        if version_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Target serving version '{body.serving_id}' not found.",
            )

        # Perform the alias activation using raw SQL on the registry's DB.
        # We use the same tables as ServingRegistry.promote() but without
        # requiring a full promote cycle (no new version creation).
        from datetime import datetime, timezone

        now_utc: str = datetime.now(tz=timezone.utc).isoformat()
        actor: str = "api"

        with serving_registry._connect() as conn:
            previous_row = conn.execute(
                "SELECT serving_id FROM serving_aliases WHERE alias = ?",
                (alias,),
            ).fetchone()
            from_serving_id: str | None = (
                str(previous_row["serving_id"]) if previous_row is not None else None
            )

            conn.execute(
                """
                INSERT INTO serving_aliases(alias, serving_id, updated_at_utc)
                VALUES(?, ?, ?)
                ON CONFLICT(alias) DO UPDATE SET
                    serving_id = excluded.serving_id,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (alias, body.serving_id, now_utc),
            )
            conn.execute(
                """
                INSERT INTO promotion_events(
                    alias,
                    from_serving_id,
                    to_serving_id,
                    actor,
                    timestamp_utc
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (alias, from_serving_id, body.serving_id, actor, now_utc),
            )
            conn.commit()

        logger.info(
            "Alias '%s' activated to serving_id='%s' (from='%s', reason='%s')",
            alias,
            body.serving_id,
            from_serving_id,
            body.reason,
        )

        return {
            "alias": alias,
            "serving_id": body.serving_id,
            "from_serving_id": from_serving_id,
            "activated_at_utc": now_utc,
            "reason": body.reason,
        }

    @router.get("/v1/serving/aliases/{alias}/history", response_model=list[ActivationEvent])
    async def alias_history(alias: str) -> list[dict[str, Any]]:
        """Return the full audit trail of alias activations, newest first."""
        # Verify alias exists.
        with serving_registry._connect() as conn:
            alias_row = conn.execute(
                "SELECT alias FROM serving_aliases WHERE alias = ?",
                (alias,),
            ).fetchone()

        if alias_row is None:
            raise HTTPException(status_code=404, detail=f"Alias '{alias}' not found.")

        return _fetch_alias_history(serving_registry, alias)

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    @router.post("/v1/serving/diff")
    async def diff_versions(body: DiffRequest) -> dict[str, Any]:
        """Compare runtime snapshots of two serving versions."""
        snapshot_a: dict[str, Any] | None = None
        snapshot_b: dict[str, Any] | None = None

        for label, sid in [("A", body.serving_id_a), ("B", body.serving_id_b)]:
            with serving_registry._connect() as conn:
                row = conn.execute(
                    "SELECT spec_path FROM serving_versions WHERE serving_id = ?",
                    (sid,),
                ).fetchone()
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Serving version {label} ('{sid}') not found.",
                )
            spec_path = Path(str(row["spec_path"]))
            try:
                from .serving_config import PublishedServingSpec

                spec = PublishedServingSpec.from_yaml(spec_path)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load spec for version {label} ('{sid}'): {exc}",
                ) from exc

            if label == "A":
                snapshot_a = spec.runtime_snapshot
            else:
                snapshot_b = spec.runtime_snapshot

        assert snapshot_a is not None and snapshot_b is not None
        return diff_runtime_snapshots(
            snapshot_a=snapshot_a,
            snapshot_b=snapshot_b,
            serving_id_a=body.serving_id_a,
            serving_id_b=body.serving_id_b,
        )

    return router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_alias_history(
    registry: ServingRegistry,
    alias: str,
) -> list[dict[str, Any]]:
    """Fetch promotion events for an alias, newest first.

    Args:
        registry: The ServingRegistry to query.
        alias: The alias name.

    Returns:
        List of activation event dicts.
    """
    with registry._connect() as conn:
        rows = conn.execute(
            """
            SELECT event_id, alias, from_serving_id, to_serving_id, actor, timestamp_utc
            FROM promotion_events
            WHERE alias = ?
            ORDER BY event_id DESC
            """,
            (alias,),
        ).fetchall()

    return [
        {
            "event_id": int(row["event_id"]),
            "alias": str(row["alias"]),
            "from_serving_id": row["from_serving_id"],
            "to_serving_id": str(row["to_serving_id"]),
            "actor": str(row["actor"]),
            "timestamp_utc": str(row["timestamp_utc"]),
        }
        for row in rows
    ]
