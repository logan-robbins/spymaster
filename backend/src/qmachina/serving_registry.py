"""Registry for immutable serving versions and mutable serving aliases."""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .serving_config import PublishedServingSpec, ServingSpec

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


@dataclass(frozen=True)
class PromotionResult:
    """Result metadata for a promotion write operation."""

    alias: str
    serving_id: str
    spec_path: Path
    reused_existing: bool


@dataclass(frozen=True)
class ResolvedServing:
    """Resolved serving selector (alias or ID) to immutable version spec."""

    requested_name: str
    serving_id: str
    alias: str | None
    spec: PublishedServingSpec


def validate_serving_spec_preflight(spec: ServingSpec) -> None:
    """Validate a ServingSpec before promotion or registration.

    Runs the full Pydantic model validation by round-tripping through
    model_dump/model_validate. This catches schema violations that would
    otherwise only surface at stream-time.

    Args:
        spec: The ServingSpec to validate.

    Raises:
        ValueError: If validation fails, with a descriptive message
            including the spec name and the underlying error.
    """
    try:
        data = spec.model_dump(mode="json")
        ServingSpec.model_validate(data)
    except Exception as exc:
        raise ValueError(
            f"Serving spec '{spec.name}' failed preflight validation: {exc}"
        ) from exc


def _validate_published_spec_preflight(spec: PublishedServingSpec) -> None:
    """Validate stream_schema roles in a PublishedServingSpec runtime snapshot.

    Checks that any stream_schema entries in the runtime_snapshot use valid
    StreamFieldRole enum values. This prevents invalid roles from being
    persisted into the immutable serving registry.

    Args:
        spec: The PublishedServingSpec to validate.

    Raises:
        ValueError: If any stream_schema role is not a valid StreamFieldRole.
    """
    from .serving_config import StreamFieldRole

    snapshot: dict = spec.runtime_snapshot
    schema_entries: list[dict] | None = snapshot.get("stream_schema")
    if not schema_entries:
        return

    valid_roles: set[str | None] = {r.value for r in StreamFieldRole} | {None}

    for entry in schema_entries:
        role: str | None = entry.get("role")
        if role not in valid_roles:
            raise ValueError(
                f"Published spec '{spec.serving_id}' has stream_schema field "
                f"'{entry.get('name', '?')}' with invalid role '{role}'. "
                f"Valid roles: {sorted(r.value for r in StreamFieldRole)}"
            )


class ServingRegistry:
    """SQLite-backed registry for serving versions and aliases."""

    def __init__(self, lake_root: Path) -> None:
        self._lake_root = Path(lake_root)
        self._registry_path = (
            self._lake_root / "research" / "harness" / "serving_registry.sqlite"
        )
        self._versions_dir = PublishedServingSpec.configs_dir(self._lake_root)
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @property
    def registry_path(self) -> Path:
        return self._registry_path

    @property
    def versions_dir(self) -> Path:
        return self._versions_dir

    def build_serving_id(
        self,
        *,
        experiment_name: str,
        run_id: str,
        config_hash: str,
    ) -> str:
        """Build an immutable serving version ID."""
        exp_slug = _slug(experiment_name)
        run8 = _slug(run_id)[:8]
        cfg8 = _slug(config_hash)[:8]
        return f"srv_{exp_slug}_{run8}_{cfg8}"

    def promote(
        self,
        *,
        alias: str,
        spec: PublishedServingSpec,
        actor: str = "cli",
    ) -> PromotionResult:
        """Register a published serving version and move alias to it.

        Dedup policy:
        - ``(source_run_id, source_config_hash)`` is unique.
        - repeated promote of same run/config reuses existing serving_id and
          only updates alias mapping + audit event.
        """
        alias = _validate_name(alias, kind="alias")
        serving_id = _validate_name(spec.serving_id, kind="serving_id")

        # Preflight: validate stream_schema roles before any DB writes.
        _validate_published_spec_preflight(spec)

        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT serving_id, spec_path
                FROM serving_versions
                WHERE source_run_id = ? AND source_config_hash = ?
                """,
                (spec.source.run_id, spec.source.config_hash),
            ).fetchone()

            if existing is not None:
                existing_id = str(existing["serving_id"])
                existing_path = Path(str(existing["spec_path"]))
                previous_alias_row = conn.execute(
                    "SELECT serving_id FROM serving_aliases WHERE alias = ?",
                    (alias,),
                ).fetchone()
                from_serving_id = (
                    str(previous_alias_row["serving_id"])
                    if previous_alias_row is not None
                    else None
                )
                now_utc = _now_utc_iso()
                conn.execute(
                    """
                    INSERT INTO serving_aliases(alias, serving_id, updated_at_utc)
                    VALUES(?, ?, ?)
                    ON CONFLICT(alias) DO UPDATE SET
                        serving_id = excluded.serving_id,
                        updated_at_utc = excluded.updated_at_utc
                    """,
                    (alias, existing_id, now_utc),
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
                    (alias, from_serving_id, existing_id, actor, now_utc),
                )
                conn.commit()
                logger.info(
                    "Serving promotion reused existing version: alias=%s serving_id=%s",
                    alias,
                    existing_id,
                )
                return PromotionResult(
                    alias=alias,
                    serving_id=existing_id,
                    spec_path=existing_path,
                    reused_existing=True,
                )

        spec_path = self._versions_dir / f"{serving_id}.yaml"
        if spec_path.exists():
            raise ValueError(
                f"Cannot promote serving_id={serving_id}: spec already exists at {spec_path}"
            )
        spec.to_yaml(spec_path)
        spec_sha256 = _sha256_file(spec_path)

        try:
            with self._connect() as conn:
                now_utc = _now_utc_iso()
                conn.execute(
                    """
                    INSERT INTO serving_versions(
                        serving_id,
                        spec_path,
                        spec_sha256,
                        source_run_id,
                        source_experiment_name,
                        source_config_hash,
                        created_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        serving_id,
                        str(spec_path),
                        spec_sha256,
                        spec.source.run_id,
                        spec.source.experiment_name,
                        spec.source.config_hash,
                        now_utc,
                    ),
                )

                previous_alias_row = conn.execute(
                    "SELECT serving_id FROM serving_aliases WHERE alias = ?",
                    (alias,),
                ).fetchone()
                from_serving_id = (
                    str(previous_alias_row["serving_id"])
                    if previous_alias_row is not None
                    else None
                )

                conn.execute(
                    """
                    INSERT INTO serving_aliases(alias, serving_id, updated_at_utc)
                    VALUES(?, ?, ?)
                    ON CONFLICT(alias) DO UPDATE SET
                        serving_id = excluded.serving_id,
                        updated_at_utc = excluded.updated_at_utc
                    """,
                    (alias, serving_id, now_utc),
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
                    (alias, from_serving_id, serving_id, actor, now_utc),
                )
                conn.commit()
        except Exception:
            # File write already happened. Remove orphaned spec if DB transaction failed.
            spec_path.unlink(missing_ok=True)
            raise

        logger.info(
            "Serving promotion created new version: alias=%s serving_id=%s spec=%s",
            alias,
            serving_id,
            spec_path,
        )
        return PromotionResult(
            alias=alias,
            serving_id=serving_id,
            spec_path=spec_path,
            reused_existing=False,
        )

    def resolve(self, name: str) -> ResolvedServing:
        """Resolve alias or immutable serving_id to a published serving spec."""
        requested_name = name.strip()
        if not requested_name:
            raise ValueError("serving selector cannot be empty.")

        with self._connect() as conn:
            alias_row = conn.execute(
                "SELECT alias, serving_id FROM serving_aliases WHERE alias = ?",
                (requested_name,),
            ).fetchone()
            alias: str | None = None
            if alias_row is not None:
                alias = str(alias_row["alias"])
                serving_id = str(alias_row["serving_id"])
            else:
                version_row = conn.execute(
                    "SELECT serving_id FROM serving_versions WHERE serving_id = ?",
                    (requested_name,),
                ).fetchone()
                if version_row is None:
                    raise ValueError(
                        f"Unknown serving selector '{requested_name}'. "
                        "Expected alias or immutable serving_id."
                    )
                serving_id = str(version_row["serving_id"])

            row = conn.execute(
                """
                SELECT serving_id, spec_path
                FROM serving_versions
                WHERE serving_id = ?
                """,
                (serving_id,),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"Serving version '{serving_id}' resolved from '{requested_name}' "
                    "was not found in registry."
                )
            spec_path = Path(str(row["spec_path"]))

        spec = PublishedServingSpec.from_yaml(spec_path)
        if spec.serving_id != serving_id:
            raise ValueError(
                f"Serving spec ID mismatch. Registry={serving_id}, YAML={spec.serving_id}, "
                f"path={spec_path}"
            )
        return ResolvedServing(
            requested_name=requested_name,
            serving_id=serving_id,
            alias=alias,
            spec=spec,
        )

    def preferred_alias_for_run(self, run_id: str) -> tuple[str, str] | None:
        """Return the latest alias mapped to the run's promoted serving version."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT a.alias, a.serving_id, a.updated_at_utc
                FROM serving_aliases a
                JOIN serving_versions v ON v.serving_id = a.serving_id
                WHERE v.source_run_id = ?
                ORDER BY a.updated_at_utc DESC, a.alias ASC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return str(row["alias"]), str(row["serving_id"])

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS serving_versions (
                    serving_id TEXT PRIMARY KEY,
                    spec_path TEXT NOT NULL UNIQUE,
                    spec_sha256 TEXT NOT NULL,
                    source_run_id TEXT NOT NULL,
                    source_experiment_name TEXT NOT NULL,
                    source_config_hash TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS ux_serving_versions_run_cfg
                ON serving_versions(source_run_id, source_config_hash);

                CREATE TABLE IF NOT EXISTS serving_aliases (
                    alias TEXT PRIMARY KEY,
                    serving_id TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    FOREIGN KEY(serving_id) REFERENCES serving_versions(serving_id)
                );

                CREATE TABLE IF NOT EXISTS promotion_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alias TEXT NOT NULL,
                    from_serving_id TEXT,
                    to_serving_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._registry_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _validate_name(raw: str, *, kind: str) -> str:
    name = raw.strip().lower()
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid {kind} '{raw}'. Expected lowercase [a-z0-9_], "
            "starting with alphanumeric."
        )
    return name


def _slug(raw: str) -> str:
    token = raw.strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        token = "x"
    return token


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
