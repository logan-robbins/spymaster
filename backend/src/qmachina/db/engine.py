"""Async SQLAlchemy engine and session factory for the control-plane database.

Reads ``DATABASE_URL`` from environment with a sensible local default.
The engine is created lazily on first access and can be disposed via
``dispose_engine()``.
"""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_DATABASE_URL: str = (
    "postgresql+asyncpg://qmachina:qmachina@localhost:5432/qmachina"
)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_database_url() -> str:
    """Return the configured DATABASE_URL, falling back to the local default.

    Returns:
        A SQLAlchemy-compatible async database URL string.
    """
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine() -> AsyncEngine:
    """Return the singleton async engine, creating it on first call.

    Returns:
        The shared ``AsyncEngine`` instance.
    """
    global _engine  # noqa: PLW0603
    if _engine is None:
        url: str = get_database_url()
        _engine = create_async_engine(
            url,
            echo=False,
            pool_pre_ping=True,
        )
        logger.info("Created async engine for %s", _redact_url(url))
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the singleton async session factory.

    Returns:
        An ``async_sessionmaker`` bound to the shared engine.
    """
    global _session_factory  # noqa: PLW0603
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async session for a unit of work, committing on success.

    Usage::

        async with get_db_session() as session:
            result = await session.execute(...)

    On exception the session is rolled back automatically.
    """
    factory = get_session_factory()
    session: AsyncSession = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def dispose_engine() -> None:
    """Dispose the async engine and reset module-level singletons.

    Safe to call even if no engine was ever created.
    """
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        logger.info("Disposed async engine")
    _engine = None
    _session_factory = None


def reset_engine(engine: AsyncEngine | None = None) -> None:
    """Replace the module-level engine singleton (for testing only).

    Args:
        engine: An ``AsyncEngine`` to use, or ``None`` to clear the
                singleton so ``get_engine()`` recreates from env.
    """
    global _engine, _session_factory  # noqa: PLW0603
    _engine = engine
    _session_factory = None


def _redact_url(url: str) -> str:
    """Redact password from a database URL for safe logging."""
    try:
        at_idx: int = url.index("@")
        schema_end: int = url.index("://") + 3
        return url[:schema_end] + "***@" + url[at_idx + 1 :]
    except ValueError:
        return url
