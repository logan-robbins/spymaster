"""Redis Streams-backed job queue with graceful in-process fallback.

When ``REDIS_URL`` is set and the broker is reachable, jobs are persisted
to a Redis Stream using consumer groups (``XADD`` / ``XREADGROUP``).
When Redis is unavailable the module falls back to an ``asyncio.Queue``
that lives only for the lifetime of the current process.

Usage::

    queue = await get_job_queue()
    await queue.enqueue("job-123", {"spec_ref": "baseline_sweep"})
    item = await queue.dequeue()  # ("job-123", {...}) or None
    await queue.close()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

STREAM_KEY: str = "jobs:experiments"
CONSUMER_GROUP: str = "qmachina_workers"
CONSUMER_NAME: str = "worker-0"


class _RedisJobQueue:
    """Redis Streams-backed implementation of the job queue.

    Uses ``XADD`` for enqueue and ``XREADGROUP`` for dequeue with
    automatic acknowledgement on successful read.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379``).
    """

    def __init__(self, redis_url: str) -> None:
        import redis.asyncio as aioredis

        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=True,
        )
        self._group_created: bool = False

    async def _ensure_group(self) -> None:
        """Create the consumer group if it does not already exist."""
        if self._group_created:
            return
        try:
            await self._redis.xgroup_create(
                name=STREAM_KEY,
                groupname=CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )
        except Exception as exc:
            # BUSYGROUP means the group already exists -- safe to ignore.
            if "BUSYGROUP" not in str(exc):
                raise
        self._group_created = True

    async def enqueue(self, job_id: str, payload: dict[str, Any]) -> None:
        """Add a job to the Redis stream.

        Args:
            job_id: Unique job identifier.
            payload: JSON-serializable job payload.
        """
        await self._ensure_group()
        await self._redis.xadd(
            STREAM_KEY,
            {"job_id": job_id, "payload": json.dumps(payload)},
        )
        logger.info("Enqueued job %s to Redis stream %s", job_id, STREAM_KEY)

    async def dequeue(self) -> tuple[str, dict[str, Any]] | None:
        """Read the next pending job from the consumer group.

        Non-blocking: returns ``None`` immediately if no messages are
        available.

        Returns:
            A ``(job_id, payload)`` tuple, or ``None`` if the queue is empty.
        """
        await self._ensure_group()
        result = await self._redis.xreadgroup(
            groupname=CONSUMER_GROUP,
            consumername=CONSUMER_NAME,
            streams={STREAM_KEY: ">"},
            count=1,
            block=100,  # 100 ms block to avoid busy-loop
        )
        if not result:
            return None

        # result is [[stream_key, [(message_id, fields_dict), ...]]]
        _stream_key, messages = result[0]
        if not messages:
            return None

        message_id, fields = messages[0]
        job_id: str = fields["job_id"]
        payload: dict[str, Any] = json.loads(fields["payload"])

        # Acknowledge immediately so the message is not redelivered.
        await self._redis.xack(STREAM_KEY, CONSUMER_GROUP, message_id)
        return job_id, payload

    async def close(self) -> None:
        """Close the underlying Redis connection."""
        await self._redis.aclose()
        logger.info("Closed Redis job queue connection")


class _InProcessJobQueue:
    """``asyncio.Queue``-backed fallback when Redis is unavailable.

    Jobs are lost on process exit. Suitable for single-process
    development and testing.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

    async def enqueue(self, job_id: str, payload: dict[str, Any]) -> None:
        """Add a job to the in-process queue.

        Args:
            job_id: Unique job identifier.
            payload: JSON-serializable job payload.
        """
        await self._queue.put((job_id, payload))
        logger.debug("Enqueued job %s to in-process queue", job_id)

    async def dequeue(self) -> tuple[str, dict[str, Any]] | None:
        """Read the next job from the in-process queue.

        Non-blocking: returns ``None`` immediately when empty.

        Returns:
            A ``(job_id, payload)`` tuple, or ``None`` if empty.
        """
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def close(self) -> None:
        """No-op for the in-process queue."""


# Type alias covering both implementations.
JobQueue = _RedisJobQueue | _InProcessJobQueue

_singleton: JobQueue | None = None


async def get_job_queue() -> JobQueue:
    """Return the module-level singleton job queue.

    On the first call, attempts to connect to Redis via ``REDIS_URL``.
    Falls back to an in-process ``asyncio.Queue`` if Redis is not
    configured or the connection probe fails.

    Returns:
        A ``JobQueue`` instance (Redis-backed or in-process).
    """
    global _singleton  # noqa: PLW0603
    if _singleton is not None:
        return _singleton

    redis_url: str | None = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            queue = _RedisJobQueue(redis_url)
            # Probe connectivity.
            await queue._redis.ping()
            logger.info("Connected to Redis at %s", redis_url)
            _singleton = queue
            return _singleton
        except Exception as exc:
            logger.warning(
                "Redis unavailable (%s: %s). Falling back to in-process queue.",
                type(exc).__name__,
                exc,
            )

    logger.warning(
        "REDIS_URL not set or Redis unreachable. Using in-process job queue."
    )
    _singleton = _InProcessJobQueue()
    return _singleton


def reset_job_queue() -> None:
    """Reset the singleton for testing. Not for production use."""
    global _singleton  # noqa: PLW0603
    _singleton = None
