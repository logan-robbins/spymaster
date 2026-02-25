"""Application composition root for VP REST + stream APIs."""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api_experiments import register_experiment_routes
from .api_gold_dsl import router as gold_dsl_router
from .api_jobs import register_job_routes
from .api_modeling import create_modeling_router
from .api_serving import create_serving_router
from .api_stream import register_stream_routes
from .serving_registry import ServingRegistry

logger: logging.Logger = logging.getLogger(__name__)


def create_app(
    lake_root: Path | None = None,
    perf_latency_jsonl: Path | None = None,
    perf_window_start_et: str | None = None,
    perf_window_end_et: str | None = None,
    perf_summary_every_bins: int = 200,
) -> FastAPI:
    """Create the FastAPI app for canonical fixed-bin streaming."""
    if lake_root is None:
        lake_root = Path(__file__).resolve().parents[2] / "lake"
    if perf_summary_every_bins <= 0:
        raise ValueError(f"perf_summary_every_bins must be > 0, got {perf_summary_every_bins}")
    if perf_latency_jsonl is None and (
        perf_window_start_et is not None or perf_window_end_et is not None
    ):
        raise ValueError("perf_window_start_et/perf_window_end_et require perf_latency_jsonl")

    serving_registry = ServingRegistry(lake_root)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan: probe DB, start job worker, dispose on shutdown."""
        import asyncio

        await _probe_control_plane_db()

        # Start the background job worker.
        worker_cancel = asyncio.Event()
        worker_task: asyncio.Task[None] | None = None
        try:
            from ..jobs.queue import get_job_queue
            from ..jobs.worker import job_worker_loop

            queue = await get_job_queue()
            worker_task = asyncio.create_task(
                job_worker_loop(queue, worker_cancel, lake_root=lake_root),
                name="job-worker",
            )
            logger.info("Job worker task started")
        except Exception as exc:
            logger.warning(
                "Failed to start job worker (%s: %s). "
                "Job orchestration will be unavailable.",
                type(exc).__name__, exc,
            )

        yield

        # Shutdown: stop worker, close queue, dispose DB engine.
        worker_cancel.set()
        if worker_task is not None:
            try:
                await asyncio.wait_for(worker_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Job worker did not stop within 10s; cancelling")
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass

        try:
            from ..jobs.queue import get_job_queue as _get_q

            q = await _get_q()
            await q.close()
        except Exception:
            pass

        await _shutdown_control_plane_db()

    app = FastAPI(
        title="qMachina Research Platform",
        version="4.0.0",
        description="Fixed-bin dense-grid model streaming from event feed",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "qmachina"}

    register_experiment_routes(
        app,
        lake_root=lake_root,
        serving_registry=serving_registry,
    )
    register_job_routes(app, lake_root=lake_root)
    app.include_router(create_modeling_router(lake_root))
    app.include_router(create_serving_router(lake_root=lake_root, serving_registry=serving_registry))
    app.include_router(gold_dsl_router)
    register_stream_routes(
        app,
        lake_root=lake_root,
        serving_registry=serving_registry,
        perf_latency_jsonl=perf_latency_jsonl,
        perf_window_start_et=perf_window_start_et,
        perf_window_end_et=perf_window_end_et,
        perf_summary_every_bins=perf_summary_every_bins,
    )
    return app


async def _probe_control_plane_db() -> None:
    """Attempt a lightweight connection to the control-plane database.

    Logs a warning and continues if the database is unreachable.
    The app is fully functional without Postgres (graceful degradation).
    """
    try:
        from .db.engine import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(
                __import__("sqlalchemy").text("SELECT 1"),
            )
        logger.info("Control-plane database connection verified")
    except Exception as exc:
        logger.warning(
            "Control-plane database is not available (%s: %s). "
            "The app will continue without Postgres-backed features.",
            type(exc).__name__,
            exc,
        )


async def _shutdown_control_plane_db() -> None:
    """Dispose the async engine on shutdown (if it was created)."""
    try:
        from .db.engine import dispose_engine

        await dispose_engine()
    except Exception as exc:
        logger.warning(
            "Error disposing control-plane engine on shutdown: %s", exc,
        )
