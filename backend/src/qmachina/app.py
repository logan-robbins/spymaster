"""Application composition root for VP REST + stream APIs."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api_experiments import register_experiment_routes
from .api_stream import register_stream_routes
from .serving_registry import ServingRegistry


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
    app = FastAPI(
        title="qMachina Research Platform",
        version="4.0.0",
        description="Fixed-bin dense-grid model streaming from event feed",
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
