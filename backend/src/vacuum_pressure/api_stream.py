"""WebSocket stream route registration."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket

from .serving_registry import ServingRegistry
from .stream_session import prepare_stream_session, run_stream_session

logger = logging.getLogger(__name__)


def register_stream_routes(
    app: FastAPI,
    *,
    lake_root: Path,
    serving_registry: ServingRegistry,
    perf_latency_jsonl: Path | None,
    perf_window_start_et: str | None,
    perf_window_end_et: str | None,
    perf_summary_every_bins: int,
) -> None:
    """Register websocket stream route."""

    @app.websocket("/v1/vacuum-pressure/stream")
    async def vacuum_pressure_stream(
        websocket: WebSocket,
        serving: str,
    ) -> None:
        """Stream fixed-bin dense-grid updates from the canonical event engine."""
        await websocket.accept()

        try:
            extra_params = sorted(
                key for key in websocket.query_params.keys() if key != "serving"
            )
            if extra_params:
                raise ValueError(
                    "Unsupported stream query params. In parity mode only "
                    f"'serving' is allowed. Got: {extra_params}"
                )
            session = prepare_stream_session(
                serving_registry=serving_registry,
                serving=serving,
                perf_latency_jsonl=perf_latency_jsonl,
                perf_window_start_et=perf_window_start_et,
                perf_window_end_et=perf_window_end_et,
                perf_summary_every_bins=perf_summary_every_bins,
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.error("VP stream setup failed: %s", exc)
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(exc),
            }))
            await websocket.close(code=1008, reason=str(exc))
            return

        await run_stream_session(
            websocket=websocket,
            lake_root=lake_root,
            session=session,
        )
