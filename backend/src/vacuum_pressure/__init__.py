"""Canonical vacuum-pressure dense-grid streaming package.

Canonical modules:
    config: Runtime instrument configuration resolver.
    replay_source: Event ingest adapter (PRE-PROD DBN today, live feed later).
    event_engine: In-memory event-driven force-field computation.
    core_pipeline: Math-first full-grid pressure core pipeline.
    stream_pipeline: Ingest->compute pipeline orchestration.
    server: FastAPI websocket endpoint for dense-grid streaming.
"""
