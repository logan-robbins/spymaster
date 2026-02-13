"""Live vacuum-pressure dense-grid streaming package.

Canonical modules:
    config: Runtime instrument configuration resolver.
    replay_source: Event ingest adapter (DBN today, live feed later).
    event_engine: In-memory event-driven force-field computation.
    stream_pipeline: Live ingest->compute pipeline orchestration.
    server: FastAPI websocket endpoint for dense-grid streaming.
"""
