# Serving Perm Runtime Integration (Live Task)

1. [completed] Discover current backend/frontend runtime paths and identify integration points for live model scoring.
2. [completed] Implement incremental backend `perm_derivative` runtime scorer with tunable parameters and validation.
3. [completed] Wire scorer outputs into websocket `grid_update` payload contract.
4. [completed] Add runtime parameter sourcing from config defaults + stream-time overrides.
5. [completed] Update frontend ingest/render path to consume backend model metrics and drive projection bands.
6. [completed] Add/adjust tests for runtime math, stream contract, and config override parsing.
7. [completed] Verify with `uv run pytest` and frontend typecheck.
8. [completed] Update `README.md` with new runtime settings and usage.
