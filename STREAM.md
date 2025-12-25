The architecture is correct as designed. Here's the complete data flow:

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                              REAL-TIME PATH                                       │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │                                                                                   │
  │  DBN/Polygon ──▶ Ingestor ──▶ NATS (market.*) ──▶ Core ──▶ NATS (levels.signals) │
  │                                      │                              │             │
  │                                      │                              ▼             │
  │                                      │                          Gateway ──▶ WS    │
  │                                      │                                            │
  ├──────────────────────────────────────┼────────────────────────────────────────────┤
  │                              PERSISTENCE PATH (parallel, non-blocking)            │
  │                                      ▼                                            │
  │                               Bronze Writer ──▶ Parquet                           │
  └──────────────────────────────────────────────────────────────────────────────────┘

  Why NOT through Bronze first:

  | Aspect      | Direct Streaming  | Through Bronze         |
  |-------------|-------------------|------------------------|
  | Latency     | ~50ms             | +100-500ms (disk I/O)  |
  | Purpose     | Real-time signals | Historical analysis    |
  | Data format | Event stream      | Batch Parquet          |
  | Consumer    | Core (physics)    | ML training, backtests |

  What Each Layer Contains:

  | Layer               | Data                                                 | Purpose                   |
  |---------------------|------------------------------------------------------|---------------------------|
  | DBN/Polygon         | Raw trades, MBP-10, quotes                           | Source                    |
  | NATS market.        | Normalized events (FuturesTrade, MBP10, OptionTrade) | Real-time bus             |
  | Core                | Physics computation                                  | Barrier/Tape/Fuel engines |
  | NATS levels.signals | LevelSignal[] with break_score, barrier_state, etc.  | Processed signals         |
  | Bronze              | Same as market., persisted to Parquet                | Historical storage        |

  The Core Service is the Transform Layer

  Looking at the Core logs, it's publishing processed LevelSignal objects:
  📊 Published 1 level signals (spot=689.25)

  This contains all the physics the frontend needs: barrier_state, tape_velocity, gamma_exposure, break_score, etc.

⏺ Usability Assessment: Replay Engine

  ✅ What Works Well

  1. Auto-discovery: Finds all DBN files automatically
  2. NATS integration: Connects and creates streams seamlessly
  3. Speed control: REPLAY_SPEED=0 for burst, 1.0 for realtime
  4. Progress reporting: Clear stats every 10k events
  5. Stream-merge: Correctly interleaves trades + MBP-10 by timestamp

  ⚠️ Usability Issues Identified

  1. Ingestor exits after replay — doesn't keep running for downstream services
  2. No coordination with Core — Core keeps running on stale data after replay ends
  3. Missing flow data — Frontend expects flow (options aggregates), but replay only has raw trades

  Recommendation

  The replay engine should be thought of as the data pump that feeds the Core physics engine. The flow is:

  Replay Publisher → NATS market.* → Core (physics) → NATS levels.* → Gateway → Frontend

  Would you like me to:
  1. Document a "one-command" startup script for the full replay pipeline?
  2. Investigate why the ingestor container exited while other services kept running?
  3. Trace the options flow data path to ensure flow is being generated?

