# Swift App - Live Data Integration Plan

## Problem
The Swift MarketParticlePhysicsApp uses synthetic data (`MarketDataSource`) instead of real market data from the backend WebSocket stream.

## Goal
Connect to `ws://localhost:8000/v1/hud/stream?symbol=ESH6&dt=2026-01-06` and parse Arrow IPC binary data to drive the physics simulation with real market data.

## Plan

### Phase 1: Arrow IPC Parser [IN PROGRESS]
1. Research Arrow IPC streaming format specification
2. Implement FlatBuffers metadata parser (Arrow uses FlatBuffers for schema/metadata)
3. Implement Arrow IPC message reader (continuation marker, metadata, body)
4. Implement column readers for required types: Int64, Float64, Bool, String
5. Create typed row readers for each surface schema

### Phase 2: WebSocket Client
1. Create URLSession-based WebSocket client
2. Handle JSON control frames (batch_start, surface_header)
3. Route binary frames to Arrow IPC parser based on pending surface header
4. Emit parsed data via delegate/callback pattern

### Phase 3: Data Source Adapter
1. Create `LiveMarketDataSource` conforming to same interface as `MarketDataSource`
2. Transform parsed snap/wall/vacuum/physics data into `MarketFrame` format
3. Handle connection lifecycle (connect, reconnect, disconnect)

### Phase 4: ViewModel Integration
1. Add data source toggle (Live vs Synthetic)
2. Update `EngineViewModel` to accept either data source
3. Handle backpressure (if data arrives faster than rendering)

### Phase 5: Testing & Verification
1. Unit tests for Arrow IPC parser
2. Integration test with real backend stream
3. Visual verification of spot price (~6800 for ESH6)

## Arrow IPC Format Reference

### Message Structure
```
[4 bytes] Continuation marker: 0xFFFFFFFF
[4 bytes] Metadata length (little-endian int32)
[N bytes] FlatBuffers metadata
[padding] Align to 8 bytes
[M bytes] Body (column buffers)
```

### Message Types
- Schema (type=1): Column names, types, nullability
- RecordBatch (type=3): Actual data with validity bitmaps + value buffers

### Required Column Types
- Int64: window_end_ts_ns, spot_ref_price_int
- Float64: mid_price, depth_qty_rest, vacuum_score, physics_score
- Int32: rel_ticks
- Bool: book_valid
- String (Utf8): side

## Current Status
- Phase 1: [COMPLETE] Arrow IPC parser implemented and tested
- Phase 2: [COMPLETE] WebSocket client implemented
- Phase 3: [COMPLETE] LiveMarketDataSource adapter created
- Phase 4: [COMPLETE] ViewModel integration with Live/Synthetic toggle
- Phase 5: [IN PROGRESS] Testing & Verification
