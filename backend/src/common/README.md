# Common Module

**Role**: Foundational infrastructure with zero dependencies on other backend modules  
**Audience**: All backend developers  
**Interface**: [INTERFACES.md](INTERFACES.md)

---

## Purpose

Provides shared contracts that enable parallel development and deterministic replay across the entire system. All other backend modules depend on `common`, but `common` depends on nothing else.

**Architectural principle**: This creates a stable foundation where interface changes are deliberate and versioned.

---

## What's Included

### Event Types (`event_types.py`)
Canonical dataclasses for runtime message passing. Every event carries `ts_event_ns` (event time) and `ts_recv_ns` (receive time) in Unix nanoseconds UTC.

**Types**: `OptionTrade`, `FuturesTrade`, `MBP10`, `BidAskLevel`

### Configuration (`config.py`)
Single source of truth for all tunable parameters. Centralized CONFIG singleton accessed by all modules.

**Categories**: Physics windows, monitoring bands, thresholds, score weights, smoothing parameters

### Price Utilities (`price_converter.py`)
ES price normalization utilities used by core analytics and features.

### Storage Schemas (`schemas/`)
Pydantic + PyArrow schema definitions for Bronze/Silver/Gold tiers. Includes SchemaRegistry for version management.

**Tiers**: Bronze (raw), Silver (clean), Gold (derived)

### NATS Bus (`bus.py`)
JetStream wrapper for pub/sub messaging between services.

**Phase**: Used in Phase 2+ (microservices). Phase 1 used in-process queues.

### Run Manifest Manager (`run_manifest_manager.py`)
Tracks run metadata for reproducibility. Captures git commit, config hash, event counts, and output files.

**Use case**: "Re-run with exact same config/code" for debugging or validation.

---

## Key Design Decisions

### Why Separate Event Types from Schemas?

**Event types**: Lightweight dataclasses for fast runtime message passing  
**Schemas**: Heavy Pydantic + PyArrow for storage validation and Parquet writing

This separation allows fast event routing without Pydantic overhead in hot paths.

### Why Flatten Gold Schema?

Parquet columnar format is optimized for flat schemas. Nested JSON loses compression efficiency and requires complex readers. ML frameworks expect flat feature vectors.

### Why EWMA Smoothing?

EWMA is parameter-free (only τ half-life) and deterministic. Kalman filters require process/observation noise estimation → calibration burden. EWMA with τ=2–5s provides sufficient smoothing for 250ms snap ticks.

---

## Testing

Run unit tests for common module:

```bash
cd backend
uv run pytest tests/test_price_converter.py -v    # 16 tests
uv run pytest tests/test_schemas.py -v            # 31 tests
uv run pytest tests/test_run_manifest_manager.py -v  # 24 tests
```

---

## Common Pitfalls

1. **Timestamp conversion**: Databento data arrives in nanoseconds (no conversion needed)
2. **Config mutation**: Don't modify CONFIG during runtime (breaks manifest tracking)
3. **Gold schemas**: Use flat fields, not nested dicts for Parquet efficiency

---

## Evolution Path

**Phase 1** (Complete): Local M4, in-process queues  
**Phase 2** (Current): Single-machine server, NATS JetStream  
**Phase 3** (Future): Colocation, Redpanda/Kafka, Iceberg metadata layer

Schema versioning ensures controlled changes across phases.

---

## References

- **Interface contract**: [INTERFACES.md](INTERFACES.md)
- **PLAN.md**: §2 (Dataset Contracts), §9 (Configuration), §11 (Vendor Contracts)
- **Tests**: `backend/tests/test_*.py`

---

**Ownership**: Agent A (completed ✅)  
**Dependencies**: None  
**Consumers**: All other backend modules
