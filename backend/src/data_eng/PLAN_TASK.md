1. DONE - Review current selector and pipeline touchpoints for selection usage.
2. DONE - Implement volume-roll selector and align selection consumers.
3. DONE - Run selection build, silver one-date run + schema check, then full silver/gold runs for 2025-10-01:2026-01-09 with 8 workers.
4. DONE - Add premarket-trade filter so outage dates are treated like closed sessions.
5. DONE - Rebuild selection map and confirm 2025-11-28 is skipped.
6. DONE - Remove stale silver/gold/index data and rebuild selection, silver, vectors, index, and gold with 8 workers.
7. DONE - Verify selection excludes 2025-11-28 and all gold partitions exist for selected sessions.
8. DONE - Run data sanity checks on vectors and premarket trade coverage.
9. DONE - Report index schema contract status.
