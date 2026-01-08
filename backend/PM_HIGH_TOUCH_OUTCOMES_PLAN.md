## PM_HIGH Touch → Break/Bounce Baseline (2‑Minute Candles, Futures Only)

### Goal (what this baseline answers)
- **Goal**: Build a **pure historical outcome baseline** around **PM_HIGH** using **only 2‑minute candle OHLC** (derived from trade prints) with **stop-loss-aware** outcome categories in both directions:
  - **Up-from-below**: `BREAK_UP`, `BREAK_UP_STRONG`, `BOUNCE_DOWN`
  - **Down-from-above**: `BREAK_DOWN`, `BREAK_DOWN_STRONG`, `BOUNCE_UP`
- **Purpose**: Establish the unconditional empirical frequencies and provide a clean, deterministic touch/outcome labeling primitive that later feature work can join to.

### Scope (explicit constraints)
- **Product**: futures only (**NOT** futures_options).
- **Level**: `PM_HIGH` only (for now).
- **Dates**: 2025-06-04 to 2025-09-30 (inclusive), using whatever `dt=` partitions actually exist in the lake.
- **Per-contract processing**: run per front-month contract symbol partition (e.g., `ESU5`, `ESZ5`), do not mix roll regimes into a single distribution unless explicitly requested later.

### Source-of-truth data (what to read)
- **Primary lake table**: `silver.future.market_by_price_10_with_levels`
  - **Dataset key**: `silver.future.market_by_price_10_with_levels` (configured in `backend/src/data_eng/config/datasets.yaml`)
  - **Lake path pattern**: `lake/silver/product_type=future/symbol={symbol}/table=market_by_price_10_with_levels/dt={dt}/`
  - **Level source**: column `pm_high` (scalar/broadcast per day).
- **Trade prints source**: MBP-10 includes trade events:
  - **Contract**: `backend/src/data_eng/contracts/bronze/future/market_by_price_10.avsc` (+ silver variants)
  - **Fields used**: `ts_event`, `action`, `price`, `size`, `sequence`
  - **Trade action encoding**: this repo treats trade as `action == "T"` (see `backend/src/data_eng/stages/silver/future/mbp10_bar5s/constants.py`).

### Non-negotiable prerequisite (fail fast)
- **Price unit sanity**: Verify that trade `price` is already in **points** scale (e.g., ~5000 for ES), not a scaled integer (~5e12).
  - **If price is scaled**: stop immediately and resolve the unit conversion at the ingestion layer; do not paper over it in the baseline.
  - **Why**: thresholds (±5/±10 points) must be applied in real point units.

### Time handling (session windows)
- **Time zone**: America/New_York.
- **Session window for touch + outcome evaluation**: 09:30:00–12:30:00 NY (first 3 hours).
- **Outcome horizon**: 4 minutes = 240 seconds after the touch candle close.
- **PM_HIGH definition window**: already handled upstream (05:00–09:30 NY) and stored as `pm_high`.

### Canonical definitions (the math)
- Let \(L\) = `pm_high` for the day.
- **Touch band**: \([L-5,\, L+5]\).
- **Two-minute candle** \(i\): derived from trade prices in a fixed 2‑minute bin, with OHLC:
  - \(O_i, H_i, L_i, C_i\) = open/high/low/close trade prices in that 2‑minute bin.
- **Touch candle**: candle \(i\) where \(C_i \in [L-5,\, L+5]\) **and** prior close is outside band on the correct side (see below).
- **Outcome evaluation window**: the **next 2 candles** after a touch (exactly 4 minutes):
  - window candles: \(i+1\) and \(i+2\)
  - window extrema:
    - \(H_\text{win} = \max(H_{i+1}, H_{i+2})\)
    - \(L_\text{win} = \min(L_{i+1}, L_{i+2})\)

### Step 1 — Build 2‑minute candles (only place trade-level data is used)
- **Trade filter**: keep only trade events (`action == "T"`).
- **Session filter**: keep only trades whose timestamps fall within the session window.
- **Bin alignment**:
  - Two-minute bins must be aligned to clock boundaries starting at 09:30:00 (e.g., 09:30–09:31:59.999…, 09:32–09:33:59.999…, …).
  - Use NY timestamps to define bins, but store candle timestamps as ns (consistent with the lake).
- **Empty bins**:
  - If a bin has no trades, **do not fabricate a candle** (no forward-fill).
- **Per-candle fields**:
  - Required: candle start/end ts, OHLC
  - Recommended for debugging: trade count, total trade volume

### Step 2 — Touch detection (two permutations)

#### A) Up-from-below permutation (price approaching upward)
- **Touch condition**: candle \(i\) is a touch iff:
  - **close in band**: \(C_i \in [L-5,\, L+5]\)
  - **previous close outside band and below**: \(C_{i-1} < L-5\)
- **Touch timestamp**: the touch candle **end** timestamp.
- **Permutation label**: `UP_FROM_BELOW`.

#### B) Down-from-above permutation (price approaching downward)
- **Touch condition**: candle \(i\) is a touch iff:
  - **close in band**: \(C_i \in [L-5,\, L+5]\)
  - **previous close outside band and above**: \(C_{i-1} > L+5\)
- **Touch timestamp**: the touch candle **end** timestamp.
- **Permutation label**: `DOWN_FROM_ABOVE`.

### Step 3 — Debounce (OHLC-only; no further trade scanning)
This debounce logic must be implemented purely from candle OHLC so it’s stable and does not depend on micro-tick sequencing.

- **Concept**: once price is “working the level” (candles intersect the band), we are in-zone and we do not emit another touch until price exits the zone and re-enters from the correct side.
- **Band intersection test (OHLC)**:
  - candle intersects band iff \(L_i \le L+5\) AND \(H_i \ge L-5\)
- **State machine**:
  - **Enter** `in_zone = True` on the first candle that intersects the band.
  - **Exit** `in_zone = False` only when a candle **does not** intersect the band.
  - **Emit touch** only when `in_zone == False` and the relevant permutation’s touch condition (Step 2) is met.

### Step 4 — Outcome labeling (4 minutes ahead, OHLC-only)
Outcomes are defined by whether the window crosses thresholds **at any time** within the next 4 minutes. Ordering within the window is intentionally ignored.

#### A) Up-from-below outcomes (PM_HIGH)
Given touch candle \(i\) and window extrema \(H_\text{win}, L_\text{win}\):
- **BOUNCE_DOWN**:
  - if \(L_\text{win} \le L-5\)
  - Rationale: stop-loss breach within 4 minutes → day-trader stop would be hit.
- **BREAK_UP_STRONG**:
  - else if \(H_\text{win} \ge L+10\)
- **BREAK_UP**:
  - else if \(H_\text{win} \ge L+5\)
- **NO_EVENT**:
  - else

#### B) Down-from-above outcomes (PM_HIGH)
Mirror definitions:
- **BOUNCE_UP**:
  - if \(H_\text{win} \ge L+5\)
- **BREAK_DOWN_STRONG**:
  - else if \(L_\text{win} \le L-10\)
- **BREAK_DOWN**:
  - else if \(L_\text{win} \le L-5\)
- **NO_EVENT**:
  - else

#### Window completeness
- A touch is **evaluable** only if both window candles \(i+1\) and \(i+2\) exist.
  - If missing due to no-trade candles, mark `is_window_complete = False` and exclude from headline stats (but keep for audit if desired).

### Outputs (deliverables for baseline + future joining)

#### Touch-level table (one row per touch)
- **Identity / join keys**:
  - `dt`, `symbol`, `level_type="PM_HIGH"`, `level_price=L`
  - `permutation` in {`UP_FROM_BELOW`, `DOWN_FROM_ABOVE`}
  - `touch_candle_end_ts`, `touch_candle_close`
- **Outcome**:
  - `outcome` (one of the six directional outcomes or `NO_EVENT`)
  - `is_window_complete`
- **Diagnostics (high value for debugging)**:
  - `window_high`, `window_low`
  - `max_favorable_pts`, `max_adverse_pts` computed vs \(L\)
  - `touch_candle_ohlc` (optional but helpful)

#### Summary table (counts + rates)
- Group by: `symbol` (optional), `permutation`, `outcome`
- Report:
  - count, percentage within permutation
  - total touches per permutation
  - optionally, exclude `NO_EVENT` for a “conditional on movement” view (must be explicit if done)

### Failure modes (must be explicit and loud)
- **Missing input partitions**: log and skip date(s) with no `dt=` partition for the symbol.
- **Missing required columns**: hard-fail (contract drift).
- **NaN / invalid pm_high**: skip that day with explicit reason.
- **Too few trades to form candles**: skip that day with explicit reason.
- **Price unit mismatch**: hard-fail and fix upstream; do not continue.

### Verification (minimum high-signal checks)
- **Unit sanity**: print/inspect a handful of trade prices and one day’s \(L\); confirm plausible magnitudes.
- **Candle sanity**:
  - confirm candles align to 2‑minute boundaries starting at 09:30
  - confirm OHLC reflect the first/last/max/min trade prices in each bin
- **Touch sanity**:
  - confirm previous close is outside band on the correct side
  - confirm debounce prevents multiple touches during continuous band intersection
- **Outcome sanity**:
  - hand-check a few touches by looking at the next 2 candles:
    - if window low crosses \(L-5\), outcome must be bounce (directionally correct)
    - else if window high crosses \(L+10\), outcome must be strong break, etc.

### Notes / future extensions (do not implement unless explicitly requested)
- **Other levels**: the same machinery can be reused for `PM_LOW`, `OR_HIGH`, `OR_LOW` by substituting \(L\).
- **Different candle size**: 1‑min, 5‑min, etc. should be a parameter only after this baseline is validated.
- **Trade-level ordering**: intentionally excluded here; if needed later, it should be a separate "high-resolution" baseline with explicit semantics.

---

## RESULTS (2025-06-04 to 2025-09-30)

### Implementation
- Script: `src/data_eng/analysis/v2/pm_high_baseline.py`
- Output: `/tmp/pm_high_touches.csv`

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total Touches | 158 |
| Evaluable (window complete) | 156 |
| ESU5 touches | 121 |
| ESZ5 touches | 35 |

### UP_FROM_BELOW (Approaching PM_HIGH from below)

| Outcome | Count | Percentage |
|---------|-------|------------|
| BOUNCE_DOWN | 60 | 58.3% |
| NO_EVENT | 42 | 40.8% |
| BREAK_UP | 1 | 1.0% |
| BREAK_UP_STRONG | 0 | 0.0% |
| **TOTAL** | **103** | |

### DOWN_FROM_ABOVE (Approaching PM_HIGH from above)

| Outcome | Count | Percentage |
|---------|-------|------------|
| BOUNCE_UP | 42 | 79.2% |
| NO_EVENT | 9 | 17.0% |
| BREAK_DOWN_STRONG | 2 | 3.8% |
| BREAK_DOWN | 0 | 0.0% |
| **TOTAL** | **53** | |

### Key Finding
**PM_HIGH is a strong structural level:**
- **From below**: Only 1% break through (strong resistance)
- **From above**: 79% bounce back up (strong support)
- The level exhibits asymmetric behavior depending on approach direction

### Verification Checks Passed
1. Candle alignment: 2-minute bins start at 09:30, 09:32, etc.
2. Touch detection: Debounce prevents multiple touches during continuous band intersection
3. Outcome labeling: All sampled outcomes correctly match threshold logic
