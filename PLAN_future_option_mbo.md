# Technical Implementation Plan: Dynamic Strike-Level GEX Pipeline

**Target:** Build independent options GEX pipeline that computes gamma exposure at top 5 strikes above/below current price (no P_ref dependency).

**Goal:** Real-time GEX heatmap that moves with price, showing dealer hedging pressure at immediate strike levels surrounding current market price.

**CRITICAL:** Dynamic price-referenced GEX focused on strike-level physics, not broad zones around fixed reference levels.

---

## 1. Raw Data Discovery ✅ COMPLETE

### 1.1 Raw Data Location
Raw future options MBO data is stored under:
`backend/lake/raw/source=databento/product_type=future_option/symbol=ES/table=market_by_order_dbn/`

Contains standard Databento artifacts:
- Daily compressed DBN files (glbx-mdp3-YYYYMMDD.mbo.dbn.zst)
- Metadata files (condition.json, manifest.json, metadata.json, symbology.json)

### 1.2 Data Characteristics
- Date range: 2026-01-06 through 2026-01-19
- Schema: MBO (rtype=160)
- Parent symbol access: ES.OPT provides all ES options
- Underlying contracts: Quarterly ES futures (ESH6, ESM6, ESU6, ESZ6)

### 1.3 Option Symbol Anatomy
**Valid quarterly options follow pattern:** `ES[HMUZ]\d{1,2}\s+[CP]\d+`
- Underlying: `ESH6` (March 2026 contract)
- Right: `C` (call) or `P` (put)
- Strike: Index points (6000, 6005, 6010, etc.)

**Filter criteria:**
- Keep: Standard quarterly options only
- Exclude: User-defined spreads (UD:* patterns), calendar spreads (containing `-`)

### 1.4 Option-Specific Characteristics
**Key differences from futures:**
- **Strike prices:** Quoted in ES index points (6000, 6005, 6010...) - same scale as P_ref
- **Premium prices:** Quoted in index points, converted to dollars via multiplier
- **Contract multiplier:** 50 (same as futures)
- **Expiration:** 3rd Friday of contract month
- **Tick sizes:** Premium ticks vary by price level ($0.05/$0.25)

---

## 2. Option Contract Selection (DIFFERENT FROM FUTURES) ✅ COMPLETE

### 2.1 Why Futures Selection Does NOT Apply
Futures selection (`mbo_contract_day_selection.parquet`) prioritizes volume dominance for the underlying contract. Options require different criteria:

1. **DTE (Days To Expiration)** - Options near expiration have highest gamma exposure
2. **Front-month priority** - Most liquid options with greatest market impact
3. **Strike relevance** - Focus on strikes near current price levels

### 2.2 Option Expiration Logic
Options expire on 3rd Friday of contract month. For front-month selection:
- Calculate DTE for each option series
- Filter to options expiring within 45 days (meaningful gamma)
- Select shortest DTE (highest gamma impact)
- Exclude expired options

### 2.3 Dynamic Price-Referenced GEX (No P_ref Dependency)
**Core Concept:** GEX physics should be computed relative to current market price, not a fixed reference level. As price moves throughout the session, GEX distribution moves with it.

**Reference Price Calculation:**
- Use NBBO midpoint or candle average as dynamic reference point
- Update reference price every 5-second window based on current market conditions
- No dependency on futures P_ref or premarket data

**Real-Time Visualization:**
- As price moves on chart, GEX heatmap moves with it
- See GEX density at strikes immediately above/below current price
- Track how GEX distribution evolves as price approaches different strike levels

---

## 3. Bronze Stage: `BronzeIngestFutureOptionMbo` ✅ COMPLETE

### 3.1 File Location
`backend/src/data_eng/stages/bronze/future_option_mbo/ingest_preview.py`

### 3.2 Key Differences from Futures Bronze
1. **Symbol parsing** extracts: underlying (ESH6), right (C/P), strike (index points)
2. **Partitioning by underlying** (ESH6, ESM6) NOT by option_symbol (too granular)
3. **Expiration date** calculated from underlying code
4. **Strike is in index points** (6000, 6005) - same magnitude as P_ref

### 3.3 Processing Requirements
- Parse option symbols to extract underlying contract, call/put indicator, strike price
- Filter to standard quarterly options only (exclude spreads, exotics)
- Calculate expiration dates from contract codes
- Partition data by underlying contract for efficient processing
- Maintain strike prices in index points (matching P_ref scale)

### 3.4 Data Schema Requirements
Bronze contract must include all standard MBO fields plus option-specific metadata:
- Standard fields: timestamps, order_id, price, size, action, side, sequence
- Option fields: underlying contract, full option symbol, expiration date, strike price, call/put indicator
- Strike prices stored in index points (matching futures P_ref scale)
- Premium prices maintained in fixed-point representation

---

## 4. Silver Stage: GEX Feature Computation ✅ COMPLETE

### 4.1 Purpose
Compute Gamma Exposure (GEX) features from option order book data, measuring dealer hedging pressure above and below key price levels.

### 4.2 Core Concept: Dynamic GEX Physics
GEX quantifies how much market makers must hedge option positions around current price:
- **Positive GEX**: Dealers are net long options → must buy deltas when spot moves up
- **Negative GEX**: Dealers are net short options → must sell deltas when spot moves up
- **Dynamic**: GEX heatmap moves with price throughout the session

### 4.3 Strike-Focused GEX (Relative to Current Price)
**Top 5 Strikes Above/Below Current Price:**
- Identify 5 nearest strikes above and below current NBBO midpoint
- Calculate GEX at each specific strike level (not broad zones)
- Focus on immediate price action: strikes within ~$5-10 of current price
- GEX density heatmap centered on current market price

**Reference Price Options:**
- **NBBO Midpoint**: (best_bid + best_ask) / 2
- **Candle Average**: Volume-weighted average price over window
- **Last Trade Price**: Most recent transaction price

### 4.4 Dynamic GEX Calculation Fundamentals
For each option position in each 5-second window:
1. Calculate current reference price (NBBO midpoint or candle average)
2. Identify top 5 strikes above and below current reference price
3. Calculate Black-Scholes gamma using: spot (current reference), strike, DTE, IV, risk-free rate
4. Scale by position size and contract multiplier (50)
5. Apply standard GEX scaling (×100 for per-1% move convention)
6. Aggregate GEX at each specific strike level (not broad distance buckets)

### 4.5 Strike-Level GEX Feature Set
**Per-Strike GEX Features:**
- `gex_strike_N_call`: GEX from calls at Nth strike above current price
- `gex_strike_N_put`: GEX from puts at Nth strike above current price
- `gex_strike_negN_call`: GEX from calls at Nth strike below current price
- `gex_strike_negN_put`: GEX from puts at Nth strike below current price
- Net GEX at each strike: `gex_strike_N_net = gex_strike_N_call - gex_strike_N_put`

**Aggregate GEX Features:**
- Total GEX above current price: sum of call+put GEX at strikes +1 to +5
- Total GEX below current price: sum of call+put GEX at strikes -1 to -5
- GEX imbalance ratio: `(above_GEX - below_GEX) / (above_GEX + below_GEX + epsilon)`

**Option Flow Features (Strike-Specific):**
- Add/pull intensity at each strike level
- Order flow imbalance by strike proximity to current price
- Repricing activity (orders moving toward/away from current price level)

**Derived Features (Same as Vacuum):**
- First differences: `d1_*` (momentum)
- Second differences: `d2_*` (acceleration)
- Third differences: `d3_*` (jerk/change in acceleration)

### 4.6 Dynamic Strike-Based Processing Architecture
**Real-Time Price Tracking:**
- Calculate current reference price every 5-second window (NBBO midpoint or candle average)
- Identify top 5 strikes above and below current reference price
- Update strike set dynamically as price moves throughout session

**Per-Strike GEX Calculation:**
- For each of the 10 key strikes (±5 from current price), compute:
  - Call GEX from orders at that strike
  - Put GEX from orders at that strike
  - Net GEX (calls - puts) at that strike
  - Order flow metrics (adds, pulls, repricing) at that strike

**Visualization Concept:**
- As price moves on chart, GEX heatmap moves with it
- See GEX "pressure" at strikes immediately surrounding current price
- Color-coded GEX intensity: red for negative (resistance), green for positive (support)
- Real-time evolution as price approaches different strike levels

**Key Transfer Insight from Futures Vacuum:**
Futures vacuum captures whether approaching price creates "resistance" (orders pushing back) or "vacuum" (orders pulling away). For options, GEX captures whether dealer hedging creates "resistance" (net GEX pushing against direction) or "support" (net GEX aligned with direction). The same windowed order book simulation and flow intensity metrics apply - but now dynamically centered on current price rather than fixed P_ref.

**Independent Processing (No Futures Dependencies):**
- Calculate current reference price from option MBO data itself
- No dependency on futures silver vacuum or P_ref
- Self-contained options GEX calculation using NBBO/candle data
- Filter options to front-month (shortest DTE) with meaningful gamma (≤45 days)

**Real-time Processing:**
- Simulate option order book evolution across 5-second windows
- Track order additions, modifications, cancellations, and fills
- Accumulate order flow metrics (add intensity, pull intensity)
- Compute GEX snapshots at each window boundary

**Feature Computation:**
- Aggregate GEX by strike buckets relative to P_ref
- Calculate net positioning (call - put GEX) at key levels
- Measure GEX asymmetry above vs below P_ref
- Track option order flow imbalances and intensities

**Feature Enhancement:**
- Add derivative features (d1, d2, d3) matching futures vacuum pattern
- Compute approach direction from option trade stream (when available)
- Maintain compatibility with existing futures feature schema

### 4.7 Silver Data Contract

Silver contract defines all GEX features with double precision, including base features, derivatives (d1/d2/d3), and window metadata. Schema matches futures vacuum pattern for integration.

---

## 5. Optional Gold Stage Integration

### 5.1 Enhanced Trigger Vectors (Optional)

**Integration Approach:**
- Join dynamic GEX features with futures vacuum features for comprehensive analysis
- Combine strike-level GEX physics with futures-level vacuum physics
- Enable ML models to see both immediate strike pressures and broader market structure
- Maintain backward compatibility with existing futures-only trigger vectors

### 5.2 Feature Schema Expansion (Optional)

**Combined Feature Set:**
- Futures vacuum features (existing displacement, slope, flow metrics)
- Dynamic GEX features (strike-level pressures, net positioning)
- Cross-domain features (GEX response to vacuum conditions)
- Maintain derivatives (d1/d2/d3) for momentum analysis across all features

---

## 6. Implementation Structure ✅ COMPLETE

### 6.1 New Components Required
**Stage Implementations:**
- Bronze ingestor for option MBO data parsing and partitioning
- Silver GEX computer for real-time gamma exposure calculations

**Data Contracts:**
- Bronze schema: Extended MBO fields with option metadata
- Silver schema: GEX features with derivatives and window metadata

### 6.2 Modified Components
**Configuration:**
- Add option dataset definitions to datasets.yaml
- Register option pipeline stages in pipeline.py
- Extend vector schema with GEX features

**Integration:**
- Modify gold trigger vectors to join GEX features
- Maintain backward compatibility with existing vacuum features

---

## 7. Execution Dependencies

### 7.1 Independent Pipeline Processing
**Simplified Dependency Chain:**
1. Options bronze (processes raw MBO, independent)
2. Options silver (computes dynamic GEX relative to current price, independent)
3. Gold integration (joins futures vacuum + options GEX, optional)

**Parallel Processing:**
- Options bronze and silver run completely independently
- No dependency on futures pipeline timing or P_ref computation
- Can process options data immediately upon availability
- Gold integration only needed if combining with futures vacuum features

### 7.2 Implementation Phases
**Phase 1: Core Implementation**
- Build bronze and silver stages with contracts
- Update configuration and pipeline registration

**Phase 2: Data Processing**
- Process options bronze (completely independent)
- Process options silver immediately after (no dependencies)
- Optional: Update gold integration if combining with futures vacuum

**Phase 3: Validation & Visualization**
- Verify dynamic strike identification and GEX calculation
- Test real-time GEX heatmap movement with price
- Validate GEX physics around key strikes
- Optional: Test gold integration with futures vacuum features

### 7.2 Real-Time GEX Visualization Concept
**Dynamic Heatmap:**
- Price moves on chart → GEX heatmap moves with it
- Color-coded GEX at each of top 5 strikes above/below current price
- Red: Negative GEX (dealer resistance to upward moves)
- Green: Positive GEX (dealer support for upward moves)
- Intensity: Magnitude of GEX pressure

**Strike-Level Focus:**
- Strike +1: Immediate next resistance/support
- Strike +2: Near-term battleground
- Strike +3: Medium-term positioning
- Strikes -1, -2, -3: Equivalent below current price

**Real-Time Evolution:**
- As price approaches strike +2, see GEX building there
- Monitor how GEX distribution shifts as price moves through strike levels
- Track option order flow changes around key strikes
