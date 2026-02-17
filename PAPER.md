# Paper Outline: Regime Change Detection in CME Microstructure via Vacuum-Pressure Lattice Features

## Target

Journal submission (Quantitative Finance, Journal of Financial Economics, or similar peer-reviewed venue). Cross-list to ArXiv q-fin.TR. Publishable dataset included. Engine internals NOT disclosed — describe transformations conceptually with formulas, not source code or architecture. Conservative framing: preliminary results requiring multi-session validation before strong claims.

---

## 1. Title & Abstract

**Working title**: *Vacuum-Pressure Lattice Features for Sub-Second Regime Detection in Index Futures Microstructure*

**Abstract** (~250 words):
- Problem: detecting directional regime changes at 100ms resolution from limit order book data
- Approach: transform raw MBO (Market-by-Order) tick data into a fixed-width time-binned lattice of physics-inspired features (mass, velocity, acceleration, jerk of order flow) with a two-force composite model and per-cell spectrum scoring
- Dataset: 1 hour of MNQH6 RTH data (3.6M cells), published as an immutable parquet dataset for reproducibility
- Experiments: 6 independent signal generators evaluated with identical TP/SL harness (8:4 tick asymmetric, 2:1 R:R)
- Key result: all 6 beat 33.3% breakeven on this session; best performer (PFP) achieves 40.6% TP rate with 731 signals (+0.51 ticks/trade mean PnL)
- Framing caveat: these are preliminary findings on a single session; multi-session and multi-instrument validation is required before generalization claims
- Contribution: the lattice representation itself, the published dataset, and preliminary evidence that spatial/temporal derivative features of order flow carry short-horizon directional information

---

## 2. Introduction

- Motivation: LOB microstructure research overwhelmingly uses L2/L3 snapshots or trade-and-quote; full MBO reconstruction at sub-second resolution is underexplored
- Gap: existing regime detection (HMM, change-point) operates on aggregated price series, not on the full spatial structure of the book
- Our contribution:
  1. A physics-inspired feature lattice (price-tick x time-bin) derived from MBO event streams
  2. A published, immutable research dataset (parquet) with 33 columns x 3.6M rows enabling reproducible microstructure research
  3. Six independent experiments providing preliminary evidence of directional edge from lattice features
- Scope and caveat: single instrument (MNQH6), single session — we present a novel representation and initial experimental evidence, not a validated trading strategy; claims are conditional on future multi-session confirmation

---

## 3. Related Work

### 3.1 Limit Order Book Modeling
- LOB shape features and queue dynamics (Cont, Stoikov, & Talreja 2010; Huang, Lehalle, & Rosenbaum 2015)
- Deep LOB models (Sirignano 2019; Zhang et al. 2019; Tran et al. 2017)
- Order flow imbalance as a directional predictor (Cont, Kukanov, & Stoikov 2014)

### 3.2 Regime Detection in Financial Markets
- Hidden Markov Models for market regimes (Hamilton 1989; Bulla & Bulla 2006)
- Change-point detection (Andreou & Ghysels 2002; Fryzlewicz 2014)
- Information-theoretic approaches: entropy in microstructure (Cont 2001; Easley, Lopez de Prado, & O'Hara — VPIN 2012)

### 3.3 Physics-Inspired Models in Finance
- Potential field / pressure metaphors for order book dynamics
- Agent-based models with force analogies
- Distinction from our approach: we use physics as a feature engineering language, not as a generative model

### 3.4 MBO-Level Analysis
- Rarity of full MBO studies vs aggregated L2 data
- Databento and similar vendors making MBO data accessible
- Gap: no prior work (to our knowledge) constructs a full price-tick × time-bin lattice with derivative chain features from MBO events

---

## 4. Data

### 4.1 Source Data
- Vendor: Databento
- Schema: MBO (Market-by-Order) — individual order lifecycle events (add, cancel/modify, fill) with nanosecond timestamps, order IDs, price, quantity, side
- Instrument: MNQH6 (Micro E-mini Nasdaq 100 Futures, March 2026 expiry)
- Exchange: CME Globex
- Date: 2026-02-06
- Capture window: 09:25–10:25 ET (first hour of Regular Trading Hours with 5 min pre-open warmup)
- Tick size: $0.25

### 4.2 Transformation Pipeline (Conceptual)
- **Stage 1 — Book Reconstruction**: MBO events replayed in timestamp order to maintain a full price-level depth map. Each event (add, modify, cancel, fill) updates per-price-level bid/ask depth and tracks individual order IDs for accurate cancellation/modification accounting.
- **Stage 2 — Fixed-Width Time Binning**: Continuous event stream discretized into 100ms bins. Within each bin, all events are processed; at bin boundaries, a snapshot of the full lattice state is emitted.
- **Stage 3 — Derivative Chain**: Per price tick, exponentially-weighted moving averages compute velocity (d1), acceleration (d2), and jerk (d3) of three mass channels (add, pull, fill) and rest depth. Continuous-time EMA with configurable time constants (tau_v=2s, tau_a=5s, tau_j=10s).
- **Stage 4 — Two-Force Model**: Per tick, composite pressure (liquidity-building) and vacuum (liquidity-draining) scores computed from weighted combinations of velocities and accelerations.
- **Stage 5 — Spectrum Scoring**: Per-cell independent kernel: multi-window rollup of pressure-vacuum composite → derivative chain → robust z-score normalization → tanh-compressed score in [-1, +1] → three-state classification {pressure, neutral, vacuum}.
- **Stage 6 — Serve-Time Grid Extraction**: Full engine grid sliced to ±50 ticks around BBO midpoint, yielding 101 columns per bin.

### 4.3 Published Dataset Schema
- The raw Databento MBO data is licensed and NOT redistributed. Researchers can independently acquire the same data from Databento using the instrument, date, and schema specified in Section 4.1.
- What IS published: the computed (post-transformation) immutable dataset — the output of Stages 1–6 above. This is our original derived work, not Databento's raw data.
- `bins.parquet`: 35,999 rows x 11 columns (bin metadata: timestamps, mid_price, BBO, book validity)
- `grid_clean.parquet`: 3,635,899 rows x 33 columns (101 ticks x 35,999 bins, one row per (bin, k) cell)
- Table of all 33 columns with descriptions and derivative orders
- Dataset available at [repository TBD]
- SHA256 checksums provided for integrity verification

---

## 5. Feature Definitions

### 5.1 Coordinate System
- Relative tick index k: k=0 is BBO midpoint, k<0 bid side, k>0 ask side
- 101-column grid: k ∈ [-50, +50], tick size $0.25, covers ±$12.50 around spot

### 5.2 Mass Channels and Exponential Decay
- Three mass accumulators per tick: add (new orders), pull (cancellations), fill (executions)
- Exponential decay: `mass(t) = mass(t-1) * exp(-dt/tau) + delta`, tau_rest_decay = 30s
- Motivation: decay prevents stale order activity from dominating current state; 30s chosen to span typical CME queue refresh cycles

### 5.3 Derivative Chain
- Continuous-time EMA-smoothed derivatives
- `alpha(dt, tau) = 1 - exp(-dt/tau)` — adapts to irregular event spacing
- Velocity: `v = alpha * (delta/dt) + (1-alpha) * v_prev`
- Acceleration: same chain applied to velocity
- Jerk: same chain applied to acceleration
- Time constants: tau_v=2s (fast, tracks individual events), tau_a=5s (medium, smooths over micro-noise), tau_j=10s (slow, captures structural inflection)
- Applied independently to add, pull, fill mass channels AND rest depth

### 5.4 Two-Force Composite Model

#### 5.4.1 Formulas
- `pressure = 1.0*v_add + 0.5*max(v_rest_depth,0) + 0.3*max(a_add,0)` — liquidity building
- `vacuum = 1.0*v_pull + 1.5*v_fill + 0.5*max(-v_rest_depth,0) + 0.3*max(a_pull,0)` — liquidity draining

#### 5.4.2 Coefficient Selection Rationale

Coefficients are derived from physics-inspired first-principles reasoning, not empirical optimization:

- **c1=1.0 (v_add) and c4=1.0 (v_pull)**: The primary signal channels — new orders arriving and orders being cancelled are the baseline forces, weighted equally at unit scale
- **c5=1.5 (v_fill)**: Fills are weighted higher than pulls because a fill permanently removes liquidity (the order is consumed by a trade), whereas a pull just relocates it. A fill is a stronger vacuum signal than a cancellation
- **c2=0.5 (max(v_rest_depth,0)) and c6=0.5 (max(-v_rest_depth,0))**: Net resting depth velocity is a secondary confirmation — if depth is growing (positive), that supports pressure; if shrinking (negative), that supports vacuum. Weighted at half the primary signal because it is partially redundant with v_add and v_pull
- **c3=0.3 (max(a_add,0)) and c7=0.3 (max(a_pull,0))**: Acceleration terms are only used when positive (i.e., the rate is increasing). These act as momentum boosters — if adding is accelerating, pressure is building faster. Weighted lowest because they are noisier higher-order derivatives

These coefficients are set from first principles, NOT optimized on the evaluation dataset. We note this as a limitation and discuss robustness in Section 9.

### 5.5 Spectrum Score
- Composite: `(P - V) / (|P| + |V| + eps)` — normalized net pressure in [-1,1]
- Multi-window trailing mean rollup (windows: 5, 10, 20, 40 bins = 0.5s to 4s)
- Derivative chain on rollup → robust z-score (median/MAD, window=300 bins, min_periods=75) → tanh compression (scale=3.0)
- Final weighted combination of d1/d2/d3 z-scores (weights: 0.55, 0.30, 0.15)
- State thresholding: pressure if score ≥ 0.15, vacuum if score ≤ -0.15, neutral otherwise

---

## 6. Evaluation Methodology

### 6.1 Baseline
- Existing directional edge approach (aggregated spectrum-score regime detection) achieves 28.5% TP rate on this dataset
- Below 33.3% breakeven for 2:1 R:R

### 6.2 Signal Detection Protocol
- Threshold crossing with direction change detection
- Cooldown enforcement (configurable per experiment, 20–40 bins = 2–4 seconds)
- 300-bin warmup exclusion (30 seconds) for rolling window fill
- Pseudocode provided in Appendix B

### 6.3 TP/SL Evaluation
- Asymmetric: TP = 8 ticks ($2.00), SL = 4 ticks ($1.00), 2:1 reward-to-risk
- Breakeven TP rate: 33.3%
- Max hold: 1200 bins (120 seconds)
- Forward-looking evaluation from signal bin
- Entry price = mid_price at signal bin (assumes immediate fill at mid — a simplification discussed in limitations)

### 6.4 Shared Harness Guarantee
- All 6 experiments use identical signal detection and outcome evaluation implementations
- Prevents evaluation methodology drift across experiments
- Common utilities: rolling OLS slope, robust z-score (median/MAD with 1.4826 MAD-to-sigma scaling factor)

---

## 7. Experiments

Each experiment is described as: thesis → mathematical construction → parameter choices → results table → interpretation.

### 7.1 Asymmetric Derivative Slope (ADS)
- **Thesis**: OLS slopes of bid-vs-ask velocity asymmetry are earliest leading indicators of regime change
- **Construction**: Three spatial bands (inner 3-tick / mid 8-tick / outer 12-tick) → per-band add/pull asymmetry → inverse-sqrt bandwidth weighting → multi-scale OLS slopes (w=10,25,50 bins) → robust z-score → tanh-compressed weighted combination
- **Design choices**: inverse-sqrt bandwidth weighting (narrower bands resolve faster, get more weight); multi-scale slope ensemble captures both fast and slow asymmetry shifts
- **Results table**: threshold sweep (0.02–0.20), best at 0.02 → 40.2% TP, 951 signals, +0.30 ticks/trade
- **Interpretation**: consistent profitability across all thresholds; high signal volume makes it statistically robust

### 7.2 Spatial Pressure Gradient (SPG)
- **Thesis**: Spatial derivative dP/dk of pressure/vacuum fields encodes directional intent — walls (high pressure gradient) block price, vacuums (negative gradient) attract it
- **Construction**: Central differences along k-axis → mean gradient by side (±16 ticks) → wall and pull directional signals → dual EMA smoothing (fast 5-bin + slow 20-bin) → spatial curvature correction (second derivative around spot)
- **Results table**: threshold sweep, best at 2.606 → 39.3% TP, 863 signals, +0.03 ticks/trade
- **Notable finding**: 99% short-biased (854/863 signals were short) — likely capturing persistent downward microstructure asymmetry during this specific session rather than symmetric regime detection. Discussed as a limitation.

### 7.3 Entropy Regime Detector (ERD)
- **Thesis**: Shannon entropy spike in the {pressure, neutral, vacuum} state distribution precedes regime transitions — the book fragments from ordered to disordered before reversing
- **Construction**: Per-bin Shannon entropy (full grid, above-spot, below-spot) → entropy asymmetry (H_above - H_below) → robust z-score spike gate (z > 0.5) → two directional variants tested (spectrum-score-based vs entropy-asymmetry-based)
- **Results table**: threshold sweep (0.05–2.00), bimodal behavior — 40.3% TP at 0.30 threshold (77 signals), 53.8% TP at 1.50 threshold (13 signals)
- **Interpretation**: the entropy-spike thesis shows promise but the highest TP rates coincide with very low signal counts (n=13), making them statistically unreliable. The moderate-threshold regime (n=77–110) provides more credible evidence.

### 7.4 Pressure Front Propagation (PFP)
- **Thesis**: Aggressive order activity propagates spatially from inner ticks (near BBO) outward; when inner-tick velocity leads outer-tick velocity by more than baseline lag, an aggressive participant is acting directionally
- **Construction**: Inner zone (±1–3 ticks) and outer zone (±5–12 ticks) intensity → EMA cross-product lead-lag metrics with 5-bin lag → add/pull channel blend (60/40)
- **Results table**: threshold sweep, best at 0.098 → 40.6% TP, 731 signals, +0.51 ticks/trade
- **Key result**: best balanced performer across all experiments — highest TP rate among high-volume strategies and highest per-trade PnL among statistically robust strategies

### 7.5 Jerk-Acceleration Divergence (JAD)
- **Thesis**: When jerk (d3) of add/pull diverges between bid and ask sides, it signals the inflection point BEFORE acceleration changes sign; jerk-acceleration agreement confirms direction, disagreement signals early reversal
- **Construction**: Distance-weighted (1/|k|) spatial aggregation → bid-ask divergence for jerk and acceleration channels → adaptive agreement/disagreement weighting (0.4/0.6 when agreeing, 0.8/0.2 when disagreeing — trust jerk more at disagreement) → robust z-score → tanh compression
- **Results table**: threshold sweep (0.05–0.30), best at 0.05 → 38.5% TP, 1253 signals, +0.10 ticks/trade
- **Interpretation**: highest signal volume of all experiments; the agreement/disagreement weighting adds interpretability but the overall edge per signal is modest

### 7.6 Intensity Imbalance Rate-of-Change (IIRC)
- **Thesis**: The rate of change of the add-to-pull intensity ratio captures order-flow toxicity momentum
- **Construction**: Side-summed velocity (±16 ticks) → Laplace-smoothed (+1.0) intensity ratio → log imbalance → dual-scale OLS slope (fast 10-bin + slow 30-bin, 60/40 blend) → noise floor filter (|imbalance| ≥ 0.1, removes 94.6% of bins)
- **Results table**: threshold sweep (0.001–0.05), best at 0.02 → 38.5% TP, 26 signals, +2.77 ticks/trade
- **Interpretation**: highest per-trade PnL but only 26 signals — too few for statistical confidence; the aggressive noise floor filter produces highly selective but rare signals

---

## 8. Results

### 8.1 Summary Tables
- **Table 1**: Ranking by best TP rate (minimum 20 signals) — all 6 experiments, best threshold, TP%, signal count, mean PnL, events/hr, vs baseline, vs breakeven
- **Table 2**: Ranking by mean PnL per trade (minimum 20 signals)

### 8.2 Cross-Experiment Analysis
- All 6 beat both baseline (28.5%) and breakeven (33.3%) at their optimal threshold
- Trade-off frontier: signal volume vs TP rate vs PnL/trade
  - High-volume, consistent edge: PFP (731 signals, +0.51t), ADS (951 signals, +0.30t)
  - Moderate-volume, moderate edge: SPG (863 signals, +0.03t), JAD (1253 signals, +0.10t)
  - Low-volume, high selectivity: ERD (77 signals, +0.14t), IIRC (26 signals, +2.77t)
- SPG directional bias finding: persistent spatial asymmetry during this session warrants investigation across sessions

### 8.3 Statistical Analysis
- Binomial confidence intervals on TP rates for each experiment at optimal threshold
- Multiple comparison correction (Bonferroni or Holm-Bonferroni) across 6 experiments × threshold sweeps
- Expected PnL under null hypothesis (random 50/50 directional signals → ~33.3% TP at 2:1 R:R if price is a random walk)
- Signal autocorrelation analysis: are signals clustered or uniformly distributed across the session?

### 8.4 Robustness Checks
- Sensitivity to cooldown parameter (vary ±50%)
- Sensitivity to spatial band definitions (vary by ±2 ticks)
- In-sample vs held-out split within the 1-hour window (first 30 min calibrate, second 30 min evaluate)

---

## 9. Discussion

### 9.1 Interpretation of Results
- The lattice representation preserves spatial structure lost in aggregated order-flow metrics (OFI, VPIN)
- Physics-inspired derivative chain (v/a/j) captures rate-of-change dynamics invisible in level data
- Per-cell independence in spectrum scoring avoids cross-tick information leakage
- PFP's lead-lag signal is consistent with institutional execution patterns (aggress at BBO, then sweep deeper levels)

### 9.2 Two-Force Model Sensitivity
- How robust are results to coefficient perturbation? (Monte Carlo perturbation ±20% around hand-tuned values)
- Would data-driven coefficient optimization via cross-validation improve or overfit?
- The hand-tuned coefficients reflect microstructure priors — this transparency is a feature, not a bug

### 9.3 Practical Considerations
- Latency budget: 100ms bins are achievable in production with modern hardware
- Feature computation is O(n_ticks) per bin, dominated by constant-factor EMA/decay operations
- The lattice representation is instrument-agnostic (parameterized by tick size and grid width)
- Transaction cost gap: mid-price entry assumption vs real queue position and spread crossing

### 9.4 Relationship to Prior Work
- How our lattice features relate to Cont et al.'s OFI — OFI aggregates across the book, we preserve per-tick structure
- How spectrum scoring relates to HMM regime detection — our approach is local and memoryless per cell vs global state
- How entropy experiment relates to information-theoretic microstructure models
- How PFP lead-lag relates to Hasbrouck's information share and price discovery literature

### 9.5 Limitations
- **Single session**: 1 hour, 1 instrument, 1 date — results may not generalize; multi-session validation is the critical next step
- **No out-of-sample testing**: all experiments evaluated on the same session; in-sample overfitting cannot be ruled out despite independent experiment designs
- **Mid-price fill assumption**: TP/SL evaluation assumes immediate fill at mid, ignoring spread, slippage, and queue position — actual edge would be reduced
- **Hand-tuned coefficients**: two-force model weights were set by intuition, not optimized — this avoids overfitting but may leave edge on the table
- **No ensemble testing**: experiments are independent; combining signals might improve or degrade performance
- **Market conditions**: 2026-02-06 09:25–10:25 ET represents one specific microstructure environment; results are contingent on the volatility, participation, and event rate of this session

---

## 10. Conclusion and Future Work

### 10.1 Summary
- We introduce the vacuum-pressure lattice representation: a physics-inspired feature space that transforms raw MBO events into a structured (time-bin × price-tick) grid with derivative chain, two-force composite, and spectrum scoring features
- Six independent experiments provide preliminary evidence that this feature space carries short-horizon directional information, with all strategies exceeding the 33.3% breakeven TP rate on the studied session
- We publish the computed dataset (35K bins × 101 ticks × 33 features) as an immutable, checksummed parquet archive to enable independent verification and extension

### 10.2 Future Work
- **Multi-session validation**: replicate across multiple trading days, instruments (ES, NQ, GC, CL), and market conditions (high/low volatility, trend/range days)
- **Out-of-sample protocol**: strict train/test separation across sessions, not within-session splits
- **Ensemble methods**: combine best-performing signals (PFP, ADS) with learned weights
- **Transaction cost modeling**: incorporate realistic spread, slippage, and queue priority
- **Coefficient optimization**: cross-validated optimization of two-force model weights with regularization
- **Alternative bin widths**: test 50ms and 200ms bin widths for latency-accuracy tradeoffs
- **Extension to equity microstructure**: apply lattice representation to equity LOB data with different tick structures

---

## 11. Appendices

### A. Complete Feature Schema
- Full table of all 33 grid_clean columns with types, units, and formulas

### B. Evaluation Harness Pseudocode
- `detect_signals` algorithm (exact pseudocode)
- `evaluate_tp_sl` algorithm (exact pseudocode)
- `rolling_ols_slope` formula: `slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)`
- `robust_zscore` formula: `z = (x - median) / (1.4826 * MAD)`, trailing window, min_periods guard

### C. Per-Experiment Full Threshold Sweep Tables
- Complete results tables for all 6 experiments at all tested thresholds

### D. Dataset Access and Reproducibility
- Download instructions / repository link
- File format specifications (parquet schema, column types)
- SHA256 checksums for integrity verification
- Experiment reproduction commands

---

## Disclosure Boundary

| Published | Not Published |
|---|---|
| Data source identified (Databento MBO), instrument, date, capture window | Raw Databento MBO data (licensed, not redistributable) |
| Transformation pipeline described as formulas (all 6 stages) | Engine source code / implementation architecture |
| All feature formulas with coefficient rationale | Runtime optimizations (pre-allocation, buffer management) |
| Computed immutable dataset: grid_clean + bins parquet with checksums | Streaming pipeline / server infrastructure |
| Evaluation harness (exact pseudocode + algorithms) | Production deployment tooling |
| All experiment formulas, parameters, and full results | Agent workspace / experiment management tooling |
| Two-force model coefficient values AND selection rationale | Databento download/caching scripts |
