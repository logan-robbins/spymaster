Looking at the context, all tasks from the previous session are complete. Both Round 1 (6 statistical experiments) and Round 2 (6 ML experiments) have been fully executed, results collected, comparison scripts run, and EXPERIMENT.md updated.

Here's a summary of where we stand:

Round 1 (Statistical/Derivative-based) — All beat 33.3% breakeven:

ERD (Entropy): 50.0% TP, 8 signals
PFP (Pressure Front): 45.3% TP, 75 signals
JAD (Jerk-Accel): 42.6% TP, 54 signals
ADS (Asymmetric Slope): 40.5% TP, 1,071 signals
SPG (Spatial Gradient): 39.3% TP, 1,120 signals
IIRC (Intensity Imbalance): 38.7% TP, 1,084 signals
Round 2 (ML models) — All beat 33.3% breakeven:

GBM_MF (LightGBM): 48.2% TP, 56 signals, +1.30t PnL
XGB_SNAP (XGBoost): 42.2% TP, 334 signals, +0.94t PnL
KNN_CL (KNN): 41.1% TP, 1,140 signals, +0.42t PnL
PCA_AD (Rolling PCA): 40.7% TP, 405 signals, +0.50t PnL
SVM_SP (LinearSVC): 40.5% TP, 1,071 signals, +0.23t PnL
LSVM_DER (SGD/SVM): 40.4% TP, 1,137 signals, +0.28t PnL
All 12 experiments beat the baseline (28.5% TP) and the breakeven threshold (33.3%). Everything is documented in EXPERIMENT.md and all results are in their respective agents/{name}/outputs/results.json files. Ready for whatever direction you'd like to take next.

Now, which of these can be "micro online trained" so that they can keep providing valuable insights over micro regimes, fast enough for our inference ideals (we dont presume to go more than maybe 2 minutes forward projection).



Let me think about this carefully — this is about evaluating which of our 12 experiments can be adapted for online/incremental learning within the 100ms bin cycle, fast enough for real-time inference with projections up to ~2 minutes (1,200 bins). Let me review the experiment implementations to assess their online-trainability.

Let me review the key experiment scripts to assess their computational profiles:


Read 12 files, searched 2 patterns

Now let me also look at the Round 1 statistical experiments and the current streaming infrastructure to understand our inference constraints:


Read 3 files

Now I have a thorough picture of all 12 experiments. Let me also check the remaining Round 1 experiments quickly:


Reviewed four market microstructure trading agent experiments: ERD, JAD, SPG, and IIRC

Now I have a complete picture. Here's my analysis:

Online Trainability Assessment
The constraint: must inference at or near every 100ms bin, with projections up to ~2 minutes (1,200 bins). The model must update/adapt within that bin cycle — no batch retraining that blocks the hot path.

Tier 1: Natively Online — Drop-in Ready
LSVM_DER (SGDClassifier) — Best candidate for online learning.

Already uses SGDClassifier with warm_start=True and partial_fit()
partial_fit() is O(n_features) per sample — 60 features × one dot product = sub-microsecond
Inference is a single dot product: w · x + b → sub-microsecond
It already does incremental updates between full refits — this is literally online SGD
Feature computation (band asymmetries, rolling slopes) is all incremental: keep running sums, update in O(1) per bin
Verdict: production-ready as-is. Just wire partial_fit() into the bin callback. 40.4% TP, 1,137 signals — solid volume.
Tier 2: Trivially Convertible to Online
ADS (Asymmetric Derivative Slope) — Pure arithmetic, no model at all.

Rolling OLS slope is a running sum update: O(1) per bin per window
Robust z-score is running median/MAD: O(1) amortized with a sorted buffer
tanh blend is one expression
Total per-bin cost: ~10 μs
Verdict: already online by nature. Just maintain the rolling buffers. 40.5% TP, 1,071 signals.
SPG (Spatial Pressure Gradient) — Pure EMA + central differences.

Spatial gradient = P[k+1] - P[k-1], already computed per bin in the engine
Dual EMA is α*x + (1-α)*prev — two multiplies per channel
Curvature = second central difference — one subtraction
Total: ~5 μs per bin
Verdict: trivially online. 39.3% TP, 1,120 signals.
PFP (Pressure Front Propagation) — EMA cross-products with lag buffer.

Zone intensity = mean of 3-8 grid cells (already in cache)
Lead-lag metric = EMA of lagged products — just need a 5-bin ring buffer for the lag
All EMA: α*x + (1-α)*prev
Total: ~8 μs per bin
Verdict: trivially online. 45.3% TP, 75 signals — lower volume but high accuracy.
IIRC (Intensity Imbalance RoC) — Rolling sums + OLS slope.

Sum v_add/v_pull/v_fill by side: sum of ~16 cells per side, already in cache
Log-ratio is one expression
Rolling OLS slope: maintain running sums (Σx, Σy, Σxy, Σx²) — O(1) update
Total: ~8 μs per bin
Verdict: trivially online. 38.7% TP, 1,084 signals.
JAD (Jerk-Acceleration Divergence) — Weighted aggregation + z-score.

Distance-weighted spatial sums (w=1/|k|): weights are fixed, dot product against 24 cells per side
Divergence is a subtraction
Robust z-score: rolling buffer
tanh blend
Total: ~12 μs per bin
Verdict: trivially online. 42.6% TP, 54 signals.
Tier 3: Online-Adaptable with Modification
KNN_CL — Needs a sliding window KD-tree, but doable.

Can't re-fit sklearn KNN every bin — fit() rebuilds the tree
Solution: use a ring-buffer ball tree (fixed-size, FIFO). Pre-allocate 1,200 slots (2 minutes of history). Each bin, push the new sample and evict the oldest.
At query time: brute-force K=11 search over 1,200 standardized vectors of 35 dimensions = 35 × 1200 = 42,000 multiplies → ~50 μs on M4.
Running standardization: maintain running mean/var, update incrementally.
Verdict: online with a fixed-window brute-force KNN. Drop sklearn, use numpy. 41.1% TP, 1,140 signals — highest volume.
PCA_AD (Rolling PCA Anomaly) — Online PCA is a solved problem.

Replace sklearn PCA with incremental PCA (rank-1 updates via GROUSE, or just maintain a covariance matrix and do truncated eigen every N bins)
Practical approach: maintain a 600-bin buffer of 101-dim PV-diff vectors. Refit PCA every 100 bins (every 10 seconds). PCA on 600×101 matrix takes ~2ms on M4 — amortized to 20 μs/bin over 100 bins.
Between refits, projection/reconstruction is a matrix multiply: scores = (x - μ) @ V, recon = scores @ V.T + μ — 101×10 + 10×101 = ~2,000 FLOPs → sub-microsecond.
Verdict: online with periodic refit. 40.7% TP, 405 signals.
Tier 4: NOT Online-Trainable (Batch-Only)
GBM_MF (LightGBM) — ❌ Cannot be incrementally trained.

LightGBM has no partial_fit(). Each retrain builds all trees from scratch.
Retraining 200 boosted trees on 2,400+ samples with 53 features takes ~500ms — blocks 5 bin cycles.
Could amortize by retraining on a background thread every N seconds, but the model itself is frozen between retrains — it's not truly learning online, just periodically batch-updating.
Inference is fast (~10 μs for 200 trees), so you could deploy a stale model and retrain async. But it won't adapt to micro-regime shifts within its retrain window.
Verdict: inference-only online, retrain async. Best TP rate (48.2%) but only 56 signals — low volume makes it a confirmation filter, not a primary signal.
XGB_SNAP (XGBoost) — ❌ Same story as GBM.

No incremental training. 150 trees × 163 features × 3,000+ samples = ~800ms retrain.
Inference is fast (~15 μs).
Verdict: same as GBM — inference online, retrain async. 42.2% TP, 334 signals.
SVM_SP (LinearSVC) — ⚠️ Almost online, but LinearSVC lacks partial_fit().

Unlike SGDClassifier, sklearn's LinearSVC uses liblinear (coordinate descent), which requires the full dataset.
Fix: replace with SGDClassifier(loss='hinge') — mathematically equivalent linear SVM, but with partial_fit().
45 features, single dot product inference → sub-microsecond.
Verdict: trivially fixable by swapping to SGDClassifier. Then it's Tier 1. 40.5% TP, 1,071 signals.
ERD (Entropy Regime Detector) — ✅ Actually online despite not being ML.

Shannon entropy over 101 state codes per bin: O(101) = trivial
Asymmetry between bid/ask halves: one pass
Robust z-score: rolling buffer
Total: ~15 μs
Verdict: trivially online. 50.0% TP but only 8 signals — extreme filter.
Summary: What Can Run Online at 100ms
Experiment	Online Method	Per-Bin Cost	TP%	Signals	Role
LSVM_DER	partial_fit() native	<1 μs	40.4%	1,137	Primary workhorse
ADS	Rolling sums	~10 μs	40.5%	1,071	Primary workhorse
KNN_CL	Brute-force ring buffer	~50 μs	41.1%	1,140	Primary workhorse
SPG	EMA update	~5 μs	39.3%	1,120	Primary workhorse
PFP	EMA + lag buffer	~8 μs	45.3%	75	High-confidence filter
JAD	Weighted sums + z-score	~12 μs	42.6%	54	High-confidence filter
ERD	Entropy calculation	~15 μs	50.0%	8	Extreme filter
IIRC	Rolling sums + OLS	~8 μs	38.7%	1,084	Primary workhorse
PCA_AD	Periodic refit (10s)	~20 μs avg	40.7%	405	Anomaly gate
SVM_SP	Swap to SGD → partial_fit()	<1 μs	40.5%	1,071	Primary (after swap)
GBM_MF	Inference only, async retrain	~10 μs inf	48.2%	56	Async confirmation
XGB_SNAP	Inference only, async retrain	~15 μs inf	42.2%	334	Async confirmation
The total budget if you ran all 12 simultaneously: ~150 μs per bin. The bin width is 100ms = 100,000 μs. We'd use 0.15% of the bin budget. Even an ensemble of all 12 is trivially feasible.

The natural architecture: 5 online primary signals (LSVM_DER, ADS, KNN_CL, SPG, IIRC) running every bin, gated by 3 high-confidence filters (PFP, JAD, ERD) and 1 anomaly detector (PCA_AD), with 2 async tree models (GBM_MF, XGB_SNAP) providing background confirmation that retrains every 30-60 seconds on a separate thread.

---

Here are the MSD experiment results. The experiment completed in 8.4s, producing 4 JSON outputs.

---

## Forensic Attribution: The 09:29:58 Prediction

At the critical bin (09:29:58 ET, mid_price=$24,763.25), the forward excursion was **-173.5 ticks ($43.38 drop)** — an extreme move. Here's what each signal was doing:

**Signal values at 09:29:58:**

| Signal | Raw Value | Weight | Weighted Contribution |
|--------|-----------|--------|----------------------|
| **ADS** | +0.135 | 0.35 | **+0.047** (dominant) |
| **PFP** | -0.046 | 0.40 | -0.019 |
| **ERD** | 0.0 | 0.25 | 0.0 (spike_gate=0) |

The composite at that bin was actually weakly **positive** (+0.029) — meaning the existing signal combination did NOT cleanly predict the drop at that exact instant. However, **spatial vacuum** was strongly negative: `spatial_vac_a = -56.9`, `spatial_vac_c = -23.7`, meaning significantly more vacuum below spot than above. This is exactly the "vertical view" you asked about — the system could see spatial vacuum opening below spot in a major way.

**Key observations from the 200ms preceding 09:29:58:**
- At 09:29:57.9, `spatial_vac_a` spiked to **-134.0** (extremely bearish vacuum asymmetry)
- `vac_below_sum=500` vs `vac_above_sum=366` — 36% more vacuum below spot
- PFP was already mildly bearish (lead_ask > lead_bid, meaning ask-side inner was leading outer)
- ERD spike_gate was 0 — entropy wasn't elevated enough to activate

**Conclusion**: The accurate prediction was primarily driven by **spatial vacuum asymmetry** — a feature that our current composite signal doesn't include. The three production signals (PFP, ADS, ERD) were actually mixed/neutral at that bin.

---

## Move-Size Stratified Results

Tier definitions (max favorable excursion in $0.25 ticks): micro (<4), small (4-8), medium (8-16), large (16-32), extreme (32+).

**Best performing threshold per signal (≥5 signals):**

| Signal | Best TP% | N | PnL/trade | Large+Extreme Selectivity | Best Threshold |
|--------|----------|---|-----------|--------------------------|----------------|
| **ADS** | **43.4%** | 106 | +0.97t | 84.0% | 0.550 |
| Spatial Vac (weighted) | 41.4% | 331 | +0.40t | 85.8% | 237.6 |
| PFP | 40.6% | 731 | +0.51t | 81.8% | 0.098 |
| Spatial Vac (sum) | 39.4% | 728 | +0.18t | 86.7% | 1411.3 |
| Composite | 38.1% | 312 | +0.08t | 88.1% | 0.132 |
| ERD | 36.7% | 436 | -0.30t | 85.1% | 0.049 |

**Critical finding — the extreme-move tier is where signals shine:**

At ADS best threshold (0.550), the **extreme tier** (32+ tick moves) hit **51.5% TP** with +2.71t PnL/trade on 68 signals. Medium tier was 42.9%, and micro/small were always 0% (these small moves get stopped out before reaching 8-tick TP).

For composite at threshold 0.132, the **medium tier** hit **71.4% TP** (14 signals) — very small sample but notable. The extreme tier was 40.4% on 228 signals.

---

## Key Takeaways

1. **Spatial vacuum was the real predictor at 09:29:58**, not the composite. The current composite (PFP/ADS/ERD) missed the signal because none of the three components captured the vertical vacuum asymmetry.

2. **ADS is the strongest individual signal** — 43.4% TP, highest PnL per trade. Its 10-bin OLS slope z-score was the most informative sub-component.

3. **Large-move selectivity is high across all signals** (~82-88%) — signals fire preferentially when subsequent moves are large. This is good: it means the signals aren't wasting fires on micro/small moves.

4. **Micro/small tiers always produce 0% TP** — the 8-tick TP target is unreachable in <4-tick or 4-8 tick moves. This suggests a possible gate: if we could estimate "expected move size" ahead of time, we could suppress fires during low-volatility periods.

5. **Spatial vacuum deserves inclusion in the composite** — with 41.4% TP and the highest selectivity, the distance-weighted vacuum variant (variant_c) should be evaluated as a 4th composite component.


---

• I’m using the quant-grunt skill here because this is signal math parity and tuning. Next I’ll mirror the frontend ADS/PFP/SVac formulas exactly into one harness signal with tunable runtime
  parameters, then run it on the gold dataset.

• Updated Plan
  └ Living task doc for adding live ADS+PFP+SVac composite into harness and running it.
    □ 1. Discover frontend composite math and current harness signal interfaces/registry constraints.
    □ 2. Implement a new harness statistical signal that matches frontend ADS+PFP+SVac with tunable parameters.
    □ 3. Register the signal and add a harness config for tunable sweeps on the gold dataset.
    □ 4. Verify locally (list-signals, targeted harness tests, and a real campaign run with MLflow).
    □ 5. Update README.md with new runtime/harness commands and signal description.
    □ 6. Report best run metrics and exact parameter settings from the campaign.