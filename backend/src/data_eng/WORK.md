Implement this in three passes: **(A) leak-proof holdout evaluation**, **(B) cluster discovery**, **(C) “does it trade?” sanity backtests**. Do it per **{level_id} × {approach_dir}** (since your indices are already partitioned that way).

---

## A) Leak-proof holdout retrieval evaluation (the minimum viable “is there signal?” test)

### A1) Inventory what you can actually predict (define the target)

1. In `gold.future_mbo.mbo_trigger_vectors`, enumerate **all label/outcome columns** produced by `build_trigger_vectors.py`.
2. Pick **one canonical target** to start (don’t multi-target yet). Prefer something like:

   * binary: “breaks through level by +X ticks before −Y ticks within H seconds”
   * or continuous: forward return / MFE / MAE over a fixed horizon
3. Lock the *exact* definition (X,Y,H, direction conventions) in a small **eval spec** object (YAML/JSON) saved with results.

**Verification:** for that label, compute unconditional base rate by {approach_dir} and by time-of-day bins. If it varies massively, you’ll need TOD controls later.

---

### A2) Create a holdout split that cannot leak

Because your FAISS index is pooled across contract-days, you must ensure the query never retrieves “future” information.

Implement **one** of these two evaluation modes (do both eventually):

**Mode 1 — Fixed train/test split (fast, clean):**

* Define train session_dates = `[start … split_date]`, test session_dates = `(split_date … end]`.
* Build a **train-only index** for each {level_id, approach_dir}.
* Evaluate all test triggers against that train index.

**Mode 2 — Walk-forward (most realistic):**

* For each test day D: query using an index built from all days `< D`.
* Practical shortcut: keep one “all prior days” index and **filter neighbors by session_date < query_date** (requires over-retrieving; see below).

**Critical:** normalization stats must be train-only.

* Your feature doc says: per-dim robust scaling (median/MAD), clip to ±8, zero-MAD dims→0, then L2 normalize.
* For Mode 1: use the train index’s `norm_stats.json` to normalize all test queries.
* For Mode 2: if stats change over time, either (i) freeze stats from an initial training window, or (ii) maintain rolling stats (harder). Start with frozen stats.

**Verification:** confirm that for every test query, no neighbor in the final neighbor set has `session_date >= query.session_date` (and optionally `contract == query.contract` if you want cross-contract generalization).

---

### A3) Retrieval → prediction: define the kNN estimator

For each test trigger vector **q**:

1. FAISS search with **K_raw** (e.g., 500).
2. Apply **causal filters** using stored metadata:

   * drop neighbors with same `(session_date, trigger_id)` (self-hit)
   * drop neighbors violating holdout rules (e.g., `neighbor.session_date >= query.session_date`)
   * optional: drop same `contract` to test cross-contract generalization
3. Keep first **K** after filtering (evaluate K ∈ {5, 10, 20, 50, 100}).

Convert neighbor labels to a prediction:

* If FAISS is cosine/IP on L2-normalized vectors, use similarity `s_j` and weights
  `w_j = exp((s_j - s_1)/τ)` with a small τ (start τ≈0.02–0.05), or just uniform weights.
* Prediction:

  * binary: `p_hat = Σ w_j y_j / Σ w_j`
  * continuous: `ŷ = Σ w_j y_j / Σ w_j`
* Also compute **neighbor dispersion** diagnostics:

  * effective neighbor count `N_eff = (Σw)^2 / Σ(w^2)`
  * similarity gap `s_1 - s_K`
  * entropy of neighbor outcome distribution

**Verification:** run a “null retrieval” where you replace neighbors with random training vectors matched on time-of-day bin; your method must beat this.

---

### A4) Metrics: measure “predictive power” without pretending it’s a strategy

Compute on test set (per {level_id, approach_dir}, and also by TOD bins):

**If binary target**

* log loss, Brier score, ROC-AUC
* calibration curve + ECE (expected calibration error)
* lift: mean(y) in top decile of p_hat vs unconditional base rate

**If continuous target**

* correlation (Spearman + Pearson), MAE/MSE
* directional hit rate (sign agreement) if meaningful
* conditional quantiles (e.g., median outcome vs p_hat bins)

**Always include baselines**

* unconditional predictor (base rate / mean)
* “time-of-day only” predictor (base rate by TOD bin)
* optionally: a tiny linear/logistic model on a handful of interpretable features (sanity comparator)

**Deliverable artifacts**

* `backend/lake/eval/retrieval/{level_id}/{approach_dir}/split={id}/metrics.parquet`
* `…/predictions.parquet` with: query metadata, p_hat/ŷ, realized label, top-K neighbor ids, similarity stats

---

## B) Cluster discovery that actually means something (not just pretty UMAP)

### B1) Build a kNN graph from the *training* vectors

Per {level_id, approach_dir}:

1. For each train vector, get top M neighbors (M≈30–50) from FAISS (within-train).
2. Create weighted edges with weights = similarity (or exp(sim/τ)).

### B2) Community detection on the graph (preferred) + stability checks

* Run **Leiden** (or Louvain as fallback) on the weighted kNN graph.
* For each cluster:

  * size, median similarity within cluster
  * **label enrichment** vs baseline (effect size + significance via permutation/bootstrap; control FDR)
  * representative medoids (top 10 closest-to-cluster-center vectors with metadata)

**Stability verification**

* rerun clustering across:

  * different M (30/40/50)
  * bootstrap subsamples of train days
* report ARI/NMI stability; ignore clusters that aren’t stable.

### B3) Out-of-sample validation of clusters

Assign each **test** query to a cluster using neighbor majority (or highest total edge weight to cluster).

* Compare cluster-conditioned outcome distribution on test vs train.
* The clusters you care about are those where:

  * enrichment persists OOS
  * cluster assignment confidence is high (dominant cluster share among neighbors)

**Deliverables**

* `…/clusters_train.parquet` (vector_id → cluster_id + cluster stats)
* `…/clusters_test_assignments.parquet`
* `…/cluster_enrichment_report.parquet`

---

## C) Minimal “does it trade?” sanity backtest (optional, but clarifies whether the signal matters)

Once you have a binary outcome target (e.g., break vs reject), define a toy decision rule:

* take “breakout” if `p_hat > θ`, “rejection” if `p_hat < 1-θ`, else no-trade
  Evaluate:
* hit rate vs baseline
* average outcome in traded subset
* sensitivity to θ and K
* robustness by TOD bin and by volatility regime (use simple realized vol context from the day)

This is not a production strategy; it’s a **signal strength diagnostic**.

---

## The first thing I would have the agent implement tomorrow

1. **Mode 1 fixed split** for one `{level_id=pm_high, approach_dir=approach_up}`.
2. Produce:

   * metrics table for K ∈ {10, 50} with both uniform and similarity-weighted neighbors
   * null matched-TOD baseline
   * 20 example “retrieval audit cards” (query + top neighbors + outcome histogram + similarity stats)

If that doesn’t beat the null baseline, don’t cluster yet—fix target definition, normalization, or leakage first.
