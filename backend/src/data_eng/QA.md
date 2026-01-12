## Plan (live)

1. Add index artifacts (manifest, feature list, vectors) and retrieval schema locks.
2. Build the index QA runner that enforces items 0-11 and writes the JSON report.
3. Rebuild indices and run QA to verify every partition.

## Progress
- [x] 0) Freeze the invariants
- [x] 1) Artifact inventory and loadability
- [x] 2) Count consistency
- [x] 3) Metadata contract checks
- [x] 4) Vector integrity inside the index
- [x] 5) FAISS id ↔ metadata row alignment
- [x] 6) Duplicate and near-duplicate detection
- [x] 7) Retrieval sanity tests
- [x] 8) Approximate index quality
- [x] 9) Cross-partition corpus integrity
- [x] 10) Determinism + reproducibility
- [x] 11) Operational pristine checks

## 0) Freeze the invariants (agent must treat these as non-negotiable)

Tell the agent to start by writing down the invariants the index must satisfy, and every check either validates an invariant or fails hard:

1. **Vector space**

   * `D = 952` everywhere.
   * Similarity metric must be **cosine** implemented as **inner product over L2-normalized vectors**.
   * All vectors stored/queried must be float32 at the FAISS boundary.

2. **Segmentation**

   * Indices are partitioned exactly by `level_id × approach_dir` (and only that).
   * No cross-contamination: an index partition must not contain rows from a different `(level_id, approach_dir)`.

3. **Row identity**

   * Each vector row has a stable unique identifier (whatever your schema uses: `row_id`, `vector_id`, `episode_id`, etc.).
   * FAISS internal ids (0..N-1 or explicit IDMap ids) must map **perfectly** to the metadata row mapping used at retrieval.

4. **Normalization contract**

   * The `norm_stats.json` used to build the index must be the same one used at query time.
   * `norm_stats.json` corresponds to the same feature ordering as the vector builder.
   * Constant dims: you already observed **116 dims with MAD=0** — these must behave as “always 0 after robust scaling,” and must not inject NaN/Inf or dominate norms.

---

## 1) Artifact inventory and loadability (fast, fail-fast)

For each `(level_id, approach_dir)` partition:

1. Confirm the partition directory exists and contains *exactly* the expected set of files:

   * FAISS index file(s) (e.g., `index.faiss`)
   * Metadata mapping file (e.g., `meta.parquet` / `rows.parquet`)
   * Any stored normalized vectors file if you persist it (highly recommended for QA): `vectors.npy` / `vectors.parquet`
   * The normalization stats file used: `norm_stats.json` (either per-partition reference or global reference)
   * A build manifest (must include: build timestamp, git SHA, input partitions, counts)

2. Load each artifact:

   * FAISS index must load without warnings.
   * Metadata must load and be non-empty.
   * If vectors file exists: it must load and have shape `(N, 952)` matching metadata and FAISS counts.

**Fail if:**

* Any artifact missing.
* Any file loads but yields inconsistent shapes/dtypes.
* Any partition has N=0 unless your system explicitly expects it (then the partition must be explicitly declared “absent” and retrieval must handle it deterministically).

---

## 2) Count consistency: “N must agree everywhere”

For each partition:

1. Compute:

   * `N_faiss = index.ntotal`
   * `N_meta = number of rows in metadata`
   * `N_vec = number of vectors in stored vectors file` (if present)

2. Require:

   * `N_faiss == N_meta`
   * and if vectors persisted: `N_faiss == N_vec`

3. Across all partitions:

   * Sum of partition Ns must equal the expected global corpus size **if** your index is a full partitioning of the corpus (no row belongs to multiple partitions).
   * If your design duplicates rows across partitions (it shouldn’t), then the duplication rule must be explicitly validated.

**Fail if:** any mismatch, even by 1.

---

## 3) Metadata contract checks (index-level, not upstream)

For each partition metadata table:

1. Schema presence:

   * Confirm all required columns exist for retrieval-time reconstruction:

     * Partition keys: `level_id`, `approach_dir`
     * Identity keys: `row_id` / `episode_id`
     * Time keys: `session_date`, `ts` or `window_end_ts` / `trigger_ts`
     * Label keys needed for outcome distributions
     * Any display keys needed for UI payload (pressure stream join keys, etc.)

2. Schema types:

   * Ensure types are stable: timestamps are timestamps, ints are ints, no accidental strings, no object dtype garbage.

3. Uniqueness:

   * `row_id` (or equivalent) must be unique within a partition.
   * Prefer also unique globally across all partitions (if that’s your contract).

4. Partition purity:

   * Validate every row in this metadata has exactly the partition’s `(level_id, approach_dir)` values.

5. Time sanity (index-relevant):

   * Timestamps must lie within the session’s allowed window (whatever your platform defines for the index corpus; likely first 3 hours RTH).
   * No future timestamps relative to session date.
   * No timezone mixing (agent should check that all timestamps are either tz-aware in the same tz or all tz-naive but consistently interpreted).

**Fail if:** any missing columns, mixed types, non-unique ids, or partition impurity.

---

## 4) Vector integrity inside the index (NOT the gold table)

This is where “pristine” usually breaks.

### 4A) L2 normalization invariants

For vectors inside each FAISS partition:

1. Sample vectors directly from:

   * persisted `vectors.npy` if present, AND/OR
   * FAISS `reconstruct()` (if index supports it; most do)

2. Compute norms and require:

   * `abs(norm - 1.0) <= 1e-4` for almost all vectors (set a tiny tolerance for float32)
   * There must be **no** zero norms and **no** exploding norms (e.g., > 1.001)

**Fail if:** normalization tolerance violated beyond a tiny fraction (e.g., > 0.01%); even a small systemic deviation indicates the wrong thing was indexed.

### 4B) Similarity range sanity

1. Run random pair dot-products on reconstructed vectors.
2. For cosine/inner product:

   * Similarities should fall in [-1, +1] with tiny numeric slop.

**Fail if:** you see values meaningfully > 1 or < -1, which implies vectors are not truly L2-normalized or are corrupted.

### 4C) Constant-dimension behavior (your MAD=0 dims)

1. Identify the 116 constant dims from `norm_stats.json`.
2. On a sample of reconstructed vectors:

   * Those dims must be exactly 0 after scaling (and still 0 after L2 norm).

**Fail if:** constant dims vary (means feature ordering mismatch or stats mismatch).

### 4D) “No label leakage into vector payload”

This is subtle and critical.

Tell the agent to validate that the 952-d vector is composed only of allowed feature columns (as per SPEC), and that:

* no label columns,
* no forward-looking hit timing fields,
* no future-derived aggregates,
  have been accidentally concatenated into the vector.

**How to validate without touching upstream MBO:**

* Compare the vector feature-name list (the canonical ordered list used by the builder) against a denylist of label fields.
* Ensure the vector builder stores the feature list and that the index QA asserts equality.

**Fail if:** feature list is missing or any forbidden field is present.

---

## 5) FAISS id ↔ metadata row alignment (the biggest “silent killer”)

This must be checked rigorously.

For each partition:

1. Determine whether FAISS is:

   * implicit ids (0..N-1), or
   * `IndexIDMap` / explicit ids.

2. Validate mapping correctness:

   * Pick 200 random ids.
   * For each id:

     * reconstruct vector `v_faiss`
     * fetch the corresponding metadata row
     * fetch the corresponding stored vector row (if persisted)
     * check `cos(v_faiss, v_stored) >= 0.999999` (float32 tolerance)

3. Validate ordering stability:

   * Ensure metadata rows are stored in the same order assumed by FAISS ids.
   * Ensure retrieval pipeline uses the same mapping file produced at build time.

**Fail if:** any reconstructed vector doesn’t match its mapped row. This indicates the index is effectively unusable even if “counts look right.”

---

## 6) Duplicate and near-duplicate detection (index quality + retrieval pathologies)

For each partition:

1. Exact duplicates:

   * Hash vectors (or quantize then hash) and compute duplicate rate.

2. Near duplicates:

   * For a random sample, compute nearest neighbor similarity distribution.
   * Flag if a large fraction of vectors have top-1 similarity extremely close to 1.0 (excluding self), e.g. > 0.99999.

3. Interpret:

   * Some duplication is expected if the market state repeats, but “walls” of duplicates usually indicate:

     * a bug in feature assembly,
     * constant/zeroed features dominating,
     * time window not advancing,
     * metadata collapse (many rows mapping to the same vector).

**Fail if:** duplicate rate is unexpectedly high (agent should report rate per partition and global); you decide the threshold, but anything like double-digit % is a red flag.

---

## 7) Retrieval sanity tests (index behaves like a metric space)

These are black-box tests that catch issues that pure file checks won’t.

For each partition:

1. Self-query test:

   * Query with a vector that is known to be in the index.
   * Ensure top-1 neighbor is itself (or if you explicitly exclude self, ensure it appears at rank 2+ with similarity ≈ 1).

2. “Monotonicity” test:

   * Top-K similarities must be non-increasing and within valid range.

3. Distance distribution test:

   * For random queries, capture the distribution of:

     * top-1 similarity
     * top-10 similarity mean
     * similarity gap (top-1 minus top-10 mean)
   * Look for partitions where results are degenerate:

     * everything returns ~the same similarity
     * or top-1 is always ~1 (suggesting duplicates)
     * or similarities are extremely low (index built from unnormalized vectors or wrong stats)

**Fail if:** any partition’s retrieval distribution looks degenerate relative to others.

---

## 8) Approximate index quality (if not Flat)

If any partition uses IVF/PQ/HNSW (anything approximate):

1. Build an exact baseline for a small sample:

   * Take ~500 queries.
   * Compute exact top-K via brute force dot-product against the stored vectors for that partition.

2. Compare FAISS results:

   * Compute recall@K (e.g., K=10, 50).
   * Track recall separately for each partition.

3. Require minimum recall:

   * Your threshold depends on design, but for “research-quality retrieval,” you usually want recall@50 very high.

**Fail if:** recall is poor or wildly variable across partitions; this often indicates training instability, wrong nprobe, or wrong normalization.

---

## 9) Cross-partition corpus integrity (global view)

Across all indices:

1. Coverage:

   * Count vectors by `session_date` across the entire index corpus.
   * Look for missing dates, spikes, or days contributing implausibly high fractions.

2. Outcome distribution sanity:

   * Compute label distributions per partition and globally.
   * Look for partitions where one class dominates at absurd rates (often means label join mismatch or wrong horizon).

3. Stratification invariants:

   * If your platform expects “first 3 hours only,” validate that the timestamp distribution respects it everywhere.

4. Selection purity:

   * Confirm that every indexed row corresponds to an allowed contract-day selection (use the selection map as an allowlist).
   * Confirm there are zero rows from excluded days.

**Fail if:** any excluded day appears in the index, even once.

---

## 10) Determinism + reproducibility (pristine means rebuild = identical)

Have the agent perform a controlled rebuild and compare:

1. Rebuild the indices from the same input corpus with the same `norm_stats.json`.
2. Compare:

   * per-partition N
   * per-partition metadata hashes
   * per-partition vectors hashes (if persisted)
   * retrieval results for a fixed set of query vectors

Expected:

* If Flat: identical.
* If IVF/PQ: counts and mapping must be identical; retrieved neighbors should be highly consistent; if not, the build is nondeterministic and needs fixed seeds + stable ordering.

**Fail if:** mapping changes or retrieval changes substantially between rebuilds.

---

## 11) Operational “pristine” checks (will it stay correct when used)

1. Cold-load test:

   * Load all indices fresh in a clean process and run a small retrieval battery.
   * This catches “works in memory, fails on disk” corruption issues.

2. Concurrent queries:

   * Run parallel retrieval calls to ensure no race conditions / shared mutable state issues in your retrieval engine.

3. Version locking:

   * Ensure the retrieval engine refuses to load an index if:

     * `D` mismatches,
     * `norm_stats.json` hash mismatches,
     * feature list hash mismatches,
     * or metadata schema version mismatches.

**Fail if:** the system silently loads mismatched versions.

---

Produce ONE canonical “index QA runner” that:

1. Runs the entire checklist across every `(level_id, approach_dir)` partition.

2. Emits:

   * a single pass/fail exit code,
   * a compact JSON report with per-partition metrics + failures (no markdown docs),
   * and prints a short summary to stdout.

3. Fails hard on:

   * id/metadata mismatch,
   * normalization violations,
   * feature-order/stats mismatch,
   * excluded days included,
   * low recall (if approximate).
