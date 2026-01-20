# Databricks Pipeline Code Review Report

---

## 1. Architecture & Design

### Strengths
- **Clear medallion architecture**: Bronze → Silver → Gold layering is correctly applied
- **Separation of concerns**: Historical batch jobs (`hist__*`) and real-time streaming jobs (`rt__*`) are properly separated
- **Consistent path conventions**: Uses `abfss://` protocol with parameterized storage accounts
- **Multi-output streaming**: The `rt__silver_to_gold` and `rt__gold_to_inference` jobs correctly implement fan-out to both Delta and Event Hubs

### Critical Gaps

**1.1 Missing Delta Live Tables / Lakeflow Declarative Pipelines**
```python
# Current: Manual Delta writes
df_parsed.writeStream.format("delta").outputMode("append")...

# 2025-2026 Best Practice: Lakeflow Declarative Pipelines
@dlt.table(name="bronze_mbo")
def bronze_mbo():
    return spark.readStream.format("eventhubs").load()
```
Lakeflow provides automatic dependency management, schema enforcement, data quality expectations, and enhanced autoscaling specifically designed for streaming workloads.

**1.2 No Unity Catalog Integration in Code**
While WORK.md mentions Unity Catalog is enabled, the streaming code writes directly to ABFSS paths rather than using Unity Catalog managed tables:
```python
# Current
.start("abfss://lake@storage.dfs.core.windows.net/bronze/stream")

# Best Practice
.toTable("spymaster.bronze.mbo_stream")
```
This loses governance features, lineage tracking, and schema evolution capabilities.

**1.3 Hardcoded Configuration**
```python
# rt__mbo_raw_to_bronze.py line 17-21
EVENTHUB_NAMESPACE = "ehnspymasterdevoxxrlojskvxey"
EVENTHUB_NAME = "mbo_raw"
CHECKPOINT_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/..."
```
Configuration should use Databricks widgets, environment variables, or job parameters for environment portability.

---

## 2. Streaming Implementation

### Strengths
- Correctly uses `readStream` / `writeStream` patterns
- Appropriate trigger interval (`processingTime="10 seconds"`)
- Partitions output by `session_date` for query optimization

### Critical Gaps

**2.1 Deprecated Stateful API**
The `rt__bronze_to_silver.py` uses `applyInPandasWithState`:
```python
# Line 188-195
df_silver = (
    df_bronze
    .groupBy("contract_id")
    .applyInPandasWithState(
        process_orderbook_updates,
        outputStructType=bar_5s_schema,
        stateStructType=state_output_schema,
        outputMode="append",
        timeoutConf=GroupStateTimeout.NoTimeout,
    )
)
```

**Issues:**
1. `applyInPandasWithState` is considered legacy as of Spark 4.0 (2025)
2. `GroupStateTimeout.NoTimeout` means state grows unbounded forever
3. Should use `transformWithStateInPandas` with TTL support

**2.2 Missing Watermarking**
No watermarks are defined on any streaming query:
```python
# Missing - should be present
df_bronze.withWatermark("event_time", "10 minutes")
```
Without watermarks:
- State cannot be automatically cleaned up
- Memory will grow unbounded over time
- Stream-stream joins would fail

**2.3 No Rate Limiting**
```python
# Current
spark.readStream.format("delta").load(BRONZE_PATH)

# Should include
.option("maxFilesPerTrigger", 1000)
.option("maxBytesPerTrigger", "1g")
```
Without rate limiting, backpressure situations can cause OOM failures.

**2.4 Missing RocksDB State Store Configuration**
For stateful streaming with large state, RocksDB is required:
```python
# Missing from spark config
spark.conf.set("spark.sql.streaming.stateStore.providerClass", 
               "org.apache.spark.sql.execution.streaming.state.RocksDBStateStoreProvider")
spark.conf.set("spark.sql.streaming.stateStore.rocksdb.changelogCheckpointing.enabled", "true")
```

---

## 3. State Management

### Critical Issues

**3.1 State Stored as JSON String**
```python
# rt__bronze_to_silver.py lines 91-102, 166
if state.exists:
    book_state = OrderbookState.from_dict(json.loads(state.get))
...
state.update(json.dumps(book_state.to_dict()))
```
This pattern:
- Incurs JSON serialization/deserialization overhead on every row
- Loses type safety
- Makes state evolution difficult
- Should use proper state schema types

**3.2 Dictionary-Based Order Book**
```python
# Lines 38-44
@dataclass
class OrderbookState:
    bids: dict  # price -> size
    asks: dict  # price -> size
```
This is inefficient for:
- Large orderbooks (thousands of price levels)
- Frequent updates
- Serialization/deserialization

Should use sorted data structures or external state stores like Redis for complex state.

**3.3 No State Expiration**
The state persists indefinitely. For market data, stale contract states (after futures roll) should be expired.

---

## 4. Error Handling & Resilience

### Critical Gaps

**4.1 No Exception Handling in Stateful Function**
```python
# rt__bronze_to_silver.py process_orderbook_updates function
for _, row in pdf.iterrows():
    action = row["action"]
    # No try/except - single bad row can crash entire stream
```

**4.2 Silent Failure in Inference**
```python
# rt__gold_to_inference.py lines 83-90
except Exception as e:
    for _ in batch:
        results.append({
            "prediction": -1,
            "prob_0": 0.0,
            "prob_1": 0.0,
        })
    print(f"Inference error: {e}")  # print() is not production logging
```
Issues:
- `print()` is not captured in production logs
- Silent failures with sentinel values (-1) can pollute downstream
- No retry logic for transient failures

**4.3 No Dead Letter Queue**
Malformed events are not captured anywhere for investigation.

**4.4 No Metrics/Alerting**
No integration with:
- Spark metrics (via listener)
- Azure Monitor
- Custom progress reporters

---

## 5. Schema & Data Contracts

### Strengths
- Explicit StructType definitions for input/output schemas
- Schema documented in WORK.md data contract section

### Gaps

**5.1 No Schema Enforcement**
```python
# Current: implicit schema from Delta
df_bronze = spark.readStream.format("delta").load(BRONZE_PATH)

# Should explicitly enforce
df_bronze = (
    spark.readStream.format("delta")
    .schema(expected_schema)  # Fail fast on schema mismatch
    .load(BRONZE_PATH)
)
```

**5.2 Nullable Fields Inconsistency**
```python
# bar_5s_schema has True for nullable
StructField("contract_id", StringType(), True)

# But na.fill(0.0) is applied downstream
df_vectors.na.fill(0.0)
```
This hides data quality issues rather than surfacing them.

---

## 6. Performance Optimization

### Critical Gaps

**6.1 Row-by-Row Iteration**
```python
# rt__bronze_to_silver.py line 114
for _, row in pdf.iterrows():
```
`iterrows()` is the slowest way to process pandas DataFrames. Should use vectorized operations or `apply()`.

**6.2 No Shuffle Partition Tuning**
Default `spark.sql.shuffle.partitions` (200) is not tuned for the workload.

**6.3 Single Worker Clusters**
Job definitions specify `"num_workers": 0` (driver-only) or `"num_workers": 1`:
```json
"new_cluster": {
    "node_type_id": "Standard_D2s_v3",
    "num_workers": 0
}
```
This is insufficient for production streaming workloads.

**6.4 No Auto-Optimize Enabled**
Delta table properties should include:
```sql
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact' = 'true'
)
```

---

## 7. Security Practices

### Strengths
- Uses Databricks secret scope backed by Key Vault
- Service principal authentication pattern

### Gaps

**7.1 Hardcoded User Identity**
```json
"single_user_name": "logan@qmachina.com"
```
Should use service principals for production jobs.

**7.2 No IP Restrictions**
Event Hub and ML endpoint calls have no network-level protection.

---

## 8. Testing & Observability

### Critical Gap: Zero Test Coverage
- No unit tests for transformation logic
- No integration tests for streaming pipelines
- No data quality tests (Great Expectations / DLT Expectations)

### Missing Observability
```python
# Should have stream progress monitoring
query.recentProgress  # Not used
query.status  # Not logged

# Should emit custom metrics
spark.sparkContext.setLocalProperty("spark.scheduler.pool", "streaming")
```

---

## 9. Job Definition Analysis

### Strengths
- Properly uses `"continuous"` mode for streaming
- Includes Maven coordinates for Event Hubs library

### Gaps

**9.1 No Retry Policy**
```json
// Missing from job definition
"retry_on_timeout": true,
"max_retries": 10
```

**9.2 No Alerts Configuration**
```json
// Missing
"email_notifications": {
    "on_failure": ["alerts@qmachina.com"]
}
```

**9.3 No Cluster Tags**
No cost attribution or environment tags.

---

## 10. Historical Batch Jobs Assessment

The `backend/databricks/jobs/` batch implementations are cleaner:

### Strengths
- Widget-based parameterization
- `replaceWhere` for idempotent partition writes
- Proper partitioning strategy

### Gaps
- Empty DataFrame handling uses `.rdd.isEmpty()` (expensive full scan)
- No Delta `MERGE` for upserts where applicable
- Window functions without explicit `ROWS BETWEEN` can be non-deterministic

---

# Detailed Remediation Guide with Best Practices

Each section below contains factual best practices with implementation steps and source citations.

---

## PRIORITY 1: Add Watermarking to All Streaming Queries

**Current Gap:** No watermarks defined on any streaming query, causing unbounded state growth.

### Implementation Steps

1. **Add `withWatermark()` before any groupBy or stateful operation:**
   - Apply to the event-time column (e.g., `event_time`)
   - Set threshold based on acceptable late data (e.g., "10 minutes")
   - Place BEFORE the `groupBy` or `window` operation

2. **For stream-stream joins, define watermarks on BOTH sides:**
   - Both input streams must have `withWatermark`
   - Include a time-range join condition to enable state cleanup

3. **Use `dropDuplicatesWithinWatermark` instead of `dropDuplicates`:**
   - Automatically clears dedup state once keys fall outside watermark boundary
   - Far more scalable for high-cardinality streams

4. **Monitor state metrics:**
   - Check `stateOperators.numRowsInMemory` in StreamingQueryProgress
   - If growing indefinitely, watermark or join conditions are misconfigured

### Sources
- **Databricks Docs:** https://docs.databricks.com/aws/en/structured-streaming/watermarks
- **Azure Databricks:** https://learn.microsoft.com/en-us/azure/databricks/ldp/stateful-processing
- **Databricks Community:** https://community.databricks.com/t5/technical-blog/deep-dive-streaming-deduplication/ba-p/105062

---

## PRIORITY 2: Replace `applyInPandasWithState` with `transformWithStateInPandas`

**Current Gap:** Using legacy API without TTL support; state grows unbounded.

### Implementation Steps

1. **Upgrade to Databricks Runtime 16.2+ (17.0 recommended for Spark 4.0):**
   - `transformWithStateInPandas` requires DBR 16.2 or above
   - Full feature support available in DBR 17.0

2. **Refactor from function-based to class-based `StatefulProcessor`:**
   - Create class extending `StatefulProcessor`
   - Implement `init()`, `handleInputRows()`, and `close()` methods
   - Initialize state variables using `StateFactory` in `init()`

3. **Use native state types instead of JSON serialization:**
   - `ValueState` for single values
   - `ListState` for append-efficient lists  
   - `MapState` for key-value lookups within a grouping key

4. **Enable Time-to-Live (TTL) for automatic state expiration:**
   - Define `ttlDuration` when creating state variables
   - Eliminates need for manual timeout logic

5. **Ensure RocksDB state store is enabled (REQUIRED):**
   - `transformWithState` only works with RocksDB provider

### Alternative: Migrate to Lakeflow Declarative Pipelines

For new pipelines, Databricks recommends Lakeflow over manual Structured Streaming. It automates infrastructure scaling, schema evolution, and state management.

### Sources
- **Databricks Blog:** https://www.databricks.com/blog/evolution-arbitrary-stateful-stream-processing-spark
- **Databricks Blog:** https://www.databricks.com/blog/events-insights-complex-state-processing-schema-evolution-transformwithstate
- **Databricks Docs:** https://docs.databricks.com/aws/en/stateful-applications/
- **Spark 4.0 Migration Guide:** https://downloads.apache.org/spark/docs/4.0.0/streaming/ss-migration-guide.html
- **DBR 17.0 Release Notes:** https://learn.microsoft.com/en-us/azure/databricks/release-notes/runtime/17.0

---

## PRIORITY 3: Implement Dead Letter Queue (DLQ) for Error Handling

**Current Gap:** No exception handling in stateful functions; malformed events crash streams.

### Implementation Steps

1. **Split valid and invalid records using `from_json` parsing:**
   - Parse incoming JSON with expected schema
   - Filter where `parsed.isNull()` to identify failures
   - Write valid records to main sink, invalid to DLQ sink

2. **Store raw payloads in DLQ with metadata:**
   - `raw_record` (original bytes/string)
   - `error_message` (parse exception type)
   - `source_topic` or `source_partition`
   - `ingestion_timestamp`
   - `retry_count` (for reprocessing logic)

3. **Use `badRecordsPath` for file-based sources (Auto Loader):**
   - Set `.option("badRecordsPath", "/mnt/dlq/bad_records")`
   - Automatically captures malformed files with exception details

4. **Implement separate checkpoints for DLQ stream:**
   - DLQ writes need independent checkpoint locations
   - Prevents main stream failure from affecting DLQ

5. **Build reprocessing logic:**
   - Read from DLQ table
   - Filter records under retry threshold
   - Attempt re-parsing with updated schema
   - Increment retry_count on failure

### Sources
- **AWS Plain English:** https://aws.plainenglish.io/handling-bad-records-in-streaming-pipelines-using-dead-letter-queues-in-pyspark-265e7a55eb29
- **AWS Plain English (Part 2):** https://aws.plainenglish.io/productionizing-dead-letter-queues-in-pyspark-streaming-pipelines-part-2-fd228fb99fe5
- **Databricks Docs:** https://docs.databricks.com/en/ingestion/bad-records.html
- **Confluent Guide:** https://www.confluent.io/learn/kafka-dead-letter-queue/

---

## PRIORITY 4: Add Rate Limiting for Backpressure Control

**Current Gap:** No `maxFilesPerTrigger` or `maxBytesPerTrigger`; OOM risk on data spikes.

### Implementation Steps

1. **For Delta Lake sources, add rate limiting options:**
   - `.option("maxFilesPerTrigger", 1000)` - cap file count per batch
   - `.option("maxBytesPerTrigger", "1g")` - soft cap on bytes per batch
   - When both set, batch stops at whichever limit is reached first

2. **For Auto Loader (cloudFiles), use prefixed options:**
   - `.option("cloudFiles.maxFilesPerTrigger", 500)`
   - `.option("cloudFiles.maxBytesPerTrigger", "2g")`

3. **For Kafka sources, use offset-based limiting:**
   - `.option("maxOffsetsPerTrigger", 10000)` - cap records per batch

4. **Monitor backlog metrics:**
   - Check `numFilesOutstanding` and `numBytesOutstanding`
   - Available in Raw Data tab of streaming query progress

5. **Tune based on cluster capacity:**
   - Target partition size: 128MB - 200MB
   - If partitions < 100MB, increase batch size (overhead per task)
   - If partitions > 1GB, reduce batch size (spilling risk)

### Sources
- **Azure Databricks:** https://learn.microsoft.com/en-us/azure/databricks/structured-streaming/batch-size
- **Databricks Docs:** https://docs.databricks.com/aws/en/ingestion/cloud-object-storage/auto-loader/options
- **Auto Loader Production:** https://learn.microsoft.com/en-us/azure/databricks/ingestion/cloud-object-storage/auto-loader/production

---

## PRIORITY 5: Enable RocksDB State Store with Changelog Checkpointing

**Current Gap:** Using default in-memory state store; checkpoint duration and GC pressure issues.

### Implementation Steps

1. **Enable RocksDB as state store provider:**
   ```
   spark.conf.set("spark.sql.streaming.stateStore.providerClass",
                  "com.databricks.sql.streaming.state.RocksDBStateStoreProvider")
   ```

2. **Enable changelog checkpointing (DBR 13.3 LTS+):**
   ```
   spark.conf.set("spark.sql.streaming.stateStore.rocksdb.changelogCheckpointing.enabled", "true")
   ```
   - Only writes changed records to checkpoint instead of full snapshots
   - Dramatically reduces checkpoint duration for large state

3. **For latency-critical workloads, enable async checkpointing:**
   ```
   spark.conf.set("spark.databricks.streaming.statefulOperator.asyncCheckpoint.enabled", "true")
   ```
   - Next micro-batch can start without waiting for checkpoint completion
   - WARNING: Recovery may take longer on failure

4. **Set configurations at cluster level (not in notebook):**
   - Add to job cluster definition or cluster policy
   - Ensures consistency across all streaming jobs

5. **Monitor RocksDB metrics:**
   - `rocksdbCommitCheckpointLatency`
   - `rocksdbCommitFileSyncLatencyMs`
   - Available in StreamingQueryProgress JSON

### Sources
- **Azure Databricks:** https://learn.microsoft.com/en-us/azure/databricks/structured-streaming/rocksdb-state-store
- **Databricks Docs:** https://docs.databricks.com/aws/en/structured-streaming/rocksdb-state-store
- **Async Checkpointing:** https://docs.azure.cn/en-us/databricks/structured-streaming/async-checkpointing
- **Databricks Community:** https://community.databricks.com/t5/technical-blog/demystifying-spark-s-statestore-api-for-robust-stateful/ba-p/119239

---

## PRIORITY 6: Add Streaming Metrics and Alerting

**Current Gap:** No monitoring integration; `print()` statements instead of proper logging.

### Implementation Steps

1. **Configure native Databricks job notifications:**
   - Add `email_notifications` to job JSON definition
   - Configure for `on_start`, `on_success`, `on_failure`
   - Add `streaming_backlog_threshold` alert for streaming jobs

2. **Use System Tables for account-wide monitoring:**
   - Query `system.lakeflow` schema for job runs and tasks
   - Query `system.lakeflow.pipeline_update_timeline` (Public Preview Sept 2025)
   - Join with `system.billing` for cost-per-stream analysis

3. **Export to Azure Monitor:**
   - Configure Azure Monitoring Extension for Databricks
   - Export to Log Analytics for long-term retention
   - Create dashboards in Azure Monitor or Power BI

4. **Add custom StreamingQueryListener:**
   - Log `recentProgress` and `status` to structured logging
   - Emit custom metrics to Azure Application Insights
   - Alert on `numRowsDroppedByWatermark` increases

5. **Replace `print()` with proper logging:**
   - Use `logging` module with appropriate handlers
   - Logs appear in Spark driver stdout which is captured by Databricks

### Sources
- **Azure Databricks:** https://learn.microsoft.com/en-us/azure/databricks/jobs/monitor
- **Observability Best Practices:** https://learn.microsoft.com/en-us/azure/databricks/data-engineering/observability-best-practices
- **Job Notifications:** https://learn.microsoft.com/en-us/azure/databricks/jobs/notifications
- **Pipeline Monitoring UI:** https://learn.microsoft.com/en-us/azure/databricks/ldp/monitoring-ui

---

## PRIORITY 7: Create Unit and Integration Test Suites

**Current Gap:** Zero test coverage for transformation logic.

### Implementation Steps

1. **Reorganize code for testability:**
   - Move transformation logic from notebooks to `.py` modules
   - Place in `src/` directory with tests in `tests/`
   - Use Databricks Git Folders to sync

2. **Create unit tests with local SparkSession:**
   - Use `pytest` with `pyspark` library
   - Create fixtures for SparkSession lifecycle
   - Test individual transformation functions in isolation
   - Use `unittest.mock` to isolate external dependencies

3. **Create integration tests with Databricks Connect:**
   - Install `databricks-connect` package
   - Configure profile to connect to test cluster
   - Test against dedicated "test" catalog in Unity Catalog
   - Use `pytest` fixtures with `yield` for cleanup

4. **Test streaming logic with micro-batch simulation:**
   - Treat single micro-batch as static DataFrame
   - Write small files to temp directory for `readStream` testing
   - Use temporary checkpoint locations

5. **Automate with Databricks Asset Bundles (DABs):**
   - Use `databricks bundle run` to trigger tests
   - Integrate with GitHub Actions or GitLab CI
   - Run unit tests on every PR, integration tests on merge

6. **Add coverage reporting:**
   - Use `pytest-cov` to track test coverage
   - Set minimum threshold (e.g., 80%)

### Sources
- **Databricks Community:** https://community.databricks.com/t5/technical-blog/integration-testing-for-lakeflow-jobs-with-pytest-and-databricks/ba-p/141683
- **Databricks Docs:** https://docs.databricks.com/aws/en/notebooks/testing
- **Unit Testing Blog:** https://community.databricks.com/t5/technical-blog/writing-unit-tests-for-pyspark-in-databricks-approaches-and-best/ba-p/122398
- **VS Code pytest:** https://docs.databricks.com/aws/en/dev-tools/vscode-ext/pytest

---

## PRIORITY 8: Migrate to Unity Catalog Managed Tables

**Current Gap:** Writing to ABFSS paths directly; losing governance features.

### Implementation Steps

1. **Use `.toTable()` instead of `.start()` with paths:**
   - Replace: `.start("abfss://lake@storage.dfs.core.windows.net/bronze/stream")`
   - With: `.toTable("spymaster.bronze.mbo_stream")`

2. **Enable Predictive Optimization on managed tables:**
   - AI-driven automatic `OPTIMIZE` and `VACUUM`
   - No manual scheduling required
   - Available for Unity Catalog managed tables only

3. **Store checkpoints in Unity Catalog External Locations:**
   - Creates governed, auditable checkpoint storage
   - Accessible by service principals with proper grants

4. **Use streaming tables for bronze ingestion:**
   - Create via SQL: `CREATE OR REFRESH STREAMING TABLE`
   - Benefits from serverless execution and Auto Loader integration

5. **Configure service principals for job execution:**
   - Set "Run as" to service principal, not individual user
   - Grant minimal required UC permissions (SELECT, MODIFY)

6. **Leverage UniForm for interoperability:**
   - Managed tables support Universal Format
   - Readable by Iceberg, Delta, and Hudi clients

### Sources
- **Unity Catalog Best Practices:** https://docs.databricks.com/aws/en/data-governance/unity-catalog/best-practices
- **Managed Tables:** https://docs.databricks.com/aws/en/tables/managed
- **Predictive Optimization:** https://www.databricks.com/blog/how-unity-catalog-managed-tables-automate-performance-scale
- **UC Streaming:** https://docs.azure.cn/en-us/databricks/structured-streaming/unity-catalog

---

## ADDITIONAL ITEMS

### Item 1.1: Migrate to Lakeflow Declarative Pipelines

**Implementation Steps:**

1. **Create streaming tables with `CREATE OR REFRESH STREAMING TABLE`**
2. **Use `@dlt.table` decorator for Python pipelines**
3. **Enable AUTO CDC for Change Data Capture workloads**
4. **Leverage serverless execution for automatic scaling**

**Sources:**
- https://learn.microsoft.com/en-us/azure/databricks/ldp/
- https://learn.microsoft.com/en-us/azure/databricks/ldp/streaming-tables
- https://docs.databricks.com/aws/en/dlt/concepts

---

### Item 6.1: Replace `iterrows()` with Vectorized Operations

**Implementation Steps:**

1. **Use native Pandas/NumPy operations instead of row iteration**
2. **For UDFs, use `@pandas_udf` with Series-to-Series signature**
3. **Enable Arrow optimization:**
   ```
   spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
   ```
4. **For stateful operations, use Iterator-of-Series pattern to load model once per partition**

**Sources:**
- https://www.canadiandataguy.com/p/why-your-pyspark-udf-is-slowing-everything
- https://docs.databricks.com/en/udf/pandas.html
- https://spark.apache.org/docs/latest/api/python/tutorial/pandas_on_spark/best_practices.html

---

### Item 6.2: Tune `spark.sql.shuffle.partitions`

**Implementation Steps:**

1. **Enable Adaptive Query Execution (AQE):**
   - Default in DBR 7.3+
   - Dynamically coalesces shuffle partitions
2. **Enable auto-optimized shuffle:**
   ```
   spark.conf.set("spark.databricks.adaptive.autoOptimizeShuffle.enabled", "true")
   ```
3. **For Real-time Mode (DBR 16.4+), calculate required task slots:**
   - Slots needed = source partitions + shuffle partitions
4. **Target partition size: 128MB - 200MB**

**Sources:**
- https://docs.databricks.com/aws/en/structured-streaming/real-time
- https://www.e6data.com/query-and-cost-optimization-hub/databricks-performance-optimization-complete-query-tuning-guide-2025

---

### Item 6.4: Enable Auto-Optimize for Delta Tables

**Implementation Steps:**

1. **Enable via table properties:**
   ```sql
   ALTER TABLE my_table SET TBLPROPERTIES (
     'delta.autoOptimize.optimizeWrite' = 'true',
     'delta.autoOptimize.autoCompact' = 'true'
   );
   ```
2. **For new tables, consider Liquid Clustering:**
   ```sql
   CREATE TABLE sales CLUSTER BY (customer_id);
   ```
3. **For Unity Catalog tables, enable Predictive Optimization**

**Sources:**
- https://docs.databricks.com/aws/en/delta/tune-file-size
- https://delta.io/blog/delta-lake-optimize/
- https://docs.delta.io/latest/optimizations-oss.html

---

### Item 7.1: Use Service Principals for Production Jobs

**Implementation Steps:**

1. **Create service principal at account level (not workspace)**
2. **Configure "Run as" in job definition to service principal**
3. **Use OAuth tokens instead of Personal Access Tokens**
4. **Grant minimal Unity Catalog permissions to groups (not individuals)**
5. **Enable `RestrictWorkspaceAdmins` for segregation of duties**

**Sources:**
- https://docs.databricks.com/aws/en/admin/users-groups/best-practices
- https://docs.databricks.com/aws/en/jobs/privileges
- https://www.databricks.com/blog/whats-new-security-and-compliance-data-ai-summit-2025

---

### Items 9.1-9.3: Job Definition Configuration

**Implementation Steps:**

1. **Add retry policy to task definitions:**
   ```json
   "retry_policy": {
     "max_retries": 3,
     "interval_ms": 5000,
     "retry_on_timeout": true
   }
   ```

2. **Add email and webhook notifications:**
   ```json
   "email_notifications": {
     "on_failure": ["oncall@company.com"]
   },
   "notification_settings": {
     "alert_on_last_attempt": true
   }
   ```

3. **Add cluster tags for cost attribution:**
   ```json
   "custom_tags": {
     "Environment": "Production",
     "Project": "Spymaster",
     "Owner": "DataEng"
   }
   ```

**Sources:**
- https://docs.databricks.com/aws/en/jobs/configure-job
- https://docs.databricks.com/aws/en/jobs/notifications
- https://learn.microsoft.com/en-us/azure/databricks/workflows/jobs/job-notifications

---

# Summary

| Priority | Item | Primary Documentation |
|----------|------|----------------------|
| 1 | Watermarking | https://docs.databricks.com/aws/en/structured-streaming/watermarks |
| 2 | transformWithStateInPandas | https://docs.databricks.com/aws/en/stateful-applications/ |
| 3 | Dead Letter Queue | https://docs.databricks.com/en/ingestion/bad-records.html |
| 4 | Rate Limiting | https://learn.microsoft.com/en-us/azure/databricks/structured-streaming/batch-size |
| 5 | RocksDB + Changelog | https://learn.microsoft.com/en-us/azure/databricks/structured-streaming/rocksdb-state-store |
| 6 | Metrics & Alerting | https://learn.microsoft.com/en-us/azure/databricks/data-engineering/observability-best-practices |
| 7 | Testing | https://docs.databricks.com/aws/en/notebooks/testing |
| 8 | Unity Catalog | https://docs.databricks.com/aws/en/data-governance/unity-catalog/best-practices |
