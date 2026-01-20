Below is **one** Azure-native blueprint that moves your working local Python pipeline into an enterprise-grade system with (a) **medallion lakehouse**, (b) **repeatable ML experimentation + MLflow + managed inference endpoint**, and (c) a **built-in real-time dashboard/UI with anomaly detection** (no custom UI required).

I’m locking the stack to:

* **ADLS Gen2** for immutable storage + medallion tables
* **Azure Databricks** for Bronze/Silver/Gold processing (batch + streaming)
* **Azure Data Factory** for orchestration of historical jobs
* **Azure Machine Learning** for experiments (Hyperopt), MLflow tracking/registry, and managed online inference endpoints ([Microsoft Learn][1])
* **Azure Event Hubs** as the real-time event log/bus
* **Microsoft Fabric Real-Time Intelligence (Eventstream + Eventhouse + Real-Time Dashboards/Power BI)** for built-in real-time visualization and anomaly detection ([Microsoft Learn][2])

---

# Execution Plan (Living)

1. Deploy Databricks streaming jobs (secret scope, notebook upload, job creation). [COMPLETE]
2. Configure Fabric Real-Time Intelligence (Eventhouse, Eventstreams, dashboard). [COMPLETE]
3. Run end-to-end pipeline test with synthetic events and validate outputs. [IN_PROGRESS]
4. Apply production hardening (monitoring, cost, security). [PENDING]

# 0) Target runtime contract 

**Core rule:** the *same* logical stages exist in both batch and streaming:

* **Bronze** = parse/normalize raw MBO events (no feature logic, no book state)
* **Silver** = session logic + stateful book reconstruction + canonical rollups (5s bars, etc.)
* **Gold** = model-ready feature vectors (your multi-window lookback vectors) + labels (for historical)

Your existing stage classes map 1:1 into Databricks tasks (batch) and into streaming queries (real-time). For real-time, Silver uses **stateful processing in PySpark** (supported via `applyInPandasWithState`) to persist book state per instrument across micro-batches. ([spark.apache.org][3])

---

# 1) Foundation setup (do this once)

## 1.1 Create the Azure resources (single region, single subscription boundary)

Create these resources in one Resource Group:

1. **Azure Storage account (ADLS Gen2 enabled)** [COMPLETE]

   * Containers:

     * `raw-dbn/` (uploaded `.dbn.zst`)
     * `lake/bronze/`
     * `lake/silver/`
     * `lake/gold/`
     * `lake/checkpoints/` (streaming checkpoints)
     * `ml/artifacts/` (exported train datasets, model backtest outputs)

2. **Azure Databricks workspace** [COMPLETE]

   * Enable **Unity Catalog** (governance) and register ADLS as an External Location. [COMPLETE]
   * Create 3 catalogs/schemas (or equivalent UC structure): `bronze`, `silver`, `gold`. [COMPLETE]

3. **Azure Data Factory** [COMPLETE]

   * This will only orchestrate historical (batch) runs and backfills.

4. **Azure Machine Learning workspace** [COMPLETE]

   * This becomes your system of record for: MLflow tracking, experiment runs, model registry, managed endpoints. ([Microsoft Learn][1])

5. **Azure Event Hubs namespace** [COMPLETE]

   * Event Hubs (topics):

     * `mbo_raw` (raw events from Databento)
     * `features_gold` (real-time Gold vectors + feature series)
     * `inference_scores` (real-time inference outputs)
   * Create distinct **consumer groups** for:

     * Databricks Bronze stream
     * Fabric Eventstream ingestion
     * Any downstream analytics/QA

6. **Microsoft Fabric workspace (Real-Time Intelligence)** [COMPLETE]

   * You will create:

     * one **Eventhouse** (KQL database) ([Microsoft Learn][4])
     * one **Eventstream** to ingest from Event Hubs ([Microsoft Learn][5])
     * one **Real-Time Dashboard** (tiles backed by KQL queries) ([Microsoft Learn][6])
     * Power BI report(s) for anomaly detection on feature/inference time-series ([Microsoft Learn][7])

7. **Azure Key Vault** [COMPLETE]

   * Store:

     * Databento API key(s)
     * Any broker keys later
   * Use managed identity access for runtime services.

---

## 1.2 Define the enterprise data contract (required for everything downstream) [COMPLETE]

Create a shared schema contract (documented in repo) for:

### Raw MBO event envelope (Bronze input)

* `event_time` (exchange or venue timestamp)
* `ingest_time` (Azure arrival timestamp)
* `venue`
* `symbol`
* `instrument_type` (`FUT`, `OPT`, `STK`)
* `underlier` (e.g., `ES`, `TSLA`)
* `contract_id` (canonical ID you define)
* `action` (add/cancel/modify/trade/etc.)
* `order_id`
* `side`, `price`, `size`
* `sequence` (monotonic per venue feed if available)
* `payload` (raw fields you keep for fidelity)

### Gold feature vector record (for training + inference)

* `vector_time` (the time the vector is valid for)
* `model_id` (`ES_MODEL`, `TSLA_MODEL`)
* `contract_id` (or `symbol`)
* `lookback_spec` (your window definitions)
* `feature_vector` (dense array)
* `feature_hash` (deterministic hash for reproducibility)
* `label` (historical only; omitted in real-time)
* `data_version` (bronze/silver/gold lineage pointers)

This contract is what keeps “historical” and “real-time” identical.

---

# 2) Phase 1 — Build Model (Historical)

## Step 1 — Upload `.dbn.zst` to **ADLS Gen2** [COMPLETE]

* Landing path convention:

  * `raw-dbn/{underlier}/{instrument_type}/{yyyymmdd}/{file}.dbn.zst`
* Maintain a small manifest table (Delta in `bronze`) that lists:

  * file path, checksum, session date, underlier, instrument_type, ingestion batch id

## Step 2 — Process to **Bronze** using **Azure Databricks Job** [COMPLETE]

Create a Databricks Job: `hist__dbn_to_bronze`

* Task: run your `BronzeProcessDBN`
* Reads `.dbn.zst` from ADLS
* Writes **Delta** table(s) to `lake/bronze/`

  * Partition by: `session_date`, `underlier`, `instrument_type`
* Guarantees:

  * schema enforced
  * dedupe policy (based on `venue+sequence` or `order_id+event_time+action` per your rules)
  * append-only

## Step 3 — Process to **Silver** using **Azure Databricks Job** [COMPLETE]

Create a Databricks Job: `hist__bronze_to_silver`

Map tasks exactly to your stage classes in order:

1. `SilverAddSessionLevels`
2. `SilverFilterFirst4Hours`
3. `SilverComputeBar5sFeatures`
4. `SilverExtractLevelApproach2m`

**Silver outputs (Delta in `lake/silver/`)**

* `silver.mbo_normalized` (post session tagging + filters)
* `silver.orderbook_state` (snapshots and/or state transitions you persist)
* `silver.bar_5s` (canonical rollups used downstream)
* `silver.feature_primitives` (intermediate computed fields)

## Step 4 — Process to **Gold** using **Azure Databricks Job** [COMPLETE]

Create a Databricks Job: `hist__silver_to_gold`

* Task: `GoldExtractSetupVectors`
* Output:

  * `gold.setup_vectors` (your multi-window vectors)
  * `gold.labels` (if you label events/outcomes)
  * `gold.training_view` (join of vectors + labels + metadata)

## Step 5 — Prepare ML features using **Databricks → Export snapshot to ADLS** [COMPLETE]

To make Azure ML training deterministic and versioned:

* Each training run writes a **frozen snapshot** (Parquet) to:

  * `ml/artifacts/datasets/{model_id}/{run_date}/`
* Snapshot includes:

  * vectors, labels, and the exact `data_version` lineage pointers

(Delta stays the system of record; the snapshot is the immutable training input.)

## Step 6 — Split train/test/eval using **Azure Machine Learning Job** [COMPLETE]

In Azure ML:

* Register the snapshot folder as a **Data Asset** version (for the model_id).
* Create an Azure ML pipeline/job component that:

  * does a **time-ordered split** by `session_date` (never random for market microstructure)
  * emits `train/`, `val/`, `test/` partitions back to `ml/artifacts/datasets/...`

## Step 7 — Run experiments w/Hyperopt using **Azure Machine Learning + MLflow** [COMPLETE]

In Azure ML:

* Create a **training job** that uses your model training entrypoint.
* Use **Hyperopt** inside an Azure ML hyperparameter sweep job (implementt the sweep wrapper).
* Configure MLflow tracking to the Azure ML workspace (Azure ML workspaces are MLflow-compatible). ([Microsoft Learn][8])

Result:

* Every run logs params/metrics/artifacts to MLflow
* The best run produces a logged MLflow model artifact

**Completion notes:**
- Sweep job `train_hyperopt_sweep_v3` completed successfully (2 trials).
- Best validation accuracy: ~55.67% (loss: 0.4433)
- Hyperopt parameters: C (regularization), max_iter optimized via random search.
- Model output saved to Azure ML outputs.

## Step 8 — View results and promote model to an Inference Endpoint using **Azure ML Studio + Managed Online Endpoint** [COMPLETE]

* Review runs in **Azure ML Studio** (experiment list + MLflow run details). ([Microsoft Learn][1]) [COMPLETE]
* Register the winning model to the Azure ML registry (MLflow model registry supported). ([Microsoft Learn][1]) [COMPLETE]
* Deploy as a **Managed Online Endpoint** using MLflow "no-code deployment" for real-time inference. ([Microsoft Learn][9]) [COMPLETE]

**Completion notes:**
- Model registered: `es_logreg_model:1` in Azure ML model registry.
- Endpoint deployed: `es-model-endpoint` with deployment `blue` (100% traffic).
- Scoring URI: `https://es-model-endpoint.westus.inference.ml.azure.com/score`
- Inference tested successfully - returns predictions and probabilities.
- Previous `SubscriptionNotRegistered` error resolved after resource provider registration.

At this point you have:

* `ES_MODEL` registered and deployed to a managed online endpoint

---

# 3) Phase 2 — Real-Time Analysis (Databento → features → inference batches → dashboard)

## Step 1 — Ingest via Databento API into **Azure Event Hubs** [DEFERRED - CODE READY]

Create a runtime service: `databento_stream_ingestor`

* Host it as an Azure-managed container runtime (set up the deployment)
* Responsibilities:

  * Connect to Databento live stream
  * Filter to:

    * ES futures + ES options chain
    * TSLA stock + TSLA options chain
  * Wrap events in the **Bronze envelope schema**
  * Publish to Event Hub: `mbo_raw`
* Secrets:

  * Databento API key pulled at runtime from **Key Vault** using managed identity

**Status:** Code ready at `infra/containers/databento_ingestor/`. Deployment deferred until live Databento subscription is active. Currently using DBN file uploads for historical processing.

## Step 2 — Bronze stream using **Databricks Structured Streaming** [COMPLETE]

Create a Databricks streaming job: `rt__mbo_raw_to_bronze`

* Source: Event Hubs `mbo_raw`
* Output: Delta append-only to `lake/bronze/stream/`
* Checkpoint: `lake/checkpoints/rt__bronze/`
* Guarantees:

  * schema enforcement
  * idempotent writes per your dedupe key
  * partitions aligned to `underlier`, `instrument_type`, `session_date`

**Implementation:** `infra/databricks/streaming/rt__mbo_raw_to_bronze.py`

## Step 3 — Silver stream (stateful orderbook + rollups) using **Databricks stateful streaming in Python** [COMPLETE]

Create a Databricks streaming job: `rt__bronze_to_silver`

* Source: `bronze` Delta stream
* Use **stateful processing** keyed by `contract_id` (or symbol) to persist:

  * order_id map
  * depth levels
  * session resets/roll rules
* Implement with PySpark `applyInPandasWithState` so the book state persists across micro-batches in a supported way. ([spark.apache.org][3])
* Outputs (Delta):

  * `silver.orderbook_state_stream`
  * `silver.bar_5s_stream`
  * `silver.feature_primitives_stream`

**Implementation:** `infra/databricks/streaming/rt__bronze_to_silver.py`

## Step 4 — Gold stream (feature vectors) using **Databricks Structured Streaming** [COMPLETE]

Create a Databricks streaming job: `rt__silver_to_gold`

* Source: Silver outputs
* Computes:

  * your multi-window derivatives, OFI, "market physics" features, etc.
  * emits *both*:

    1. a compact **feature time-series** (for dashboards)
    2. the **model input vectors** (for inference batches)
* Outputs:

  * Delta `gold.setup_vectors_stream`
  * Delta `gold.feature_series_stream`
  * Publish to Event Hub `features_gold` (for Fabric ingestion)

**Implementation:** `infra/databricks/streaming/rt__silver_to_gold.py`

## Step 5 — Inference every N seconds in batches using **Databricks → Azure ML Online Endpoint** [COMPLETE]

Create a Databricks streaming job: `rt__gold_to_inference`

* Source: `gold.setup_vectors_stream`
* Micro-batch trigger: `N seconds` (set this as a streaming trigger)
* For each micro-batch:

  * group by `model_id` (ES vs TSLA)
  * call the corresponding **Azure ML Managed Online Endpoint**
  * attach `model_version` and `feature_hash` to every prediction record
* Outputs:

  * Event Hub `inference_scores`
  * Delta `gold.inference_scores_stream`

(Endpoint behavior is standard Azure ML online inference over HTTPS. ([Microsoft Learn][10]))

**Implementation:** `infra/databricks/streaming/rt__gold_to_inference.py`
**Secret:** AML endpoint key stored in Key Vault `kvspymasterdevrtoxxrlojs` as `aml-endpoint-key`

---

## Step 6 — Show features + inference on a built-in dashboard using **Fabric Real-Time Intelligence** [COMPLETE - CONFIG READY]

This gives you "TradingView-like monitoring" without building a UI.

### 6.1 Create the Eventhouse

In Fabric (Real-Time Intelligence):

* Create **Eventhouse** `trading_eventhouse` (KQL database). ([Microsoft Learn][4])
* Create KQL tables:

  * `features` (from `features_gold`)
  * `scores` (from `inference_scores`)

### 6.2 Create the Eventstream ingest

* Create **Eventstream** `trading_stream`
* Sources:

  * Azure Event Hubs `features_gold`
  * Azure Event Hubs `inference_scores`
* Destination:

  * `trading_eventhouse.features`
  * `trading_eventhouse.scores` ([Microsoft Learn][5])

### 6.3 Create the Real-Time Dashboard (Fabric)

* Build a Fabric **Real-Time Dashboard** backed by KQL queries:

  * feature time-series tiles (selectable by `underlier`, `contract_id`)
  * inference score time-series tiles
  * "top movers" tables (largest change in OFI / pressure / etc.)
    Fabric dashboards are first-class RTI artifacts where tiles are backed by queries. ([Microsoft Learn][6])

### 6.4 Add anomaly detection (no custom ML UI)

Do both of these:

1. **Power BI line-chart anomaly detection** on feature and score series (built-in). ([Microsoft Learn][7])
2. KQL-based anomaly flags in Eventhouse using `series_decompose_anomalies()` for features like OFI spikes, liquidity vacuums, etc. (applies to Microsoft Fabric KQL). ([Microsoft Learn][11])

**Implementation:**
- KQL schema: `infra/fabric/eventhouse_schema.kql`
- Dashboard queries: `infra/fabric/dashboard_queries.kql`
- Setup guide: `infra/fabric/SETUP_GUIDE.md`

Result: users get real-time visuals + anomaly highlighting immediately.

---

# 4) Orchestration & run control (how the whole system stays deterministic)

## 4.1 Historical orchestration with **Azure Data Factory**

Create one ADF pipeline per model family:

* `adf_hist_es_pipeline`
* `adf_hist_tsla_pipeline`

Each ADF pipeline runs Databricks jobs in order:

1. `hist__dbn_to_bronze`
2. `hist__bronze_to_silver`
3. `hist__silver_to_gold`
4. “export training snapshot” task
5. triggers Azure ML training pipeline/job

ADF is responsible for “when” and “which inputs”; Databricks is responsible for “how the transforms compute”.

## 4.2 ML promotion gate is in **Azure ML**

Azure ML owns:

* experiment lineage (MLflow)
* model registry versions
* endpoint deployments ([Microsoft Learn][1])

Databricks real-time inference always calls a **specific model version** (passed via config), so rollout is controlled and reversible.

---

# 5) Implementation backlog in order

1. **Repo packaging**

   * Convert your local pipeline into a package installable in Databricks and Azure ML.
   * Preserve your stage class contract; adapt internals to Spark DataFrames for batch and streaming.

2. **Databricks batch jobs**

   * Implement the three historical jobs and ensure Delta tables land in Bronze/Silver/Gold paths exactly as specified.

3. **Azure ML training job**

   * Implement the training entrypoint that reads the frozen dataset snapshot, runs Hyperopt sweeps, logs to MLflow in AML. ([Microsoft Learn][8])

4. **Azure ML endpoint deployment**

   * Deploy MLflow model to managed online endpoint; expose endpoint URIs + auth to Databricks via Key Vault. ([Microsoft Learn][9])

5. **Real-time ingestion**

   * Implement Databento ingest container → Event Hubs.

6. **Databricks streaming**

   * Bronze stream
   * Silver stateful book stream using `applyInPandasWithState` ([spark.apache.org][3])
   * Gold feature vector stream
   * Inference micro-batch stream calling AML endpoint

7. **Fabric RTI dashboards**

   * Eventstream → Eventhouse tables → real-time dashboard
   * Power BI anomaly detection on selected series ([Microsoft Learn][7])

---

[1]: https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2&utm_source=chatgpt.com "MLflow and Azure Machine Learning"
[2]: https://learn.microsoft.com/en-us/fabric/real-time-intelligence/?utm_source=chatgpt.com "Real-Time Intelligence documentation in Microsoft Fabric"
[3]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.applyInPandasWithState.html?utm_source=chatgpt.com "pyspark.sql.GroupedData.applyInPandasWithState"
[4]: https://learn.microsoft.com/en-us/fabric/real-time-intelligence/eventhouse?utm_source=chatgpt.com "Eventhouse overview - Microsoft Fabric"
[5]: https://learn.microsoft.com/en-us/fabric/real-time-intelligence/event-streams/overview?utm_source=chatgpt.com "Microsoft Fabric Eventstreams Overview"
[6]: https://learn.microsoft.com/en-us/fabric/real-time-intelligence/dashboard-real-time-create?utm_source=chatgpt.com "Create a Real-Time Dashboard - Microsoft Fabric"
[7]: https://learn.microsoft.com/en-us/power-bi/visuals/power-bi-visualization-anomaly-detection?utm_source=chatgpt.com "Anomaly Detection Tutorial for Power BI"
[8]: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-configure-tracking?view=azureml-api-2&utm_source=chatgpt.com "Configure MLflow for Azure Machine Learning"
[9]: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models-online-endpoints?view=azureml-api-2&utm_source=chatgpt.com "Deploy MLflow models to online endpoints - Azure"
[10]: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&utm_source=chatgpt.com "Deploy Machine Learning Models to Online Endpoints"
[11]: https://learn.microsoft.com/en-us/kusto/query/series-decompose-anomalies-function?view=microsoft-fabric&utm_source=chatgpt.com "series_decompose_anomalies() - Kusto"

---

# 6) Current State & Handoff


## Completed Work

### Phase 1 - Historical Pipeline
- **Steps 1-4**: Databricks batch jobs executed (`hist__dbn_to_bronze`, `hist__bronze_to_silver`, `hist__silver_to_gold`)
- **Step 5**: Training snapshot exported to `ml-artifacts/datasets/ES_MODEL/2025-10-01/`
- **Step 6**: Train/val/test split completed in Azure ML
- **Step 7**: Hyperopt sweep `train_hyperopt_sweep_v3` completed (55.67% val accuracy)
- **Step 8**: Model `es_logreg_model:1` deployed to `es-model-endpoint`

### Phase 2 - Streaming Pipeline 
- **Step 1**: Databento ingestor code ready at `infra/containers/databento_ingestor/` (test with mock service/data)
- **Steps 2-5**: Streaming notebooks created at `infra/databricks/streaming/`
- **Step 6**: Fabric RTI configuration at `infra/fabric/`

### Infrastructure as Code
- Bicep modules: `infra/main.bicep` + `infra/modules/*.bicep`
- Parameters: `infra/main.parameters.json`
- All resources deployable via: `az deployment group create --template-file main.bicep --parameters main.parameters.json`


---

# 7) Next Engineer Actions

## Priority 1: Deploy Databricks Streaming Jobs

### 1.1 Create Databricks Secret Scope [COMPLETE]
```bash
# Install Databricks CLI and configure
databricks configure --token

# Create secret scope linked to Key Vault
databricks secrets create-scope --scope spymaster \
  --scope-backend-type AZURE_KEYVAULT \
  --resource-id /subscriptions/70464868-52ea-435d-93a6-8002e83f0b89/resourceGroups/rg-spymaster-dev/providers/Microsoft.KeyVault/vaults/kvspymasterdevrtoxxrlojs \
  --dns-name https://kvspymasterdevrtoxxrlojs.vault.azure.net/
```

### 1.2 Upload Streaming Notebooks [COMPLETE]
Upload all files from `infra/databricks/streaming/` to Databricks workspace:
- `rt__mbo_raw_to_bronze.py`
- `rt__bronze_to_silver.py`
- `rt__silver_to_gold.py`
- `rt__gold_to_inference.py`

### 1.3 Create Databricks Jobs [COMPLETE]
For each notebook, create a job with:
- Cluster: Shared job cluster with Event Hubs library (`com.microsoft.azure:azure-eventhubs-spark_2.12:2.3.22`)
- Schedule: Continuous (streaming)
- Retries: Unlimited

## Priority 2: Configure Fabric Real-Time Intelligence

### 2.1 Create Eventhouse [COMPLETE]
1. Open Fabric workspace `qfabric`
2. Create Eventhouse: `trading_eventhouse`
3. Run KQL from `infra/fabric/eventhouse_schema.kql` to create tables

### 2.2 Create Eventstreams [COMPLETE]
1. Create Eventstream: `features_stream`
   - Source: Event Hubs `features_gold` (consumer group: `fabric_stream`)
   - Destination: `trading_eventhouse.features`
2. Create Eventstream: `scores_stream`
   - Source: Event Hubs `inference_scores` (consumer group: `fabric_stream`)
   - Destination: `trading_eventhouse.scores`

### 2.3 Create Real-Time Dashboard [COMPLETE]
1. Create dashboard: `Spymaster Trading Intelligence`
2. Add tiles using queries from `infra/fabric/dashboard_queries.kql`
3. Set auto-refresh: 30 seconds

See `infra/fabric/SETUP_GUIDE.md` for detailed instructions.

## Priority 3: End-to-End Testing

### 3.1 Test with Synthetic Data [COMPLETE]
```python
# Publish test events to Event Hub mbo_raw
from azure.eventhub import EventHubProducerClient, EventData
import json

conn_str = "Endpoint=sb://ehnspymasterdevoxxrlojskvxey..."
producer = EventHubProducerClient.from_connection_string(conn_str, eventhub_name="mbo_raw")

test_event = {
    "event_time": 1705000000000000000,
    "ingest_time": 1705000000000000000,
    "venue": "GLBX.MDP3",
    "symbol": 12345,
    "instrument_type": "FUT",
    "underlier": "ES",
    "contract_id": "ESH6",
    "action": "T",
    "order_id": 1,
    "side": "B",
    "price": 5000.0,
    "size": 1,
    "sequence": 1,
    "payload": "{}"
}

batch = producer.create_batch()
batch.add(EventData(json.dumps(test_event)))
producer.send_batch(batch)
```

Completion notes:
- Sent a synthetic event with `backend/scripts/send_eventhub_test.py` using Key Vault secret `eventhub-connection-string`.

### 3.2 Validate Pipeline Flow [IN_PROGRESS]
1. Verify Bronze Delta table populated in `lake/bronze/stream/`
2. Verify Silver bar_5s stream in `lake/silver/bar_5s_stream/`
3. Verify Gold vectors in `lake/gold/setup_vectors_stream/`
4. Verify inference scores in `lake/gold/inference_scores_stream/`
5. Verify Fabric dashboard shows real-time data



## Continuation (Next Engineer)

1. Confirm `rt__mbo_raw_to_bronze` is running and processing events; check run output for streaming errors.
2. Verify Bronze data files exist under `lake/bronze/stream/` (not just `_delta_log`).
3. Start and verify Silver, Gold, and Inference streaming jobs; confirm outputs land in their Delta paths.
4. Validate Fabric Eventhouse tables and dashboard tiles show live data from `features_gold` and `inference_scores`.
5. Mark Execution Plan step 3 as [COMPLETE] once Bronze → Silver → Gold → Inference → Fabric is validated end-to-end.
