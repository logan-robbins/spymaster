Below is **one** Azure-native blueprint that moves your working local Python pipeline into an enterprise-grade system with (a) **medallion lakehouse**, (b) **repeatable ML experimentation + MLflow + managed inference endpoint**, and (c) a **built-in real-time dashboard/UI with anomaly detection** (no custom UI required).

I’m locking the stack to:

* **ADLS Gen2** for immutable storage + medallion tables
* **Azure Databricks** for Bronze/Silver/Gold processing (batch + streaming)
* **Azure Data Factory** for orchestration of historical jobs
* **Azure Machine Learning** for experiments (Hyperopt), MLflow tracking/registry, and managed online inference endpoints ([Microsoft Learn][1])
* **Azure Event Hubs** as the real-time event log/bus
* **Microsoft Fabric Real-Time Intelligence (Eventstream + Eventhouse + Real-Time Dashboards/Power BI)** for built-in real-time visualization and anomaly detection ([Microsoft Learn][2])

---

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

## Step 7 — Run experiments w/Hyperopt using **Azure Machine Learning + MLflow**

In Azure ML:

* Create a **training job** that uses your model training entrypoint.
* Use **Hyperopt** inside an Azure ML hyperparameter sweep job (implementt the sweep wrapper).
* Configure MLflow tracking to the Azure ML workspace (Azure ML workspaces are MLflow-compatible). ([Microsoft Learn][8])

Result:

* Every run logs params/metrics/artifacts to MLflow
* The best run produces a logged MLflow model artifact

## Step 8 — View results and promote model to an Inference Endpoint using **Azure ML Studio + Managed Online Endpoint**

* Review runs in **Azure ML Studio** (experiment list + MLflow run details). ([Microsoft Learn][1])
* Register the winning model to the Azure ML registry (MLflow model registry supported). ([Microsoft Learn][1])
* Deploy as a **Managed Online Endpoint** using MLflow “no-code deployment” for real-time inference. ([Microsoft Learn][9])

At this point you have:

* `ES_MODEL` endpoint
* `TSLA_MODEL` endpoint

---

# 3) Phase 2 — Real-Time Analysis (Databento → features → inference batches → dashboard)

## Step 1 — Ingest via Databento API into **Azure Event Hubs**

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

## Step 2 — Bronze stream using **Databricks Structured Streaming**

Create a Databricks streaming job: `rt__mbo_raw_to_bronze`

* Source: Event Hubs `mbo_raw`
* Output: Delta append-only to `lake/bronze/stream/`
* Checkpoint: `lake/checkpoints/rt__bronze/`
* Guarantees:

  * schema enforcement
  * idempotent writes per your dedupe key
  * partitions aligned to `underlier`, `instrument_type`, `session_date`

## Step 3 — Silver stream (stateful orderbook + rollups) using **Databricks stateful streaming in Python**

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

## Step 4 — Gold stream (feature vectors) using **Databricks Structured Streaming**

Create a Databricks streaming job: `rt__silver_to_gold`

* Source: Silver outputs
* Computes:

  * your multi-window derivatives, OFI, “market physics” features, etc.
  * emits *both*:

    1. a compact **feature time-series** (for dashboards)
    2. the **model input vectors** (for inference batches)
* Outputs:

  * Delta `gold.setup_vectors_stream`
  * Delta `gold.feature_series_stream`
  * Publish to Event Hub `features_gold` (for Fabric ingestion)

## Step 5 — Inference every N seconds in batches using **Databricks → Azure ML Online Endpoint**

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

---

## Step 6 — Show features + inference on a built-in dashboard using **Fabric Real-Time Intelligence**

This gives you “TradingView-like monitoring” without building a UI.

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
  * “top movers” tables (largest change in OFI / pressure / etc.)
    Fabric dashboards are first-class RTI artifacts where tiles are backed by queries. ([Microsoft Learn][6])

### 6.4 Add anomaly detection (no custom ML UI)

Do both of these:

1. **Power BI line-chart anomaly detection** on feature and score series (built-in). ([Microsoft Learn][7])
2. KQL-based anomaly flags in Eventhouse using `series_decompose_anomalies()` for features like OFI spikes, liquidity vacuums, etc. (applies to Microsoft Fabric KQL). ([Microsoft Learn][11])

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

## Handoff summary

- Databricks batch jobs are created and executed: hist__dbn_to_bronze (job_id 34416110842711), hist__bronze_to_silver (job_id 616566000685834), hist__silver_to_gold (job_id 66228837418678); outputs landed in ADLS lake/bronze, lake/silver, and lake/gold with session_date=2025-10-01 partitions.
- External locations created in Unity Catalog for raw and ml containers: spymaster_raw (raw-dbn) and spymaster_ml (ml-artifacts) using storage credential spymaster_lake_cred.
- Training snapshot export job hist__export_training_snapshot (job_id 1031675175331261) succeeded; snapshot stored at ml-artifacts/datasets/ES_MODEL/2025-10-01/.
- Azure ML data asset registered: es_training_snapshot version 2025-10-01 pointing to ml_artifacts datastore.
- Azure ML split job succeeded (split_training_snapshot_retry); train/val/test files at ml-artifacts/datasets/ES_MODEL/2025-10-01/splits/{train,val,test}/part-00000.parquet.
- Azure ML environment spymaster-hyperopt version 2 created with numpy/pandas/pyarrow/scikit-learn/hyperopt/mlflow/azureml-mlflow; sweep job train_hyperopt_sweep_run is running as of last check.

## Next engineer actions

- Check status of Azure ML sweep job train_hyperopt_sweep_run; if failed, stream logs and fix, then rerun sweep until completed.
- Once sweep completes, identify best run in MLflow, register model, and proceed with managed online endpoint deployment.
- Continue remaining WORK.md steps: endpoint integration, real-time ingestion, Databricks streaming, Fabric RTI artifacts, and validation.
