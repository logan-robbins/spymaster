# Infrastructure - AI Agent Reference

**PURPOSE**: This document is optimized for AI coding agents. All commands are executable as-is.

**CONTEXT**: 
- Python environment: `../backend/.venv/bin/python3`
- Azure CLI user: `logan@qmachina.com`
- Subscription: `70464868-52ea-435d-93a6-8002e83f0b89`
- Tenant: `2843abed-8970-461e-a260-a59dc1398dbf`
- Resource Group: `rg-spymaster-dev`
- Location: `westus`

---

## Directory Structure

```
infra/
├── main.bicep                      # Main infrastructure deployment
├── modules/                        # Bicep modules for each service
│   ├── storage.bicep              # ADLS Gen2 for data lake
│   ├── databricks.bicep           # Databricks workspace + Unity Catalog
│   ├── eventhubs.bicep            # Event Hubs for streaming
│   ├── aml.bicep                  # Azure ML workspace + endpoints
│   ├── keyvault.bicep             # Key Vault for secrets
│   ├── datafactory.bicep          # Data Factory for orchestration
│   ├── fabric-capacity.bicep      # Microsoft Fabric capacity
│   ├── loganalytics.bicep         # Log Analytics workspace
│   └── ...
├── databricks/                     # Git submodule → spymaster-databricks repo
│   ├── streaming/                 # Real-time streaming notebooks
│   │   ├── rt__mbo_raw_to_bronze.py
│   │   ├── rt__bronze_to_silver.py
│   │   ├── rt__silver_to_gold.py
│   │   └── rt__gold_to_inference.py
│   ├── batch/                      # Historical batch notebooks
│   │   ├── hist_dbn_to_bronze.py
│   │   ├── hist_bronze_to_silver.py
│   │   ├── hist_silver_to_gold.py
│   │   └── hist_export_training_snapshot.py
│   └── jobs/                       # Job definitions (JSON)
│       ├── streaming/              # Streaming job configs
│       │   └── rt__*.json
│       └── batch/                  # Batch job configs
│           └── hist__*.json
├── aml/
│   ├── deploy_endpoint.py         # Script to deploy ML endpoint
│   ├── endpoints/es_model/        # Endpoint configuration
│   ├── models/                    # Model definitions
│   ├── jobs/                      # AML job definitions
│   └── environments/              # Conda environments
├── containers/
│   └── databento_ingestor/        # Container for streaming ingestion
├── fabric/
│   ├── eventhouse_schema.kql      # KQL schema for Eventhouse
│   ├── dashboard_queries.kql      # Dashboard KQL queries
│   └── SETUP_GUIDE.md             # Fabric setup instructions
├── azure-resources.json           # Complete resource inventory
└── README.md                      # This file

```

**IMPORTANT**: `infra/databricks/` is a **Git submodule** pointing to:
- https://github.com/qmachina/spymaster-databricks

To initialize after cloning:
```bash
git submodule update --init --recursive
```

---

## System Architecture

### Data Flow
```
Raw DBN Files → Event Hubs → Bronze (Delta) → Silver (Delta) → Gold (Delta) → Inference
                                ↓              ↓               ↓
                            Unity Catalog  Unity Catalog  Unity Catalog
                                                               ↓
                                                          Event Hubs → Fabric
```

### Layers
1. **Bronze**: Raw MBO events from Event Hubs, with DLQ for malformed records
2. **Silver**: Stateful orderbook reconstruction with 5-second bars
3. **Gold**: Feature vectors with multi-window lookbacks
4. **Inference**: Real-time ML predictions via AML endpoint

### Services
- **Azure Databricks**: All data processing (streaming + batch)
- **Azure Machine Learning**: Model training, registry, inference endpoints
- **Event Hubs**: Real-time event streaming
- **ADLS Gen2**: Data lake storage (medallion architecture)
- **Unity Catalog**: Data governance and lineage
- **Microsoft Fabric**: Real-time dashboards and analytics
- **Key Vault**: Secret management
- **Data Factory**: Batch orchestration

---

## Quick Reference: Resource Details

**Storage Account (Lake)**: `spymasterdevlakeoxxrlojs`
- Containers: `lake`, `raw-dbn`, `ml-artifacts`

**Databricks Workspace**: `adbspymasterdevoxxrlojskvxey`
- URL: `https://adb-7405608014630062.2.azuredatabricks.net`
- Unity Catalog enabled

**Event Hubs Namespace**: `ehnspymasterdevoxxrlojskvxey`
- Hubs: `mbo_raw`, `features_gold`, `inference_scores`

**AML Workspace**: `mlwspymasterdevpoc`
- Endpoint: `es-model-endpoint`
- Scoring URI: `https://es-model-endpoint.westus.inference.ml.azure.com/score`

**Key Vaults**:
- Runtime: `kvspymasterdevrtoxxrlojs` (for runtime secrets)
- AML: `kvspymasterdevoxxrlojskv` (for AML workspace)

**Fabric Capacity**: `qfabric` (F8)

Full details in `azure-resources.json`.

---

## Deployment Commands

### Prerequisites Check
```bash
# Verify az cli login
az account show --query "{subscription:name, user:user.name}" -o table

# Verify Python environment
../backend/.venv/bin/python3 --version

# Install required tools
az extension add --name databricks --upgrade
az extension add --name ml --upgrade
```

### 1. Deploy Azure Infrastructure (Bicep)

```bash
cd infra

# Deploy all resources
az deployment group create \
  --resource-group rg-spymaster-dev \
  --template-file main.bicep \
  --parameters @params/dev.bicepparam \
  --parameters deployFabric=false

# Verify deployment
az deployment group show \
  --resource-group rg-spymaster-dev \
  --name main \
  --query properties.provisioningState -o tsv
```

### 2. Deploy Databricks Notebooks via Git (Recommended)

**IMPORTANT**: The `databricks/` directory is a Git submodule pointing to:
- Repository: https://github.com/qmachina/spymaster-databricks
- Local path: `infra/databricks/` (submodule)

**Setup Git integration in Databricks:**
```bash
# Authenticate with Databricks
databricks configure --token --host https://adb-7405608014630062.2.azuredatabricks.net

# Create Repos integration (one-time setup)
databricks repos create \
  --url https://github.com/qmachina/spymaster-databricks \
  --provider github \
  --path /Repos/logan@qmachina.com/spymaster-databricks

# Verify
databricks repos list --output json | jq '.repos[] | {path:.path, url:.url, branch:.branch}'
```

**Notebooks will be available at:**
```
/Repos/logan@qmachina.com/spymaster-databricks/
├── streaming/
│   ├── rt__mbo_raw_to_bronze
│   ├── rt__bronze_to_silver
│   ├── rt__silver_to_gold
│   └── rt__gold_to_inference
└── batch/
    ├── hist_dbn_to_bronze
    ├── hist_bronze_to_silver
    ├── hist_silver_to_gold
    └── hist_export_training_snapshot
```

**Updates**: When code changes in the spymaster-databricks repo:
```bash
# Update to latest
databricks repos update \
  --repo-id <id> \
  --branch main

# Or enable auto-sync in Databricks UI
```

### 3. Create Databricks Secret Scope

```bash
# Create secret scope backed by Key Vault
databricks secrets create-scope \
  --scope spymaster-runtime \
  --scope-backend-type AZURE_KEYVAULT \
  --resource-id /subscriptions/70464868-52ea-435d-93a6-8002e83f0b89/resourceGroups/rg-spymaster-dev/providers/Microsoft.KeyVault/vaults/kvspymasterdevrtoxxrlojs \
  --dns-name https://kvspymasterdevrtoxxrlojs.vault.azure.net/

# Verify secrets are accessible
databricks secrets list-secrets --scope spymaster-runtime
```

### 4. Create Unity Catalog Tables

```bash
# Create catalog and schemas (if not exist)
databricks sql execute \
  --query "CREATE CATALOG IF NOT EXISTS spymaster"

databricks sql execute \
  --query "CREATE SCHEMA IF NOT EXISTS spymaster.bronze"

databricks sql execute \
  --query "CREATE SCHEMA IF NOT EXISTS spymaster.silver"

databricks sql execute \
  --query "CREATE SCHEMA IF NOT EXISTS spymaster.gold"

# Tables will be created automatically by streaming jobs
# But you can pre-create them:
databricks sql execute \
  --query "CREATE TABLE IF NOT EXISTS spymaster.bronze.mbo_stream (
    action STRING,
    price DOUBLE,
    size BIGINT,
    side STRING,
    order_id BIGINT,
    contract_id STRING,
    event_time BIGINT,
    publisher_id BIGINT,
    session_date STRING,
    underlier STRING,
    instrument_type STRING,
    ingestion_timestamp TIMESTAMP,
    event_time_ts TIMESTAMP
  ) USING DELTA"
```

### 5. Create Databricks Jobs

**IMPORTANT**: Job definitions reference Git paths, not manual uploads.

**Verify job definitions have correct paths:**
```bash
# Should reference /Repos/logan@qmachina.com/spymaster-databricks/...
cat databricks/jobs/streaming/rt__mbo_raw_to_bronze.json | jq '.tasks[].notebook_task.notebook_path'
```

**Create streaming jobs:**
```bash
databricks jobs create --json-file databricks/jobs/streaming/rt__mbo_raw_to_bronze.json
databricks jobs create --json-file databricks/jobs/streaming/rt__bronze_to_silver.json
databricks jobs create --json-file databricks/jobs/streaming/rt__silver_to_gold.json
databricks jobs create --json-file databricks/jobs/streaming/rt__gold_to_inference.json
```

**Create historical batch jobs:**
```bash
databricks jobs create --json-file databricks/jobs/batch/hist__dbn_to_bronze.json
databricks jobs create --json-file databricks/jobs/batch/hist__bronze_to_silver.json
databricks jobs create --json-file databricks/jobs/batch/hist__silver_to_gold.json
databricks jobs create --json-file databricks/jobs/batch/hist__export_training_snapshot.json
```

**List all jobs:**
```bash
databricks jobs list --output json | jq '.jobs[] | {name:.settings.name, id:.job_id}'
```

### 6. Deploy Azure ML Endpoint

```bash
cd aml

# Deploy using Python script
../backend/.venv/bin/python3 deploy_endpoint.py

# Or manually via az ml
az ml online-endpoint create \
  --resource-group rg-spymaster-dev \
  --workspace-name mlwspymasterdevpoc \
  --file endpoints/es_model/endpoint.yaml

az ml online-deployment create \
  --resource-group rg-spymaster-dev \
  --workspace-name mlwspymasterdevpoc \
  --file endpoints/es_model/deployment.yaml \
  --all-traffic

# Get endpoint key
az ml online-endpoint get-credentials \
  --resource-group rg-spymaster-dev \
  --workspace-name mlwspymasterdevpoc \
  --name es-model-endpoint \
  --query primaryKey -o tsv

# Store in Key Vault
az keyvault secret set \
  --vault-name kvspymasterdevrtoxxrlojs \
  --name aml-endpoint-key \
  --value "<key-from-above>"
```

### 7. Setup Fabric (Manual - API not available yet)

Follow `fabric/SETUP_GUIDE.md` for:
1. Create Eventhouse
2. Create Eventstream connections
3. Setup Real-Time Dashboard
4. Configure KQL queries from `fabric/dashboard_queries.kql`

---

## Testing Commands

### Test Databricks Notebooks Locally
```bash
cd ../backend

# Run unit tests
uv run pytest tests/streaming/ -v

# Run specific test
uv run pytest tests/streaming/test_bronze_transformations.py::test_mbo_schema_validation -v

# Run with coverage
uv run pytest tests/streaming/ --cov=../infra/databricks/streaming --cov-report=html
```

### Test Streaming Pipeline End-to-End

**1. Send test event to Event Hub:**
```bash
../backend/.venv/bin/python3 << 'EOF'
from azure.eventhub import EventHubProducerClient, EventData
import json

connection_str = "<from-keyvault>"
producer = EventHubProducerClient.from_connection_string(
    connection_str, 
    eventhub_name="mbo_raw"
)

test_event = {
    "action": "A",
    "price": 6050.25,
    "size": 100,
    "side": "B",
    "order_id": 12345,
    "contract_id": "ESZ5",
    "event_time": 1700000000000000000,
    "publisher_id": 1,
    "session_date": "2024-01-19",
    "underlier": "ES",
    "instrument_type": "FUT"
}

event_data_batch = producer.create_batch()
event_data_batch.add(EventData(json.dumps(test_event)))
producer.send_batch(event_data_batch)
producer.close()
print("Test event sent")
EOF
```

**2. Query Unity Catalog tables:**
```bash
databricks sql execute \
  --query "SELECT * FROM spymaster.bronze.mbo_stream LIMIT 10"

databricks sql execute \
  --query "SELECT * FROM spymaster.silver.orderbook_5s ORDER BY bar_ts DESC LIMIT 10"

databricks sql execute \
  --query "SELECT contract_id, COUNT(*) as row_count FROM spymaster.gold.feature_vectors GROUP BY contract_id"
```

**3. Check DLQ for errors:**
```bash
databricks sql execute \
  --query "SELECT * FROM spymaster.bronze.mbo_stream_dlq LIMIT 10"
```

**4. Monitor streaming job:**
```bash
# Get job run status
databricks jobs run-now --job-id <job-id>

# Get latest run
databricks runs list --job-id <job-id> --limit 1 --output json | jq '.'

# View logs
databricks runs get-output --run-id <run-id>
```

### Test ML Endpoint
```bash
../backend/.venv/bin/python3 << 'EOF'
import requests
import json

endpoint_uri = "https://es-model-endpoint.westus.inference.ml.azure.com/score"
api_key = "<from-keyvault>"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "input_data": {
        "data": [[0.01, 0.02, -0.01, 1.5, 0.8]]  # sample feature vector
    }
}

response = requests.post(endpoint_uri, headers=headers, json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
EOF
```

---

## Monitoring & Observability

### Check Databricks Cluster Status
```bash
databricks clusters list --output json | jq '.clusters[] | {name:.cluster_name, state:.state}'
```

### View Streaming Query Progress
```bash
# Via Databricks SQL
databricks sql execute \
  --query "SELECT * FROM system.streaming.query_progress WHERE query_name LIKE '%mbo%' ORDER BY timestamp DESC LIMIT 10"
```

### Check Storage Usage
```bash
az storage account show-usage \
  --account-name spymasterdevlakeoxxrlojs \
  --resource-group rg-spymaster-dev
```

### View Key Vault Secrets (metadata only)
```bash
az keyvault secret list \
  --vault-name kvspymasterdevrtoxxrlojs \
  --query "[].{name:name, enabled:attributes.enabled}" -o table
```

### Check Event Hub Metrics
```bash
az eventhubs namespace show \
  --resource-group rg-spymaster-dev \
  --name ehnspymasterdevoxxrlojskvxey \
  --query "{name:name, sku:sku.name, capacity:sku.capacity}" -o table
```

---

## Troubleshooting

### Streaming Job Not Starting
```bash
# Check cluster status
databricks clusters get --cluster-id <cluster-id>

# Check job configuration
databricks jobs get --job-id <job-id>

# View run logs
databricks runs get --run-id <run-id> --include-history
```

### State Store Issues
```bash
# Clear checkpoint to restart from beginning (CAUTION: loses state)
az storage fs file delete \
  --file-system lake \
  --path "checkpoints/rt__bronze_main" \
  --account-name spymasterdevlakeoxxrlojs \
  --recursive \
  --yes

# Check checkpoint metadata
az storage fs file list \
  --file-system lake \
  --path "checkpoints/" \
  --account-name spymasterdevlakeoxxrlojs
```

### Unity Catalog Permission Issues
```bash
# Grant permissions
databricks sql execute \
  --query "GRANT ALL PRIVILEGES ON CATALOG spymaster TO \`logan@qmachina.com\`"

databricks sql execute \
  --query "GRANT ALL PRIVILEGES ON SCHEMA spymaster.bronze TO \`logan@qmachina.com\`"
```

### Event Hub Connection Issues
```bash
# Test connection string
../backend/.venv/bin/python3 << 'EOF'
from azure.eventhub import EventHubConsumerClient

connection_str = "<from-keyvault>"
consumer = EventHubConsumerClient.from_connection_string(
    connection_str,
    consumer_group="$Default",
    eventhub_name="mbo_raw"
)
print("Connection successful")
consumer.close()
EOF
```

---

## Cleanup Commands

### Delete Specific Resources
```bash
# Delete a Databricks job
databricks jobs delete --job-id <job-id>

# Delete Unity Catalog table
databricks sql execute --query "DROP TABLE IF EXISTS spymaster.bronze.mbo_stream"

# Delete ML endpoint
az ml online-endpoint delete \
  --resource-group rg-spymaster-dev \
  --workspace-name mlwspymasterdevpoc \
  --name es-model-endpoint \
  --yes
```

### Delete All Infrastructure (DESTRUCTIVE)
```bash
# WARNING: This deletes everything
az group delete \
  --name rg-spymaster-dev \
  --yes \
  --no-wait
```

---

## Data Pipeline Execution Order

### One-Time Setup
1. Deploy infrastructure (Bicep)
2. Create secret scope
3. Upload notebooks
4. Create Unity Catalog schemas
5. Deploy ML endpoint
6. Create Databricks jobs

### Historical Backfill
```bash
# Run in order:
1. hist__dbn_to_bronze      # Parse raw DBN files
2. hist__bronze_to_silver   # Build 5s bars
3. hist__silver_to_gold     # Generate features
4. hist__export_training_snapshot  # Create training dataset
```

### Real-Time Pipeline
```bash
# Start streaming jobs (run continuously):
1. rt__mbo_raw_to_bronze    # Event Hubs → Bronze
2. rt__bronze_to_silver     # Stateful orderbook
3. rt__silver_to_gold       # Feature engineering
4. rt__gold_to_inference    # ML predictions
```

---

## Environment Variables

Set these for Python scripts:

```bash
export AZURE_SUBSCRIPTION_ID=70464868-52ea-435d-93a6-8002e83f0b89
export AZURE_TENANT_ID=2843abed-8970-461e-a260-a59dc1398dbf
export RESOURCE_GROUP=rg-spymaster-dev
export DATABRICKS_HOST=https://adb-7405608014630062.2.azuredatabricks.net
export AML_WORKSPACE=mlwspymasterdevpoc
export STORAGE_ACCOUNT=spymasterdevlakeoxxrlojs
export EVENTHUB_NAMESPACE=ehnspymasterdevoxxrlojskvxey
export KEYVAULT_NAME=kvspymasterdevrtoxxrlojs
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `azure-resources.json` | Complete inventory of deployed resources |
| `main.bicep` | Infrastructure as Code (IaC) template |
| `databricks/streaming/rt__*.py` | Real-time streaming notebooks |
| `databricks/hist/hist_*.py` | Historical batch processing notebooks |
| `databricks/jobs/*.json` | Job definitions with cluster configs |
| `aml/deploy_endpoint.py` | Script to deploy ML inference endpoint |
| `fabric/eventhouse_schema.kql` | KQL schema for Fabric Eventhouse |

---

## Best Practices for AI Agents

1. **Always check current state before deploying**:
   ```bash
   az resource list --resource-group rg-spymaster-dev --query "[].{name:name, type:type}" -o table
   ```

2. **Use idempotent commands** (--overwrite, IF NOT EXISTS, etc.)

3. **Verify deployments** after each step:
   ```bash
   az deployment group show --resource-group rg-spymaster-dev --name <deployment-name>
   ```

4. **Check logs for errors**:
   ```bash
   databricks runs get-output --run-id <run-id> | jq '.error'
   ```

5. **Use JSON output for parsing**:
   ```bash
   databricks jobs list --output json | jq '.'
   ```

6. **Always reference `../backend/.venv/bin/python3`** for Python execution

7. **Use `azure-resources.json`** as single source of truth for resource names/IDs

---

**LAST UPDATED**: 2026-01-20
**MAINTAINED BY**: AI Agents (auto-generated)
