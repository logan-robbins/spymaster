# Spymaster

Enterprise-grade market microstructure analytics platform for real-time price level interaction analysis.

## Overview

Spymaster is a cloud-native data engineering platform that processes high-frequency market data to identify and analyze price interactions at technically significant levels. The system ingests market-by-order data, computes orderbook dynamics, and performs similarity-based retrieval to surface historical precedents for current market configurations.

**Key Capabilities:**
- Real-time orderbook reconstruction from MBO streams
- Multi-layer feature engineering (Bronze → Silver → Gold)
- Vector-based similarity search across historical market states
- ML-driven outcome prediction at technical price levels
- Sub-second latency for critical market hours

**Primary Use Case:**  
When price approaches a defined level (e.g., previous day high, opening range boundaries), the system retrieves historically similar setups and presents empirical outcome distributions to quantify break/reject/chop probabilities.

## Architecture

### Cloud Infrastructure (Azure)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Databento Live API  ──▶  Event Hubs (mbo_raw)                      │
│  DBN File Uploads    ──▶  ADLS Gen2 (lake/raw/)                     │
│                                                                       │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER (Databricks)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   BRONZE     │───▶│   SILVER     │───▶│    GOLD      │          │
│  │              │    │              │    │              │          │
│  │ Raw MBO      │    │ Orderbook    │    │ Feature      │          │
│  │ Events       │    │ 5s Bars      │    │ Vectors      │          │
│  │ (Delta)      │    │ Level Calcs  │    │ Episodes     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                    │                    │                 │
│         └────────────────────┴────────────────────┘                 │
│                              │                                       │
│                    Unity Catalog (Governance)                        │
│                                                                       │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Azure ML Workspace                                                  │
│  ├─ Model Training (Hyperopt sweeps)                                │
│  ├─ Model Registry (MLflow)                                         │
│  └─ Managed Endpoints ──▶ Real-time predictions                     │
│                                                                       │
│  Gold Events ──▶ Event Hubs (inference_results)                     │
│                         │                                            │
│                         └──▶ Microsoft Fabric (Dashboards)           │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────┐
                    │    SECRETS & GOVERNANCE  │
                    ├──────────────────────────┤
                    │  Key Vault (API keys)    │
                    │  Managed Identities      │
                    │  Log Analytics           │
                    └──────────────────────────┘
```

### Data Flow

**Historical Processing (Batch):**
```
DBN Files → Bronze (MBO events) → Silver (5s bars + orderbook features) 
  → Gold (episode vectors) → Training snapshots → Azure ML
```

**Real-time Processing (Streaming):**
```
Databento Live → Event Hubs → Bronze (streaming) → Silver (stateful) 
  → Gold (triggers) → AML Endpoint → Event Hubs → Fabric Dashboard
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Ingestion** | Event Hubs, ADLS Gen2 | Stream buffering, blob storage |
| **Processing** | Databricks (Structured Streaming) | Orderbook reconstruction, feature engineering |
| **Storage** | Delta Lake, Unity Catalog | ACID tables, schema evolution, lineage |
| **ML** | Azure ML, MLflow | Hyperparameter tuning, model versioning |
| **Inference** | Managed Online Endpoints | Real-time scoring (<100ms p99) |
| **Analytics** | Microsoft Fabric, KQL | Real-time dashboards, ad-hoc analysis |
| **Orchestration** | Data Factory | Batch pipeline scheduling |
| **IaC** | Bicep | Declarative infrastructure |

## Project Structure

```
spymaster/
├── backend/               # Local Python processing
│   ├── src/
│   │   ├── data_eng/     # Pipeline stages (Bronze/Silver/Gold)
│   │   ├── ml/           # Model training scripts
│   │   ├── gateway/      # WebSocket streaming service
│   │   └── ingestion/    # DBN file handlers
│   ├── lake/             # Local data lake mirror
│   ├── scripts/          # Data download, validation, replay
│   └── pyproject.toml    # Python dependencies (uv)
│
├── infra/                # Azure infrastructure
│   ├── main.bicep        # Root infrastructure template
│   ├── modules/          # Modular Bicep components
│   ├── databricks/       # Databricks notebooks (Git submodule)
│   │   ├── streaming/    # Real-time notebooks
│   │   ├── batch/        # Historical processing
│   │   └── jobs/         # Job definitions (JSON)
│   ├── aml/              # Azure ML configs
│   │   ├── endpoints/    # Model deployment specs
│   │   ├── jobs/         # Training job definitions
│   │   └── environments/ # Conda environments
│   └── scripts/          # Deployment automation
│
└── docs/                 # Additional documentation
```

## Data Engineering Pipeline

The system implements a medallion architecture with specialized stages:

### Bronze Layer
- Raw MBO events partitioned by contract/date
- Schema: `ts_event`, `price`, `size`, `action`, `side`, `order_id`
- Dead-letter queue for malformed records
- Retention: Indefinite (source of truth)

### Silver Layer
- 5-second orderbook snapshots with computed features:
  - **Ladder**: Bid/ask depth at 10 levels
  - **Flow**: Trade imbalance, aggressor classification
  - **Shape**: Book skew, weighted mid, liquidity concentration
  - **Wall**: Large order detection, cumulative size metrics
  - **Vacuum**: Distance to technical levels, approach velocity
- Session-level indicators (PM High/Low, OR High/Low, moving averages)
- RTH filtering (09:30-16:00 ET)

### Gold Layer
- Episode construction: Multi-bar windows centered on level approaches
- Feature vectors: 144-dimensional representations combining:
  - Geometry (distance, velocity, time-to-level)
  - Orderbook state (depth, imbalance, shape)
  - Options flow (call/put pressure, dealer hedging signals)
- Outcome labels: BREAK, REJECT, CHOP (first-crossing semantics)

### Retrieval System
- FAISS similarity search (60 partitions: 6 levels × 2 directions × 5 time buckets)
- Normalized embeddings (z-score by feature group)
- Query-time filtering (same level type, direction, time-of-day)
- Returns: Top-K historical episodes with outcome distributions

## Machine Learning

**Model Training:**
- Hyperparameter optimization via Hyperopt (Azure ML compute clusters)
- Walk-forward validation preserving temporal ordering
- Logistic regression baseline → Transformer models (future)
- MLflow tracking for all experiments

**Inference:**
- Managed Online Endpoints with autoscaling
- Input: 144-dim feature vector + metadata
- Output: Class probabilities {BREAK, REJECT, CHOP}
- Latency target: <50ms p95

## Local Development

### Prerequisites
- Python 3.11+ with `uv` package manager
- Azure CLI (authenticated)
- Databento API key (stored in `backend/.env`)

### Setup
```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/qmachina/spymaster.git
cd spymaster/backend

# Install dependencies
uv sync

# Download historical data
uv run python scripts/download_es_futures_fast.py --start 2024-11-01 --end 2024-12-31

# Run pipeline locally
uv run python -m data_eng bronze silver gold --date 2024-12-01
```

### Testing Replay
```bash
# Simulate live market with historical data
bash scripts/run_replay.sh
```

## Infrastructure Deployment

### Deploy to Azure
```bash
cd infra

# Deploy all resources
bash scripts/deploy_infrastructure.sh

# Verify deployment
az deployment group show --resource-group rg-spymaster-dev --name main
```

### Configure Databricks
```bash
# Link Git repo for notebooks
databricks repos create \
  --url https://github.com/qmachina/spymaster-databricks \
  --provider github

# Add secrets to Key Vault
az keyvault secret set \
  --vault-name <vault-name> \
  --name databento-api-key \
  --value <your-key>
```

## Monitoring

**Key Metrics:**
- Event Hubs: Incoming messages/sec, consumer lag
- Databricks: Stream processing rate, checkpoint latency
- AML Endpoint: Request rate, latency (p50/p95/p99), errors
- Storage: Data volume by layer, query performance

**Dashboards:**
- Microsoft Fabric: Real-time inference results, outcome distributions
- Azure Monitor: Infrastructure health, cost tracking
- MLflow: Training metrics, model lineage

## Security

- **Secrets**: All API keys stored in Azure Key Vault
- **Access**: Managed identities for service-to-service auth
- **Network**: Private endpoints for storage and Event Hubs (optional)
- **Data**: No PII; market data only
- **Compliance**: .env files gitignored, no credentials in code

## License

Proprietary - QMachina LLC

## Contact

For inquiries: logan@qmachina.com
