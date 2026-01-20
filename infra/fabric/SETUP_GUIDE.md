# Fabric Real-Time Intelligence Setup Guide

This guide walks through setting up Fabric RTI components for the Spymaster trading dashboard.

## Prerequisites

- Microsoft Fabric workspace with RTI enabled (Fabric capacity: F8 or higher)
- Event Hubs namespace: `ehnspymasterdevoxxrlojskvxey`
- Event Hubs: `features_gold`, `inference_scores`

## Step 1: Create Eventhouse

1. Navigate to your Fabric workspace (`qfabric`)
2. Click **New** → **Eventhouse**
3. Name: `trading_eventhouse`
4. Wait for provisioning to complete

### Create Tables

In the Eventhouse KQL editor, run the commands from `eventhouse_schema.kql`:

```kql
// Create features table
.create table features (
    contract_id: string,
    vector_time: long,
    model_id: string,
    close_price: real,
    volume: long,
    return_1: real,
    ma_5: real,
    vol_ratio: real,
    spread: real,
    momentum_5: real,
    bid_ask_imbalance: real,
    ingestion_time: datetime
)

// Create scores table
.create table scores (
    contract_id: string,
    vector_time: long,
    model_id: string,
    model_version: string,
    close_price: real,
    prediction: int,
    prob_0: real,
    prob_1: real,
    inference_time: datetime,
    ingestion_time: datetime
)
```

## Step 2: Create Eventstream for Features

1. Click **New** → **Eventstream**
2. Name: `features_stream`
3. Add Source:
   - Type: **Azure Event Hubs**
   - Connection: Select or create connection to `ehnspymasterdevoxxrlojskvxey`
   - Event Hub: `features_gold`
   - Consumer Group: `fabric_stream`
   - Data format: JSON
4. Add Destination:
   - Type: **Eventhouse**
   - Database: `trading_eventhouse`
   - Table: `features`
   - Input data format: JSON
   - Mapping: Use `features_mapping` (created above)
5. Click **Publish**

## Step 3: Create Eventstream for Scores

1. Click **New** → **Eventstream**
2. Name: `scores_stream`
3. Add Source:
   - Type: **Azure Event Hubs**
   - Connection: Same as above
   - Event Hub: `inference_scores`
   - Consumer Group: `fabric_stream`
   - Data format: JSON
4. Add Destination:
   - Type: **Eventhouse**
   - Database: `trading_eventhouse`
   - Table: `scores`
   - Input data format: JSON
   - Mapping: Use `scores_mapping`
5. Click **Publish**

## Step 4: Create Real-Time Dashboard

1. Click **New** → **Real-Time Dashboard**
2. Name: `Spymaster Trading Intelligence`
3. Add tiles using queries from `dashboard_queries.kql`

### Recommended Dashboard Layout

| Tile | Query | Visualization |
|------|-------|---------------|
| Price Time Series | TILE 1 | Line Chart |
| Prediction Distribution | TILE 2 | Pie Chart |
| Confidence Over Time | TILE 3 | Area Chart |
| Feature Anomalies | TILE 4 | Table with conditional formatting |
| Volume Ratio Heatmap | TILE 5 | Heatmap |
| Spread Analysis | TILE 6 | Box Plot |
| Latest Predictions | TILE 7 | Table |
| Momentum Gauge | TILE 8 | KPI Card |
| Top Movers | TILE 9 | Table |
| Event Count | TILE 10 | KPI Card |

### Dashboard Settings

- Auto-refresh: **30 seconds**
- Time range: **Last 1 hour** (adjustable)
- Theme: Dark (for trading)

## Step 5: Add Power BI Anomaly Detection

1. Open Power BI Desktop or Fabric Power BI
2. Connect to Eventhouse: `trading_eventhouse`
3. Create visuals for:
   - `features` table time series
   - `scores` table predictions
4. For each line chart:
   - Click **Analytics** pane
   - Enable **Find anomalies**
   - Configure sensitivity (recommended: 90%)
5. Publish to Fabric workspace

## Event Hub Connection Details

- Namespace: `ehnspymasterdevoxxrlojskvxey.servicebus.windows.net`
- Features Hub: `features_gold`
- Scores Hub: `inference_scores`
- Consumer Group: `fabric_stream`

## Troubleshooting

### No data appearing in Eventhouse

1. Verify Eventstream is running (green status)
2. Check Event Hub has messages (Azure Portal → Event Hubs → Metrics)
3. Verify table mapping matches JSON schema
4. Check consumer group is correct

### Slow dashboard refresh

1. Add time filters to queries (e.g., `where ingestion_time > ago(1h)`)
2. Use materialized views for aggregations
3. Consider reducing auto-refresh interval

### Anomaly detection not working

1. Ensure at least 12 data points in the series
2. Verify time column is properly formatted
3. Check for NULL values in the series data
