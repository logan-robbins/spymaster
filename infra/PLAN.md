# Plan
1. [COMPLETE] Inventory infra and pipeline assets; confirm Bronze/Gold schema contracts.
2. [COMPLETE] Align infra Bicep modules/params to WORK.md (storage, Databricks, Event Hubs, Data Factory, AML, Key Vault, Fabric capacity) and remove non-required components.
3. [COMPLETE] Deploy infra with az CLI (what-if + deploy) and verify resources via Azure APIs.
4. [COMPLETE] Validate Fabric workspace access and finish remaining foundation items in WORK.md (Key Vault + data contract).
5. [COMPLETE] Build a minimal DBN sample (09:30–11:30 ET) from raw DBN, write to disk, upload to ADLS raw-dbn with a manifest row.
6. [IN PROGRESS] Databricks batch jobs and snapshot export complete; package backend for Databricks/AML next.
7. Implement Azure ML training job + sweep, register model, deploy managed endpoint, verify inference API.
8. Implement real-time ingestion service, Databricks streaming pipeline, and inference stream.
9. Configure Fabric RTI artifacts (Eventhouse, Eventstream, dashboard, anomaly detection) and validate data flow.
10. Mark infra/WORK.md tasks with [COMPLETE] as each item is finished.
