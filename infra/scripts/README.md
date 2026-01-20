# Infrastructure Scripts

## generate_azure_resources_inventory.py

Auto-generates `azure-resources.json` by querying Azure CLI.

### Usage

```bash
../backend/.venv/bin/python3 generate_azure_resources_inventory.py > ../azure-resources.json
```

### What It Does

Queries Azure CLI for complete resource inventory:
- Resource groups
- Storage accounts (with containers)
- Event Hubs (with consumer groups)
- Databricks workspace
- Azure ML (workspaces, endpoints, models, datastores)
- Key Vaults (with secret metadata)
- Data Factory
- Container Registry
- Microsoft Fabric
- Monitoring (Log Analytics, App Insights)
- Purview
- All managed identities

### Output Structure

```json
{
  "_metadata": {...},
  "quick_reference": {...},
  "storage": {...},
  "eventhubs": {...},
  "databricks": {...},
  "machine_learning": {...},
  ...
}
```

### When to Regenerate

- Before any deployment
- After infrastructure changes
- When troubleshooting resource access
- To verify current state

### Requirements

- Azure CLI logged in (`az login`)
- Proper RBAC permissions on resources
- Python 3.12+

### Notes for AI Agents

- Always regenerate before deployments to ensure current state
- Use `azure-resources.json` as single source of truth for resource names/IDs
- Do not hardcode resource values in code or documentation
- Script handles missing permissions gracefully (returns empty dicts)
