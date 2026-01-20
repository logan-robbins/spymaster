#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$INFRA_DIR")"

echo "=== Spymaster Post-Deployment Setup ==="
echo "Working directory: $INFRA_DIR"
echo ""

cd "$INFRA_DIR"

PYTHON="$REPO_ROOT/backend/.venv/bin/python3"

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Run: cd backend && uv sync"
    exit 1
fi

if ! command -v az &> /dev/null; then
    echo "ERROR: az CLI not found"
    exit 1
fi

if ! command -v databricks &> /dev/null; then
    echo "ERROR: databricks CLI not found"
    echo "Install: pip install databricks-cli"
    exit 1
fi

echo "[1/8] Generating resource inventory..."
$PYTHON scripts/generate_azure_resources_inventory.py > azure-resources.json
echo "Generated azure-resources.json"

DATABRICKS_HOST="https://$(jq -r '.quick_reference.databricks_workspace_url' azure-resources.json)"
KEYVAULT_NAME=$(jq -r '.quick_reference.keyvault_runtime' azure-resources.json)
EVENTHUB_NS=$(jq -r '.quick_reference.eventhubs_namespace' azure-resources.json)
STORAGE_ACCOUNT=$(jq -r '.quick_reference.storage_account_lake' azure-resources.json)
RG_NAME=$(jq -r '._metadata.primary_resource_group' azure-resources.json)

echo ""
echo "Resources:"
echo "  Databricks: $DATABRICKS_HOST"
echo "  Key Vault: $KEYVAULT_NAME"
echo "  Event Hub NS: $EVENTHUB_NS"
echo "  Storage: $STORAGE_ACCOUNT"
echo ""

echo "[2/8] Storing Event Hub connection string in Key Vault..."
EVENTHUB_CONN=$(az eventhubs namespace authorization-rule keys list \
    --resource-group "$RG_NAME" \
    --namespace-name "$EVENTHUB_NS" \
    --name RootManageSharedAccessKey \
    --query primaryConnectionString -o tsv 2>/dev/null || echo "")

if [[ -n "$EVENTHUB_CONN" ]]; then
    az keyvault secret set \
        --vault-name "$KEYVAULT_NAME" \
        --name eventhub-connection-string \
        --value "$EVENTHUB_CONN" \
        --output none
    echo "Stored eventhub-connection-string"
else
    echo "WARN: Could not get Event Hub connection string"
fi

echo ""
echo "[3/8] Configuring Databricks CLI..."
export DATABRICKS_HOST
if ! databricks auth describe &>/dev/null; then
    echo "Databricks auth not configured. Run:"
    echo "  databricks configure --token --host $DATABRICKS_HOST"
    echo ""
    echo "Skipping Databricks steps..."
    SKIP_DATABRICKS=true
else
    SKIP_DATABRICKS=false
    echo "Databricks CLI configured"
fi

if [[ "$SKIP_DATABRICKS" != "true" ]]; then
    echo ""
    echo "[4/8] Creating Databricks secret scope..."
    KV_ID=$(jq -r '.keyvaults.runtime.id' azure-resources.json)
    KV_URI="https://${KEYVAULT_NAME}.vault.azure.net/"

    if databricks secrets list-scopes 2>/dev/null | grep -q "spymaster"; then
        echo "Secret scope 'spymaster' already exists"
    else
        databricks secrets create-scope \
            --scope spymaster \
            --scope-backend-type AZURE_KEYVAULT \
            --resource-id "$KV_ID" \
            --dns-name "$KV_URI" 2>/dev/null || echo "Scope may already exist"
        echo "Created secret scope 'spymaster'"
    fi

    echo ""
    echo "[5/8] Creating Databricks Git Repos integration..."
    if databricks repos list 2>/dev/null | grep -q "spymaster-databricks"; then
        echo "Repo integration already exists"
    else
        databricks repos create \
            --url https://github.com/qmachina/spymaster-databricks \
            --provider github \
            --path "/Repos/logan@qmachina.com/spymaster-databricks" 2>/dev/null || echo "Repo may already exist"
        echo "Created Git Repos integration"
    fi

    echo ""
    echo "[6/8] Creating Unity Catalog structure..."
    for catalog in bronze silver gold; do
        databricks unity-catalog catalogs create "$catalog" 2>/dev/null || echo "Catalog $catalog may exist"
    done

    for catalog in bronze silver gold; do
        databricks unity-catalog schemas create default "$catalog" 2>/dev/null || echo "Schema $catalog.default may exist"
    done

    for catalog in bronze silver gold; do
        databricks unity-catalog schemas create future_mbo "$catalog" 2>/dev/null || echo "Schema $catalog.future_mbo may exist"
    done
    echo "Unity Catalog structure created"

    echo ""
    echo "[7/8] Creating Databricks jobs..."
    cd "$INFRA_DIR/databricks"

    for job_file in jobs/streaming/*.json; do
        if [[ -f "$job_file" ]]; then
            job_name=$(jq -r '.name // .settings.name' "$job_file" 2>/dev/null || echo "unknown")
            if databricks jobs list 2>/dev/null | grep -q "$job_name"; then
                echo "Job '$job_name' already exists"
            else
                databricks jobs create --json-file "$job_file" 2>/dev/null || echo "Could not create job: $job_file"
                echo "Created job: $job_name"
            fi
        fi
    done

    for job_file in jobs/batch/*.json; do
        if [[ -f "$job_file" ]]; then
            job_name=$(jq -r '.name // .settings.name' "$job_file" 2>/dev/null || echo "unknown")
            if databricks jobs list 2>/dev/null | grep -q "$job_name"; then
                echo "Job '$job_name' already exists"
            else
                databricks jobs create --json-file "$job_file" 2>/dev/null || echo "Could not create job: $job_file"
                echo "Created job: $job_name"
            fi
        fi
    done

    cd "$INFRA_DIR"
else
    echo "[4-7/8] Skipping Databricks steps (auth not configured)"
fi

echo ""
echo "[8/8] Summary..."
echo ""
echo "=== Post-Deployment Setup Complete ==="
echo ""
echo "Manual steps remaining:"
echo "  1. Add databento-api-key to Key Vault:"
echo "     az keyvault secret set --vault-name $KEYVAULT_NAME --name databento-api-key --value '<your-key>'"
echo ""
echo "  2. Deploy AML endpoint (after training model):"
echo "     cd aml && $PYTHON deploy_endpoint.py"
echo ""
echo "  3. Store AML endpoint key in Key Vault (after endpoint deployed):"
echo "     KEY=\$(az ml online-endpoint get-credentials --resource-group $RG_NAME --workspace-name \$(jq -r '.quick_reference.aml_workspace' azure-resources.json) --name es-model-endpoint --query primaryKey -o tsv)"
echo "     az keyvault secret set --vault-name $KEYVAULT_NAME --name aml-endpoint-key --value \"\$KEY\""
echo ""
echo "  4. Configure Microsoft Fabric (manual - see fabric/SETUP_GUIDE.md)"
echo ""
if [[ "$SKIP_DATABRICKS" == "true" ]]; then
    echo "  5. Configure Databricks CLI and re-run this script:"
    echo "     databricks configure --token --host $DATABRICKS_HOST"
    echo "     bash scripts/post_deployment_setup.sh"
    echo ""
fi
