#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Spymaster Environment Destruction ==="
echo ""
echo "WARNING: This will DELETE all Azure resources in rg-spymaster-dev"
echo "Data in the data lake will be PERMANENTLY LOST"
echo ""

read -p "Type 'DESTROY' to confirm: " CONFIRM
if [[ "$CONFIRM" != "DESTROY" ]]; then
    echo "Aborted."
    exit 1
fi

cd "$INFRA_DIR"

RG_NAME="rg-spymaster-dev"

echo ""
echo "[1/3] Deleting Databricks jobs (preserving notebooks in Git)..."
if command -v databricks &> /dev/null && databricks auth describe &>/dev/null 2>&1; then
    JOB_IDS=$(databricks jobs list --output json 2>/dev/null | jq -r '.[].job_id' || echo "")
    for job_id in $JOB_IDS; do
        echo "Deleting job: $job_id"
        databricks jobs delete --job-id "$job_id" 2>/dev/null || true
    done
    echo "Databricks jobs deleted"
else
    echo "Skipping Databricks cleanup (CLI not configured)"
fi

echo ""
echo "[2/3] Deleting Azure resource group..."
az group delete \
    --name "$RG_NAME" \
    --yes \
    --no-wait

echo "Resource group deletion initiated (async)"

echo ""
echo "[3/3] Waiting for deletion..."
echo "This can take 10-20 minutes. Checking every 30 seconds..."

while az group show --name "$RG_NAME" &>/dev/null; do
    echo "  Still deleting... $(date +%H:%M:%S)"
    sleep 30
done

echo ""
echo "=== Destruction Complete ==="
echo ""
echo "To redeploy:"
echo "  1. Create resource group:"
echo "     az group create --name $RG_NAME --location westus"
echo ""
echo "  2. Deploy infrastructure:"
echo "     USER_OID=\$(az ad signed-in-user show --query id -o tsv)"
echo "     az deployment group create \\"
echo "       --resource-group $RG_NAME \\"
echo "       --template-file main.bicep \\"
echo "       --parameters @params/dev.bicepparam \\"
echo "       --parameters deployingUserObjectId=\$USER_OID"
echo ""
echo "  3. Run post-deployment:"
echo "     bash scripts/post_deployment_setup.sh"
