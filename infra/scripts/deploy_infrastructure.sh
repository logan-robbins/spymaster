#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Spymaster Infrastructure Deployment ==="
echo ""

cd "$INFRA_DIR"

RG_NAME="rg-spymaster-dev"
LOCATION="westus"

if ! az account show &>/dev/null; then
    echo "ERROR: Not logged in to Azure CLI"
    echo "Run: az login"
    exit 1
fi

SUBSCRIPTION=$(az account show --query name -o tsv)
USER_NAME=$(az account show --query user.name -o tsv)
echo "Subscription: $SUBSCRIPTION"
echo "User: $USER_NAME"
echo ""

echo "[1/4] Checking resource group..."
if az group show --name "$RG_NAME" &>/dev/null; then
    echo "Resource group '$RG_NAME' exists"
else
    echo "Creating resource group '$RG_NAME' in '$LOCATION'..."
    az group create --name "$RG_NAME" --location "$LOCATION" --output none
    echo "Created resource group"
fi

echo ""
echo "[2/4] Getting deploying user object ID..."
USER_OID=$(az ad signed-in-user show --query id -o tsv 2>/dev/null || echo "")

if [[ -z "$USER_OID" ]]; then
    echo "WARN: Could not get user object ID. AML Key Vault access will need manual setup."
    DEPLOY_PARAMS=""
else
    echo "User Object ID: $USER_OID"
    DEPLOY_PARAMS="deployingUserObjectId=$USER_OID"
fi

echo ""
echo "[3/4] Deploying Bicep template..."
echo "This may take 15-30 minutes..."

DEPLOY_CMD="az deployment group create \
    --resource-group $RG_NAME \
    --template-file main.bicep \
    --parameters @params/dev.bicepparam"

if [[ -n "$DEPLOY_PARAMS" ]]; then
    DEPLOY_CMD="$DEPLOY_CMD --parameters $DEPLOY_PARAMS"
fi

eval "$DEPLOY_CMD"

DEPLOY_STATE=$(az deployment group show \
    --resource-group "$RG_NAME" \
    --name main \
    --query properties.provisioningState -o tsv 2>/dev/null || echo "Unknown")

echo ""
if [[ "$DEPLOY_STATE" == "Succeeded" ]]; then
    echo "Bicep deployment SUCCEEDED"
else
    echo "ERROR: Deployment state: $DEPLOY_STATE"
    echo "Check Azure portal for details"
    exit 1
fi

echo ""
echo "[4/4] Running post-deployment setup..."
bash "$SCRIPT_DIR/post_deployment_setup.sh"

echo ""
echo "=== Deployment Complete ==="
