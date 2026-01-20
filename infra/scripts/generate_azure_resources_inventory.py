#!/usr/bin/env python3
"""
Generate comprehensive Azure resource inventory for AI agents.

This script queries Azure CLI to build a complete resource inventory
that can be used by AI agents for deployments, testing, and operations.

Usage:
    python3 generate_azure_resources_inventory.py > ../azure-resources.json
    
Requirements:
    - az cli logged in
    - jq installed (optional, for pretty printing)
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def run_az_command(command: List[str]) -> Any:
    """Execute az cli command and return JSON output."""
    try:
        result = subprocess.run(
            ["az"] + command + ["--output", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout.strip():
            return json.loads(result.stdout)
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}", file=sys.stderr)
        print(f"Error: {e.stderr}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from command: {' '.join(command)}", file=sys.stderr)
        print(f"Output: {result.stdout}", file=sys.stderr)
        return None


def get_account_info() -> Dict[str, Any]:
    """Get current Azure account information."""
    account = run_az_command(["account", "show"])
    if not account:
        raise RuntimeError("Failed to get account info. Ensure 'az login' is complete.")
    
    return {
        "subscription_id": account["id"],
        "subscription_name": account["name"],
        "tenant_id": account["tenantId"],
        "user": account["user"]["name"],
        "user_type": account["user"]["type"]
    }


def get_resource_groups() -> List[Dict[str, Any]]:
    """Get all resource groups."""
    rgs = run_az_command(["group", "list"]) or []
    
    return [
        {
            "name": rg["name"],
            "id": rg["id"],
            "location": rg["location"],
            "purpose": "Primary resource group for all Spymaster resources" if "spymaster-dev" in rg["name"]
            else "Databricks managed resource group" if "adbspymaster" in rg["name"]
            else "Application Insights managed resource group" if "ai_appis" in rg["name"]
            else "Purview managed resource group" if "pvspymaster" in rg["name"]
            else "Managed resource group"
        }
        for rg in rgs
        if any(x in rg["name"].lower() for x in ["spymaster", "databricks", "purview", "appis"])
    ]


def get_storage_accounts(rg_name: str) -> Dict[str, Any]:
    """Get storage account details."""
    accounts = run_az_command(["storage", "account", "list", "--resource-group", rg_name]) or []
    
    storage_info = {}
    
    for account in accounts:
        name = account["name"]
        
        # Get containers
        containers = run_az_command([
            "storage", "container", "list",
            "--account-name", name,
            "--auth-mode", "login"
        ]) or []
        
        container_list = [
            {
                "name": c["name"],
                "purpose": "Medallion architecture data lake (bronze/silver/gold)" if c["name"] == "lake"
                else "ML model artifacts and training outputs" if "ml" in c["name"]
                else "Raw Databento DBN files" if "raw" in c["name"]
                else "Power BI datasets" if "powerbi" in c["name"].lower()
                else "Container"
            }
            for c in containers
        ]
        
        account_info = {
            "name": name,
            "id": account["id"],
            "location": account["location"],
            "kind": account["kind"],
            "sku": account["sku"]["name"],
            "is_hns_enabled": account.get("isHnsEnabled", False),
            "minimum_tls_version": account.get("minimumTlsVersion"),
            "allow_blob_public_access": account.get("allowBlobPublicAccess", False),
            "allow_shared_key_access": account.get("allowSharedKeyAccess", True),
            "endpoints": {
                "blob": account["primaryEndpoints"].get("blob"),
                "dfs": account["primaryEndpoints"].get("dfs"),
                "file": account["primaryEndpoints"].get("file"),
                "queue": account["primaryEndpoints"].get("queue"),
                "table": account["primaryEndpoints"].get("table"),
                "web": account["primaryEndpoints"].get("web"),
            },
            "containers": container_list
        }
        
        if "lake" in name:
            storage_info["lake"] = account_info
            storage_info["lake"]["purpose"] = "Data lake storage for medallion architecture"
        elif "mls" in name or "aml" in name:
            storage_info["aml"] = account_info
            storage_info["aml"]["purpose"] = "AML workspace default storage"
    
    return storage_info


def get_eventhubs(rg_name: str) -> Dict[str, Any]:
    """Get Event Hubs namespace and hub details."""
    namespaces = run_az_command(["eventhubs", "namespace", "list", "--resource-group", rg_name]) or []
    
    if not namespaces:
        return {}
    
    ns = namespaces[0]
    ns_name = ns["name"]
    
    # Get all Event Hubs in namespace
    hubs = run_az_command(["eventhubs", "eventhub", "list", "--resource-group", rg_name, "--namespace-name", ns_name]) or []
    
    hubs_info = {}
    for hub in hubs:
        hub_name = hub["name"]
        
        # Get consumer groups
        consumer_groups = run_az_command([
            "eventhubs", "eventhub", "consumer-group", "list",
            "--resource-group", rg_name,
            "--namespace-name", ns_name,
            "--eventhub-name", hub_name
        ]) or []
        
        cg_list = [
            {
                "name": cg["name"],
                "id": cg["id"]
            }
            for cg in consumer_groups
        ]
        
        hubs_info[hub_name] = {
            "name": hub_name,
            "id": hub["id"],
            "partition_count": hub.get("partitionCount", 4),
            "partition_ids": hub.get("partitionIds", []),
            "retention_days": hub.get("messageRetentionInDays", 7),
            "purpose": "Raw MBO market events from Databento" if hub_name == "mbo_raw"
            else "Real-time feature vectors from gold layer" if hub_name == "features_gold"
            else "Model inference predictions/scores" if hub_name == "inference_scores"
            else "Event hub",
            "consumer_groups": cg_list
        }
    
    # Get authorization rules
    auth_rules = run_az_command([
        "eventhubs", "namespace", "authorization-rule", "list",
        "--resource-group", rg_name,
        "--namespace-name", ns_name
    ]) or []
    
    auth_rules_info = {}
    for rule in auth_rules:
        auth_rules_info[rule["name"]] = {
            "id": rule["id"],
            "rights": rule["rights"]
        }
    
    return {
        "namespace": {
            "name": ns_name,
            "id": ns["id"],
            "location": ns["location"],
            "sku": ns["sku"]["name"],
            "capacity": ns["sku"].get("capacity", 1),
            "kafka_enabled": ns.get("kafkaEnabled", True),
            "minimum_tls_version": ns.get("minimumTlsVersion", "1.2"),
            "service_bus_endpoint": ns.get("serviceBusEndpoint"),
            "metric_id": ns.get("metricId")
        },
        "hubs": hubs_info,
        "authorization_rules": auth_rules_info
    }


def get_databricks(rg_name: str) -> Dict[str, Any]:
    """Get Databricks workspace details."""
    workspaces = run_az_command(["databricks", "workspace", "list", "--resource-group", rg_name]) or []
    
    if not workspaces:
        return {}
    
    ws = workspaces[0]
    
    # Get access connector
    access_connectors = run_az_command([
        "databricks", "access-connector", "list",
        "--resource-group", rg_name
    ]) or []
    
    acc_info = {}
    if access_connectors:
        acc = access_connectors[0]
        acc_info = {
            "name": acc["name"],
            "id": acc["id"],
            "location": acc["location"],
            "identity": {
                "type": acc["identity"]["type"],
                "principal_id": acc["identity"]["principalId"],
                "tenant_id": acc["identity"]["tenantId"]
            }
        }
    
    return {
        "workspace": {
            "name": ws["name"],
            "id": ws["id"],
            "location": ws["location"],
            "workspace_id": str(ws["workspaceId"]),
            "workspace_url": ws["workspaceUrl"],
            "sku": ws["sku"]["name"],
            "managed_resource_group_id": ws["managedResourceGroupId"],
            "is_uc_enabled": ws.get("parameters", {}).get("enableNoPublicIp", {}).get("value") is not None,
            "public_network_access": ws.get("publicNetworkAccess", "Enabled"),
            "internal_storage_account": ws.get("parameters", {}).get("storageAccountName", {}).get("value", "unknown")
        },
        "access_connector": acc_info
    }


def get_machine_learning(rg_name: str, subscription_id: str) -> Dict[str, Any]:
    """Get Azure Machine Learning workspace details."""
    workspaces = run_az_command(["ml", "workspace", "list", "--resource-group", rg_name]) or []
    
    if not workspaces:
        return {}
    
    ws = workspaces[0]
    ws_name = ws["name"]
    
    # Get compute clusters
    computes = run_az_command([
        "ml", "compute", "list",
        "--resource-group", rg_name,
        "--workspace-name", ws_name
    ]) or []
    
    compute_info = {}
    for compute in computes:
        if compute["type"] == "amlcompute":
            compute_info[compute["name"].replace("-", "_")] = {
                "name": compute["name"],
                "id": compute["id"],
                "location": compute["location"],
                "type": compute["type"],
                "size": compute.get("properties", {}).get("vmSize"),
                "tier": compute.get("properties", {}).get("vmPriority", "dedicated"),
                "min_instances": compute.get("properties", {}).get("scaleSettings", {}).get("minNodeCount", 0),
                "max_instances": compute.get("properties", {}).get("scaleSettings", {}).get("maxNodeCount", 2),
                "idle_time_before_scale_down_seconds": compute.get("properties", {}).get("scaleSettings", {}).get("nodeIdleTimeBeforeScaleDown", 600),
                "identity": compute.get("identity", {})
            }
    
    # Get online endpoints
    endpoints = run_az_command([
        "ml", "online-endpoint", "list",
        "--resource-group", rg_name,
        "--workspace-name", ws_name
    ]) or []
    
    endpoints_info = {}
    for endpoint in endpoints:
        ep_name = endpoint["name"]
        
        # Get deployments
        deployments = run_az_command([
            "ml", "online-deployment", "list",
            "--resource-group", rg_name,
            "--workspace-name", ws_name,
            "--endpoint-name", ep_name
        ]) or []
        
        deployments_info = {}
        for deployment in deployments:
            deployments_info[deployment["name"]] = {
                "name": deployment["name"],
                "id": deployment.get("id", f"deployment-{deployment['name']}"),
                "instance_type": deployment.get("sku", {}).get("name"),
                "model": deployment.get("model")
            }
        
        endpoints_info[ep_name.replace("-", "_")] = {
            "name": ep_name,
            "id": endpoint.get("id", f"endpoint-{ep_name}"),
            "location": endpoint.get("location", "westus"),
            "auth_mode": endpoint.get("properties", {}).get("authMode", "key"),
            "scoring_uri": endpoint.get("properties", {}).get("scoringUri"),
            "openapi_uri": endpoint.get("properties", {}).get("openApiUri"),
            "description": endpoint.get("properties", {}).get("description", ""),
            "traffic": endpoint.get("properties", {}).get("traffic", {}),
            "identity": endpoint.get("identity", {}),
            "deployments": deployments_info
        }
    
    # Get models
    models = run_az_command([
        "ml", "model", "list",
        "--resource-group", rg_name,
        "--workspace-name", ws_name
    ]) or []
    
    models_info = {}
    for model in models:
        model_name = model["name"]
        models_info[model_name] = {
            "name": model_name,
            "id": f"azureml:/subscriptions/{subscription_id}/resourceGroups/{rg_name}/providers/Microsoft.MachineLearningServices/workspaces/{ws_name}/models/{model_name}",
            "latest_version": str(model.get("version", "1"))
        }
    
    # Get datastores
    datastores = run_az_command([
        "ml", "datastore", "list",
        "--resource-group", rg_name,
        "--workspace-name", ws_name
    ]) or []
    
    datastores_info = {}
    for ds in datastores:
        ds_name = ds["name"]
        datastores_info[ds_name.replace("-", "_")] = {
            "name": ds_name,
            "id": f"/subscriptions/{subscription_id}/resourceGroups/{rg_name}/providers/Microsoft.MachineLearningServices/workspaces/{ws_name}/datastores/{ds_name}",
            "type": ds.get("type", "unknown"),
            "account_name": ds.get("accountName") or ds.get("containerName", "unknown"),
            "filesystem": ds.get("fileSystem") or ds.get("containerName")
        }
    
    return {
        "workspace": {
            "name": ws_name,
            "id": ws["id"],
            "location": ws["location"],
            "display_name": ws.get("friendlyName", ws_name),
            "description": ws.get("description", f"{ws_name} workspace"),
            "mlflow_tracking_uri": f"azureml://{ws['location']}.api.azureml.ms/mlflow/v1.0/subscriptions/{subscription_id}/resourceGroups/{rg_name}/providers/Microsoft.MachineLearningServices/workspaces/{ws_name}",
            "discovery_url": ws.get("discoveryUrl", f"https://{ws['location']}.api.azureml.ms/discovery"),
            "identity": ws.get("identity", {}),
            "linked_resources": {
                "storage_account": ws.get("storageAccount"),
                "key_vault": ws.get("keyVault"),
                "application_insights": ws.get("applicationInsights"),
                "container_registry": ws.get("containerRegistry")
            }
        },
        "compute": compute_info,
        "endpoints": endpoints_info,
        "models": models_info,
        "datastores": datastores_info
    }


def get_keyvaults(rg_name: str) -> Dict[str, Any]:
    """Get Key Vault details."""
    vaults = run_az_command(["keyvault", "list", "--resource-group", rg_name]) or []
    
    vaults_info = {}
    for vault in vaults:
        vault_name = vault["name"]
        
        # Try to get secrets (metadata only) - may fail if no permission
        secrets = run_az_command([
            "keyvault", "secret", "list",
            "--vault-name", vault_name
        ])
        
        secrets_list = []
        if secrets:
            secrets_list = [
                {
                    "name": secret["name"],
                    "uri": secret["id"]
                }
                for secret in secrets
            ]
        
        key = "aml" if "kv" in vault_name and vault_name.endswith("kv") else "runtime"
        
        vaults_info[key] = {
            "name": vault_name,
            "id": vault["id"],
            "location": vault["location"],
            "vault_uri": vault["properties"]["vaultUri"],
            "sku": vault["properties"]["sku"]["name"],
            "enable_rbac_authorization": vault["properties"].get("enableRbacAuthorization", False),
            "enable_purge_protection": vault["properties"].get("enablePurgeProtection", True),
            "soft_delete_retention_days": vault["properties"].get("softDeleteRetentionInDays", 90),
            "purpose": "AML workspace key vault" if key == "aml" else "Runtime secrets for applications",
            "secrets": secrets_list if key == "runtime" else []
        }
    
    return vaults_info


def get_data_factory(rg_name: str) -> Dict[str, Any]:
    """Get Data Factory details."""
    factories = run_az_command(["datafactory", "list", "--resource-group", rg_name]) or []
    
    if not factories:
        return {}
    
    df = factories[0]
    df_name = df["name"]
    
    # Get linked services
    linked_services = run_az_command([
        "datafactory", "linked-service", "list",
        "--resource-group", rg_name,
        "--factory-name", df_name
    ]) or []
    
    ls_info = {}
    for ls in linked_services:
        ls_info[ls["name"]] = {
            "name": ls["name"],
            "id": ls["id"],
            "type": ls["properties"].get("type"),
            "url": ls["properties"].get("typeProperties", {}).get("url")
        }
    
    return {
        "name": df_name,
        "id": df["id"],
        "location": df["location"],
        "version": df.get("version", "2018-06-01"),
        "identity": df.get("identity", {}),
        "linked_services": ls_info
    }


def get_container_registry(rg_name: str) -> Dict[str, Any]:
    """Get Container Registry details."""
    registries = run_az_command(["acr", "list", "--resource-group", rg_name]) or []
    
    if not registries:
        return {}
    
    acr = registries[0]
    acr_name = acr["name"]
    
    # Get repositories
    repos = run_az_command([
        "acr", "repository", "list",
        "--name", acr_name
    ]) or []
    
    return {
        "name": acr_name,
        "id": acr["id"],
        "location": acr["location"],
        "login_server": acr["loginServer"],
        "sku": acr["sku"]["name"],
        "admin_user_enabled": acr.get("adminUserEnabled", False),
        "repositories": repos
    }


def get_fabric_capacity(rg_name: str) -> Dict[str, Any]:
    """Get Microsoft Fabric capacity details."""
    capacities = run_az_command(["fabric", "capacity", "list", "--resource-group", rg_name]) or []
    
    if not capacities:
        return {}
    
    capacity = capacities[0]
    
    return {
        "capacity": {
            "name": capacity.get("name", "qfabric"),
            "id": capacity.get("id", ""),
            "location": capacity.get("location", "West US 2"),
            "sku": capacity.get("sku", {}).get("name", "F8"),
            "tier": capacity.get("sku", {}).get("tier", "Fabric"),
            "state": capacity.get("properties", {}).get("state", "Active"),
            "administrators": capacity.get("properties", {}).get("administration", {}).get("members", [])
        }
    }


def get_monitoring(rg_name: str) -> Dict[str, Any]:
    """Get Log Analytics and Application Insights details."""
    # Log Analytics
    law_workspaces = run_az_command([
        "monitor", "log-analytics", "workspace", "list",
        "--resource-group", rg_name
    ]) or []
    
    law_info = {}
    if law_workspaces:
        law = law_workspaces[0]
        law_info = {
            "name": law["name"],
            "id": law["id"],
            "location": law["location"],
            "customer_id": law["customerId"],
            "sku": law["sku"]["name"],
            "retention_in_days": law.get("retentionInDays", 30)
        }
    
    # Application Insights
    app_insights = run_az_command([
        "monitor", "app-insights", "component", "show",
        "--resource-group", rg_name,
        "--query", "[0]"
    ]) or run_az_command([
        "resource", "list",
        "--resource-group", rg_name,
        "--resource-type", "Microsoft.Insights/components",
        "--query", "[0]"
    ])
    
    appins_info = {}
    if app_insights:
        appins_info = {
            "name": app_insights["name"],
            "id": app_insights["id"],
            "location": app_insights["location"],
            "app_id": app_insights.get("properties", {}).get("AppId"),
            "instrumentation_key": app_insights.get("properties", {}).get("InstrumentationKey"),
            "connection_string": app_insights.get("properties", {}).get("ConnectionString"),
            "application_type": app_insights.get("properties", {}).get("Application_Type", "web"),
            "retention_in_days": app_insights.get("properties", {}).get("RetentionInDays", 90),
            "ingestion_mode": app_insights.get("properties", {}).get("IngestionMode", "LogAnalytics")
        }
    
    return {
        "log_analytics": law_info,
        "application_insights": appins_info
    }


def get_purview(rg_name: str) -> Dict[str, Any]:
    """Get Purview account details."""
    accounts = run_az_command(["purview", "account", "list", "--resource-group", rg_name]) or []
    
    if not accounts:
        return {}
    
    pv = accounts[0]
    
    return {
        "name": pv["name"],
        "id": pv["id"],
        "location": pv["location"],
        "sku": pv["sku"]["name"],
        "managed_resource_group": pv.get("managedResourceGroupName"),
        "endpoints": pv.get("endpoints", {}),
        "identity": pv.get("identity", {})
    }


def get_managed_identities(resources: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all managed identities from resources."""
    identities = {}
    
    # AML workspace
    if "machine_learning" in resources and "workspace" in resources["machine_learning"]:
        ws_identity = resources["machine_learning"]["workspace"].get("identity", {})
        if ws_identity.get("principal_id"):
            identities["aml_workspace"] = {
                "resource": resources["machine_learning"]["workspace"]["name"],
                "principal_id": ws_identity["principal_id"],
                "type": ws_identity["type"]
            }
    
    # AML compute
    if "machine_learning" in resources and "compute" in resources["machine_learning"]:
        for compute_name, compute_data in resources["machine_learning"]["compute"].items():
            if compute_data.get("identity", {}).get("principal_id"):
                identities[f"aml_{compute_name}"] = {
                    "resource": compute_data["name"],
                    "principal_id": compute_data["identity"]["principal_id"],
                    "type": compute_data["identity"]["type"]
                }
    
    # AML endpoints
    if "machine_learning" in resources and "endpoints" in resources["machine_learning"]:
        for ep_name, ep_data in resources["machine_learning"]["endpoints"].items():
            if ep_data.get("identity", {}).get("principal_id"):
                identities[f"aml_{ep_name}"] = {
                    "resource": ep_data["name"],
                    "principal_id": ep_data["identity"]["principal_id"],
                    "type": ep_data["identity"]["type"]
                }
    
    # Databricks access connector
    if "databricks" in resources and "access_connector" in resources["databricks"]:
        acc = resources["databricks"]["access_connector"]
        if acc.get("identity", {}).get("principal_id"):
            identities["databricks_access_connector"] = {
                "resource": acc["name"],
                "principal_id": acc["identity"]["principal_id"],
                "type": acc["identity"]["type"]
            }
    
    # Data Factory
    if "data_factory" in resources and resources["data_factory"].get("identity", {}).get("principalId"):
        identities["data_factory"] = {
            "resource": resources["data_factory"]["name"],
            "principal_id": resources["data_factory"]["identity"]["principalId"],
            "type": resources["data_factory"]["identity"]["type"]
        }
    
    # Purview
    if "purview" in resources and resources["purview"].get("identity", {}).get("principal_id"):
        identities["purview"] = {
            "resource": resources["purview"]["name"],
            "principal_id": resources["purview"]["identity"]["principal_id"],
            "type": resources["purview"]["identity"]["type"]
        }
    
    return {
        "summary": "All managed identities used by Spymaster resources for RBAC-based access",
        "identities": identities
    }


def build_quick_reference(resources: Dict[str, Any]) -> Dict[str, str]:
    """Build quick reference map of key resource names."""
    quick_ref = {
        "resource_group": resources.get("_metadata", {}).get("primary_resource_group", ""),
    }
    
    if "storage" in resources and "lake" in resources["storage"]:
        quick_ref["storage_account_lake"] = resources["storage"]["lake"]["name"]
    
    if "storage" in resources and "aml" in resources["storage"]:
        quick_ref["storage_account_aml"] = resources["storage"]["aml"]["name"]
    
    if "eventhubs" in resources:
        quick_ref["eventhubs_namespace"] = resources["eventhubs"]["namespace"]["name"]
    
    if "databricks" in resources and "workspace" in resources["databricks"]:
        quick_ref["databricks_workspace"] = resources["databricks"]["workspace"]["name"]
        quick_ref["databricks_workspace_url"] = resources["databricks"]["workspace"]["workspace_url"]
    
    if "machine_learning" in resources and "workspace" in resources["machine_learning"]:
        quick_ref["aml_workspace"] = resources["machine_learning"]["workspace"]["name"]
    
    if "machine_learning" in resources and "endpoints" in resources["machine_learning"]:
        for ep_name, ep_data in resources["machine_learning"]["endpoints"].items():
            quick_ref[f"aml_{ep_name}"] = ep_data["name"]
            quick_ref[f"aml_scoring_uri_{ep_name}"] = ep_data.get("scoring_uri", "")
    
    if "keyvaults" in resources:
        for kv_name, kv_data in resources["keyvaults"].items():
            quick_ref[f"keyvault_{kv_name}"] = kv_data["name"]
    
    if "data_factory" in resources:
        quick_ref["data_factory"] = resources["data_factory"]["name"]
    
    if "container_registry" in resources:
        quick_ref["container_registry"] = resources["container_registry"]["name"]
        quick_ref["acr_login_server"] = resources["container_registry"]["login_server"]
    
    if "fabric" in resources and "capacity" in resources["fabric"]:
        quick_ref["fabric_capacity"] = resources["fabric"]["capacity"]["name"]
    
    if "monitoring" in resources:
        if "log_analytics" in resources["monitoring"]:
            quick_ref["log_analytics"] = resources["monitoring"]["log_analytics"]["name"]
        if "application_insights" in resources["monitoring"]:
            quick_ref["app_insights"] = resources["monitoring"]["application_insights"]["name"]
    
    return quick_ref


def main():
    """Generate comprehensive Azure resource inventory."""
    print("Generating Azure resource inventory...", file=sys.stderr)
    
    # Get account info
    account_info = get_account_info()
    subscription_id = account_info["subscription_id"]
    
    # Primary resource group
    rg_name = "rg-spymaster-dev"
    
    # Build inventory
    inventory = {
        "_metadata": {
            "description": "Comprehensive Azure resource inventory for Spymaster project. Use this file for quick reference to resource IDs, names, endpoints, and identities without running Azure CLI commands.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "subscription_id": subscription_id,
            "subscription_name": account_info["subscription_name"],
            "tenant_id": account_info["tenant_id"],
            "environment": "dev",
            "primary_location": "westus",
            "primary_resource_group": rg_name
        }
    }
    
    print("Fetching resource groups...", file=sys.stderr)
    inventory["resource_groups"] = get_resource_groups()
    
    print("Fetching storage accounts...", file=sys.stderr)
    inventory["storage"] = get_storage_accounts(rg_name)
    
    print("Fetching Event Hubs...", file=sys.stderr)
    inventory["eventhubs"] = get_eventhubs(rg_name)
    
    print("Fetching Databricks...", file=sys.stderr)
    inventory["databricks"] = get_databricks(rg_name)
    
    print("Fetching Azure ML...", file=sys.stderr)
    inventory["machine_learning"] = get_machine_learning(rg_name, subscription_id)
    
    print("Fetching Key Vaults...", file=sys.stderr)
    inventory["keyvaults"] = get_keyvaults(rg_name)
    
    print("Fetching Data Factory...", file=sys.stderr)
    inventory["data_factory"] = get_data_factory(rg_name)
    
    print("Fetching Container Registry...", file=sys.stderr)
    inventory["container_registry"] = get_container_registry(rg_name)
    
    print("Fetching Fabric Capacity...", file=sys.stderr)
    inventory["fabric"] = get_fabric_capacity(rg_name)
    
    print("Fetching monitoring resources...", file=sys.stderr)
    inventory["monitoring"] = get_monitoring(rg_name)
    
    print("Fetching Purview...", file=sys.stderr)
    inventory["purview"] = get_purview(rg_name)
    
    print("Extracting managed identities...", file=sys.stderr)
    inventory["managed_identities"] = get_managed_identities(inventory)
    
    print("Building quick reference...", file=sys.stderr)
    inventory["quick_reference"] = build_quick_reference(inventory)
    
    # Add role definitions
    inventory["role_definitions"] = {
        "reference": "Common Azure RBAC role definition IDs used in this deployment",
        "storage_blob_data_contributor": "ba92f5b4-2d11-453d-a403-e96b0029c9fe",
        "event_hubs_data_sender": "2b629674-e913-4c01-ae53-ef4638d8f975",
        "event_hubs_data_receiver": "a638d3c7-ab3a-418d-83e6-5f17a39d4fde",
        "key_vault_secrets_user": "4633458b-17de-408a-b874-0445c86b69e6"
    }
    
    # Add connection strings templates
    storage_lake = inventory["storage"].get("lake", {}).get("name", "")
    eventhubs_ns = inventory["eventhubs"].get("namespace", {}).get("name", "")
    databricks_url = inventory["databricks"].get("workspace", {}).get("workspace_url", "")
    
    inventory["connection_strings"] = {
        "note": "These are template patterns. Actual connection strings may require secrets from Key Vault.",
        "eventhubs_namespace": f"Endpoint=sb://{eventhubs_ns}.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=<from-keyvault>",
        "storage_lake_abfss": f"abfss://{{container}}@{storage_lake}.dfs.core.windows.net/",
        "storage_lake_https": f"https://{storage_lake}.dfs.core.windows.net/{{container}}/",
        "databricks_host": f"https://{databricks_url}",
        "aml_scoring": inventory["machine_learning"].get("endpoints", {}).get("es_model_endpoint", {}).get("scoring_uri", "")
    }
    
    # Add data paths
    inventory["data_paths"] = {
        "lake_bronze": f"abfss://lake@{storage_lake}.dfs.core.windows.net/bronze/",
        "lake_silver": f"abfss://lake@{storage_lake}.dfs.core.windows.net/silver/",
        "lake_gold": f"abfss://lake@{storage_lake}.dfs.core.windows.net/gold/",
        "raw_dbn": f"abfss://raw-dbn@{storage_lake}.dfs.core.windows.net/",
        "ml_artifacts": f"abfss://ml-artifacts@{storage_lake}.dfs.core.windows.net/"
    }
    
    # Output JSON
    print(json.dumps(inventory, indent=2))
    print(f"\n✅ Inventory generated successfully", file=sys.stderr)
    print(f"Total resource groups: {len(inventory['resource_groups'])}", file=sys.stderr)
    print(f"Storage accounts: {len(inventory.get('storage', {}))}", file=sys.stderr)
    print(f"Event Hubs: {len(inventory.get('eventhubs', {}).get('hubs', {}))}", file=sys.stderr)
    print(f"Managed identities: {len(inventory.get('managed_identities', {}).get('identities', {}))}", file=sys.stderr)


if __name__ == "__main__":
    main()
