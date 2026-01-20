targetScope = 'resourceGroup'

@description('Location for all resources.')
param location string = 'eastus'

@description('Prefix for resource names (lowercase, alphanumeric).')
@minLength(3)
@maxLength(12)
param namePrefix string = 'spymaster'

@description('Environment tag for naming.')
@minLength(2)
@maxLength(6)
param environment string = 'dev'

@description('Fabric capacity SKU name, for example F2 or F4.')
param fabricCapacitySku string = 'F2'

@description('Fabric capacity name.')
param fabricCapacityName string

@description('Fabric capacity location.')
param fabricCapacityLocation string = location

@description('Fabric capacity admin members (UPNs).')
param fabricAdminMembers array

@description('Whether to deploy Fabric capacity in this run.')
param deployFabric bool = true

@description('AML compute VM size.')
param amlComputeVmSize string = 'Standard_DS3_v2'

@description('Azure ML workspace name.')
param amlWorkspaceName string

@description('AML compute cluster min nodes.')
param amlComputeMinNodes int = 0

@description('AML compute cluster max nodes.')
param amlComputeMaxNodes int = 2

@description('AML compute idle time before scale down.')
param amlComputeIdleTime string = 'PT10M'

@description('Event Hubs SKU name.')
param eventHubsSkuName string = 'Standard'

@description('Event Hubs capacity units.')
param eventHubsSkuCapacity int = 1

@description('Event Hubs partition count per hub.')
param eventHubPartitionCount int = 4

@description('Event Hubs retention in days.')
param eventHubRetentionDays int = 7

@description('Databricks workspace SKU name.')
param databricksSkuName string = 'premium'

@description('Databricks public network access.')
param databricksPublicNetworkAccess string = 'Enabled'

@description('Role definition id for Azure Event Hubs Data Sender.')
param eventHubsDataSenderRoleId string

@description('Role definition id for Azure Event Hubs Data Receiver.')
param eventHubsDataReceiverRoleId string

@description('Role definition id for Key Vault Secrets User.')
param keyVaultSecretsUserRoleId string

var nameSeed = uniqueString(resourceGroup().id)
var storageBase = toLower('${namePrefix}${environment}lake${nameSeed}')
var amlAcrBase = toLower('acr${namePrefix}${environment}${nameSeed}')
var amlStorageBase = toLower('${namePrefix}${environment}mls${nameSeed}')
var storageAccountName = substring(storageBase, 0, min(length(storageBase), 24))
var amlStorageAccountName = substring(amlStorageBase, 0, min(length(amlStorageBase), 24))
var dataFactoryName = toLower('adf${namePrefix}${environment}${nameSeed}')
var amlKeyVaultName = toLower(substring('kv${namePrefix}${environment}${nameSeed}', 0, 24))
var amlContainerRegistryName = substring(amlAcrBase, 0, min(length(amlAcrBase), 50))
var amlAppInsightsName = toLower('appi${namePrefix}${environment}${nameSeed}')
var eventHubsNamespaceName = toLower('ehn${namePrefix}${environment}${nameSeed}')
var databricksWorkspaceBase = toLower('adb${namePrefix}${environment}${nameSeed}')
var databricksWorkspaceName = substring(databricksWorkspaceBase, 0, min(length(databricksWorkspaceBase), 64))
var databricksManagedRgName = 'rg-${databricksWorkspaceName}-managed'
var databricksAccessConnectorBase = toLower('adbacc${namePrefix}${environment}${nameSeed}')
var databricksAccessConnectorName = substring(databricksAccessConnectorBase, 0, min(length(databricksAccessConnectorBase), 64))
var runtimeKeyVaultName = toLower(substring('kv${namePrefix}${environment}rt${nameSeed}', 0, 24))

var lakeContainers = [
  'raw-dbn'
  'lake'
  'ml-artifacts'
]

var amlDatastoreContainers = [
  {
    name: 'raw_dbn'
    filesystem: 'raw-dbn'
  }
  {
    name: 'lake'
    filesystem: 'lake'
  }
  {
    name: 'ml_artifacts'
    filesystem: 'ml-artifacts'
  }
]

var eventHubEntities = [
  {
    name: 'mbo_raw'
    partitionCount: eventHubPartitionCount
    messageRetentionInDays: eventHubRetentionDays
    consumerGroups: [
      'databricks_bronze'
      'fabric_stream'
      'analytics'
    ]
  }
  {
    name: 'features_gold'
    partitionCount: eventHubPartitionCount
    messageRetentionInDays: eventHubRetentionDays
    consumerGroups: [
      'databricks_features'
      'fabric_stream'
      'analytics'
    ]
  }
  {
    name: 'inference_scores'
    partitionCount: eventHubPartitionCount
    messageRetentionInDays: eventHubRetentionDays
    consumerGroups: [
      'databricks_inference'
      'fabric_stream'
      'analytics'
    ]
  }
]

module storage './modules/storage.bicep' = {
  name: 'storage'
  params: {
    location: location
    storageAccountName: storageAccountName
    containers: lakeContainers
  }
}

module datafactory './modules/datafactory.bicep' = {
  name: 'datafactory'
  params: {
    location: location
    dataFactoryName: dataFactoryName
    storageAccountName: storage.outputs.storageAccountName
  }
}

module keyvault './modules/keyvault.bicep' = {
  name: 'runtime-kv'
  params: {
    location: location
    keyVaultName: runtimeKeyVaultName
  }
}

module eventhubs './modules/eventhubs.bicep' = {
  name: 'eventhubs'
  params: {
    location: location
    namespaceName: eventHubsNamespaceName
    skuName: eventHubsSkuName
    skuCapacity: eventHubsSkuCapacity
    eventHubs: eventHubEntities
  }
}

module databricks './modules/databricks.bicep' = {
  name: 'databricks'
  params: {
    location: location
    workspaceName: databricksWorkspaceName
    managedResourceGroupName: databricksManagedRgName
    skuName: databricksSkuName
    publicNetworkAccess: databricksPublicNetworkAccess
    accessConnectorName: databricksAccessConnectorName
  }
}

module aml './modules/aml.bicep' = {
  name: 'aml'
  params: {
    location: location
    amlWorkspaceName: amlWorkspaceName
    amlStorageAccountName: amlStorageAccountName
    lakeStorageAccountName: storage.outputs.storageAccountName
    keyVaultName: amlKeyVaultName
    containerRegistryName: amlContainerRegistryName
    appInsightsName: amlAppInsightsName
    datastores: amlDatastoreContainers
    computeVmSize: amlComputeVmSize
    computeMinNodes: amlComputeMinNodes
    computeMaxNodes: amlComputeMaxNodes
    computeIdleTime: amlComputeIdleTime
  }
}

module fabric './modules/fabric-capacity.bicep' = if (deployFabric) {
  name: 'fabric'
  params: {
    location: fabricCapacityLocation
    capacityName: fabricCapacityName
    skuName: fabricCapacitySku
    adminMembers: fabricAdminMembers
  }
}

module rbac './modules/rbac.bicep' = {
  name: 'rbac'
  params: {
    storageAccountName: storage.outputs.storageAccountName
    dataFactoryPrincipalId: datafactory.outputs.dataFactoryPrincipalId
    amlWorkspacePrincipalId: aml.outputs.amlWorkspacePrincipalId
    amlComputePrincipalId: aml.outputs.amlComputePrincipalId
    databricksAccessConnectorPrincipalId: databricks.outputs.accessConnectorPrincipalId
    eventHubsNamespaceName: eventhubs.outputs.namespaceName
    runtimeKeyVaultName: keyvault.outputs.keyVaultName
    eventHubsDataSenderRoleId: eventHubsDataSenderRoleId
    eventHubsDataReceiverRoleId: eventHubsDataReceiverRoleId
    keyVaultSecretsUserRoleId: keyVaultSecretsUserRoleId
  }
}

output storageAccountName string = storage.outputs.storageAccountName
output storageAccountId string = storage.outputs.storageAccountId
output dataFactoryName string = datafactory.outputs.dataFactoryName
output dataFactoryId string = datafactory.outputs.dataFactoryId
output eventHubsNamespaceName string = eventhubs.outputs.namespaceName
output eventHubNames array = eventhubs.outputs.eventHubNames
output runtimeKeyVaultName string = keyvault.outputs.keyVaultName
output amlWorkspaceName string = aml.outputs.amlWorkspaceName
output amlWorkspaceId string = aml.outputs.amlWorkspaceId
output amlComputeName string = aml.outputs.amlComputeName
output databricksWorkspaceName string = databricks.outputs.workspaceName
output databricksWorkspaceUrl string = databricks.outputs.workspaceUrl
output fabricCapacityName string = deployFabric ? fabric.outputs.capacityName : ''
output fabricCapacityId string = deployFabric ? fabric.outputs.capacityId : ''
