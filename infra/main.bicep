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

@description('Fabric capacity admin members (UPNs).')
param fabricAdminMembers array

@description('Whether to deploy Fabric capacity in this run.')
param deployFabric bool = true

@description('AML compute VM size.')
param amlComputeVmSize string = 'Standard_DS3_v2'

@description('AML compute cluster min nodes.')
param amlComputeMinNodes int = 0

@description('AML compute cluster max nodes.')
param amlComputeMaxNodes int = 2

@description('AML compute idle time before scale down.')
param amlComputeIdleTime string = 'PT10M'

var nameSeed = uniqueString(resourceGroup().id)
var storageBase = toLower('${namePrefix}${environment}lake${nameSeed}')
var amlWorkspaceBase = toLower('mlw${namePrefix}${environment}${nameSeed}')
var amlStorageBase = toLower('${namePrefix}${environment}mls${nameSeed}')
var amlAcrBase = toLower('acr${namePrefix}${environment}${nameSeed}')
var storageAccountName = substring(storageBase, 0, min(length(storageBase), 24))
var amlStorageAccountName = substring(amlStorageBase, 0, min(length(amlStorageBase), 24))
var purviewAccountName = toLower('pv${namePrefix}${environment}${nameSeed}')
var dataFactoryName = toLower('adf${namePrefix}${environment}${nameSeed}')
var amlWorkspaceName = substring(amlWorkspaceBase, 0, min(length(amlWorkspaceBase), 32))
var fabricCapacityName = toLower('fabric${namePrefix}${environment}${nameSeed}')
var purviewManagedRgName = 'rg-${purviewAccountName}-mrg'
var amlKeyVaultName = toLower(substring('kv${namePrefix}${environment}${nameSeed}', 0, 24))
var amlContainerRegistryName = substring(amlAcrBase, 0, min(length(amlAcrBase), 50))
var amlAppInsightsName = toLower('appi${namePrefix}${environment}${nameSeed}')

var lakeContainers = [
  'raw'
  'bronze'
  'silver'
  'gold'
  'mlstore'
  'staging'
]

var amlDatastoreContainers = [
  'raw'
  'bronze'
  'silver'
  'gold'
  'mlstore'
]

module storage './modules/storage.bicep' = {
  name: 'storage'
  params: {
    location: location
    storageAccountName: storageAccountName
    containers: lakeContainers
  }
}

module purview './modules/purview.bicep' = {
  name: 'purview'
  params: {
    location: location
    purviewAccountName: purviewAccountName
    managedResourceGroupName: purviewManagedRgName
  }
}

module datafactory './modules/datafactory.bicep' = {
  name: 'datafactory'
  params: {
    location: location
    dataFactoryName: dataFactoryName
    purviewResourceId: purview.outputs.purviewAccountId
    storageAccountName: storage.outputs.storageAccountName
  }
}

module aml './modules/aml.bicep' = {
  name: 'aml'
  params: {
    location: location
    amlWorkspaceName: amlWorkspaceName
    lakeStorageAccountName: storage.outputs.storageAccountName
    amlStorageAccountName: amlStorageAccountName
    keyVaultName: amlKeyVaultName
    containerRegistryName: amlContainerRegistryName
    appInsightsName: amlAppInsightsName
    datastoreContainers: amlDatastoreContainers
    computeVmSize: amlComputeVmSize
    computeMinNodes: amlComputeMinNodes
    computeMaxNodes: amlComputeMaxNodes
    computeIdleTime: amlComputeIdleTime
  }
}

module fabric './modules/fabric-capacity.bicep' = if (deployFabric) {
  name: 'fabric'
  params: {
    location: location
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
    purviewPrincipalId: purview.outputs.purviewPrincipalId
    amlWorkspacePrincipalId: aml.outputs.amlWorkspacePrincipalId
    amlComputePrincipalId: aml.outputs.amlComputePrincipalId
  }
}

output storageAccountName string = storage.outputs.storageAccountName
output storageAccountId string = storage.outputs.storageAccountId
output purviewAccountName string = purview.outputs.purviewAccountName
output purviewAccountId string = purview.outputs.purviewAccountId
output dataFactoryName string = datafactory.outputs.dataFactoryName
output dataFactoryId string = datafactory.outputs.dataFactoryId
output amlWorkspaceName string = aml.outputs.amlWorkspaceName
output amlWorkspaceId string = aml.outputs.amlWorkspaceId
output amlComputeName string = aml.outputs.amlComputeName
output fabricCapacityName string = deployFabric ? fabric.outputs.capacityName : ''
output fabricCapacityId string = deployFabric ? fabric.outputs.capacityId : ''
