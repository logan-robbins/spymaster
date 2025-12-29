@description('Location for AML resources.')
param location string

@description('Azure ML workspace name.')
param amlWorkspaceName string

@description('Lake storage account name for ADLS Gen2 datastores.')
param lakeStorageAccountName string

@description('AML system storage account name.')
param amlStorageAccountName string

@description('Key Vault name for AML workspace.')
param keyVaultName string

@description('Container Registry name for AML workspace.')
param containerRegistryName string

@description('Application Insights name for AML workspace.')
param appInsightsName string

@description('Container names to register as AML datastores.')
param datastoreContainers array

@description('Compute VM size for AML AmlCompute cluster.')
param computeVmSize string

@description('Minimum node count for AML compute.')
param computeMinNodes int

@description('Maximum node count for AML compute.')
param computeMaxNodes int

@description('Idle time before AML compute scales down.')
param computeIdleTime string

var storageSuffix = environment().suffixes.storage
var storageEndpoint = storageSuffix

// AML-managed dependencies required by the workspace.
resource amlStorage 'Microsoft.Storage/storageAccounts@2025-01-01' = {
  name: amlStorageAccountName
  location: location
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    accessTier: 'Hot'
  }
}

resource amlKeyVault 'Microsoft.KeyVault/vaults@2024-11-01' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    enablePurgeProtection: true
    enableRbacAuthorization: false
    accessPolicies: []
    sku: {
      name: 'standard'
      family: 'A'
    }
  }
}

resource amlAcr 'Microsoft.ContainerRegistry/registries@2025-04-01' = {
  name: containerRegistryName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

resource amlAppInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
  }
}

// AML workspace manages ML pipelines; dependent resources are service-managed.
resource amlWorkspace 'Microsoft.MachineLearningServices/workspaces@2025-06-01' = {
  name: amlWorkspaceName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'Basic'
  }
  properties: {
    description: 'Spymaster ML workspace'
    friendlyName: 'Spymaster ML'
    keyVault: amlKeyVault.id
    applicationInsights: amlAppInsights.id
    containerRegistry: amlAcr.id
    storageAccount: amlStorage.id
  }
}

resource amlCompute 'Microsoft.MachineLearningServices/workspaces/computes@2025-06-01' = {
  name: '${amlWorkspace.name}/cpu-cluster'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: computeVmSize
      vmPriority: 'Dedicated'
      osType: 'Linux'
      scaleSettings: {
        minNodeCount: computeMinNodes
        maxNodeCount: computeMaxNodes
        nodeIdleTimeBeforeScaleDown: computeIdleTime
      }
    }
  }
}

resource amlDatastores 'Microsoft.MachineLearningServices/workspaces/datastores@2025-06-01' = [
  for containerName in datastoreContainers: {
    name: '${amlWorkspace.name}/${containerName}'
    properties: {
      datastoreType: 'AzureDataLakeGen2'
      accountName: lakeStorageAccountName
      filesystem: containerName
      protocol: 'https'
      endpoint: storageEndpoint
      resourceGroup: resourceGroup().name
      subscriptionId: subscription().subscriptionId
      serviceDataAccessAuthIdentity: 'WorkspaceSystemAssignedIdentity'
      credentials: {
        credentialsType: 'None'
      }
    }
  }
]

output amlWorkspaceId string = amlWorkspace.id
output amlWorkspaceName string = amlWorkspace.name
output amlWorkspacePrincipalId string = amlWorkspace.identity.principalId
output amlComputeName string = amlCompute.name
output amlComputePrincipalId string = amlCompute.identity.principalId
