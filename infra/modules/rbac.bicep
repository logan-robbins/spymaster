@description('Storage account name to scope data access roles.')
param storageAccountName string

@description('ADF managed identity principal ID.')
param dataFactoryPrincipalId string

@description('Purview managed identity principal ID.')
param purviewPrincipalId string

@description('AML workspace managed identity principal ID.')
param amlWorkspacePrincipalId string

@description('AML compute managed identity principal ID.')
param amlComputePrincipalId string

// Data plane access for storage is required for ADF/AML/Purview interoperability.
var storageBlobDataContributorRoleId = '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/ba92f5b4-2d11-453d-a403-e96b0029c9fe'
var storageBlobDataReaderRoleId = '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'

resource storage 'Microsoft.Storage/storageAccounts@2025-01-01' existing = {
  name: storageAccountName
}

resource adfStorageContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, dataFactoryPrincipalId, storageBlobDataContributorRoleId)
  scope: storage
  properties: {
    roleDefinitionId: storageBlobDataContributorRoleId
    principalId: dataFactoryPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource amlWorkspaceStorageContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, amlWorkspacePrincipalId, storageBlobDataContributorRoleId)
  scope: storage
  properties: {
    roleDefinitionId: storageBlobDataContributorRoleId
    principalId: amlWorkspacePrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource amlComputeStorageContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, amlComputePrincipalId, storageBlobDataContributorRoleId)
  scope: storage
  properties: {
    roleDefinitionId: storageBlobDataContributorRoleId
    principalId: amlComputePrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource purviewStorageReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, purviewPrincipalId, storageBlobDataReaderRoleId)
  scope: storage
  properties: {
    roleDefinitionId: storageBlobDataReaderRoleId
    principalId: purviewPrincipalId
    principalType: 'ServicePrincipal'
  }
}
