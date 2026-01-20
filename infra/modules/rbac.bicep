@description('Storage account name to scope data access roles.')
param storageAccountName string

@description('ADF managed identity principal ID.')
param dataFactoryPrincipalId string

@description('AML workspace managed identity principal ID.')
param amlWorkspacePrincipalId string

@description('AML compute managed identity principal ID.')
param amlComputePrincipalId string

@description('Databricks access connector principal ID.')
param databricksAccessConnectorPrincipalId string

@description('Event Hubs namespace name.')
param eventHubsNamespaceName string

@description('Runtime Key Vault name.')
param runtimeKeyVaultName string

@description('Azure Event Hubs Data Sender role definition GUID.')
param eventHubsDataSenderRoleId string

@description('Azure Event Hubs Data Receiver role definition GUID.')
param eventHubsDataReceiverRoleId string

@description('Key Vault Secrets User role definition GUID.')
param keyVaultSecretsUserRoleId string

// Data plane access for storage is required for ADF/AML/Purview interoperability.
var storageBlobDataContributorRoleId = '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/ba92f5b4-2d11-453d-a403-e96b0029c9fe'
var eventHubsDataSenderRoleDefinitionId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', eventHubsDataSenderRoleId)
var eventHubsDataReceiverRoleDefinitionId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', eventHubsDataReceiverRoleId)
var keyVaultSecretsUserRoleDefinitionId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', keyVaultSecretsUserRoleId)

resource storage 'Microsoft.Storage/storageAccounts@2025-01-01' existing = {
  name: storageAccountName
}

resource eventHubsNamespace 'Microsoft.EventHub/namespaces@2025-05-01-preview' existing = {
  name: eventHubsNamespaceName
}

resource runtimeKeyVault 'Microsoft.KeyVault/vaults@2024-11-01' existing = {
  name: runtimeKeyVaultName
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

resource databricksAccessConnectorStorageContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, databricksAccessConnectorPrincipalId, storageBlobDataContributorRoleId)
  scope: storage
  properties: {
    roleDefinitionId: storageBlobDataContributorRoleId
    principalId: databricksAccessConnectorPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource databricksEventHubsSender 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(eventHubsNamespace.id, databricksAccessConnectorPrincipalId, eventHubsDataSenderRoleDefinitionId)
  scope: eventHubsNamespace
  properties: {
    roleDefinitionId: eventHubsDataSenderRoleDefinitionId
    principalId: databricksAccessConnectorPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource databricksEventHubsReceiver 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(eventHubsNamespace.id, databricksAccessConnectorPrincipalId, eventHubsDataReceiverRoleDefinitionId)
  scope: eventHubsNamespace
  properties: {
    roleDefinitionId: eventHubsDataReceiverRoleDefinitionId
    principalId: databricksAccessConnectorPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource adfKeyVaultSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(runtimeKeyVault.id, dataFactoryPrincipalId, keyVaultSecretsUserRoleDefinitionId)
  scope: runtimeKeyVault
  properties: {
    roleDefinitionId: keyVaultSecretsUserRoleDefinitionId
    principalId: dataFactoryPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource databricksKeyVaultSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(runtimeKeyVault.id, databricksAccessConnectorPrincipalId, keyVaultSecretsUserRoleDefinitionId)
  scope: runtimeKeyVault
  properties: {
    roleDefinitionId: keyVaultSecretsUserRoleDefinitionId
    principalId: databricksAccessConnectorPrincipalId
    principalType: 'ServicePrincipal'
  }
}
