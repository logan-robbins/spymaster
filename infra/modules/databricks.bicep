@description('Location for Databricks workspace resources.')
param location string

@description('Databricks workspace name.')
param workspaceName string

@description('Managed resource group name for the workspace.')
param managedResourceGroupName string

@description('Databricks workspace SKU name.')
param skuName string = 'premium'

@description('Databricks public network access setting.')
param publicNetworkAccess string = 'Enabled'

@description('Databricks access connector name.')
param accessConnectorName string

resource accessConnector 'Microsoft.Databricks/accessConnectors@2024-05-01' = {
  name: accessConnectorName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {}
}

resource workspace 'Microsoft.Databricks/workspaces@2024-05-01' = {
  name: workspaceName
  location: location
  sku: {
    name: skuName
    tier: skuName
  }
  properties: {
    managedResourceGroupId: subscriptionResourceId('Microsoft.Resources/resourceGroups', managedResourceGroupName)
    publicNetworkAccess: publicNetworkAccess
  }
}

output workspaceId string = workspace.id
output workspaceName string = workspace.name
output workspaceUrl string = workspace.properties.workspaceUrl
output accessConnectorId string = accessConnector.id
output accessConnectorPrincipalId string = accessConnector.identity.principalId
