@description('Location for the Purview account.')
param location string

@description('Purview account name.')
param purviewAccountName string

@description('Managed resource group name for Purview.')
param managedResourceGroupName string

// Purview provides the catalog and lineage backbone.
resource purview 'Microsoft.Purview/accounts@2021-12-01' = {
  name: purviewAccountName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedResourceGroupName: managedResourceGroupName
  }
}

output purviewAccountId string = purview.id
output purviewAccountName string = purview.name
output purviewPrincipalId string = purview.identity.principalId
