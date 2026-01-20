@description('Location for the storage account.')
param location string

@description('Globally unique storage account name.')
param storageAccountName string

@description('Container names to create in the lake account.')
param containers array

// ADLS Gen2 account is the system-of-record for Bronze/Silver/Gold.
resource storage 'Microsoft.Storage/storageAccounts@2025-01-01' = {
  name: storageAccountName
  location: location
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    isHnsEnabled: true
    accessTier: 'Hot'
    minimumTlsVersion: 'TLS1_2'
  }
}

resource lakeContainers 'Microsoft.Storage/storageAccounts/blobServices/containers@2025-01-01' = [
  for containerName in containers: {
    name: '${storage.name}/default/${containerName}'
    properties: {
      publicAccess: 'None'
    }
  }
]

output storageAccountId string = storage.id
output storageAccountName string = storage.name
