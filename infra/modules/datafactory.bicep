@description('Location for the Data Factory.')
param location string

@description('Data Factory name.')
param dataFactoryName string

@description('Purview resource ID to link for lineage.')
param purviewResourceId string

@description('Storage account name for ADLS Gen2.')
param storageAccountName string

var storageSuffix = environment().suffixes.storage
var dfsEndpoint = 'https://${storageAccountName}.dfs.${storageSuffix}'

// ADF orchestrates ingestion/backfill and triggers AML pipelines.
resource factory 'Microsoft.DataFactory/factories@2018-06-01' = {
  name: dataFactoryName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    purviewConfiguration: {
      purviewResourceId: purviewResourceId
    }
  }
}

resource adlsLinkedService 'Microsoft.DataFactory/factories/linkedservices@2018-06-01' = {
  name: '${factory.name}/LakeStorage'
  properties: {
    type: 'AzureBlobFS'
    typeProperties: {
      url: dfsEndpoint
    }
  }
}

output dataFactoryId string = factory.id
output dataFactoryName string = factory.name
output dataFactoryPrincipalId string = factory.identity.principalId
