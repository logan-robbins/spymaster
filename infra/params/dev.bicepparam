using '../main.bicep'

param location = 'westus'
param namePrefix = 'spymaster'
param environment = 'dev'
param fabricCapacitySku = 'F8'
param fabricCapacityName = 'qfabric'
param fabricCapacityLocation = 'westus2'
param deployFabric = false
param fabricAdminMembers = [
  'logan@qmachina.com'
  'ljrweb@gmail.com'
]
param amlComputeVmSize = 'Standard_DS3_v2'
param amlWorkspaceName = 'mlwspymasterdevpoc'
param amlComputeMinNodes = 0
param amlComputeMaxNodes = 2
param amlComputeIdleTime = 'PT10M'
param eventHubsDataSenderRoleId = '2b629674-e913-4c01-ae53-ef4638d8f975'
param eventHubsDataReceiverRoleId = 'a638d3c7-ab3a-418d-83e6-5f17a39d4fde'
param keyVaultSecretsUserRoleId = '4633458b-17de-408a-b874-0445c86b69e6'
