using '../main.bicep'

param location = 'eastus'
param namePrefix = 'spymaster'
param environment = 'dev'
param fabricCapacitySku = 'F2'
param fabricAdminMembers = [
  'ljrweb@gmail.com'
]
param amlComputeVmSize = 'Standard_DS3_v2'
param amlComputeMinNodes = 0
param amlComputeMaxNodes = 2
param amlComputeIdleTime = 'PT10M'
