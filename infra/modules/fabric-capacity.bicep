@description('Location for the Fabric capacity.')
param location string

@description('Fabric capacity name.')
param capacityName string

@description('Fabric capacity SKU name (F2, F4, etc).')
param skuName string

@description('Fabric capacity admin members (UPNs).')
param adminMembers array

// Capacity is required before creating Fabric workspaces and lakehouses.
resource capacity 'Microsoft.Fabric/capacities@2023-11-01' = {
  name: capacityName
  location: location
  sku: {
    name: skuName
    tier: 'Fabric'
  }
  properties: {
    administration: {
      members: adminMembers
    }
  }
}

output capacityName string = capacity.name
output capacityId string = capacity.id
