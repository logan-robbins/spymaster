@description('Location for the Event Hubs namespace.')
param location string

@description('Event Hubs namespace name.')
param namespaceName string

@description('Event Hubs SKU name.')
param skuName string = 'Standard'

@description('Event Hubs capacity units.')
param skuCapacity int = 1

@description('Event hub definitions with partition counts, retention, and consumer groups.')
param eventHubs array

var consumerGroupDefinitions = reduce(
  map(eventHubs, hub => map(hub.consumerGroups, cg => {
    hubName: hub.name
    name: cg
  })),
  [],
  (acc, next) => concat(acc, next)
)

// Enforce TLS 1.2 for Event Hubs clients.
resource namespace 'Microsoft.EventHub/namespaces@2025-05-01-preview' = {
  name: namespaceName
  location: location
  sku: {
    name: skuName
    tier: skuName
    capacity: skuCapacity
  }
  properties: {
    minimumTlsVersion: '1.2'
    publicNetworkAccess: 'Enabled'
    isAutoInflateEnabled: false
  }
}

resource hubs 'Microsoft.EventHub/namespaces/eventHubs@2025-05-01-preview' = [
  for hub in eventHubs: {
    parent: namespace
    name: hub.name
    properties: {
      messageRetentionInDays: hub.messageRetentionInDays
      partitionCount: hub.partitionCount
      status: 'Active'
    }
  }
]

resource consumerGroups 'Microsoft.EventHub/namespaces/eventHubs/consumerGroups@2025-05-01-preview' = [
  for cg in consumerGroupDefinitions: {
    name: '${namespace.name}/${cg.hubName}/${cg.name}'
    properties: {}
    dependsOn: [
      hubs
    ]
  }
]

output namespaceId string = namespace.id
output namespaceName string = namespace.name
output eventHubNames array = [for hub in eventHubs: hub.name]
