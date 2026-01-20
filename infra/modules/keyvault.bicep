@description('Location for the Key Vault.')
param location string

@description('Key Vault name.')
param keyVaultName string

// RBAC authorization with purge protection enforced.
resource keyVault 'Microsoft.KeyVault/vaults@2024-11-01' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    enablePurgeProtection: true
    enableRbacAuthorization: true
    accessPolicies: []
    sku: {
      name: 'standard'
      family: 'A'
    }
  }
}

output keyVaultId string = keyVault.id
output keyVaultName string = keyVault.name
