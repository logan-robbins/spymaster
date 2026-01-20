from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
import uuid
import time

subscription_id = "70464868-52ea-435d-93a6-8002e83f0b89"
resource_group = "rg-spymaster-dev"
workspace_name = "mlwspymasterdevpoc"

credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

unique_suffix = str(uuid.uuid4())[:6]
endpoint_name = f"es-model-{unique_suffix}"
print(f"Creating endpoint: {endpoint_name}")

endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="ES model inference endpoint",
    auth_mode="key",
)

try:
    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    print(f"Endpoint {endpoint_name} created successfully")
except Exception as e:
    print(f"Failed to create endpoint: {e}")
    exit(1)

endpoint = ml_client.online_endpoints.get(name=endpoint_name)
print(f"Endpoint provisioning state: {endpoint.provisioning_state}")

if endpoint.provisioning_state != "Succeeded":
    print(f"Endpoint failed to provision: {endpoint.provisioning_state}")
    exit(1)

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model="es_logreg_model:1",
    code_configuration=CodeConfiguration(
        code="endpoints/es_model",
        scoring_script="score.py",
    ),
    environment=Environment(
        name="es-inference-env",
        version="1",
        conda_file="endpoints/es_model/conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

try:
    ml_client.online_deployments.begin_create_or_update(deployment).wait()
    print(f"Deployment blue created successfully")
except Exception as e:
    print(f"Failed to create deployment: {e}")

endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).wait()

endpoint = ml_client.online_endpoints.get(name=endpoint_name)
print(f"Endpoint scoring URI: {endpoint.scoring_uri}")
print(f"Endpoint keys: {ml_client.online_endpoints.get_keys(endpoint_name)}")
