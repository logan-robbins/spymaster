from pydantic import BaseModel


class FeatureStoreConfig(BaseModel):
    enabled: bool = False
    project: str = "qmachina"
