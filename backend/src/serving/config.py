
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# Load env
backend_dir = Path(__file__).parent.parent.parent
load_dotenv(backend_dir / '.env')

class AppSettings(BaseModel):
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_origins: str = "*" # Comma separated
    
    # Paths
    lake_root: Path = backend_dir / "lake"
    physics_params_path: Path = Path(
        os.getenv(
            "PHYSICS_PARAMS_PATH",
            str(backend_dir / "data" / "physics" / "physics_beta_gamma.json"),
        )
    )
    
    # Databento
    databento_api_key: Optional[str] = os.getenv("DATABENTO_API_KEY")

    @classmethod
    def load(cls) -> "AppSettings":
        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            allowed_origins=os.getenv("ALLOWED_ORIGINS", "*"),
        )

settings = AppSettings.load()
