
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting SpyMaster HUD Server on {settings.host}:{settings.port}")
    yield
    # Shutdown
    print("Shutting down SpyMaster HUD Server")

app = FastAPI(
    title="SpyMaster HUD API",
    version="0.1.0",
    lifespan=lifespan
)

# CORS
origins = settings.allowed_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "env": settings.app_env}

# Register routers
from .routers import hud
app.include_router(hud.router)

if __name__ == "__main__":
    uvicorn.run(
        "src.serving.main:app",
        host=settings.host,
        port=settings.port,
        reload=(settings.app_env == "development")
    )
