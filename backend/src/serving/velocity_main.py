"""Minimal FastAPI server for velocity streaming to frontend2."""
import sys
from pathlib import Path

# Ensure backend is on path
backend_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(backend_root))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import velocity

app = FastAPI(title="Velocity Stream Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(velocity.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "src.serving.velocity_main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
