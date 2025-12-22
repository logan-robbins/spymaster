# Dockerfile for Spymaster Backend Services
# Supports: Ingestor, Core, Lake, Gateway

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy application code
COPY . .

# Default command (can be overridden in docker-compose.yml)
CMD ["uv", "run", "python", "-m", "src.core.main"]

