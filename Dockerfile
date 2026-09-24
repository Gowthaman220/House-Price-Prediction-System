# Multi-stage Dockerfile for House Price Prediction MLOps System
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ /app/src/
COPY api/ /app/api/
COPY app/ /app/app/
COPY configs/ /app/configs/
COPY params.yaml /app/params.yaml
COPY data/ /app/data/
COPY models/ /app/models/
COPY reports/ /app/reports/

# Default exposed ports (FastAPI=8000, Streamlit=8501)
EXPOSE 8000 8501

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command starts FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
