"""FastAPI Model Serving Application for House Price Prediction."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from api.schemas import (
    BatchHouseFeaturesInput,
    BatchPredictionOutput,
    HealthResponse,
    HouseFeaturesInput,
    ModelInfoResponse,
    PredictionOutput,
)
from src.models.predict import get_predictor
from src.utils.config import get_logger, get_project_root, load_config

logger = get_logger("fastapi_service")
root = get_project_root()
config = load_config()

# Initialize FastAPI application
app = FastAPI(
    title="End-to-End MLOps Pipeline for House Price Prediction",
    description=(
        "Production-grade REST API serving machine learning house price predictions "
        "with complete MLOps lifecycle: automated validation, experiment tracking, "
        "and Prometheus monitoring."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend applications (Streamlit, local dev, Docker)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics Definitions
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the API",
    ["method", "endpoint", "status"]
)
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"]
)
PREDICTIONS_TOTAL = Counter(
    "house_price_predictions_total",
    "Total number of property price predictions generated"
)
PREDICTION_ERRORS_TOTAL = Counter(
    "house_price_prediction_errors_total",
    "Total number of prediction failures"
)
PREDICTION_VALUE_HISTOGRAM = Histogram(
    "house_price_prediction_usd",
    "Distribution of house price predictions in USD",
    buckets=[100000, 200000, 300000, 400000, 500000, 600000, 750000, 1000000]
)


# HTTP Request Logging & Latency Middleware
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Middleware to measure latency and record Prometheus metrics for all endpoints."""
    start_time = time.time()
    endpoint = request.url.path
    method = request.method

    try:
        response = await call_next(request)
        duration = time.time() - start_time
        status_code = str(response.status_code)
        
        # Don't clutter logs with Prometheus scrape requests
        if endpoint != "/metrics":
            HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status_code).inc()
            HTTP_REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
            logger.info("%s %s completed in %.4fs (Status: %s)", method, endpoint, duration, status_code)
            
        return response
    except Exception as exc:
        duration = time.time() - start_time
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status="500").inc()
        logger.error("%s %s failed after %.4fs: %s", method, endpoint, duration, exc)
        raise exc


@app.get("/", tags=["General"])
def root_endpoint() -> Dict[str, Any]:
    """Root endpoint providing system metadata and active endpoints."""
    return {
        "title": "End-to-End MLOps Pipeline for House Price Prediction",
        "version": "1.0.0",
        "description": "Production API for House Price Prediction and MLOps Lifecycle Demonstration",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Service health check endpoint verifying model artifact availability."""
    model_loaded = False
    try:
        predictor = get_predictor()
        model_loaded = predictor.model is not None
    except Exception as e:
        logger.warning("Health check detected model loading issue: %s", e)

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        service="house-price-prediction-api",
        version="1.0.0",
        model_loaded=model_loaded,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
def get_model_information() -> ModelInfoResponse:
    """Retrieve detailed metadata, hyperparameters, and validation metrics of the active model."""
    info_path = root / Path(config.get("models", {}).get("model_info_path", "models/model_info.json"))
    if not info_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model info metadata file not found. Ensure training has been executed."
        )

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ModelInfoResponse(**data)
    except Exception as err:
        logger.error("Failed to read model info metadata: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading model information: {str(err)}"
        )


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict(features: HouseFeaturesInput) -> PredictionOutput:
    """Predict the sale price of a single residential property given its features."""
    try:
        predictor = get_predictor()
        input_dict = features.model_dump()
        
        # Run inference
        predicted_price = float(predictor.predict(input_dict)[0])
        
        # Record Prometheus metrics
        PREDICTIONS_TOTAL.inc()
        PREDICTION_VALUE_HISTOGRAM.observe(predicted_price)

        # Retrieve model name
        info_path = root / Path(config.get("models", {}).get("model_info_path", "models/model_info.json"))
        model_name = "Best Estimator Pipeline"
        model_version = "1.0.0"
        if info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    minfo = json.load(f)
                    model_name = minfo.get("model_name", model_name)
                    model_version = minfo.get("model_version", model_version)
            except Exception:
                pass

        return PredictionOutput(
            predicted_price=predicted_price,
            formatted_price=f"${predicted_price:,.2f}",
            currency="USD",
            model_name=model_name,
            model_version=model_version,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except ValueError as val_err:
        PREDICTION_ERRORS_TOTAL.inc()
        logger.error("Validation error during prediction: %s", val_err)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as exc:
        PREDICTION_ERRORS_TOTAL.inc()
        logger.error("Unexpected prediction failure: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {str(exc)}")


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Inference"])
def predict_batch(batch_input: BatchHouseFeaturesInput) -> BatchPredictionOutput:
    """Predict sale prices for multiple residential properties in batch."""
    try:
        predictor = get_predictor()
        input_dicts = [item.model_dump() for item in batch_input.properties]
        
        # Run batch inference
        predictions = predictor.predict(input_dicts)
        
        # Retrieve model metadata
        info_path = root / Path(config.get("models", {}).get("model_info_path", "models/model_info.json"))
        model_name = "Best Estimator Pipeline"
        model_version = "1.0.0"
        if info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    minfo = json.load(f)
                    model_name = minfo.get("model_name", model_name)
                    model_version = minfo.get("model_version", model_version)
            except Exception:
                pass

        results = []
        now = datetime.now(timezone.utc).isoformat()
        for price in predictions:
            p = float(price)
            PREDICTIONS_TOTAL.inc()
            PREDICTION_VALUE_HISTOGRAM.observe(p)
            results.append(PredictionOutput(
                predicted_price=p,
                formatted_price=f"${p:,.2f}",
                currency="USD",
                model_name=model_name,
                model_version=model_version,
                timestamp=now
            ))

        return BatchPredictionOutput(
            predictions=results,
            total_count=len(results)
        )
    except Exception as exc:
        PREDICTION_ERRORS_TOTAL.inc()
        logger.error("Batch prediction failure: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Batch inference failed: {str(exc)}")


@app.get("/metrics", tags=["Monitoring"])
def metrics() -> Response:
    """Prometheus exposition format metrics endpoint for scrapers."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def start():
    """CLI starter for uvicorn server."""
    import uvicorn
    host = os.getenv("API_HOST", config.get("api", {}).get("host", "127.0.0.1"))
    port = int(os.getenv("API_PORT", config.get("api", {}).get("port", 8000)))
    logger.info("Starting FastAPI server at http://%s:%d", host, port)
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start()
