# REST API Documentation & Specification

**Service Name:** House Price Prediction Serving API  
**Default Base URL:** `http://127.0.0.1:8000`  
**Interactive Swagger UI:** `http://127.0.0.1:8000/docs`  
**ReDoc UI:** `http://127.0.0.1:8000/redoc`

---

## 1. Endpoints Overview

| Method | Path | Description | Response Model |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Root endpoint providing system status and service catalog | JSON metadata |
| `GET` | `/health` | Health check reporting service readiness and model status | `HealthResponse` |
| `GET` | `/model-info` | Metadata, hyperparameters, and validation scores of active model | `ModelInfoResponse` |
| `POST` | `/predict` | Single property market valuation inference | `PredictionOutput` |
| `POST` | `/predict/batch`| Batch property market valuation inference | `BatchPredictionOutput`|
| `GET` | `/metrics` | Prometheus metrics scrape endpoint | Prometheus text/plain |

---

## 2. Endpoint Details & Examples

### 1. Root Endpoint (`GET /`)
**Sample Request:**
```bash
curl -X GET http://127.0.0.1:8000/
```
**Sample Response (HTTP 200 OK):**
```json
{
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
```

---

### 2. Service Health Check (`GET /health`)
**Sample Request:**
```bash
curl -X GET http://127.0.0.1:8000/health
```
**Sample Response (HTTP 200 OK):**
```json
{
  "status": "healthy",
  "service": "house-price-prediction-api",
  "version": "1.0.0",
  "model_loaded": true,
  "timestamp": "2026-09-24T18:59:43.998215Z"
}
```

---

### 3. Active Model Metadata (`GET /model-info`)
**Sample Request:**
```bash
curl -X GET http://127.0.0.1:8000/model-info
```
**Sample Response (HTTP 200 OK):**
```json
{
  "model_name": "Linear Regression",
  "model_version": "1.0.0",
  "run_id": "1852953269f243b18790ed093488aee6",
  "training_date": "2026-09-24T18:57:33.271113",
  "features": {
    "numerical": ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "FullBath", "YearBuilt", "YearRemodAdd", "LotArea", "Fireplaces"],
    "categorical": ["Neighborhood", "BldgType", "HouseStyle", "CentralAir"]
  },
  "validation_metrics": {
    "mae": 13104.98,
    "rmse": 16618.21,
    "r2": 0.9704,
    "training_duration": 1.138
  }
}
```

---

### 4. Single Property Prediction (`POST /predict`)
**Request Header:** `Content-Type: application/json`

**Sample Request Body:**
```json
{
  "OverallQual": 7,
  "GrLivArea": 1850.0,
  "TotalBsmtSF": 1200.0,
  "GarageCars": 2,
  "FullBath": 2,
  "YearBuilt": 2008,
  "YearRemodAdd": 2015,
  "LotArea": 9500.0,
  "Fireplaces": 1,
  "Neighborhood": "CollegeCreek",
  "BldgType": "1Fam",
  "HouseStyle": "2Story",
  "CentralAir": "Y"
}
```

**cURL Command:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "OverallQual": 7,
       "GrLivArea": 1850.0,
       "TotalBsmtSF": 1200.0,
       "GarageCars": 2,
       "FullBath": 2,
       "YearBuilt": 2008,
       "YearRemodAdd": 2015,
       "LotArea": 9500.0,
       "Fireplaces": 1,
       "Neighborhood": "CollegeCreek",
       "BldgType": "1Fam",
       "HouseStyle": "2Story",
       "CentralAir": "Y"
     }'
```

**Sample Response (HTTP 200 OK):**
```json
{
  "predicted_price": 531356.87,
  "formatted_price": "$531,356.87",
  "currency": "USD",
  "model_name": "Linear Regression",
  "model_version": "1.0.0",
  "timestamp": "2026-09-24T18:59:44.037190Z"
}
```

---

### 5. Prometheus Metrics (`GET /metrics`)
**Sample Request:**
```bash
curl -X GET http://127.0.0.1:8000/metrics
```
**Sample Response Excerpt (HTTP 200 OK, text/plain):**
```text
# HELP http_requests_total Total HTTP requests handled by the API
# TYPE http_requests_total counter
http_requests_total{endpoint="/predict",method="POST",status="200"} 42.0
# HELP http_request_duration_seconds HTTP request latency in seconds
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{endpoint="/predict",le="0.01"} 38.0
# HELP house_price_prediction_usd Distribution of house price predictions in USD
# TYPE house_price_prediction_usd histogram
house_price_prediction_usd_bucket{le="500000.0"} 18.0
```
