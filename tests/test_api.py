"""API integration and endpoint verification tests using FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def valid_payload():
    """Valid property feature payload matching API schema."""
    return {
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


def test_root_endpoint():
    """Verify root endpoint responds with system metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_endpoint():
    """Verify health endpoint indicates healthy status and model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "timestamp" in data


def test_model_info_endpoint():
    """Verify model-info endpoint returns metadata and validation metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "validation_metrics" in data
    assert "features" in data


def test_predict_endpoint_valid(valid_payload):
    """Verify predict endpoint succeeds on valid inputs."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 50000.0
    assert data["currency"] == "USD"
    assert "formatted_price" in data
    assert "$" in data["formatted_price"]


def test_predict_endpoint_invalid_payload():
    """Verify predict endpoint returns 422 Unprocessable Entity on invalid inputs."""
    # Case 1: OverallQual out of 1-10 bounds
    bad_payload = {"OverallQual": 99, "GrLivArea": 1000}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422

    # Case 2: Missing required features
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_batch_endpoint(valid_payload):
    """Verify batch predict endpoint handles multiple records."""
    batch_payload = {"properties": [valid_payload, valid_payload]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["predicted_price"] > 0


def test_metrics_endpoint():
    """Verify Prometheus metrics exposition endpoint returns metrics text."""
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "http_requests_total" in content or "python_gc_objects_collected_total" in content
