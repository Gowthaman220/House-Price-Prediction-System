"""Tests for model loading, evaluation metrics, and inference."""

import os
from pathlib import Path
import numpy as np
import pytest
from src.models.predict import HousePricePredictor, get_predictor, predict_price
from src.models.train import calculate_metrics
from src.utils.config import get_project_root


@pytest.fixture
def sample_property():
    """Standard property dictionary for inference testing."""
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


def test_calculate_metrics():
    """Verify calculation of MAE, RMSE, and R2."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    
    metrics = calculate_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] == round(20.0 / 3.0, 2)
    assert metrics["r2"] > 0.90


def test_model_artifact_exists():
    """Ensure the trained best model artifact exists on disk."""
    root = get_project_root()
    model_path = root / "models" / "best_model.joblib"
    assert model_path.exists(), f"Model artifact missing at {model_path}"


def test_prediction_output(sample_property):
    """Verify predictor returns a valid, positive house price float."""
    pred_price = predict_price(sample_property)
    assert isinstance(pred_price, float)
    assert pred_price > 50000.0
    assert pred_price < 1500000.0


def test_prediction_missing_features(sample_property):
    """Verify predictor raises ValueError when required features are missing."""
    invalid_sample = sample_property.copy()
    del invalid_sample["OverallQual"]
    
    predictor = get_predictor()
    with pytest.raises(ValueError) as excinfo:
        predictor.predict(invalid_sample)
    assert "OverallQual" in str(excinfo.value)
