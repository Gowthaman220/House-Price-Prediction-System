# Testing Strategy & Verification Guide

**Project Title:** End-to-End MLOps Pipeline for House Price Prediction  
**Framework:** Pytest & FastAPI TestClient

---

## 1. Automated Testing Pyramid

The testing strategy ensures end-to-end reliability across data integrity, mathematical preprocessing, model persistence, and web serving:

```
          / \
         /   \      API Integration Tests (tests/test_api.py)
        /     \     - HTTP Status, Pydantic Schema Validation, Endpoints
       /-------\
      /         \   Model & Inference Tests (tests/test_model.py)
     /           \  - Artifact loading, metric correctness, bounds
    /-------------\
   /               \  Preprocessing & Data Tests (tests/test_data.py, test_preprocessing.py)
  /                 \ - Schema validation, null thresholds, split ratios, OHE
 /-------------------\
```

---

## 2. Test Catalog

### Data Ingestion & Validation (`tests/test_data.py`)
- `test_generate_benchmark_dataset`: Verifies generated dataset contains all required numerical, categorical, and target columns.
- `test_validate_dataset_success`: Verifies clean data passes all validation rules.
- `test_validate_dataset_missing_column`: Verifies missing required columns raise descriptive errors.
- `test_validate_dataset_empty_fails`: Verifies empty datasets raise `DataValidationError`.
- `test_validate_dataset_negative_target`: Ensures non-positive sale prices are flagged.

### Preprocessing & Splitting (`tests/test_preprocessing.py`)
- `test_split_data`: Verifies train, val, and test splits preserve total row count and exact split ratios.
- `test_preprocessor_imputation_and_encoding`: Verifies missing numerical values are filled and categorical values are one-hot encoded without NaNs.
- `test_preprocessor_unseen_category`: Confirms that unknown categorical values encountered during inference do not raise exceptions (`handle_unknown='ignore'`).

### Model & Prediction (`tests/test_model.py`)
- `test_calculate_metrics`: Asserts MAE, RMSE, and $R^2$ calculations match mathematical ground truth.
- `test_model_artifact_exists`: Verifies `models/best_model.joblib` is present.
- `test_prediction_output`: Asserts single-record predictions yield reasonable positive dollar estimates.
- `test_prediction_missing_features`: Confirms missing fields in inference payloads raise explicit `ValueError`.

### API & Servicing (`tests/test_api.py`)
- `test_root_endpoint`: Tests `GET /` metadata and 200 response.
- `test_health_endpoint`: Tests `GET /health` reports `status: "healthy"` and `model_loaded: true`.
- `test_model_info_endpoint`: Tests `GET /model-info` returns active architecture name and validation scores.
- `test_predict_endpoint_valid`: Tests `POST /predict` returns formatted price string and USD currency.
- `test_predict_endpoint_invalid_payload`: Confirms out-of-range or malformed inputs return `422 Unprocessable Entity`.
- `test_predict_batch_endpoint`: Tests batch inference arrays.
- `test_metrics_endpoint`: Tests Prometheus `/metrics` endpoint exports scrapeable metrics.

---

## 3. Running the Test Suite

Execute the entire test suite locally:
```bash
# Run all tests with verbose output
python -m pytest -v

# Run with test failure summaries only
python -m pytest -q

# Run a specific test module
python -m pytest tests/test_api.py -v
```
