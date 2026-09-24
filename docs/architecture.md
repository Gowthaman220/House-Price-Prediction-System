# Architecture & System Design

**Project Title:** End-to-End MLOps Pipeline for House Price Prediction  
**Problem Type:** Supervised Machine Learning (Tabular Regression)  
**Target:** Residential Property Sale Price (`SalePrice` in USD)

---

## 1. High-Level Architecture Overview

The system is designed with strict separation of concerns, decoupling the **Training & Experimentation Pipeline** from the **Inference & Serving Layer**, and establishing continuous **Operational Monitoring**.

```mermaid
flowchart TD
    subgraph Data & Pipeline Layer
        A[Raw Housing Data] --> B[Data Validation Engine]
        B --> C[Preprocessing & Split Engine]
        C --> D[Processed Train/Val/Test Splits]
        C --> E[Pretrained Preprocessor Artifact]
    end

    subgraph Experimentation & Model Lifecycle
        D --> F[Model Training Stage]
        F --> G[Linear Regression]
        F --> H[Random Forest]
        F --> I[Gradient Boosting]
        F --> J[XGBoost]
        G & H & I & J --> K[MLflow Tracking Server]
        K --> L[Model Comparison & Selection]
        L --> M[Production Model Artifact - best_model.joblib]
        L --> N[MLflow Model Registry]
    end

    subgraph Serving & Consumption Layer
        M --> O[FastAPI Serving Application]
        O --> P[REST API Endpoints: /predict, /health, /model-info]
        P --> Q[Streamlit Web Frontend]
        P --> R[External Client / Mobile / Batch Consumers]
    end

    subgraph Observability & Continuous Feedback
        O --> S[Prometheus Metrics Endpoint /metrics]
        S --> T[Prometheus Time-Series DB]
        T --> U[Grafana Operational Dashboards]
    end
```

---

## 2. Training & Experimentation Pipeline (DVC & MLflow)

The model training pipeline is versioned and executed through **Data Version Control (DVC)**. Each stage specifies its explicit code dependencies, data dependencies, configuration parameters, and output artifacts:

```mermaid
graph TD
    subgraph DVC Pipeline Stages
        S1["Stage 1: Ingest (src/data/ingestion.py)"] -->|data/raw/house_prices.csv| S2["Stage 2: Validate (src/data/validation.py)"]
        S1 -->|data/raw/house_prices.csv| S3["Stage 3: Preprocess (src/features/preprocessing.py)"]
        S3 -->|data/processed/train.csv, val.csv| S4["Stage 4: Train (src/models/train.py)"]
        S4 -->|models/best_model.joblib| S5["Stage 5: Evaluate (src/models/evaluate.py)"]
        S3 -->|data/processed/test.csv| S5
    end

    subgraph Artifacts Generated
        S4 -.-> R1["MLflow Experiments (mlflow.db & mlruns/)"]
        S4 -.-> R2["models/model_info.json"]
        S5 -.-> R3["reports/evaluation_metrics.json"]
        S5 -.-> R4["reports/figures/*.png"]
    end
```

### Stage Details

1. **Ingestion Stage (`src/data/ingestion.py`):**
   - Retrieves raw real-estate data from local storage, remote URL, or generates a realistic benchmark dataset based on Ames Housing distributions.
   - Guarantees zero hard-coded paths via `configs/config.yaml`.

2. **Validation Stage (`src/data/validation.py`):**
   - Verifies column presence, data types, null percentage thresholds, and target bounds.
   - Prevents downstream pipeline failures before computing resources are allocated.

3. **Preprocessing Stage (`src/features/preprocessing.py`):**
   - Performs train/validation/test splitting (`1050 / 150 / 300` records).
   - Fits scikit-learn `ColumnTransformer` (median numeric imputation, standard scaling, categorical one-hot encoding) **strictly on training data** to prevent data leakage.
   - Persists splits and the fitted preprocessor artifact.

4. **Training Stage (`src/models/train.py`):**
   - Builds composite scikit-learn `Pipeline` objects combining preprocessor and regressors.
   - Fits 4 candidate algorithms: Linear Regression, Random Forest, Gradient Boosting, and XGBoost.
   - Logs hyperparameters, run duration, metrics (MAE, RMSE, $R^2$), and models to **MLflow**.
   - Evaluates validation metrics, selects the best performer, and saves `models/best_model.joblib`.

5. **Evaluation Stage (`src/models/evaluate.py`):**
   - Evaluates the winning model on unseen holdout test data (`test.csv`).
   - Generates publication-ready figures under `reports/figures/` (Actual vs Predicted, Residuals Distribution, Model Comparison).

---

## 3. Real-Time Inference Architecture

```mermaid
sequenceDiagram
    autonumber
    actor User as Real Estate Analyst
    participant UI as Streamlit Web App (Port 8501)
    participant API as FastAPI Backend (Port 8000)
    participant Model as Pipeline Engine (best_model.joblib)
    participant Prom as Prometheus (Port 9090)

    User->>UI: Selects property features (sqft, quality, neighborhood, etc.)
    User->>UI: Clicks "Calculate Estimated House Price"
    UI->>API: HTTP POST /predict (JSON payload)
    Note over API: Pydantic validates ranges & data types
    API->>Model: Passes DataFrame to Pipeline.predict()
    Note over Model: Preprocessing (Impute, Scale, OHE) + Regression
    Model-->>API: Returns numpy array [estimated_price]
    API->>Prom: Increments request counter & records latency / value histogram
    API-->>UI: HTTP 200 JSON { predicted_price, formatted_price, model_name, timestamp }
    UI-->>User: Renders valuation card ($531,356.87 USD) and feature drivers
```

---

## 4. Container & Service Topology (Docker Compose)

```mermaid
graph LR
    subgraph Docker Network: mlops-net
        C1[house_price_api :8000]
        C2[house_price_streamlit :8501]
        C3[house_price_mlflow :5000]
        C4[house_price_prometheus :9090]
        C5[house_price_grafana :3000]
    end

    User((User / Browser)) -->|Port 8501| C2
    C2 -->|Internal HTTP| C1
    User -->|Port 8000 (Swagger)| C1
    User -->|Port 5000 (MLflow UI)| C3
    C4 -->|Scrapes /metrics every 5s| C1
    C5 -->|PromQL Queries| C4
    User -->|Port 3000 (Dashboards)| C5
```
