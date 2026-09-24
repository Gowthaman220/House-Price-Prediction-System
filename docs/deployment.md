# Deployment & Operational Runbook

**Project Title:** End-to-End MLOps Pipeline for House Price Prediction  
**Deployment Models:** Local Native Python | Docker Container | Docker Compose Multi-Service

---

## 1. Port Mappings & Service Catalog

When all services are running, the system exposes the following ports:

| Service | Internal Port | Host Port | URL | Description |
| :--- | :--- | :--- | :--- | :--- |
| **FastAPI Backend** | `8000` | `8000` | `http://localhost:8000` | REST API, Swagger docs (`/docs`) |
| **Streamlit UI** | `8501` | `8501` | `http://localhost:8501` | Interactive end-user frontend |
| **MLflow UI** | `5000` | `5000` | `http://localhost:5000` | Experiment runs & model registry |
| **Prometheus** | `9090` | `9090` | `http://localhost:9090` | Scraper & time-series metrics |
| **Grafana** | `3000` | `3000` | `http://localhost:3000` | Telemetry dashboards (`admin`/`admin`) |

---

## 2. Deployment Option A: Local Native Execution (No Docker Required)

This is the standard execution method for development, academic vivas, and local machines where Docker Desktop is not installed:

### Step 1: Clone & Setup Environment
```bash
# In PowerShell (Windows)
python -m venv .venv
.venv\Scripts\Activate.ps1

# In Bash (Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run End-to-End ML Pipeline
```bash
# 1. Ingest Data
python -m src.data.ingestion

# 2. Validate Data
python -m src.data.validation

# 3. Preprocess & Split
python -m src.features.preprocessing

# 4. Train Models & Track in MLflow
python -m src.models.train

# 5. Evaluate Best Model
python -m src.models.evaluate
```

### Step 3: Launch Services in Separate Terminals
```bash
# Terminal 1: Launch FastAPI Backend
python -m api.main

# Terminal 2: Launch Streamlit Web UI
streamlit run app/streamlit_app.py

# Terminal 3: Launch MLflow Dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## 3. Deployment Option B: Docker Compose Multi-Container Orchestration

For environments with Docker Desktop installed:

```bash
# Build and start all 5 microservices simultaneously
docker compose up --build

# Run in background (detached mode)
docker compose up -d

# View real-time container logs
docker compose logs -f api

# Stop all containers and teardown network
docker compose down
```

---

## 4. Production Cloud Readiness Checklist

For commercial production deployment to AWS (ECS/EKS), GCP (GKE/Cloud Run), or Azure (AKS):
1. **Reverse Proxy & SSL Termination:** Front FastAPI and Streamlit with Nginx or Traefik configured with Let's Encrypt TLS certificates.
2. **Model Storage:** Store `best_model.joblib` and DVC artifacts in cloud object storage (Amazon S3, Google Cloud Storage, or Azure Blob Storage).
3. **Tracking Server:** Configure MLflow with an Amazon RDS PostgreSQL backend and S3 artifact root.
4. **Horizontal Pod Autoscaling (HPA):** Scale FastAPI replica pods based on CPU utilization and request concurrency.
