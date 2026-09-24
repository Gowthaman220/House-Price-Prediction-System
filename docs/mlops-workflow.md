# MLOps Lifecycle & Engineering Workflow

**Project Title:** End-to-End MLOps Pipeline for House Price Prediction  
**Focus:** Full Lifecycle Traceability, Reproducibility, and Production Serving

---

## 1. The MLOps Lifecycle Explained

Traditional machine learning projects stop at a Jupyter notebook. An **MLOps system** bridges the gap between experimentation and reliable software production by establishing automated pipelines across data, model, serving, and monitoring stages:

```
[ Data Ingestion ] ──> [ Data Validation ] ──> [ Preprocessing (DVC) ]
                                                        │
                                                        ▼
[ Model Registry ] <── [ Model Selection ] <── [ Training & MLflow ]
        │
        ▼
[ FastAPI Serving ] ──> [ Streamlit UI ]
        │
        ▼
[ Prometheus Scraper ] ──> [ Grafana Dashboard ]
```

---

## 2. Why Each MLOps Tool Was Chosen

| Tool | Purpose in Pipeline | Why It Was Chosen Over Alternatives |
| :--- | :--- | :--- |
| **DVC (Data Version Control)** | Data & pipeline lineage | Git is unsuitable for multi-megabyte datasets and binary artifacts. DVC codifies reproducible DAG stages (`dvc.yaml`) while using Git for version hashes. |
| **MLflow** | Experiment tracking & model registry | Provides automated logging of metrics, hyperparameters, code commits, and serialized models. Eliminates spreadsheet-based experiment tracking. |
| **FastAPI** | High-performance inference serving | Native asynchronous support, automatic OpenAPI/Swagger interactive documentation, and sub-millisecond serialization compared to Flask or Django. |
| **Streamlit** | Interactive stakeholder interface | Enables rapid creation of clean, reactive user interfaces in pure Python without frontend JavaScript overhead. |
| **Prometheus** | Real-time metrics collection | High-throughput time-series database scraping `/metrics` to track latency distributions, error rates, and inference drift. |
| **Grafana** | Operational dashboards | Industry-standard telemetry visualization platform displaying latency quantiles (P50/P95/P99) and traffic volume. |
| **Docker & Compose** | Containerization & orchestration | Guarantees identical execution environments between developer workstations, CI runners, and production cloud servers. |
| **GitHub Actions** | Automated CI/CD | Automates linting, test suites, full ML pipeline verification, and container image builds on every pull request. |

---

## 3. Retraining Workflow & Pipeline Automation

Real-world real estate dynamics change as interest rates, inflation, and market inventory fluctuate. The system supports automated, reproducible retraining:

### Triggering Retraining Locally
```bash
# 1. Update params.yaml or add new records to data/raw/house_prices.csv
# 2. Run the complete training pipeline
python -m src.models.train
```

### Reproducing with DVC
```bash
dvc repro
```
DVC checks stage file hashes and dependencies. Only modified stages are re-executed, saving substantial compute time.

### Automated Cloud/CI Retraining Pattern
In an enterprise deployment, this workflow can be scheduled via GitHub Actions cron:
```yaml
# Conceptual GitHub Action cron trigger:
on:
  schedule:
    - cron: '0 0 1 * *' # Executes monthly retraining
```
The scheduled workflow pulls new ground-truth transaction data, executes `python -m src.models.train`, compares candidate models against the existing production champion in MLflow, and only deploys the new artifact if the new validation $R^2$ exceeds the current baseline.
