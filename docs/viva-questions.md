# Academic Viva Preparation: 35 Core Questions & Answers

**Project Title:** End-to-End MLOps Pipeline for House Price Prediction  
**Subject:** Machine Learning Operations (MLOps) & Machine Learning Engineering

---

### Q1: What is MLOps?
**Answer:** MLOps (Machine Learning Operations) is a discipline that combines Machine Learning, DevOps, and Data Engineering to automate, standardize, and monitor the entire lifecycle of ML models—from data ingestion and training to testing, deployment, versioning, and continuous observability in production.

---

### Q2: Why is MLOps needed? Why not just write a Jupyter notebook?
**Answer:** While Jupyter notebooks are excellent for rapid exploration, they suffer from hidden state bugs, lack of automated testing, poor reproducibility, and zero serving capabilities. MLOps ensures models can be reliably deployed as software services, monitored for operational health, reproduced exactly via versioned data and code, and retrained automatically.

---

### Q3: Why select House Price Prediction as the problem statement?
**Answer:** Real estate valuation is a quintessential supervised regression problem with rich tabular characteristics: multi-collinearity (area vs. rooms), physical constraints (non-negative prices), categorical features with high cardinality (neighborhoods), and non-linear interactions (quality vs. size). This makes it ideal for demonstrating full-stack MLOps techniques.

---

### Q4: Why is this formulated as a Regression problem rather than Classification?
**Answer:** The target variable (`SalePrice`) is continuous in the domain of positive real numbers ($\mathbb{R}^+$). Classification predicts discrete categorical labels, whereas regression estimates exact quantitative dollar amounts.

---

### Q5: Which models were trained and compared in this project?
**Answer:** We trained four candidate architectures representing diverse mathematical paradigms:
1. **Linear Regression:** Baseline parametric linear model.
2. **Random Forest Regressor:** Non-linear bagging ensemble of decision trees.
3. **Gradient Boosting Regressor:** Sequential boosting ensemble minimizing residual error.
4. **XGBoost Regressor:** Regularized, high-performance gradient boosting.

---

### Q6: What is Overfitting and how do you detect and prevent it?
**Answer:** Overfitting occurs when a model memorizes idiosyncrasies, noise, and outliers in the training data, leading to stellar training performance but poor generalization on unseen data. It is detected when training loss is much lower than validation loss. It is prevented using cross-validation, regularization ($L_1 / L_2$), tree pruning, bagging, and early stopping.

---

### Q7: What is Data Leakage and how did your pipeline prevent it?
**Answer:** Data leakage happens when information from the test or validation set contaminates the training set before or during model fitting. In our pipeline, train/val/test splitting is performed *first*, and all transformations (median numeric imputation, standard scaling, one-hot encoding) are fitted **strictly on the training partition** and then only applied via `.transform()` to validation/test sets.

---

### Q8: Why use scikit-learn's `Pipeline` and `ColumnTransformer`?
**Answer:** A `Pipeline` bundles preprocessing transformers and the final estimator into a single atomic Python object. This guarantees that every transformation applied during training is executed identically during live API inference, eliminating discrepancies between training and serving.

---

### Q9: What is MLflow and what specific role does it play here?
**Answer:** MLflow is an open-source platform for managing the end-to-end ML lifecycle. In this project, it:
1. Tracks experiment runs (hyperparameters, training time, metrics: MAE, RMSE, $R^2$).
2. Serializes and stores trained model artifacts.
3. Provides an interactive UI on port 5000 to compare models side-by-side.
4. Manages the Model Registry to track production model versions.

---

### Q10: What is DVC (Data Version Control) and why use it alongside Git?
**Answer:** Git is designed for tracking lightweight code diffs and becomes slow or fails when storing multi-gigabyte datasets and binary model weights. DVC stores dataset and model file hashes in tiny `.dvc` and `dvc.yaml` pointers tracked by Git, while storing actual data payloads in local cache or cloud storage (S3/GCS).

---

### Q11: Explain the purpose of `dvc.yaml` and `params.yaml`.
**Answer:**
- `params.yaml`: Centralizes all configurable hyperparameters (split ratios, imputer strategies, tree counts, learning rates) so code isn't hardcoded.
- `dvc.yaml`: Defines the Directed Acyclic Graph (DAG) of pipeline stages (ingest $\rightarrow$ validate $\rightarrow$ preprocess $\rightarrow$ train $\rightarrow$ evaluate) with explicit inputs, dependencies, parameters, and output artifacts.

---

### Q12: Why use Docker in an MLOps architecture?
**Answer:** Docker packages the application code, Python runtime, system libraries, configuration files, and model artifacts into isolated, immutable containers. This eliminates the "it works on my machine" syndrome and guarantees identical behavior across developer laptops, CI/CD runners, and cloud clusters.

---

### Q13: What is the role of Docker Compose?
**Answer:** Docker Compose is an orchestration tool that defines and executes multi-container environments. In our system, a single `docker compose up --build` command launches the FastAPI backend, Streamlit frontend, MLflow UI, Prometheus scraper, and Grafana dashboard together inside a shared virtual network.

---

### Q14: Why choose FastAPI instead of Flask or Django?
**Answer:**
1. **Performance:** Asynchronous ASGI architecture powered by Starlette and Uvicorn.
2. **Type Safety & Validation:** Native integration with Pydantic validates request payloads automatically.
3. **Self-Documenting:** Automatically generates interactive OpenAPI/Swagger UI (`/docs`).

---

### Q15: Why use Streamlit for the user interface?
**Answer:** Streamlit allows machine learning engineers to build interactive, professional web dashboards in pure Python without writing HTML, CSS, or JavaScript. It communicates directly with the FastAPI backend via REST API calls.

---

### Q16: What is CI/CD and what does your GitHub Actions workflow test?
**Answer:** CI/CD (Continuous Integration / Continuous Deployment) automates building, linting, and testing code whenever changes are committed. Our `.github/workflows/ci.yml` runs:
1. Code quality & linting (Ruff, Flake8).
2. End-to-end ML pipeline execution (Ingestion $\rightarrow$ Validation $\rightarrow$ Preprocessing $\rightarrow$ Training $\rightarrow$ Evaluation).
3. Pytest unit and integration test suite.
4. Docker container build verification.

---

### Q17: What is an MLflow Model Registry?
**Answer:** A centralized model store and collaborative lifecycle management tool. It tracks models across stages (e.g., *Staging*, *Production*, *Archived*), logs model versions, and allows production deployment systems to request the current champion model by alias or tag.

---

### Q18: What is Model Drift (Concept Drift)?
**Answer:** Model drift occurs when the statistical relationship between the input features $\mathbf{X}$ and the target variable $y$ changes over time (i.e., $P(y|\mathbf{X})$ shifts). For example, macroeconomic inflation may cause a 1,500 sq ft house to sell for $400,000 instead of $250,000, causing model predictions to degrade.

---

### Q19: What is Data Drift (Feature Drift)?
**Answer:** Data drift occurs when the distribution of the input features $P(\mathbf{X})$ changes over time, even if the underlying mapping to $y$ remains relatively stable. For example, if a developer begins building modern 5,000 sq ft smart homes in a previously modest neighborhood, the model is forced to extrapolate outside its training distribution.

---

### Q20: How does the system handle retraining?
**Answer:** The pipeline supports automated retraining via `python -m src.models.train` or `dvc repro`. New data is ingested, re-validated, preprocessed with a newly fitted preprocessor, and all 4 models are retrained. MLflow logs new runs, compares validation scores, and updates `models/best_model.joblib` only if the new model outperforms the current production baseline.

---

### Q21: What is Mean Absolute Error (MAE) and what does it convey?
**Answer:** MAE is the arithmetic average of absolute differences between actual and predicted prices:
$$\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|$$
It represents the expected error on an intuitive dollar scale without penalizing extreme outliers disproportionately.

---

### Q22: What is Root Mean Squared Error (RMSE) and why is it important here?
**Answer:** RMSE is the square root of the mean squared errors:
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$$
Because errors are squared before averaging, RMSE penalizes large mispredictions heavily. In real estate appraisal, predicting a $1,000,000 mansion at $400,000 represents severe financial risk, making RMSE a critical risk metric.

---

### Q23: What does the $R^2$ score represent? Can it be negative?
**Answer:** $R^2$ (Coefficient of Determination) measures the proportion of variance in the target variable that is explained by the features relative to a naive model that always predicts the dataset mean $\bar{y}$. It ranges from $-\infty$ to $1.0$. Yes, it can be negative if the model predicts worse than the baseline average.

---

### Q24: How was the winning model selected in your pipeline?
**Answer:** The winning model was selected programmatically by evaluating all candidate architectures on the holdout validation split (`val.csv`), selecting the model that achieved the highest $R^2$ score and lowest RMSE. In our experiments, Linear Regression achieved $R^2 = 0.9704$ on validation and $0.9664$ on the holdout test set.

---

### Q25: How is the model versioned?
**Answer:** The model is versioned using three complementary mechanisms:
1. **Semantic Versioning:** Tagged in `models/model_info.json` (e.g., `v1.0.0`).
2. **MLflow Run ID:** A unique 32-character hash linking the binary artifact to exact git commits and hyperparameters.
3. **DVC Hash:** An md5 checksum stored in `dvc.lock`.

---

### Q26: How does Prometheus collect metrics from FastAPI?
**Answer:** Prometheus operates on a pull/scrape model. The FastAPI application exposes a `/metrics` endpoint using `prometheus_client`. Every 5 seconds, Prometheus sends an HTTP GET request to `/metrics` and ingests request counts, latency histograms, error rates, and prediction values into its time-series database.

---

### Q27: How does Grafana interact with Prometheus?
**Answer:** Grafana acts as the visualization layer. It queries Prometheus using PromQL (Prometheus Query Language) and renders real-time graphs displaying request throughput, P95/P99 latency trends, and error counts.

---

### Q28: How does the system handle categorical variables with unseen values in production?
**Answer:** We configure scikit-learn's `OneHotEncoder` with `handle_unknown='ignore'`. If an API client submits an unknown neighborhood or style category, the encoder outputs an all-zero vector for those one-hot columns rather than throwing an exception.

---

### Q29: What happens if a user submits negative square footage to the API?
**Answer:** FastAPI uses Pydantic schema validation (`api/schemas.py`). The `GrLivArea` field specifies `gt=0`. The API automatically rejects the invalid request with an `HTTP 422 Unprocessable Entity` status code and a clear error message before the model is ever called.

---

### Q30: How would this system scale to handle 10,000 requests per second?
**Answer:**
1. **Stateless API Replicas:** Containerize FastAPI and deploy behind an Application Load Balancer across an auto-scaling Kubernetes cluster (HPA).
2. **Worker Concurrency:** Run multiple Uvicorn workers per container pod (`--workers 4`).
3. **Caching:** Cache repeated identical queries using Redis.
4. **ONNX Runtime / TensorRT:** Convert the scikit-learn pipeline to ONNX for accelerated C++ inference.

---

### Q31: What is the purpose of Data Validation before training?
**Answer:** Data validation prevents garbage-in, garbage-out. It verifies column completeness, data types, null percentage thresholds, duplicate counts, and value bounds before training begins, saving expensive compute hours and avoiding silent pipeline failures.

---

### Q32: Why did Linear Regression perform so strongly on this dataset?
**Answer:** In housing markets, price scales in an overwhelmingly additive and linear manner with fundamental physical dimensions (square footage, basement area, bathroom count, lot size). When categorical neighborhood and condition premiums are one-hot encoded, Ordinary Least Squares accurately captures these linear step additions without suffering from tree-based discretization error.

---

### Q33: What is the difference between Batch Inference and Real-Time Inference?
**Answer:**
- **Real-Time Inference (Online):** Single request processed on-demand with sub-second latency (our `POST /predict` endpoint).
- **Batch Inference (Offline):** Scoring large batches of records at scheduled intervals (our `POST /predict/batch` endpoint or offline database jobs).

---

### Q34: What would you change if moving this system to enterprise cloud production?
**Answer:**
1. Move MLflow backend store from local SQLite to Amazon RDS PostgreSQL.
2. Store model and DVC artifacts in Amazon S3 or Google Cloud Storage.
3. Deploy FastAPI on AWS ECS or EKS with HTTPS and API Key authentication.
4. Implement Evidently AI or Great Expectations in the monitoring stack to detect data and concept drift continuously.

---

### Q35: Summarize the key achievements of your MLOps project.
**Answer:** We implemented a complete, production-grade MLOps system that moves beyond basic modeling to cover the full lifecycle: automated ingestion, schema validation, reproducible preprocessing, multi-model experiment tracking and model registration via MLflow, pipeline versioning via DVC, REST API serving via FastAPI, interactive UI via Streamlit, Docker containerization, CI/CD automation via GitHub Actions, and operational observability via Prometheus and Grafana.
