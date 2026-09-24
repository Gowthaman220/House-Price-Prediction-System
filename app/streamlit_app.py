"""Streamlit Frontend Application for House Price Prediction MLOps System."""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Configure page metadata and aesthetic layout
st.set_page_config(
    page_title="End-to-End MLOps Pipeline for House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .price-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .price-val {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .status-badge-green {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-red {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Verify backend FastAPI connectivity or fallback to embedded model."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.status_code == 200:
            return True, resp.json(), "api"
    except Exception:
        pass

    # Check if local model artifact is available for standalone cloud deployment
    model_file = Path("models/best_model.joblib")
    if model_file.exists():
        return True, {"model_loaded": True, "service": "embedded-engine"}, "embedded"
    return False, None, "offline"


def get_model_info():
    """Fetch active model metadata from FastAPI or local model_info.json."""
    try:
        resp = requests.get(f"{API_URL}/model-info", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    info_file = Path("models/model_info.json")
    if info_file.exists():
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main():
    # Header Section
    st.markdown('<div class="main-title">🏡 End-to-End MLOps Pipeline for House Price Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">A complete, production-grade MLOps system demonstrating data validation, '
        'reproducible preprocessing, experiment tracking (MLflow), data versioning (DVC), containerization (Docker), '
        'REST API serving (FastAPI), and monitoring (Prometheus).</div>',
        unsafe_allow_html=True
    )

    # Sidebar: System Status & Architecture Overview
    is_healthy, health_data, engine_mode = check_api_health()
    model_metadata = get_model_info()

    with st.sidebar:
        st.subheader("System Status")
        if engine_mode == "api":
            st.markdown('<span class="status-badge-green">● FastAPI Backend: Connected</span>', unsafe_allow_html=True)
            st.caption(f"Endpoint: `{API_URL}`")
            if health_data and health_data.get("model_loaded"):
                st.caption("Model Artifact: **Loaded in Memory**")
        elif engine_mode == "embedded":
            st.markdown('<span class="status-badge-green">● Model Engine: Embedded (Online)</span>', unsafe_allow_html=True)
            st.caption("Mode: **Standalone Cloud Serving**")
            st.caption("Artifact: `models/best_model.joblib`")
        else:
            st.markdown('<span class="status-badge-red">● Backend & Model: Offline</span>', unsafe_allow_html=True)
            st.warning(
                f"Cannot reach `{API_URL}` or local model.\n\n"
                "Please run training:\n"
                "```bash\npython -m src.models.train\n```"
            )

        st.divider()

        # Active Model Info
        st.subheader("Active Model Info")
        if model_metadata:
            st.markdown(f"**Selected Model:** `{model_metadata.get('model_name')}`")
            st.markdown(f"**Version:** `{model_metadata.get('model_version')}`")
            v_metrics = model_metadata.get("validation_metrics", {})
            st.markdown(f"**Validation $R^2$:** `{v_metrics.get('r2', 'N/A')}`")
            st.markdown(f"**Validation RMSE:** `${v_metrics.get('rmse', 0):,.2f}`")
            st.markdown(f"**Validation MAE:** `${v_metrics.get('mae', 0):,.2f}`")
            st.caption(f"Trained on: {model_metadata.get('training_date', '')[:10]}")
        else:
            st.info("Start the FastAPI server to inspect model metadata.")

        st.divider()
        st.subheader("MLOps Stack")
        st.markdown("""
        - **Tracking:** MLflow
        - **Pipeline:** DVC
        - **API:** FastAPI & Uvicorn
        - **UI:** Streamlit
        - **Monitoring:** Prometheus & Grafana
        - **CI/CD:** GitHub Actions
        - **Containers:** Docker & Compose
        """)

    # Main Tabs: 1. Price Prediction | 2. Model Comparison & Metrics | 3. MLOps Architecture
    tab_predict, tab_models, tab_arch = st.tabs(["🎯 House Price Prediction", "📊 Model Comparison & Experiments", "⚙️ MLOps Lifecycle"])

    with tab_predict:
        st.markdown("### Enter Property Characteristics")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 📐 Physical Dimensions")
            gr_liv_area = st.number_input(
                "Ground Living Area (sq ft)",
                min_value=500.0, max_value=8000.0, value=1850.0, step=25.0,
                help="Above grade (ground) living area in square feet"
            )
            total_bsmt_sf = st.number_input(
                "Total Basement Area (sq ft)",
                min_value=0.0, max_value=4000.0, value=1200.0, step=25.0,
                help="Total square feet of basement"
            )
            lot_area = st.number_input(
                "Lot Size (sq ft)",
                min_value=1000.0, max_value=80000.0, value=9500.0, step=100.0,
                help="Lot area in square feet"
            )

        with col2:
            st.markdown("##### ⭐ Quality & Timing")
            overall_qual = st.slider(
                "Overall Quality (1-10)",
                min_value=1, max_value=10, value=7,
                help="1=Very Poor, 5=Average, 7=Good, 10=Very Excellent"
            )
            year_built = st.number_input(
                "Year Built",
                min_value=1900, max_value=datetime.now().year, value=2008, step=1,
                help="Original construction year"
            )
            year_remod = st.number_input(
                "Year Remodeled",
                min_value=1950, max_value=datetime.now().year, value=2015, step=1,
                help="Remodel date (or year built if no remodel)"
            )

        with col3:
            st.markdown("##### 🚗 Amenities & Features")
            garage_cars = st.slider("Garage Car Capacity", min_value=0, max_value=4, value=2)
            full_bath = st.slider("Full Bathrooms", min_value=1, max_value=5, value=2)
            fireplaces = st.slider("Fireplaces", min_value=0, max_value=4, value=1)

        st.markdown("##### 📍 Location & Structure Style")
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        
        with c_col1:
            neighborhood = st.selectbox(
                "Neighborhood",
                options=["NorthAmes", "CollegeCreek", "OldTown", "Edwards", "Somerset",
                         "Gilbert", "Sawyer", "Northridge", "Crawford", "Mitchell"],
                index=1
            )
        with c_col2:
            bldg_type = st.selectbox(
                "Building Type",
                options=["1Fam", "TwnhsE", "Twnhs", "Duplex", "2fmCon"],
                index=0,
                help="1Fam: Single Family, Twnhs: Townhouse, Duplex: 2 Units"
            )
        with c_col3:
            house_style = st.selectbox(
                "House Style",
                options=["1Story", "2Story", "1.5Fin", "SLvl"],
                index=1,
                help="1Story, 2Story, 1.5 Finished, Split Level"
            )
        with c_col4:
            central_air = st.radio("Central Air Conditioning", options=["Y", "N"], horizontal=True, index=0)

        st.markdown("<br>", unsafe_allow_html=True)
        
        predict_button = st.button("🚀 Calculate Estimated House Price", type="primary", use_container_width=True)

        if predict_button:
            payload = {
                "OverallQual": int(overall_qual),
                "GrLivArea": float(gr_liv_area),
                "TotalBsmtSF": float(total_bsmt_sf),
                "GarageCars": int(garage_cars),
                "FullBath": int(full_bath),
                "YearBuilt": int(year_built),
                "YearRemodAdd": int(year_remod),
                "LotArea": float(lot_area),
                "Fireplaces": int(fireplaces),
                "Neighborhood": neighborhood,
                "BldgType": bldg_type,
                "HouseStyle": house_style,
                "CentralAir": central_air
            }

            if not is_healthy:
                st.error("Model engine is offline. Please verify trained model artifacts exist.")
            else:
                with st.spinner("Executing inference pipeline..."):
                    pred_data = None
                    if engine_mode == "api":
                        try:
                            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                            if resp.status_code == 200:
                                pred_data = resp.json()
                        except Exception:
                            pred_data = None

                    if pred_data is None:
                        try:
                            from src.models.predict import predict_price
                            predicted_val = predict_price(payload)
                            m_info = model_metadata or {}
                            pred_data = {
                                "predicted_price": predicted_val,
                                "formatted_price": f"${predicted_val:,.2f}",
                                "currency": "USD",
                                "model_name": m_info.get("model_name", "Linear Regression"),
                                "model_version": m_info.get("model_version", "1.0.0"),
                            }
                        except Exception as err:
                            st.error(f"Inference error: {err}")

                    if pred_data:
                        pred_price = pred_data.get("predicted_price", 0)
                        formatted = pred_data.get("formatted_price", f"${pred_price:,.2f}")
                        model_used = pred_data.get("model_name", "Best Model")
                        
                        st.markdown(f"""
                        <div class="price-card">
                            <div style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 0.3rem;">Estimated Market Valuation</div>
                            <div class="price-val">{formatted}</div>
                            <div style="font-size: 0.9rem; opacity: 0.85; margin-top: 0.5rem;">
                                Inference served by <b>{model_used}</b> (v{pred_data.get('model_version', '1.0')})
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Prediction Breakdown / Explanatory insights
                        st.markdown("#### 💡 Valuation Breakdown & Key Drivers")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Living Space Contribution", f"{gr_liv_area:,.0f} sqft", f"Quality Rating: {overall_qual}/10")
                        col_b.metric("Basement & Garage", f"{total_bsmt_sf:,.0f} sqft basement", f"{garage_cars} Car Garage")
                        col_c.metric("Location & Age", f"{neighborhood}", f"Built in {year_built}")

    with tab_models:
        st.markdown("### Model Comparison & Experiment Tracking")
        st.markdown(
            "During the experiment tracking phase, four candidate architectures were trained and evaluated "
            "using cross-validation and holdout validation sets. All parameters, metrics, and serialized artifacts "
            "were logged into **MLflow**."
        )

        if model_metadata and "all_model_comparison" in model_metadata:
            comp = model_metadata["all_model_comparison"]
            rows = []
            for name, m in comp.items():
                rows.append({
                    "Model": name,
                    "Validation R²": f"{m.get('r2', 0):.4f}",
                    "Validation RMSE (USD)": f"${m.get('rmse', 0):,.2f}",
                    "Validation MAE (USD)": f"${m.get('mae', 0):,.2f}",
                    "Training Duration (s)": f"{m.get('training_duration', 0):.2f}s",
                    "MLflow Run ID": m.get("run_id", "N/A")[:12] + "..."
                })
            df_comp = pd.DataFrame(rows)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            st.success(f"🏆 Winning Architecture: **{model_metadata.get('model_name')}** selected automatically based on highest $R^2$ and lowest RMSE.")
        else:
            st.info("Run `python -m src.models.train` to generate the comparison matrix.")

        st.markdown("#### Evaluation Artifacts (from `reports/figures/`)")
        rep_col1, rep_col2 = st.columns(2)
        
        act_pred_img = "reports/figures/actual_vs_predicted.png"
        comp_img = "reports/figures/model_comparison.png"
        residuals_img = "reports/figures/residuals_distribution.png"

        if os.path.exists(act_pred_img):
            rep_col1.image(act_pred_img, caption="Actual vs. Predicted House Prices on Holdout Test Set")
        if os.path.exists(comp_img):
            rep_col2.image(comp_img, caption="Candidate Model Comparison (R² and RMSE)")
        if os.path.exists(residuals_img):
            st.image(residuals_img, caption="Residual Error Distribution", width=700)

    with tab_arch:
        st.markdown("### End-to-End MLOps Architecture")
        st.markdown("""
        The system implements the full industrial machine learning lifecycle:
        """)
        
        st.code("""
        +------------------------------------------------------------------------------------+
        |                               MLOps System Architecture                            |
        +------------------------------------------------------------------------------------+
        
             [Raw Data]
                 |
                 v
        +------------------+
        | Data Ingestion   |  -> Configurable source (Local / Kaggle / URL)
        +------------------+
                 |
                 v
        +------------------+
        | Data Validation  |  -> Great Expectations / Schema integrity checks
        +------------------+
                 |
                 v
        +------------------+
        | Preprocessing    |  -> Scikit-learn ColumnTransformer (Imputation, Scaling, OHE)
        | & DVC Pipeline   |  -> DVC tracks data & stage dependencies (dvc.yaml)
        +------------------+
                 |
                 v
        +------------------+
        | Model Training   |  -> Linear Regression, Random Forest, Gradient Boosting, XGBoost
        | & MLflow         |  -> Tracks parameters, metrics (MAE/RMSE/R2), duration, artifacts
        +------------------+
                 |
                 v
        +------------------+
        | Model Registry   |  -> Best model selected & registered ('HousePriceBestModel')
        +------------------+
                 |
                 v
        +------------------+
        | FastAPI Serving  |  -> Pydantic schema validation, /predict, /health, /model-info
        +------------------+
                 |
         +-------+-------+
         |               |
         v               v
    +---------+    +------------+
    |Streamlit|    | Prometheus | -> Scrapes /metrics (Latency, Requests, Value distribution)
    |Frontend |    | & Grafana  | -> Real-time operational observability dashboards
    +---------+    +------------+
        """, language="text")

        st.markdown("""
        #### Tool Roles & Rationale
        - **DVC (Data Version Control):** Enforces reproducible pipelines (`dvc.yaml`) and tracks data versions alongside Git.
        - **MLflow:** Tracks experiment runs, hyperparameter tuning, model artifacts, and registers the winning model.
        - **FastAPI:** High-performance, asynchronous REST API with automatic Swagger UI documentation and input validation.
        - **Streamlit:** Interactive web application for non-technical stakeholders and viva presentation.
        - **Prometheus & Grafana:** Continuous operational monitoring for latency, error rate, request throughput, and prediction distributions.
        - **Docker & Docker Compose:** Encapsulates dependencies for 100% reproducible deployment anywhere.
        """)


if __name__ == "__main__":
    main()
