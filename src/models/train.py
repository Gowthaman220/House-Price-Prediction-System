"""Model training and experiment tracking with MLflow for House Price Prediction."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.features.preprocessing import build_preprocessor
from src.utils.config import get_logger, get_project_root, load_config, load_params

logger = get_logger("model_training")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics: MAE, RMSE, R2.
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
        
    Returns:
        Dictionary of calculated metrics rounded to 4 decimals.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }


def get_candidate_models(params: Dict[str, Any]) -> Dict[str, Any]:
    """Instantiate candidate regression models using hyperparameters from params.yaml.
    
    Args:
        params: Parameters dictionary.
        
    Returns:
        Dictionary of {model_name: model_instance}.
    """
    train_cfg = params.get("train", {})
    models_cfg = train_cfg.get("models", {})
    random_state = train_cfg.get("random_state", 42)

    # 1. Linear Regression
    lr_cfg = models_cfg.get("linear_regression", {})
    lr = LinearRegression(fit_intercept=lr_cfg.get("fit_intercept", True))

    # 2. Random Forest Regressor
    rf_cfg = models_cfg.get("random_forest", {})
    rf = RandomForestRegressor(
        n_estimators=rf_cfg.get("n_estimators", 100),
        max_depth=rf_cfg.get("max_depth", 15),
        min_samples_split=rf_cfg.get("min_samples_split", 4),
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
        random_state=rf_cfg.get("random_state", random_state),
        n_jobs=-1
    )

    # 3. Gradient Boosting Regressor
    gb_cfg = models_cfg.get("gradient_boosting", {})
    gb = GradientBoostingRegressor(
        n_estimators=gb_cfg.get("n_estimators", 120),
        learning_rate=gb_cfg.get("learning_rate", 0.08),
        max_depth=gb_cfg.get("max_depth", 4),
        min_samples_split=gb_cfg.get("min_samples_split", 3),
        random_state=gb_cfg.get("random_state", random_state)
    )

    # 4. XGBoost Regressor
    xgb_cfg = models_cfg.get("xgboost", {})
    xgb = XGBRegressor(
        n_estimators=xgb_cfg.get("n_estimators", 120),
        learning_rate=xgb_cfg.get("learning_rate", 0.08),
        max_depth=xgb_cfg.get("max_depth", 4),
        subsample=xgb_cfg.get("subsample", 0.8),
        colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
        random_state=xgb_cfg.get("random_state", random_state),
        verbosity=0
    )

    return {
        "Linear Regression": lr,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "XGBoost": xgb
    }


def train_and_evaluate_models(
    train_path: Optional[str] = None,
    val_path: Optional[str] = None
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Train all candidate models, log to MLflow, and select the best model based on validation metrics.
    
    Args:
        train_path: Optional path to train.csv.
        val_path: Optional path to val.csv.
        
    Returns:
        Tuple of (best_model_name, best_metrics, model_comparison_dict).
    """
    root = get_project_root()
    config = load_config()
    params = load_params()

    # Configure MLflow
    mlflow_uri = config.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = config.get("mlflow", {}).get("experiment_name", "house-price-prediction")
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow configured: URI=%s, Experiment=%s", mlflow_uri, experiment_name)

    # Load data
    train_file = root / Path(train_path or config.get("data", {}).get("train_path", "data/processed/train.csv"))
    val_file = root / Path(val_path or config.get("data", {}).get("val_path", "data/processed/val.csv"))

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(f"Train/Val files not found at {train_file} or {val_file}")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    num_cols = config.get("features", {}).get("numerical", [])
    cat_cols = config.get("features", {}).get("categorical", [])
    target_col = config.get("data", {}).get("target_column", "SalePrice")

    feature_cols = num_cols + cat_cols
    X_train, y_train = train_df[feature_cols], train_df[target_col].values
    X_val, y_val = val_df[feature_cols], val_df[target_col].values

    candidate_models = get_candidate_models(params)
    logger.info("Candidate models ready for training: %s", list(candidate_models.keys()))

    results = {}
    fitted_pipelines = {}
    run_ids = {}

    for name, model_instance in candidate_models.items():
        logger.info("--- Starting training for: %s ---", name)
        
        # Build end-to-end Pipeline: preprocessor + model
        preprocessor = build_preprocessor(
            numerical_features=num_cols,
            categorical_features=cat_cols,
            num_strategy=params.get("preprocessing", {}).get("numerical_imputer_strategy", "median"),
            cat_strategy=params.get("preprocessing", {}).get("categorical_imputer_strategy", "most_frequent"),
            scale_numeric=True
        )
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model_instance)
        ])

        start_time = time.time()
        
        with mlflow.start_run(run_name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id
            run_ids[name] = run_id
            
            # Log tags and params
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("dataset_version", "v1.0")
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            mlflow.log_param("features_count", len(feature_cols))
            
            # Log model-specific hyperparameters
            if hasattr(model_instance, "get_params"):
                clean_params = {k: v for k, v in model_instance.get_params().items() if isinstance(v, (int, float, str, bool))}
                mlflow.log_params(clean_params)

            # Fit pipeline
            pipeline.fit(X_train, y_train)
            train_duration = round(time.time() - start_time, 3)
            mlflow.log_metric("training_duration_seconds", train_duration)

            # Evaluate on validation set
            y_val_pred = pipeline.predict(X_val)
            val_metrics = calculate_metrics(y_val, y_val_pred)
            
            # Log validation metrics
            mlflow.log_metric("val_mae", val_metrics["mae"])
            mlflow.log_metric("val_rmse", val_metrics["rmse"])
            mlflow.log_metric("val_r2", val_metrics["r2"])

            # Log model artifact with signature
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
                input_example=X_val.iloc[:2]
            )

            logger.info("%s results: MAE=$%s, RMSE=$%s, R2=%.4f (Time: %.2fs)",
                        name, f"{val_metrics['mae']:,.2f}", f"{val_metrics['rmse']:,.2f}",
                        val_metrics["r2"], train_duration)

            results[name] = {
                **val_metrics,
                "training_duration": train_duration,
                "run_id": run_id
            }
            fitted_pipelines[name] = pipeline

    # Select best model based on highest R2 and lowest RMSE on validation set
    best_model_name = max(results, key=lambda k: (results[k]["r2"], -results[k]["rmse"]))
    best_metrics = results[best_model_name]
    best_pipeline = fitted_pipelines[best_model_name]
    best_run_id = run_ids[best_model_name]

    logger.info("=========================================================")
    logger.info("BEST MODEL SELECTED: '%s'", best_model_name)
    logger.info("Best Validation Metrics: R2=%.4f, RMSE=$%s, MAE=$%s",
                best_metrics["r2"], f"{best_metrics['rmse']:,.2f}", f"{best_metrics['mae']:,.2f}")
    logger.info("=========================================================")

    # Save best model to disk
    models_dir = root / Path(config.get("models", {}).get("save_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_dest = root / Path(config.get("models", {}).get("best_model_path", "models/best_model.joblib"))
    joblib.dump(best_pipeline, best_model_dest)
    logger.info("Saved best model pipeline to: %s", best_model_dest)

    # Save model metadata
    model_info = {
        "model_name": best_model_name,
        "model_version": "1.0.0",
        "run_id": best_run_id,
        "training_date": datetime.now().isoformat(),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "features": {
            "numerical": num_cols,
            "categorical": cat_cols
        },
        "target": target_col,
        "validation_metrics": best_metrics,
        "all_model_comparison": results
    }
    model_info_path = root / Path(config.get("models", {}).get("model_info_path", "models/model_info.json"))
    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)
    logger.info("Saved model info metadata to: %s", model_info_path)

    # Save model comparison summary in reports/
    reports_dir = root / Path(config.get("reports", {}).get("dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    comp_file = reports_dir / "model_comparison.json"
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Attempt to register model in MLflow Model Registry
    try:
        model_uri = f"runs:/{best_run_id}/model"
        reg_model = mlflow.register_model(model_uri=model_uri, name="HousePriceBestModel")
        logger.info("Successfully registered model in MLflow Registry: %s (v%s)", reg_model.name, reg_model.version)
    except Exception as e:
        logger.warning("MLflow Model Registry registration note (local SQLite setup): %s", e)

    return best_model_name, best_metrics, results


def main():
    """CLI entrypoint for training stage."""
    parser = argparse.ArgumentParser(description="Train and compare House Price models using MLflow.")
    parser.add_argument("--train", type=str, default=None, help="Path to processed train.csv.")
    parser.add_argument("--val", type=str, default=None, help="Path to processed val.csv.")
    args = parser.parse_args()

    train_and_evaluate_models(train_path=args.train, val_path=args.val)
    print("\n=== Training and MLflow Tracking Completed Successfully ===")


if __name__ == "__main__":
    main()
