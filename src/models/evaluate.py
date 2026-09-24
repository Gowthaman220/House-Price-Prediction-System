"""Model evaluation and report generation module for House Price Prediction."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.config import get_logger, get_project_root, load_config

logger = get_logger("model_evaluation")

# Set clean aesthetic style for academic report figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300


def evaluate_model(
    test_path: Optional[str] = None,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate the trained best model on the holdout test set and generate report artifacts.
    
    Args:
        test_path: Optional path to processed test.csv.
        model_path: Optional path to saved best_model.joblib.
        output_dir: Optional path to reports directory.
        
    Returns:
        Dictionary of test metrics.
    """
    root = get_project_root()
    config = load_config()

    test_file = root / Path(test_path or config.get("data", {}).get("test_path", "data/processed/test.csv"))
    best_model_file = root / Path(model_path or config.get("models", {}).get("best_model_path", "models/best_model.joblib"))
    rep_dir = root / Path(output_dir or config.get("reports", {}).get("dir", "reports"))
    fig_dir = root / Path(config.get("reports", {}).get("figures_dir", "reports/figures"))
    
    rep_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not test_file.exists():
        raise FileNotFoundError(f"Test data not found at: {test_file}")
    if not best_model_file.exists():
        raise FileNotFoundError(f"Trained model not found at: {best_model_file}")

    logger.info("Loading holdout test data from: %s", test_file)
    test_df = pd.read_csv(test_file)

    logger.info("Loading best model from: %s", best_model_file)
    pipeline = joblib.load(best_model_file)

    num_cols = config.get("features", {}).get("numerical", [])
    cat_cols = config.get("features", {}).get("categorical", [])
    target_col = config.get("data", {}).get("target_column", "SalePrice")

    feature_cols = num_cols + cat_cols
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values

    # Run inference on test set
    y_pred = pipeline.predict(X_test)

    # Compute metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

    test_metrics = {
        "test_mae": round(mae, 2),
        "test_rmse": round(rmse, 2),
        "test_r2": round(r2, 4),
        "test_mape_percent": round(mape, 2),
        "test_samples": len(y_test),
        "min_actual_price": float(y_test.min()),
        "max_actual_price": float(y_test.max()),
        "mean_actual_price": float(y_test.mean())
    }

    logger.info("Test Evaluation Results: MAE=$%s, RMSE=$%s, R2=%.4f, MAPE=%.2f%%",
                f"{mae:,.2f}", f"{rmse:,.2f}", r2, mape)

    # Save metrics JSON
    metrics_file = rep_dir / "evaluation_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Saved evaluation metrics to: %s", metrics_file)

    # 1. Plot: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="#1f77b4", edgecolors="w", s=45, label="Holdout Properties")
    min_val = min(y_test.min(), y_pred.min()) * 0.95
    max_val = max(y_test.max(), y_pred.max()) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Prediction (45° Line)")
    plt.title("Actual vs. Predicted House Prices (Holdout Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Actual Sale Price (USD)", fontsize=11)
    plt.ylabel("Predicted Sale Price (USD)", fontsize=11)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().xaxis.set_major_formatter("${x:,.0f}")
    plt.gca().yaxis.set_major_formatter("${x:,.0f}")
    plt.annotate(
        f"$R^2 = {r2:.4f}$\n$RMSE = \\${rmse:,.0f}$\n$MAE = \\${mae:,.0f}$",
        xy=(0.05, 0.82),
        xycoords="axes fraction",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cccccc", lw=1.2)
    )
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    act_pred_path = fig_dir / "actual_vs_predicted.png"
    plt.savefig(act_pred_path, dpi=300)
    plt.close()
    logger.info("Saved Actual vs Predicted plot to: %s", act_pred_path)

    # 2. Plot: Residuals Distribution
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    n, bins, patches = plt.hist(residuals, bins=30, color="#2ca02c", alpha=0.7, edgecolor="white", density=True)
    # Fit normal distribution curve
    mu, std = np.mean(residuals), np.std(residuals)
    x_axis = np.linspace(residuals.min(), residuals.max(), 100)
    p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(- (x_axis - mu)**2 / (2 * std**2))
    plt.plot(x_axis, p, "k-", linewidth=2, label=rf"Normal Fit ($\mu$=${mu:,.0f}, $\sigma$=${std:,.0f})")
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    plt.title("Residuals (Error) Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Residual Error (Actual - Predicted) USD", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.gca().xaxis.set_major_formatter("${x:,.0f}")
    plt.legend(frameon=True)
    plt.tight_layout()
    residuals_path = fig_dir / "residuals_distribution.png"
    plt.savefig(residuals_path, dpi=300)
    plt.close()
    logger.info("Saved Residuals Distribution plot to: %s", residuals_path)

    # 3. Plot: Model Comparison
    comp_file = rep_dir / "model_comparison.json"
    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
            
        models = list(comp_data.keys())
        r2_vals = [comp_data[m]["r2"] for m in models]
        rmse_vals = [comp_data[m]["rmse"] for m in models]
        mae_vals = [comp_data[m]["mae"] for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        # R2 Bar Chart
        bars1 = ax1.bar(models, r2_vals, color="#4285F4", width=0.55, edgecolor="black", alpha=0.85)
        ax1.set_title("Model Validation $R^2$ Score (Higher is Better)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("$R^2$ Score", fontsize=11)
        ax1.set_ylim(0.8, 1.0)
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # RMSE Bar Chart
        bars2 = ax2.bar(models, rmse_vals, color="#EA4335", width=0.55, edgecolor="black", alpha=0.85)
        ax2.set_title("Model Validation RMSE (Lower is Better)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("RMSE (USD)", fontsize=11)
        ax2.yaxis.set_major_formatter("${x:,.0f}")
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 500, f"${yval:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        comp_path = fig_dir / "model_comparison.png"
        plt.savefig(comp_path, dpi=300)
        plt.close()
        logger.info("Saved Model Comparison plot to: %s", comp_path)

    return test_metrics


def main():
    """CLI entrypoint for evaluation stage."""
    parser = argparse.ArgumentParser(description="Evaluate best model on holdout test set.")
    parser.add_argument("--test", type=str, default=None, help="Path to processed test.csv.")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model artifact.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save reports.")
    args = parser.parse_args()

    evaluate_model(test_path=args.test, model_path=args.model, output_dir=args.output_dir)
    print("\n=== Model Evaluation Completed Successfully ===")


if __name__ == "__main__":
    main()
