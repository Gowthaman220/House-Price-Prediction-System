"""Inference module for House Price Prediction."""

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.config import get_logger, get_project_root, load_config

logger = get_logger("model_prediction")


class HousePricePredictor:
    """Predictor class managing model loading, feature validation, and inference."""
    
    _instance = None
    _model = None

    def __init__(self, model_path: str | None = None):
        """Initialize the predictor and load the model pipeline."""
        root = get_project_root()
        config = load_config()
        self.model_path = root / Path(model_path or config.get("models", {}).get("best_model_path", "models/best_model.joblib"))
        self.num_cols = config.get("features", {}).get("numerical", [])
        self.cat_cols = config.get("features", {}).get("categorical", [])
        self.expected_cols = self.num_cols + self.cat_cols
        self.model = self._load_model()

    def _load_model(self):
        """Load fitted scikit-learn pipeline from joblib artifact."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at {self.model_path}. "
                "Please run training first using: python -m src.models.train"
            )
        logger.info("Loading model artifact from: %s", self.model_path)
        return joblib.load(self.model_path)

    def predict(self, data: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame) -> np.ndarray:
        """Run inference on input property feature(s).
        
        Args:
            data: Single dictionary, list of dictionaries, or pandas DataFrame of features.
            
        Returns:
            Numpy array of predicted sale prices in USD.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data format for inference: {type(data)}")

        # Verify all expected columns are present
        missing_cols = [col for col in self.expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required features for prediction: {missing_cols}")

        # Ensure correct column ordering
        df_ordered = df[self.expected_cols]
        
        # Predict using full scikit-learn pipeline (handles imputation, scaling, encoding)
        raw_predictions = self.model.predict(df_ordered)
        
        # Clip to realistic lower bound (no negative prices)
        predictions = np.clip(raw_predictions, 10000.0, None)
        return np.round(predictions, 2)


_predictor_instance: HousePricePredictor | None = None


def get_predictor(model_path: str | None = None) -> HousePricePredictor:
    """Singleton getter for the HousePricePredictor instance."""
    global _predictor_instance
    if _predictor_instance is None or model_path is not None:
        _predictor_instance = HousePricePredictor(model_path=model_path)
    return _predictor_instance


def predict_price(property_features: dict[str, Any], model_path: str | None = None) -> float:
    """Convenience function to predict price for a single property dictionary."""
    predictor = get_predictor(model_path=model_path)
    preds = predictor.predict(property_features)
    return float(preds[0])


def main():
    """CLI test for model prediction."""
    parser = argparse.ArgumentParser(description="Predict house price from features.")
    parser.add_argument("--qual", type=int, default=7, help="Overall Quality (1-10)")
    parser.add_argument("--liv-area", type=float, default=1850.0, help="Ground Living Area (sqft)")
    parser.add_argument("--bsmt-sf", type=float, default=1200.0, help="Total Basement SF")
    parser.add_argument("--cars", type=int, default=2, help="Garage Cars (0-4)")
    parser.add_argument("--bath", type=int, default=2, help="Full Bathrooms (1-4)")
    parser.add_argument("--year-built", type=int, default=2008, help="Year Built")
    parser.add_argument("--year-remod", type=int, default=2015, help="Year Remodeled")
    parser.add_argument("--lot-area", type=float, default=9500.0, help="Lot Area (sqft)")
    parser.add_argument("--fireplaces", type=int, default=1, help="Fireplaces (0-3)")
    parser.add_argument("--neighborhood", type=str, default="CollegeCreek", help="Neighborhood")
    parser.add_argument("--bldg-type", type=str, default="1Fam", help="Building Type")
    parser.add_argument("--house-style", type=str, default="2Story", help="House Style")
    parser.add_argument("--central-air", type=str, default="Y", help="Central Air (Y/N)")
    args = parser.parse_args()

    sample = {
        "OverallQual": args.qual,
        "GrLivArea": args.liv_area,
        "TotalBsmtSF": args.bsmt_sf,
        "GarageCars": args.cars,
        "FullBath": args.bath,
        "YearBuilt": args.year_built,
        "YearRemodAdd": args.year_remod,
        "LotArea": args.lot_area,
        "Fireplaces": args.fireplaces,
        "Neighborhood": args.neighborhood,
        "BldgType": args.bldg_type,
        "HouseStyle": args.house_style,
        "CentralAir": args.central_air
    }

    pred = predict_price(sample)
    print("\n=== Sample Prediction ===")
    for k, v in sample.items():
        print(f"  {k}: {v}")
    print(f"\n>> PREDICTED SALE PRICE: ${pred:,.2f} USD\n")


if __name__ == "__main__":
    main()
