"""Feature engineering and preprocessing pipeline for House Price Prediction."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import get_logger, get_project_root, load_config, load_params

logger = get_logger("preprocessing")


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
    num_strategy: str = "median",
    cat_strategy: str = "most_frequent",
    scale_numeric: bool = True
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for numerical and categorical features.
    
    Args:
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.
        num_strategy: Imputation strategy for numeric columns ('median' or 'mean').
        cat_strategy: Imputation strategy for categorical columns ('most_frequent').
        scale_numeric: Whether to apply StandardScaler to numerical features.
        
    Returns:
        Configured ColumnTransformer object.
    """
    # 1. Numerical Pipeline
    num_steps = [("imputer", SimpleImputer(strategy=num_strategy))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numerical_transformer = Pipeline(steps=num_steps)

    # 2. Categorical Pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cat_strategy)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 3. Combine with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    
    return preprocessor


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets.
    
    Args:
        df: Full raw DataFrame.
        test_size: Proportion allocated to test set.
        val_size: Proportion allocated to validation set from training pool.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    logger.info("Splitting dataset: total=%d, test_size=%.2f, val_size=%.2f", len(df), test_size, val_size)
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    
    # Second split: train vs val
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state
    )
    
    logger.info("Split results: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def run_preprocessing_pipeline(
    raw_path: str | None = None,
    output_dir: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Execute end-to-end preprocessing, data splitting, and preprocessor persistence.
    
    Args:
        raw_path: Path to raw CSV file.
        output_dir: Directory where processed train/val/test CSVs will be saved.
        
    Returns:
        Tuple of (train_df, val_df, test_df, fitted_preprocessor).
    """
    root = get_project_root()
    config = load_config()
    params = load_params()

    input_file = root / Path(raw_path or config.get("data", {}).get("raw_path", "data/raw/house_prices.csv"))
    proc_dir = root / Path(output_dir or config.get("data", {}).get("processed_dir", "data/processed"))
    proc_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Raw data file not found at: {input_file}")

    logger.info("Loading raw data from: %s", input_file)
    df = pd.read_csv(input_file)

    # Load feature definitions
    num_cols = config.get("features", {}).get("numerical", [])
    cat_cols = config.get("features", {}).get("categorical", [])
    config.get("data", {}).get("target_column", "SalePrice")

    test_size = float(params.get("data", {}).get("test_size", 0.2))
    val_size = float(params.get("data", {}).get("val_size", 0.1))
    seed = int(params.get("data", {}).get("random_state", 42))

    # Split dataset
    train_df, val_df, test_df = split_data(
        df,
        test_size=test_size,
        val_size=val_size,
        random_state=seed
    )

    # Save splits to disk
    train_path = proc_dir / "train.csv"
    val_path = proc_dir / "val.csv"
    test_path = proc_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Saved processed splits to: %s, %s, %s", train_path, val_path, test_path)

    # Build and fit preprocessor on training features only (prevent data leakage!)
    X_train = train_df[num_cols + cat_cols]
    preprocessor = build_preprocessor(
        numerical_features=num_cols,
        categorical_features=cat_cols,
        num_strategy=params.get("preprocessing", {}).get("numerical_imputer_strategy", "median"),
        cat_strategy=params.get("preprocessing", {}).get("categorical_imputer_strategy", "most_frequent"),
        scale_numeric=True
    )
    preprocessor.fit(X_train)
    logger.info("Fitted preprocessor on training data (%d samples)", len(X_train))

    # Save standalone preprocessor artifact
    models_dir = root / Path(config.get("models", {}).get("save_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    prep_path = root / Path(config.get("models", {}).get("preprocessor_path", "models/preprocessor.joblib"))
    joblib.dump(preprocessor, prep_path)
    logger.info("Saved preprocessor artifact to: %s", prep_path)

    return train_df, val_df, test_df, preprocessor


def main():
    """CLI entrypoint for preprocessing stage."""
    parser = argparse.ArgumentParser(description="Preprocess and split House Price dataset.")
    parser.add_argument("--raw", type=str, default=None, help="Path to raw dataset.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save processed splits.")
    args = parser.parse_args()

    run_preprocessing_pipeline(raw_path=args.raw, output_dir=args.output_dir)
    print("\n=== Preprocessing Completed Successfully ===")


if __name__ == "__main__":
    main()
