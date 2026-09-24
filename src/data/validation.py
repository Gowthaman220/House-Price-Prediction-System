"""Data validation module for House Price Prediction MLOps Pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.config import get_logger, get_project_root, load_config

logger = get_logger("data_validation")


class DataValidationError(Exception):
    """Custom exception raised when data validation fails."""
    pass


EXPECTED_NUMERICAL_COLUMNS = [
    "OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars",
    "FullBath", "YearBuilt", "YearRemodAdd", "LotArea", "Fireplaces"
]

EXPECTED_CATEGORICAL_COLUMNS = [
    "Neighborhood", "BldgType", "HouseStyle", "CentralAir"
]

TARGET_COLUMN = "SalePrice"


def validate_dataset(
    df: pd.DataFrame,
    is_training: bool = True,
    missing_threshold: float = 0.20
) -> Tuple[bool, List[str]]:
    """Validate DataFrame against schema, ranges, data types, and null thresholds.
    
    Args:
        df: Pandas DataFrame to validate.
        is_training: If True, checks for presence and validity of target column.
        missing_threshold: Max allowed ratio of missing values in any required column.
        
    Returns:
        Tuple of (is_valid: bool, issues: List[str]).
        
    Raises:
        DataValidationError: If critical checks fail.
    """
    issues = []
    
    # 1. Check empty dataset
    if df is None or df.empty:
        msg = "Validation failed: Dataset is empty or None."
        logger.error(msg)
        raise DataValidationError(msg)
        
    # 2. Check required feature columns
    expected_features = EXPECTED_NUMERICAL_COLUMNS + EXPECTED_CATEGORICAL_COLUMNS
    missing_cols = [col for col in expected_features if col not in df.columns]
    if missing_cols:
        msg = f"Validation failed: Missing required feature columns: {missing_cols}"
        issues.append(msg)
        logger.error(msg)
        
    # 3. Check target column for training data
    if is_training:
        if TARGET_COLUMN not in df.columns:
            msg = f"Validation failed: Target column '{TARGET_COLUMN}' is missing from training data."
            issues.append(msg)
            logger.error(msg)
        else:
            # Check for null target
            null_target_count = df[TARGET_COLUMN].isnull().sum()
            if null_target_count > 0:
                msg = f"Validation failed: Target column contains {null_target_count} missing values."
                issues.append(msg)
                logger.error(msg)
                
            # Check target value range
            valid_targets = df[TARGET_COLUMN].dropna()
            if (valid_targets <= 0).any():
                msg = "Validation failed: Target column contains non-positive values."
                issues.append(msg)
                logger.error(msg)
            if (valid_targets < 10000).any() or (valid_targets > 2500000).any():
                msg = f"Warning: Some target values fall outside typical bounds ($10,000 - $2,500,000): min={valid_targets.min()}, max={valid_targets.max()}"
                logger.warning(msg)

    # 4. Check data types and ranges for numerical columns
    for col in EXPECTED_NUMERICAL_COLUMNS:
        if col in df.columns:
            # Check if column is numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                msg = f"Validation failed: Column '{col}' is expected to be numeric, got {df[col].dtype}."
                issues.append(msg)
                logger.error(msg)
            else:
                # Check for negative values where invalid
                if col in ["GrLivArea", "LotArea"] and (df[col].dropna() <= 0).any():
                    msg = f"Validation failed: Column '{col}' contains values <= 0."
                    issues.append(msg)
                    logger.error(msg)
                if col in ["TotalBsmtSF", "GarageCars", "Fireplaces"] and (df[col].dropna() < 0).any():
                    msg = f"Validation failed: Column '{col}' contains negative values."
                    issues.append(msg)
                    logger.error(msg)
                if col == "OverallQual":
                    out_of_bounds = ~df[col].dropna().between(1, 10)
                    if out_of_bounds.any():
                        msg = f"Validation failed: 'OverallQual' has {out_of_bounds.sum()} values outside [1, 10]."
                        issues.append(msg)
                        logger.error(msg)
                if col in ["YearBuilt", "YearRemodAdd"]:
                    out_of_bounds = ~df[col].dropna().between(1800, 2035)
                    if out_of_bounds.any():
                        msg = f"Validation failed: '{col}' has values outside [1800, 2035]."
                        issues.append(msg)
                        logger.error(msg)

    # 5. Check categorical columns
    for col in EXPECTED_CATEGORICAL_COLUMNS:
        if col in df.columns:
            if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                msg = f"Validation warning: Categorical column '{col}' has type {df[col].dtype}."
                logger.warning(msg)

    # 6. Check missing value ratios
    for col in df.columns:
        null_ratio = df[col].isnull().mean()
        if null_ratio > missing_threshold:
            msg = f"Validation failed: Column '{col}' has {null_ratio:.1%} missing values (exceeds threshold of {missing_threshold:.1%})."
            issues.append(msg)
            logger.error(msg)

    # 7. Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logger.warning("Found %d duplicate rows in dataset.", duplicate_count)

    is_valid = len(issues) == 0
    if is_valid:
        logger.info("Data validation passed successfully with 0 critical issues (%d rows, %d columns).", len(df), len(df.columns))
    else:
        logger.error("Data validation encountered %d critical issue(s).", len(issues))
        
    return is_valid, issues


def main():
    """CLI entrypoint for data validation stage."""
    parser = argparse.ArgumentParser(description="Validate House Price dataset.")
    parser.add_argument("--input", type=str, default=None, help="Path to CSV dataset to validate.")
    parser.add_argument("--inference", action="store_true", help="Set flag if validating inference data without target.")
    args = parser.parse_args()
    
    root = get_project_root()
    config = load_config()
    input_path = args.input or config.get("data", {}).get("raw_path", "data/raw/house_prices.csv")
    csv_file = root / Path(input_path)
    
    if not csv_file.exists():
        logger.error("Dataset not found at: %s", csv_file)
        sys.exit(1)
        
    df = pd.read_csv(csv_file)
    logger.info("Validating dataset at: %s", csv_file)
    is_valid, issues = validate_dataset(df, is_training=not args.inference)
    
    if not is_valid:
        print("\n=== Validation Failures ===")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)
    else:
        print("\n=== Validation Succeeded ===")
        print(f"Dataset at {input_path} is valid and meets schema requirements.")


if __name__ == "__main__":
    main()
