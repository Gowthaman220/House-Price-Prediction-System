"""Tests for preprocessing pipeline, transformations, and data splitting."""

import numpy as np
import pandas as pd
from src.features.preprocessing import build_preprocessor, split_data


def test_split_data():
    """Verify split_data accurately divides rows according to specified ratios."""
    df = pd.DataFrame({"feat": range(100), "SalePrice": range(100)})
    train, val, test = split_data(df, test_size=0.2, val_size=0.1, random_state=42)
    
    assert len(test) == 20
    assert len(train) + len(val) == 80
    assert len(train) + len(val) + len(test) == 100


def test_preprocessor_imputation_and_encoding():
    """Verify preprocessor handles missing numerical values and encodes categories."""
    num_cols = ["OverallQual", "GrLivArea"]
    cat_cols = ["Neighborhood", "CentralAir"]

    train_data = pd.DataFrame({
        "OverallQual": [7, 6, np.nan, 8],
        "GrLivArea": [1800.0, np.nan, 1500.0, 2200.0],
        "Neighborhood": ["CollegeCreek", "OldTown", "CollegeCreek", "Somerset"],
        "CentralAir": ["Y", "N", "Y", "Y"]
    })

    preprocessor = build_preprocessor(
        numerical_features=num_cols,
        categorical_features=cat_cols,
        num_strategy="median",
        cat_strategy="most_frequent",
        scale_numeric=True
    )

    transformed = preprocessor.fit_transform(train_data)
    assert not np.isnan(transformed).any(), "Transformed output contains NaNs"
    assert transformed.shape[0] == 4
    # 2 scaled numeric + one-hot categories > 4 columns
    assert transformed.shape[1] >= 4


def test_preprocessor_unseen_category():
    """Verify preprocessor handles unseen categories during inference gracefully."""
    num_cols = ["OverallQual"]
    cat_cols = ["Neighborhood"]

    train_data = pd.DataFrame({
        "OverallQual": [7, 8],
        "Neighborhood": ["CollegeCreek", "Somerset"]
    })

    preprocessor = build_preprocessor(
        numerical_features=num_cols,
        categorical_features=cat_cols
    )
    preprocessor.fit(train_data)

    test_unseen = pd.DataFrame({
        "OverallQual": [6],
        "Neighborhood": ["UnknownNeighborhood"]
    })

    # Should not raise exception because handle_unknown='ignore'
    transformed = preprocessor.transform(test_unseen)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()
