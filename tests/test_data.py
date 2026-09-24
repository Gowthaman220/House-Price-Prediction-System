"""Tests for data ingestion and data validation components."""

import pandas as pd
import pytest

from src.data.ingestion import generate_benchmark_dataset
from src.data.validation import (
    EXPECTED_CATEGORICAL_COLUMNS,
    EXPECTED_NUMERICAL_COLUMNS,
    TARGET_COLUMN,
    DataValidationError,
    validate_dataset,
)


def test_generate_benchmark_dataset():
    """Verify generated dataset contains expected columns and non-empty rows."""
    df = generate_benchmark_dataset(num_samples=100, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    
    # Check all expected columns are present
    for col in EXPECTED_NUMERICAL_COLUMNS + EXPECTED_CATEGORICAL_COLUMNS + [TARGET_COLUMN]:
        assert col in df.columns, f"Column '{col}' missing from generated dataset"

    # Check target price is positive and realistic
    assert (df[TARGET_COLUMN] > 0).all()
    assert df[TARGET_COLUMN].min() >= 10000


def test_validate_dataset_success():
    """Verify validation passes on a valid clean dataset."""
    df = generate_benchmark_dataset(num_samples=50, random_state=123)
    is_valid, issues = validate_dataset(df, is_training=True)
    assert is_valid is True
    assert len(issues) == 0


def test_validate_dataset_missing_column():
    """Verify validation catches missing required column."""
    df = generate_benchmark_dataset(num_samples=20, random_state=42)
    df_dropped = df.drop(columns=["OverallQual"])
    is_valid, issues = validate_dataset(df_dropped, is_training=True)
    assert is_valid is False
    assert any("OverallQual" in issue for issue in issues)


def test_validate_dataset_empty_fails():
    """Verify validation fails on empty dataframe."""
    empty_df = pd.DataFrame()
    with pytest.raises(DataValidationError):
        validate_dataset(empty_df, is_training=True)


def test_validate_dataset_negative_target():
    """Verify validation fails if target contains negative values."""
    df = generate_benchmark_dataset(num_samples=20, random_state=42)
    df.loc[0, TARGET_COLUMN] = -50000.0
    is_valid, issues = validate_dataset(df, is_training=True)
    assert is_valid is False
    assert any("non-positive" in issue for issue in issues)
