"""Tests for cleaning and feature engineering (no model or xgboost needed)."""

import numpy as np
import pandas as pd
import pytest

from src.data import (
    FEATURE_COLUMNS,
    LATE_COLS,
    RAW_FEATURES,
    TARGET,
    clean,
    engineer,
    make_synthetic,
    prepare_for_scoring,
)
from src.evaluate import assign_risk_band


@pytest.fixture(scope="module")
def raw():
    return make_synthetic(n=3000)


def test_clean_removes_invalid_ages_and_placeholder_codes(raw):
    raw = raw.copy()
    raw.loc[0, "age"] = 0
    out = clean(raw)
    assert out["age"].between(18, 100).all()
    assert (out[LATE_COLS].fillna(0) < 96).all().all()
    assert out["late_code_flag"].sum() > 0  # synthetic data contains 98 codes


def test_clean_caps_outliers(raw):
    out = clean(raw)
    assert out["RevolvingUtilizationOfUnsecuredLines"].max() <= 2.0
    assert out["DebtRatio"].max() <= 5.0


def test_engineer_returns_expected_columns(raw):
    X, y = engineer(clean(raw))
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X.columns) == 17
    assert TARGET not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert not X["total_past_due"].isna().all()


def test_prepare_for_scoring_keeps_all_rows_and_column_order(raw):
    batch = raw[RAW_FEATURES].head(50).copy()
    batch.loc[batch.index[0], "NumberOfTimes90DaysLate"] = 98
    X = prepare_for_scoring(batch)
    assert len(X) == 50
    assert list(X.columns) == FEATURE_COLUMNS
    assert X["late_code_flag"].iloc[0] == 1


def test_prepare_for_scoring_matches_training_features(raw):
    """Scoring-time features must equal training-time features for the same borrower."""
    cleaned = clean(raw)
    X_train, _ = engineer(cleaned)
    X_score = prepare_for_scoring(cleaned[RAW_FEATURES])
    pd.testing.assert_frame_equal(
        X_train.drop(columns=["late_code_flag"]).reset_index(drop=True),
        X_score.drop(columns=["late_code_flag"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_prepare_for_scoring_rejects_missing_columns(raw):
    with pytest.raises(ValueError):
        prepare_for_scoring(raw[RAW_FEATURES[:5]])


def test_risk_bands_follow_threshold():
    bands = assign_risk_band(np.array([0.01, 0.12, 0.30]), threshold=0.2)
    assert list(bands) == ["Low", "Medium", "High"]
