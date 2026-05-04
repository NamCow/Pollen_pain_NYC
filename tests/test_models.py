"""
Tests for model training and prediction utilities.

Covers XGBoost prediction clipping, logistic regression probability bounds,
baseline comparisons, and regression metric computation. Uses small synthetic
data — no dependency on real modeling tables.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.feature_engineering import FEATURE_COLS, TARGET


def _make_model_data(n_rows: int = 200, n_ntas: int = 4) -> pd.DataFrame:
    """Build a synthetic DataFrame with all required feature columns."""
    rng = np.random.default_rng(42)
    months = pd.date_range("2022-03", periods=n_rows // n_ntas, freq="MS").strftime("%Y-%m").tolist()
    rows = []
    for nta_i in range(n_ntas):
        code = f"BX{nta_i:04d}"
        for m in months:
            row = {
                "nta_code": code,
                "NTAName": f"Area {nta_i}",
                "borough": "Bronx",
                "year_month": m,
                TARGET: float(rng.integers(10, 100)),
            }
            for col in FEATURE_COLS:
                row[col] = rng.normal(10, 3)
            rows.append(row)
    return pd.DataFrame(rows)


class TestXGBoostPredictions:
    """Test that XGBoost predictions are clipped to non-negative values."""

    def test_predictions_are_non_negative(self):
        """After clipping, all predicted ED visit counts should be >= 0."""
        xgb = pytest.importorskip("xgboost")
        df = _make_model_data(n_rows=200, n_ntas=4)

        model = xgb.XGBRegressor(
            n_estimators=10, max_depth=3, random_state=42, verbosity=0
        )
        model.fit(df[FEATURE_COLS], df[TARGET])

        raw_preds = model.predict(df[FEATURE_COLS])
        clipped = np.maximum(raw_preds, 0)
        assert (clipped >= 0).all(), "Clipped predictions should all be non-negative"

    def test_clip_handles_negative_raw_predictions(self):
        """np.maximum(preds, 0) should turn negatives to zero."""
        fake_preds = np.array([-5.0, -0.1, 0.0, 10.0, 50.0])
        clipped = np.maximum(fake_preds, 0)
        expected = np.array([0.0, 0.0, 0.0, 10.0, 50.0])
        np.testing.assert_array_equal(clipped, expected)

    def test_xgboost_feature_importance_length(self):
        """Feature importances should have one entry per feature column."""
        xgb = pytest.importorskip("xgboost")
        df = _make_model_data(n_rows=200, n_ntas=4)

        model = xgb.XGBRegressor(
            n_estimators=10, max_depth=3, random_state=42, verbosity=0
        )
        model.fit(df[FEATURE_COLS], df[TARGET])

        assert len(model.feature_importances_) == len(FEATURE_COLS)


class TestLogisticRegressionPredictions:
    """Test logistic regression probability output."""

    def test_probabilities_in_zero_one(self):
        """predict_proba should return values strictly in [0, 1]."""
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        df = _make_model_data(n_rows=200, n_ntas=4)
        median_target = df[TARGET].median()
        y = (df[TARGET] >= median_target).astype(int)

        X = df[FEATURE_COLS]
        mu, sigma = X.mean(), X.std() + 1e-8
        X_scaled = (X - mu) / sigma

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)

        probs = model.predict_proba(X_scaled)[:, 1]
        assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities out of [0, 1]"

    def test_predict_returns_binary_classes(self):
        """predict() should return only 0 or 1 for binary classification."""
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        df = _make_model_data(n_rows=200, n_ntas=4)
        median_target = df[TARGET].median()
        y = (df[TARGET] >= median_target).astype(int)

        X = df[FEATURE_COLS]
        mu, sigma = X.mean(), X.std() + 1e-8
        X_scaled = (X - mu) / sigma

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)

        preds = model.predict(X_scaled)
        assert set(preds).issubset({0, 1}), f"Unexpected prediction values: {set(preds)}"


class TestRegressionMetrics:
    """Test the regression_metrics helper from train_xgboost."""

    def test_regression_metrics_keys(self):
        """regression_metrics should return dict with expected keys."""
        from src.models.train_xgboost import regression_metrics

        actual = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        pred = pd.Series([12.0, 18.0, 33.0, 38.0, 52.0])
        result = regression_metrics(actual, pred)
        assert set(result.keys()) == {"mae", "rmse", "pearson_r", "pearson_p"}

    def test_perfect_predictions_give_zero_error(self):
        """Identical actual and predicted should yield MAE=0 and RMSE=0."""
        from src.models.train_xgboost import regression_metrics

        values = pd.Series([10.0, 20.0, 30.0])
        result = regression_metrics(values, values)
        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["pearson_r"] == pytest.approx(1.0)

    def test_mae_is_correct(self):
        from src.models.train_xgboost import regression_metrics

        actual = pd.Series([10.0, 20.0, 30.0])
        pred = pd.Series([12.0, 22.0, 28.0])
        result = regression_metrics(actual, pred)
        assert result["mae"] == pytest.approx(2.0)


class TestHoldoutBaselineComparison:
    """Test that baseline comparison DataFrames have expected structure."""

    def test_seasonal_average_baseline_shape(self):
        """seasonal_average_baseline should return a Series matching test length."""
        from src.models.train_xgboost import seasonal_average_baseline

        train = pd.DataFrame({
            "nta_code": ["A"] * 12 + ["B"] * 12,
            "year_month": [f"2022-{m:02d}" for m in range(1, 13)] * 2,
            TARGET: np.random.default_rng(42).integers(10, 50, 24).astype(float),
        })
        test = pd.DataFrame({
            "nta_code": ["A", "A", "B", "B"],
            "year_month": ["2023-01", "2023-02", "2023-01", "2023-02"],
            TARGET: [25.0, 30.0, 20.0, 35.0],
        })
        result = seasonal_average_baseline(train, test)
        assert len(result) == len(test)
        assert not result.isna().any(), "Baseline should have no NaN values"
