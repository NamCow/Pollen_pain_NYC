"""
Tests for src/features/feature_engineering.py

All tests use small synthetic DataFrames — no dependency on real data files.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.feature_engineering import (
    FEATURE_COLS,
    POLLEN_SEASON_MONTHS,
    REQUIRED_MODEL_FEATURES,
    TARGET,
    add_engineered_features,
    filter_low_case_ntas,
    filter_pollen_season,
    holdout_split,
    standard_scale,
    walk_forward_splits,
)


def _make_monthly_df(n_months: int = 24, n_ntas: int = 3, start: str = "2022-03") -> pd.DataFrame:
    """Build a small synthetic DataFrame that mimics the modeling table."""
    months = pd.date_range(start, periods=n_months, freq="MS").strftime("%Y-%m").tolist()
    rows = []
    for nta_i in range(n_ntas):
        code = f"BX{nta_i:04d}"
        for m in months:
            row = {
                "nta_code": code,
                "NTAName": f"Area {nta_i}",
                "borough": "Bronx",
                "year_month": m,
                TARGET: float(20 + nta_i * 10 + np.random.default_rng(42).integers(0, 5)),
                "temp_max_mean": 25.0,
                "temp_min_mean": 15.0,
                "precip_total": 50.0,
                "wind_max_mean": 10.0,
                "pollen_composite_avg": 5.0,
                "pollen_composite_max": 12.0,
                "tree_avg": 3.0,
                "tree_max": 8.0,
                "weed_avg": 1.5,
                "grass_avg": 2.0,
                "pollen_14d_lag_avg": 4.5,
                "pollen_28d_lag_avg": 4.0,
                "pm25_mean": 8.0,
                "ozone_mean": 30.0,
                "no2_mean": 20.0,
                "tree_count": 500,
                "total_dbh": 3000.0,
                "mean_dbh": 6.0,
                "pct_good_health": 0.7,
                "chs_asthma_pct": 10.0,
            }
            rows.append(row)
    return pd.DataFrame(rows)


class TestWalkForwardSplits:
    """Verify the walk-forward CV utility produces valid temporal splits."""

    def test_train_always_before_test(self):
        """Train period must always end before the test period begins (no temporal leakage)."""
        df = _make_monthly_df(n_months=30, n_ntas=2)
        for train, test, train_end, test_start, test_end in walk_forward_splits(df, n_test_months=6):
            assert train_end < test_start, (
                f"Temporal leakage: train_end={train_end} >= test_start={test_start}"
            )

    def test_no_overlapping_months_between_train_and_test(self):
        """Train and test sets should share zero year_month values."""
        df = _make_monthly_df(n_months=30, n_ntas=2)
        for train, test, *_ in walk_forward_splits(df, n_test_months=6):
            overlap = set(train["year_month"]) & set(test["year_month"])
            assert overlap == set(), f"Train/test overlap on months: {overlap}"

    def test_minimum_train_size_is_12_months(self):
        """The first fold should have at least 12 months of training data."""
        df = _make_monthly_df(n_months=24, n_ntas=2)
        first_split = next(walk_forward_splits(df, n_test_months=6))
        train = first_split[0]
        train_months = train["year_month"].nunique()
        assert train_months >= 12, f"First fold has only {train_months} training months, expected >= 12"

    def test_yields_at_least_one_split(self):
        """With 24 months and min_train=12, we should get at least one split."""
        df = _make_monthly_df(n_months=24, n_ntas=2)
        splits = list(walk_forward_splits(df, n_test_months=6))
        assert len(splits) >= 1, "Expected at least one walk-forward split"

    def test_no_splits_when_insufficient_data(self):
        """With fewer than min_train+1 months, no splits should be produced."""
        df = _make_monthly_df(n_months=12, n_ntas=2)
        splits = list(walk_forward_splits(df, n_test_months=6))
        assert len(splits) == 0, "Should produce no splits with only 12 months"


class TestFeatureCols:
    """Verify FEATURE_COLS list integrity."""

    def test_feature_cols_non_empty(self):
        assert len(FEATURE_COLS) > 0

    def test_contains_required_model_features(self):
        """REQUIRED_MODEL_FEATURES must be a subset of FEATURE_COLS."""
        for col in REQUIRED_MODEL_FEATURES:
            assert col in FEATURE_COLS, f"Required feature {col!r} not in FEATURE_COLS"

    def test_contains_pollen_lag_features(self):
        assert "pollen_14d_lag_avg" in FEATURE_COLS
        assert "pollen_28d_lag_avg" in FEATURE_COLS

    def test_contains_cyclic_month_encodings(self):
        assert "month_sin" in FEATURE_COLS
        assert "month_cos" in FEATURE_COLS


class TestAddEngineeredFeatures:
    """Test the cyclic encoding and derived feature creation."""

    def test_cyclic_month_values_in_valid_range(self):
        """month_sin and month_cos must be in [-1, 1]."""
        df = _make_monthly_df(n_months=12, n_ntas=1)
        result = add_engineered_features(df)
        assert result["month_sin"].between(-1, 1).all(), "month_sin out of [-1, 1]"
        assert result["month_cos"].between(-1, 1).all(), "month_cos out of [-1, 1]"

    def test_engineered_columns_are_added(self):
        """add_engineered_features should create all expected derived columns."""
        df = _make_monthly_df(n_months=6, n_ntas=1)
        result = add_engineered_features(df)
        expected_new = [
            "month_sin", "month_cos", "season_progress",
            "temp_diurnal_range", "warm_dry_index",
            "pollen_change_vs_28d", "pollen_temp_interaction",
            "pollen_pm25_interaction", "pollen_chs_interaction",
            "weed_temp_interaction",
        ]
        for col in expected_new:
            assert col in result.columns, f"Missing engineered column: {col}"

    def test_diurnal_range_equals_max_minus_min(self):
        """temp_diurnal_range should be temp_max_mean - temp_min_mean."""
        df = _make_monthly_df(n_months=3, n_ntas=1)
        result = add_engineered_features(df)
        expected = result["temp_max_mean"] - result["temp_min_mean"]
        pd.testing.assert_series_equal(
            result["temp_diurnal_range"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


class TestFilterFunctions:
    """Test season filtering and low-case NTA filtering."""

    def test_filter_pollen_season_keeps_only_march_through_october(self):
        df = _make_monthly_df(n_months=12, n_ntas=1, start="2023-01")
        result = filter_pollen_season(df)
        result_months = pd.to_datetime(result["year_month"] + "-01").dt.month
        for m in result_months:
            assert m in POLLEN_SEASON_MONTHS, f"Month {m} should have been filtered out"

    def test_filter_low_case_ntas_removes_low_volume(self):
        """NTAs with average ED visits below the threshold should be dropped."""
        df = pd.DataFrame({
            "nta_code": ["A"] * 5 + ["B"] * 5,
            TARGET: [100.0] * 5 + [2.0] * 5,
        })
        result = filter_low_case_ntas(df, min_avg_monthly_cases=10.0)
        assert "A" in result["nta_code"].values
        assert "B" not in result["nta_code"].values


class TestHoldoutSplit:
    """Test the holdout_split function."""

    def test_holdout_has_correct_number_of_months(self):
        df = _make_monthly_df(n_months=24, n_ntas=2)
        _, holdout, _, _, _ = holdout_split(df, holdout_months=6)
        assert holdout["year_month"].nunique() == 6

    def test_raises_when_not_enough_months(self):
        df = _make_monthly_df(n_months=6, n_ntas=1)
        with pytest.raises(ValueError, match="Need more than"):
            holdout_split(df, holdout_months=6)


class TestStandardScale:
    """Test the standard_scale utility."""

    def test_train_has_zero_mean_unit_variance(self):
        rng = np.random.default_rng(42)
        X_train = pd.DataFrame({"a": rng.normal(10, 3, 100), "b": rng.normal(5, 2, 100)})
        X_test = pd.DataFrame({"a": rng.normal(10, 3, 20), "b": rng.normal(5, 2, 20)})
        scaled_train, _ = standard_scale(X_train, X_test)
        np.testing.assert_allclose(scaled_train.mean().values, 0, atol=1e-7)
        np.testing.assert_allclose(scaled_train.std(ddof=0).values, 1, atol=0.05)
