"""
Feature definitions and engineering utilities shared across training scripts.
"""

import pandas as pd

FEATURE_COLS = [
    "temp_max_mean", "temp_min_mean", "precip_total", "wind_max_mean",
    "pollen_composite_avg", "pollen_composite_max",
    "tree_avg", "tree_max", "weed_avg", "grass_avg",
    "pollen_14d_lag_avg", "pollen_28d_lag_avg",
    "pm25_mean", "ozone_mean", "no2_mean",
    "tree_count", "total_dbh", "mean_dbh", "pct_good_health",
    "chs_asthma_pct",
]

TARGET = "ed_visits"


def walk_forward_splits(df: pd.DataFrame, n_test_months: int = 6):
    """Yield (train, test, train_end, test_start, test_end) for walk-forward CV.

    Always trains on past data and tests on future data — never random k-fold,
    which would leak future months into training.
    """
    months = sorted(df["year_month"].unique())
    total = len(months)
    min_train = 12

    for i in range(min_train, total, n_test_months):
        test_end = min(i + n_test_months, total)
        train_months = months[:i]
        test_months = months[i:test_end]
        if not test_months:
            break
        train = df[df["year_month"].isin(train_months)]
        test = df[df["year_month"].isin(test_months)]
        yield train, test, train_months[-1], test_months[0], test_months[-1]


def standard_scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale X_test using X_train statistics (no data leakage)."""
    mu = X_train.mean()
    sigma = X_train.std() + 1e-8
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def load_modeling_table(path: str = "./data/processed/modeling_table.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["pollen_composite_avg", "temp_max_mean"])
    return df
