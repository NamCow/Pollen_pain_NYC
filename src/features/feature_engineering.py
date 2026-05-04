"""
Feature definitions and engineering utilities shared across training scripts.

All constants (feature lists, hyperparameters, thresholds) are loaded from
config/settings.yaml via src.utils.config so that every script shares a
single source of truth.
"""

import math

import pandas as pd

from src.utils.config import (
    FEATURE_COLS,
    MIN_AVG_MONTHLY_ED_VISITS,
    POLLEN_SEASON_MONTHS,
    REQUIRED_MODEL_FEATURES,
    TARGET,
    XGBOOST_PARAMS,
    CV_MIN_TRAIN_MONTHS,
    PROCESSED_DIR,
)


def walk_forward_splits(df: pd.DataFrame, n_test_months: int = 6):
    """Yield (train, test, train_end, test_start, test_end) for walk-forward CV.

    Always trains on past data and tests on future data — never random k-fold,
    which would leak future months into training.
    """
    months = sorted(df["year_month"].unique())
    total = len(months)
    min_train = CV_MIN_TRAIN_MONTHS

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


def filter_pollen_season(df: pd.DataFrame, year_month_col: str = "year_month") -> pd.DataFrame:
    """Keep only March-October rows, matching the pollen monitoring season."""
    year_month = pd.to_datetime(df[year_month_col] + "-01")
    return df[year_month.dt.month.isin(POLLEN_SEASON_MONTHS)].copy()


def add_engineered_features(df: pd.DataFrame, year_month_col: str = "year_month") -> pd.DataFrame:
    """Create low-leakage derived features from existing monthly signals."""
    out = df.copy().sort_values(["nta_code", year_month_col]).reset_index(drop=True)
    year_month = pd.to_datetime(out[year_month_col] + "-01")
    month = year_month.dt.month

    angle = 2.0 * math.pi * month.astype(float) / 12.0
    out["month_sin"] = angle.map(math.sin)
    out["month_cos"] = angle.map(math.cos)
    out["season_progress"] = (month - min(POLLEN_SEASON_MONTHS)) / (len(POLLEN_SEASON_MONTHS) - 1)

    out["temp_diurnal_range"] = out["temp_max_mean"] - out["temp_min_mean"]
    out["warm_dry_index"] = out["temp_max_mean"] / (1.0 + out["precip_total"].clip(lower=0))
    out["pollen_change_vs_28d"] = out["pollen_composite_avg"] - out["pollen_28d_lag_avg"]
    out["pollen_temp_interaction"] = out["pollen_composite_avg"] * out["temp_max_mean"]
    out["pollen_pm25_interaction"] = out["pollen_composite_avg"] * out["pm25_mean"]
    out["pollen_chs_interaction"] = out["pollen_composite_avg"] * out["chs_asthma_pct"]
    out["weed_temp_interaction"] = out["weed_avg"] * out["temp_max_mean"]
    return out


def filter_low_case_ntas(
    df: pd.DataFrame,
    target_col: str = TARGET,
    nta_col: str = "nta_code",
    min_avg_monthly_cases: float = MIN_AVG_MONTHLY_ED_VISITS,
) -> pd.DataFrame:
    """Drop NTAs whose average monthly ED burden is too small for stable modeling."""
    nta_mean = df.groupby(nta_col)[target_col].mean()
    keep_ntas = nta_mean[nta_mean >= min_avg_monthly_cases].index
    return df[df[nta_col].isin(keep_ntas)].copy()


def holdout_split(df: pd.DataFrame, holdout_months: int = 6):
    """Split rows into development and final holdout windows using the latest months."""
    months = sorted(df["year_month"].unique())
    if len(months) <= holdout_months:
        raise ValueError(
            f"Need more than {holdout_months} months to create a holdout split; found {len(months)} months."
        )
    holdout_window = months[-holdout_months:]
    dev_window = months[:-holdout_months]
    dev = df[df["year_month"].isin(dev_window)].copy()
    holdout = df[df["year_month"].isin(holdout_window)].copy()
    return dev, holdout, dev_window[-1], holdout_window[0], holdout_window[-1]


def load_modeling_table(
    path: str = str(PROCESSED_DIR / "modeling_table.csv"),
    season_only: bool = True,
    require_complete_features: bool = True,
    min_avg_monthly_cases: float | None = MIN_AVG_MONTHLY_ED_VISITS,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if season_only:
        df = filter_pollen_season(df)
    if require_complete_features:
        df = df.dropna(subset=REQUIRED_MODEL_FEATURES)
    if min_avg_monthly_cases is not None:
        df = filter_low_case_ntas(df, min_avg_monthly_cases=min_avg_monthly_cases)
    df = add_engineered_features(df)
    return df
