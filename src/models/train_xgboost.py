"""
Train XGBoost regression and Logistic Regression baseline for
predicting monthly asthma ED visits per NTA.

Walk-forward time-series CV: train on earlier months, test on later months.
Never random k-fold — that would leak future data.

Input:  data/processed/modeling_table.csv
Output: data/models/xgboost_predictions.csv
        data/models/feature_importance.csv
        data/models/fold_results.csv
        data/models/evaluation_summary.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error,
    roc_auc_score, accuracy_score,
)
from scipy.stats import pearsonr
import xgboost as xgb

INPUT = Path("./data/processed/modeling_table.csv")
OUTPUT_DIR = Path("./data/models")

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
    months = sorted(df["year_month"].unique())
    total = len(months)
    min_train = 12
    step = n_test_months

    for i in range(min_train, total, step):
        test_end = min(i + step, total)
        train_months = months[:i]
        test_months = months[i:test_end]

        if len(test_months) == 0:
            break

        train = df[df["year_month"].isin(train_months)]
        test = df[df["year_month"].isin(test_months)]
        yield train, test, train_months[-1], test_months[0], test_months[-1]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(INPUT)
    df = df.dropna(subset=["pollen_composite_avg", "temp_max_mean"])
    df = df[["nta_code", "NTAName", "borough", "year_month", TARGET] + FEATURE_COLS].copy()
    print(f"  {len(df):,} rows, {df['nta_code'].nunique()} NTAs, {df['year_month'].nunique()} months")
    print(f"  Date range: {df['year_month'].min()} to {df['year_month'].max()}")

    median_target = df[TARGET].median()
    df["ed_high"] = (df[TARGET] >= median_target).astype(int)
    print(f"  Target median: {median_target:.2f}")
    print(f"  High class balance: {df['ed_high'].mean():.2%}")

    print("\n=== Walk-Forward Cross-Validation ===")
    all_preds = []
    fold_results = []

    for fold_i, (train, test, train_end, test_start, test_end) in enumerate(
        walk_forward_splits(df, n_test_months=6), 1
    ):
        X_train = train[FEATURE_COLS]
        y_train = train[TARGET]
        X_test = test[FEATURE_COLS]
        y_test = test[TARGET]

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        y_train_bin = (y_train >= median_target).astype(int)
        y_test_bin = (y_test >= median_target).astype(int)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        train_mean = X_train.mean()
        train_std = X_train.std() + 1e-8
        X_train_scaled = (X_train.fillna(train_mean) - train_mean) / train_std
        X_test_scaled = (X_test.fillna(train_mean) - train_mean) / train_std
        lr.fit(X_train_scaled, y_train_bin)
        lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
        lr_preds = lr.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        corr, corr_p = pearsonr(y_test, preds)

        auc = roc_auc_score(y_test_bin, lr_probs) if y_test_bin.nunique() > 1 else float("nan")
        lr_acc = accuracy_score(y_test_bin, lr_preds)

        test_copy = test.copy()
        test_copy["pred"] = preds
        nta_mae = test_copy.groupby("nta_code").apply(
            lambda g: mean_absolute_error(g[TARGET], g["pred"]),
            include_groups=False,
        )
        pct_within_15 = (nta_mae / test_copy.groupby("nta_code")[TARGET].mean() < 0.15).mean()

        print(f"\nFold {fold_i}: train<={train_end} -> test {test_start}..{test_end}")
        print(f"  Train: {len(train):,} rows  Test: {len(test):,} rows")
        print(f"  XGBoost  - MAE: {mae:.2f}  RMSE: {rmse:.2f}  Pearson r: {corr:.3f} (p={corr_p:.2e})")
        print(f"  LogReg   - AUC: {auc:.3f}  Accuracy: {lr_acc:.2%}")
        print(f"  NTAs within 15% MAE: {pct_within_15:.1%}")

        fold_results.append({
            "fold": fold_i, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "train_rows": len(train), "test_rows": len(test),
            "mae": mae, "rmse": rmse, "pearson_r": corr, "pearson_p": corr_p,
            "lr_auc": auc, "lr_accuracy": lr_acc,
            "pct_ntas_within_15pct": pct_within_15,
        })

        test_out = test[["nta_code", "borough", "year_month", TARGET]].copy()
        test_out["pred"] = preds
        test_out["fold"] = fold_i
        all_preds.append(test_out)

    print("\n=== Training Final Model (all data) ===")
    X_all = df[FEATURE_COLS]
    y_all = df[TARGET]
    final_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    final_model.fit(X_all, y_all)

    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv(OUTPUT_DIR / "xgboost_predictions.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(OUTPUT_DIR / "fold_results.csv", index=False)

    summary_lines = [
        "=== EVALUATION SUMMARY ===",
        f"Total rows: {len(df):,}  NTAs: {df['nta_code'].nunique()}  Months: {df['year_month'].nunique()}",
        f"Modeling window: {df['year_month'].min()} to {df['year_month'].max()}",
        "",
        "--- XGBoost Regression (walk-forward CV) ---",
        f"Mean MAE:  {fold_df['mae'].mean():.2f}",
        f"Mean RMSE: {fold_df['rmse'].mean():.2f}",
        f"Mean Pearson r: {fold_df['pearson_r'].mean():.3f}",
        f"Mean NTAs within 15%: {fold_df['pct_ntas_within_15pct'].mean():.1%}",
        "",
        "--- Logistic Regression Baseline ---",
        f"Mean AUC-ROC: {fold_df['lr_auc'].mean():.3f}",
        f"Mean Accuracy: {fold_df['lr_accuracy'].mean():.2%}",
        "",
        "--- Feature Importance (top 10) ---",
    ]
    for _, row in importance.head(10).iterrows():
        summary_lines.append(f"  {row['feature']:30s} {row['importance']:.4f}")

    summary = "\n".join(summary_lines)
    (OUTPUT_DIR / "evaluation_summary.txt").write_text(summary)

    print(f"\n{summary}")
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
