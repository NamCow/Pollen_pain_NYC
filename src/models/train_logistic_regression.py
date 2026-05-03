"""
Logistic Regression baseline for predicting above-median asthma ED visit months per NTA.

Walk-forward time-series CV: train on earlier months, test on later months.

Input:  data/processed/modeling_table.csv
Output: data/models/logistic_regression_predictions.csv
        data/models/logistic_regression_coefficients.csv
        data/models/logistic_regression_fold_results.csv
        data/models/logistic_regression_summary.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

OUTPUT_DIR = Path("./data/models")
from src.features.feature_engineering import (
    FEATURE_COLS,
    MIN_AVG_MONTHLY_ED_VISITS,
    TARGET,
    load_modeling_table,
    walk_forward_splits,
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_modeling_table()
    df = df[["nta_code", "NTAName", "borough", "year_month", TARGET] + FEATURE_COLS].copy()
    print(f"  {len(df):,} rows, {df['nta_code'].nunique()} NTAs, {df['year_month'].nunique()} months")
    print(f"  Date range: {df['year_month'].min()} to {df['year_month'].max()}")
    print(f"  Minimum mean monthly ED filter: {MIN_AVG_MONTHLY_ED_VISITS:.0f}+ cases")

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
        y_train = train["ed_high"]
        X_test = test[FEATURE_COLS]
        y_test = test["ed_high"]

        train_mean = X_train.mean()
        train_std = X_train.std() + 1e-8
        X_train_scaled = (X_train.fillna(train_mean) - train_mean) / train_std
        X_test_scaled = (X_test.fillna(train_mean) - train_mean) / train_std

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = model.predict(X_test_scaled)

        auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else float("nan")
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(f"\nFold {fold_i}: train<={train_end} -> test {test_start}..{test_end}")
        print(f"  Train: {len(train):,} rows  Test: {len(test):,} rows")
        print(f"  AUC: {auc:.3f}  Accuracy: {acc:.2%}  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")

        fold_results.append({
            "fold": fold_i, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "train_rows": len(train), "test_rows": len(test),
            "auc": auc, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1,
        })

        test_out = test[["nta_code", "borough", "year_month", TARGET, "ed_high"]].copy()
        test_out["pred_class"] = preds
        test_out["pred_prob"] = probs
        test_out["fold"] = fold_i
        all_preds.append(test_out)

    print("\n=== Training Final Model (all data) ===")
    X_all = df[FEATURE_COLS]
    y_all = df["ed_high"]
    all_mean = X_all.mean()
    all_std = X_all.std() + 1e-8
    X_all_scaled = (X_all.fillna(all_mean) - all_mean) / all_std

    final_model = LogisticRegression(max_iter=1000, random_state=42)
    final_model.fit(X_all_scaled, y_all)

    coefficients = pd.DataFrame({
        "feature": FEATURE_COLS,
        "coefficient": final_model.coef_[0],
        "abs_coefficient": np.abs(final_model.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False)

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv(OUTPUT_DIR / "logistic_regression_predictions.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "logistic_regression_coefficients.csv", index=False)

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(OUTPUT_DIR / "logistic_regression_fold_results.csv", index=False)

    summary_lines = [
        "=== LOGISTIC REGRESSION BASELINE EVALUATION ===",
        f"Total rows: {len(df):,}  NTAs: {df['nta_code'].nunique()}  Months: {df['year_month'].nunique()}",
        f"Modeling window: {df['year_month'].min()} to {df['year_month'].max()}",
        f"Target: above-median ED visits (median = {median_target:.2f})",
        f"Class balance: {df['ed_high'].mean():.2%} positive",
        "",
        "--- Walk-Forward CV Results ---",
        f"Mean AUC-ROC:   {fold_df['auc'].mean():.3f}",
        f"Mean Accuracy:  {fold_df['accuracy'].mean():.2%}",
        f"Mean Precision: {fold_df['precision'].mean():.3f}",
        f"Mean Recall:    {fold_df['recall'].mean():.3f}",
        f"Mean F1:        {fold_df['f1'].mean():.3f}",
        "",
        "--- Per-Fold Results ---",
    ]
    for _, row in fold_df.iterrows():
        summary_lines.append(
            f"  Fold {int(row['fold'])}: AUC={row['auc']:.3f}  Acc={row['accuracy']:.2%}  "
            f"P={row['precision']:.3f}  R={row['recall']:.3f}  F1={row['f1']:.3f}"
        )

    summary_lines.extend([
        "",
        "--- Coefficients (top 10 by magnitude) ---",
    ])
    for _, row in coefficients.head(10).iterrows():
        summary_lines.append(f"  {row['feature']:30s} {row['coefficient']:+.4f}")

    summary_lines.extend([
        "",
        f"Intercept: {final_model.intercept_[0]:.4f}",
    ])

    summary = "\n".join(summary_lines)
    (OUTPUT_DIR / "logistic_regression_summary.txt").write_text(summary)

    print(f"\n{summary}")
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
