"""
Standalone Logistic Regression baseline for classifying high vs low
monthly asthma ED visits per NTA.

Label: ed_high = 1 if ed_visits >= median, else 0.
CV:    Walk-forward time-series splits (no random k-fold).

Input:  data/processed/modeling_table.csv
Output: data/models/logreg_predictions.csv
        data/models/logreg_fold_results.csv
        data/models/logreg_summary.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from src.features.feature_engineering import (
    FEATURE_COLS, TARGET, walk_forward_splits, standard_scale, load_modeling_table,
)

OUTPUT_DIR = Path("./data/models")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_modeling_table()
    median_target = df[TARGET].median()
    df["ed_high"] = (df[TARGET] >= median_target).astype(int)
    print(f"  {len(df):,} rows | {df['nta_code'].nunique()} NTAs | {df['year_month'].nunique()} months")
    print(f"  Positive class rate: {df['ed_high'].mean():.2%}")

    print("\n=== Walk-Forward Cross-Validation ===")
    all_preds = []
    fold_results = []

    for fold_i, (train, test, train_end, test_start, test_end) in enumerate(
        walk_forward_splits(df, n_test_months=6), 1
    ):
        X_train = train[FEATURE_COLS]
        y_train = train["ed_high"]
        X_test  = test[FEATURE_COLS]
        y_test  = test["ed_high"]

        X_train_s, X_test_s = standard_scale(X_train, X_test)

        lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr.fit(X_train_s, y_train)

        probs = lr.predict_proba(X_test_s)[:, 1]
        preds = lr.predict(X_test_s)

        auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else float("nan")
        acc = accuracy_score(y_test, preds)

        print(f"\nFold {fold_i}: train<={train_end}  test {test_start}..{test_end}")
        print(f"  AUC-ROC: {auc:.3f}  Accuracy: {acc:.2%}")

        fold_results.append({
            "fold": fold_i, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "auc": auc, "accuracy": acc,
            "train_rows": len(train), "test_rows": len(test),
        })

        out = test[["nta_code", "borough", "year_month", TARGET, "ed_high"]].copy()
        out["prob_high"] = probs
        out["pred_high"] = preds
        out["fold"] = fold_i
        all_preds.append(out)

    fold_df = pd.DataFrame(fold_results)
    pred_df = pd.concat(all_preds, ignore_index=True)

    fold_df.to_csv(OUTPUT_DIR / "logreg_fold_results.csv", index=False)
    pred_df.to_csv(OUTPUT_DIR / "logreg_predictions.csv", index=False)

    summary_lines = [
        "=== LOGISTIC REGRESSION EVALUATION SUMMARY ===",
        f"Total rows: {len(df):,}  NTAs: {df['nta_code'].nunique()}  Months: {df['year_month'].nunique()}",
        f"Modeling window: {df['year_month'].min()} to {df['year_month'].max()}",
        f"Target median (ed_visits): {median_target:.2f}",
        "",
        f"Mean AUC-ROC:  {fold_df['auc'].mean():.3f}",
        f"Mean Accuracy: {fold_df['accuracy'].mean():.2%}",
        "",
        "Per-fold results:",
    ]
    for _, row in fold_df.iterrows():
        summary_lines.append(
            f"  Fold {int(row['fold'])}: AUC={row['auc']:.3f}  Acc={row['accuracy']:.2%}"
            f"  ({int(row['train_rows'])} train / {int(row['test_rows'])} test)"
        )

    summary = "\n".join(summary_lines)
    (OUTPUT_DIR / "logreg_summary.txt").write_text(summary)
    print(f"\n{summary}")
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
