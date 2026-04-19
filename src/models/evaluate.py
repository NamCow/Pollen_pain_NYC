"""
Load saved fold results and print / export a consolidated evaluation report.

Input:  data/models/fold_results.csv
        data/models/logreg_fold_results.csv  (optional)
        data/models/feature_importance.csv
Output: prints summary to stdout; optionally writes data/models/full_report.txt
"""

import sys
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("./data/models")


def load(filename: str) -> pd.DataFrame | None:
    path = MODEL_DIR / filename
    if not path.exists():
        print(f"  [skip] {filename} not found")
        return None
    return pd.read_csv(path)


def evaluate_xgboost(fold_df: pd.DataFrame) -> list[str]:
    lines = [
        "--- XGBoost Regression (walk-forward CV) ---",
        f"Folds:      {len(fold_df)}",
        f"Mean MAE:   {fold_df['mae'].mean():.2f}  (±{fold_df['mae'].std():.2f})",
        f"Mean RMSE:  {fold_df['rmse'].mean():.2f}  (±{fold_df['rmse'].std():.2f})",
        f"Mean Pearson r: {fold_df['pearson_r'].mean():.3f}",
        f"Mean NTAs within 15%: {fold_df['pct_ntas_within_15pct'].mean():.1%}",
    ]
    lines.append("\nPer-fold breakdown:")
    for _, row in fold_df.iterrows():
        lines.append(
            f"  Fold {int(row['fold'])}: {row['test_start']}..{row['test_end']}"
            f"  MAE={row['mae']:.2f}  r={row['pearson_r']:.3f}"
        )
    return lines


def evaluate_logreg(fold_df: pd.DataFrame) -> list[str]:
    lines = [
        "--- Logistic Regression Baseline (walk-forward CV) ---",
        f"Mean AUC-ROC:  {fold_df['auc'].mean():.3f}  (±{fold_df['auc'].std():.3f})",
        f"Mean Accuracy: {fold_df['accuracy'].mean():.2%}",
    ]
    return lines


def top_features(imp_df: pd.DataFrame, n: int = 10) -> list[str]:
    lines = [f"--- Top {n} Features (XGBoost importance) ---"]
    for _, row in imp_df.head(n).iterrows():
        lines.append(f"  {row['feature']:30s} {row['importance']:.4f}")
    return lines


def main():
    print("Loading evaluation data...\n")
    xgb_folds  = load("fold_results.csv")
    lr_folds   = load("logreg_fold_results.csv")
    importance = load("feature_importance.csv")

    report_lines = ["=" * 50, "POLLEN & PAIN — MODEL EVALUATION REPORT", "=" * 50, ""]

    if xgb_folds is not None:
        report_lines += evaluate_xgboost(xgb_folds)
        report_lines.append("")

    if lr_folds is not None:
        report_lines += evaluate_logreg(lr_folds)
        report_lines.append("")

    if importance is not None:
        report_lines += top_features(importance)
        report_lines.append("")

    report = "\n".join(report_lines)
    print(report)

    out_path = MODEL_DIR / "full_report.txt"
    out_path.write_text(report)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
