"""
Standalone re-prediction script (NOT the canonical training pipeline).

This script trains a final XGBoost model on all available rows and generates
fitted predictions for every NTA x month in the modeling window.  It does NOT
perform walk-forward cross-validation, holdout evaluation, baseline comparison,
or any of the diagnostic checks that the canonical pipeline produces.

Use this only when you need to quickly regenerate final_predictions.csv without
re-running the full evaluation suite.  For the complete training + evaluation
pipeline, use ``train_xgboost.py`` instead, which produces CV fold results,
holdout metrics, feature importance, baseline comparisons, and the same
final_predictions.csv as a last step.

Input:  data/processed/modeling_table.csv
Output: data/models/final_predictions.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb

from src.features.feature_engineering import FEATURE_COLS, TARGET, XGBOOST_PARAMS, load_modeling_table
from src.utils.config import MODELS_DIR

OUTPUT_DIR = MODELS_DIR


def train_final_model(df: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(df[FEATURE_COLS], df[TARGET])
    return model


def build_predictions(model: xgb.XGBRegressor, df: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(df[FEATURE_COLS])
    out = df[["nta_code", "NTAName", "borough", "year_month", TARGET]].copy()
    out["pred"] = np.maximum(preds, 0)
    out["residual"] = out[TARGET] - out["pred"]
    out["fold"] = 0
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading modeling table...")
    df = load_modeling_table()
    print(f"  {len(df):,} rows | {df['nta_code'].nunique()} NTAs | {df['year_month'].nunique()} months")

    print("Training final model on all usable rows...")
    model = train_final_model(df)

    print("Generating all-data predictions...")
    result = build_predictions(model, df)

    out_path = OUTPUT_DIR / "final_predictions.csv"
    result.to_csv(out_path, index=False)

    mae = (result["residual"].abs()).mean()
    print(f"  In-sample MAE: {mae:.2f}")
    print(f"Saved: {out_path} ({len(result):,} rows)")
    print(result.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
