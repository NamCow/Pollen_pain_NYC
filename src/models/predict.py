"""
Train a final XGBoost model on all available seasonal rows and generate
predictions for every NTA x month in the usable modeling window.

Input:  data/processed/modeling_table.csv
Output: data/models/final_predictions.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb

from src.features.feature_engineering import FEATURE_COLS, TARGET, load_modeling_table

OUTPUT_DIR = Path("./data/models")


def train_final_model(df: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
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
