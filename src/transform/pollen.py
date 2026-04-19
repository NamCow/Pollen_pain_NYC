"""
Process AAAAI pollen data: compute composite index and lag features.

Pollen is city-wide (one station) — same value broadcast to all NTAs.

Input:  data/raw/aaaai_pollen_daily.csv
Output: data/processed/pollen_monthly_features.csv
"""

import pandas as pd
from pathlib import Path

INPUT = Path("./data/raw/aaaai_pollen_daily.csv")
OUTPUT = Path("./data/processed/pollen_monthly_features.csv")

CATEGORY_WEIGHTS = {"TREE": 0.4, "WEED": 0.25, "GRASS": 0.25, "MOLD": 0.1}


def main():
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])

    for cat in CATEGORY_WEIGHTS:
        if cat not in df.columns:
            df[cat] = 0
    df = df.fillna(0)

    df["composite_pollen"] = sum(
        df[cat] * w for cat, w in CATEGORY_WEIGHTS.items()
    )

    df["pollen_14d_avg"] = df["composite_pollen"].rolling(14, min_periods=1).mean()
    df["pollen_28d_avg"] = df["composite_pollen"].rolling(28, min_periods=1).mean()

    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby("year_month", as_index=False)
        .agg(
            pollen_composite_avg=("composite_pollen", "mean"),
            pollen_composite_max=("composite_pollen", "max"),
            tree_avg=("TREE", "mean"),
            tree_max=("TREE", "max"),
            weed_avg=("WEED", "mean"),
            grass_avg=("GRASS", "mean"),
            mold_avg=("MOLD", "mean"),
            pollen_14d_lag_avg=("pollen_14d_avg", "last"),
            pollen_28d_lag_avg=("pollen_28d_avg", "last"),
        )
        .sort_values("year_month")
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT, index=False)
    print(f"Saved: {OUTPUT} ({len(monthly)} rows)")
    print(monthly.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
