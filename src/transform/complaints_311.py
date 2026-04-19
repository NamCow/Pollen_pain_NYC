"""
Aggregate 311 air quality complaints → monthly count per borough.

Used as a validation signal, not a model feature.

Input:  data/raw/nyc311_air_quality_complaints.csv
Output: data/processed/311_monthly_by_borough.csv
"""

import pandas as pd
from pathlib import Path

INPUT = Path("./data/raw/nyc311_air_quality_complaints.csv")
OUTPUT = Path("./data/processed/311_monthly_by_borough.csv")


def main():
    df = pd.read_csv(INPUT, usecols=["created_date", "borough"], low_memory=False)
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df = df.dropna(subset=["created_date", "borough"])
    df = df[df["borough"] != "Unspecified"]

    df["year_month"] = df["created_date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby(["borough", "year_month"], as_index=False)
        .size()
        .rename(columns={"size": "complaint_count"})
        .sort_values(["borough", "year_month"])
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT, index=False)
    print(f"Saved: {OUTPUT} ({len(monthly)} rows)")
    print(monthly.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
