"""
Aggregate daily weather → monthly per NTA.

Input:  data/raw/openmeteo_weather_daily_by_nta.csv  (daily, per NTA)
Output: data/processed/weather_monthly_by_nta.csv    (monthly, per NTA)
"""

import pandas as pd
from pathlib import Path

INPUT = Path("./data/raw/openmeteo_weather_daily_by_nta.csv")
OUTPUT = Path("./data/processed/weather_monthly_by_nta.csv")


def main():
    df = pd.read_csv(INPUT)
    df["time"] = pd.to_datetime(df["time"])
    df["year_month"] = df["time"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby(["nta_code", "year_month"], as_index=False)
        .agg(
            temp_max_mean=("temperature_2m_max", "mean"),
            temp_min_mean=("temperature_2m_min", "mean"),
            precip_total=("precipitation_sum", "sum"),
            wind_max_mean=("wind_speed_10m_max", "mean"),
        )
        .sort_values(["nta_code", "year_month"])
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT, index=False)
    print(f"Saved: {OUTPUT} ({len(monthly):,} rows, {monthly['nta_code'].nunique()} NTAs)")


if __name__ == "__main__":
    main()
