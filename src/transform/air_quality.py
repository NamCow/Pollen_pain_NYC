"""
Merge EPA (2022-2024) and AirNow (2025) air quality into one file.

City-wide monthly means — broadcast to all NTAs during final join.

Input:  data/raw/epa_air_quality_monthly_2022_2024.csv
        data/raw/epa_air_quality_monthly_2025.csv
Output: data/processed/air_quality_monthly.csv
"""

import pandas as pd
from pathlib import Path

EPA_CSV = Path("./data/raw/epa_air_quality_monthly_2022_2024.csv")
AIRNOW_CSV = Path("./data/raw/epa_air_quality_monthly_2025.csv")
OUTPUT = Path("./data/processed/air_quality_monthly.csv")


def main():
    epa = pd.read_csv(EPA_CSV)

    rename_map = {}
    for col in epa.columns:
        if "88101" in col or "pm25" in col.lower():
            rename_map[col] = "pm25_mean"
        elif "42602" in col or "no2" in col.lower():
            rename_map[col] = "no2_mean"
        elif "44201" in col or "ozone" in col.lower():
            rename_map[col] = "ozone_mean"
    epa = epa.rename(columns=rename_map)

    if "ozone_mean" in epa.columns and epa["ozone_mean"].max() < 1:
        epa["ozone_mean"] = epa["ozone_mean"] * 1000

    epa_clean = epa[["month", "pm25_mean", "ozone_mean"]].copy()
    if "no2_mean" in epa.columns:
        epa_clean["no2_mean"] = epa["no2_mean"]

    airnow = pd.read_csv(AIRNOW_CSV)
    airnow_clean = airnow[["month", "pm25_mean", "ozone_mean"]].copy()

    merged = pd.concat([epa_clean, airnow_clean], ignore_index=True)
    merged["month"] = pd.to_datetime(merged["month"], format="%Y-%m")
    merged = merged.sort_values("month").reset_index(drop=True)
    merged["year_month"] = merged["month"].dt.to_period("M").astype(str)
    merged = merged.drop(columns=["month"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)
    print(f"Saved: {OUTPUT} ({len(merged)} rows)")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
