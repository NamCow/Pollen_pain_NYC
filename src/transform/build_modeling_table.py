"""
Joins all processed features + target into the final modeling table.

One row per NTA per month. Pollen and air quality are city-wide (broadcast).
Weather and tree canopy vary by NTA. CHS baseline is static per NTA.

Inputs (all from data/processed/):
  - asthma_ed_monthly_nta.csv      (target — monthly ED counts per NTA)
  - weather_monthly_by_nta.csv     (monthly weather per NTA)
  - pollen_monthly_features.csv    (monthly pollen, city-wide)
  - air_quality_monthly.csv        (monthly PM2.5/ozone, city-wide)
  - tree_canopy_by_nta.csv         (static canopy score per NTA)
  - chs_baseline_by_nta.csv        (static asthma prevalence per NTA)

Output:
  - data/processed/modeling_table.csv
"""

import pandas as pd
from pathlib import Path

from src.utils.config import PROCESSED_DIR

PROCESSED = PROCESSED_DIR
OUTPUT = PROCESSED / "modeling_table.csv"


def main():
    print("Loading target variable...")
    target = pd.read_csv(PROCESSED / "asthma_ed_monthly_nta.csv")
    target = target[["nta_code", "NTAName", "borough", "year_month", "estimated_count"]]
    target = target.rename(columns={"estimated_count": "ed_visits"})
    print(f"  {len(target):,} rows, {target['nta_code'].nunique()} NTAs, {target['year_month'].nunique()} months")

    print("Loading weather...")
    weather = pd.read_csv(PROCESSED / "weather_monthly_by_nta.csv")
    merged = target.merge(weather, on=["nta_code", "year_month"], how="left")
    print(f"  After weather join: {len(merged):,} rows, weather nulls: {merged['temp_max_mean'].isna().sum()}")

    print("Loading pollen (broadcast to all NTAs)...")
    pollen = pd.read_csv(PROCESSED / "pollen_monthly_features.csv")
    merged = merged.merge(pollen, on="year_month", how="left")
    print(f"  After pollen join: pollen nulls: {merged['pollen_composite_avg'].isna().sum()}")

    print("Loading air quality (broadcast to all NTAs)...")
    aq = pd.read_csv(PROCESSED / "air_quality_monthly.csv")
    merged = merged.merge(aq, on="year_month", how="left")
    print(f"  After air quality join: pm25 nulls: {merged['pm25_mean'].isna().sum()}")

    print("Loading tree canopy (static per NTA)...")
    canopy = pd.read_csv(PROCESSED / "tree_canopy_by_nta.csv")
    merged = merged.merge(canopy, on="nta_code", how="left")
    print(f"  After canopy join: canopy nulls: {merged['tree_count'].isna().sum()}")

    print("Loading CHS baseline (static per NTA)...")
    chs = pd.read_csv(PROCESSED / "chs_baseline_by_nta.csv")
    chs = chs[["nta_code", "chs_asthma_pct"]]
    merged = merged.merge(chs, on="nta_code", how="left")
    print(f"  After CHS join: chs nulls: {merged['chs_asthma_pct'].isna().sum()}")

    merged = merged.sort_values(["nta_code", "year_month"]).reset_index(drop=True)

    merged.to_csv(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT} ({len(merged):,} rows, {merged.shape[1]} columns)")
    print(f"Columns: {list(merged.columns)}")
    print(f"\nNull summary:")
    print(merged.isna().sum().to_string())
    print(f"\nSample:")
    print(merged.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
