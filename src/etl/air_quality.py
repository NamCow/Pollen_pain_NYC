import pandas as pd

airnow_2025 = pd.read_csv("./data/Input/airnow_nyc_monthly_2025.csv")
epa_2022_2024 = pd.read_csv("./data/Input/air_quality_nyc_monthly.csv")

epa_clean = epa_2022_2024.copy()

epa_clean = epa_clean.rename(columns={
    "88101_mean": "pm25_mean",
    "44201_mean": "ozone_mean"
})

epa_clean["ozone_mean"] = epa_clean["ozone_mean"] * 1000

epa_clean = epa_clean[["month", "pm25_mean", "ozone_mean"]]

airnow_clean = airnow_2025.copy()

airnow_clean = airnow_clean[["month", "pm25_mean", "ozone_mean"]]

merged = pd.concat([epa_clean, airnow_clean], ignore_index=True)

merged["month"] = pd.to_datetime(merged["month"], format="%Y-%m")
merged = merged.sort_values("month").reset_index(drop=True)

merged["month"] = merged["month"].dt.strftime("%Y-%m")

merged.to_csv("./data/processed/air_quality_nyc_monthly_2022_2025_clean.csv", index=False)

print("Done!")
print(merged.head())
print(merged.tail())