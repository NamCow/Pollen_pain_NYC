import pandas as pd


import pandas as pd
from pathlib import Path


def main():
    input_path = Path("./data/raw/openmeteo_weather_daily_by_nta.csv")
    output_path = Path("./data/processed/weather_monthly_nyc.csv")

    df = pd.read_csv(input_path)

    df["time"] = pd.to_datetime(df["time"])

    df["month"] = df["time"].dt.to_period("M").astype(str)

    df_monthly_nyc = (
        df.groupby("month", as_index=False)
        .agg({
            "temperature_2m_max": "mean",
            "temperature_2m_min": "mean",
            "precipitation_sum": "sum",
            "wind_speed_10m_max": "mean",
        })
        .sort_values("month")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_monthly_nyc.to_csv(output_path, index=False)

    print("Saved to:", output_path)
    print(df_monthly_nyc.head())


if __name__ == "__main__":
    main()