import pandas as pd

df = pd.read_csv("./data/raw/filtered_asthma_data.csv", sep=";")

df["Date"] = pd.to_datetime(df["Date"])

monthly_avg = (
    df.groupby("Date", as_index=False)["number_of_visits"]
      .mean()
      .rename(columns={
          "Date": "month_date",
          "number_of_visits": "avg_visits"
      })
)

monthly_avg["month_date"] = monthly_avg["month_date"].dt.strftime("%Y-%m-%d")

monthly_avg.to_csv("./data/processed/asthma_monthly_avg_simple.csv", index=False)

print("Done ! File saved: asthma_monthly_avg_simple.csv")