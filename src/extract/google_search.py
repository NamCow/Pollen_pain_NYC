"""
Fetch monthly Google Trends data (keep Google Trends normalized 0-100 scale)
for pollen/asthma-related keywords and export to CSV.

Usage:
    pip install pytrends pandas
    python google_trends_monthly_nyc.py

Notes:
- Google Trends is normalized 0-100 within the selected time range / location.
- This script keeps that scale as returned by Google Trends.
- pytrends is unofficial and may occasionally break if Google changes endpoints.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq


# ---------------------------
# CONFIG
# ---------------------------
KEYWORDS = [
    "pollen allergy",
    "asthma",
    "allergy symptoms",
    "pollen count",
    "asthma symptoms",
]

GEO = "US-NY"


TIMEFRAME = "2022-01-01 2025-12-31"

# tz=300 means UTC-5, matching New York standard time conventionally used in many examples.
TZ = 300

# retry / pause controls
SLEEP_SECONDS = 3
MAX_RETRIES = 3

OUTPUT_DIR = Path("./data/processed")
OUTPUT_CSV = OUTPUT_DIR / "google_trends_monthly_ny_2022_2025.csv"



def fetch_one_keyword(pytrends: TrendReq, keyword: str) -> pd.DataFrame:
    """
    Fetch daily/weekly interest over time for one keyword and aggregate to monthly mean.
    Keeps Google Trends 0-100 scale for that keyword exactly as returned.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pytrends.build_payload(
                kw_list=[keyword],
                cat=0,
                timeframe=TIMEFRAME,
                geo=GEO,
                gprop=""
            )

            df = pytrends.interest_over_time()

            if df.empty:
                raise ValueError(f"No data returned for keyword: {keyword}")

            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            # index -> date column
            df = df.reset_index().rename(columns={"date": "date"})

            # monthly aggregation
            df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

            monthly = (
                df.groupby("month", as_index=False)[keyword]
                .mean()
                .rename(columns={keyword: keyword.replace(" ", "_")})
            )

            return monthly

        except Exception as e:
            last_error = e
            print(f"[Attempt {attempt}/{MAX_RETRIES}] Failed for '{keyword}': {e}")
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_SECONDS)

    raise RuntimeError(f"Failed to fetch keyword '{keyword}' after {MAX_RETRIES} attempts") from last_error


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pytrends = TrendReq(
        hl="en-US",
        tz=TZ,
        timeout=(10, 25),
        retries=2,
        backoff_factor=0.3,
    )

    frames = []

    for keyword in KEYWORDS:
        print(f"Fetching: {keyword}")
        monthly = fetch_one_keyword(pytrends, keyword)
        frames.append(monthly)
        time.sleep(SLEEP_SECONDS)

    # outer merge all keywords on month
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="month", how="outer")

    result = result.sort_values("month").reset_index(drop=True)

    # optional composite feature, still on 0-100-ish comparable index scale
    trend_cols = [c for c in result.columns if c != "month"]
    result["search_index_mean"] = result[trend_cols].mean(axis=1)

    result.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved:", OUTPUT_CSV.resolve())
    print(result.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
