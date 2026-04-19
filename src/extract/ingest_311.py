import requests
import pandas as pd
from pathlib import Path

SOCRATA_ENDPOINT = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
OUTPUT_CSV = "./data/raw/311_air_quality.csv"
PAGE_SIZE = 50_000

COLUMNS = (
    "unique_key,created_date,closed_date,complaint_type,descriptor,"
    "incident_zip,latitude,longitude,borough,community_board,bbl"
)
WHERE = (
    "complaint_type='Air Quality' "
    "AND created_date >= '2022-03-01T00:00:00' "
    "AND created_date <= '2025-11-01T23:59:59'"
)


def fetch_all() -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        resp = requests.get(
            SOCRATA_ENDPOINT,
            params={
                "$where": WHERE,
                "$select": COLUMNS,
                "$limit": PAGE_SIZE,
                "$offset": offset,
                "$order": "created_date ASC",
            },
            timeout=90,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        print(f"  {len(rows):,} records fetched...")
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return pd.DataFrame(rows)


def main():
    print("Fetching NYC 311 Air Quality complaints...")
    df = fetch_all()
    if df.empty:
        print("No data returned.")
        return
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}  ({len(df):,} rows)")
    print(df["descriptor"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
