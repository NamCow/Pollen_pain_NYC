"""
Fetches multi-year daily pollen counts from the AAAAI National Allergy Bureau
pollen counting station in NYC via their public GraphQL API.

Station: NYC (ID 5effe609-c645-4620-bc99-a3b34934897c)
Categories: TREE, WEED, GRASS, MOLD
Date range: April–November 2022–2025

Output: daily rows with per-category pollen counts, plus monthly summary.
"""

import time
import json
import requests
import pandas as pd
from pathlib import Path

GQL_URL = "https://pollen.aaaai.org/graphql/public"
STATION_ID = "5effe609-c645-4620-bc99-a3b34934897c"

YEARS = [2022, 2023, 2024, 2025]
MONTHS = [3, 4, 5, 6, 7, 8, 9, 10]

OUTPUT_DIR = Path("./data/raw")
OUTPUT_DAILY = OUTPUT_DIR / "aaaai_pollen_daily.csv"
OUTPUT_MONTHLY = OUTPUT_DIR / "aaaai_pollen_monthly.csv"

SLEEP_BETWEEN = 0.15
MAX_RETRIES = 3


def gql(query: str, variables: dict = None) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                GQL_URL,
                json={"query": query, "variables": variables or {}},
                headers={"Content-Type": "application/json"},
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  [retry {attempt+1}/{MAX_RETRIES}] {e}")
            time.sleep(2 * (attempt + 1))
    return None


def fetch_month_ids(year: int, month: int) -> list[dict]:
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    from_date = f"{year}-{month:02d}-01"
    to_date = f"{year}-{month:02d}-{last_day}"

    result = gql(
        """
        query($f: String) {
            allergenCollectionSets(limit: 100, order: "date", filter: $f) {
                id date
            }
        }
        """,
        {"f": f'stationId=="{STATION_ID}" && date>="{from_date}" && date<="{to_date}"'},
    )
    if result and "data" in result:
        return result["data"].get("allergenCollectionSets", [])
    return []


def fetch_detail(record_id: str) -> dict | None:
    result = gql(
        """
        query ($id: ID) {
            allergenCollectionSet(id: $id) {
                id date
                allergenCollections {
                    value
                    allergen { name category }
                }
            }
        }
        """,
        {"id": record_id},
    )
    if result and "data" in result:
        return result["data"].get("allergenCollectionSet")
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for year in YEARS:
        for month in MONTHS:
            days = fetch_month_ids(year, month)
            print(f"{year}-{month:02d}: {len(days)} days found")

            for i, day in enumerate(days):
                detail = fetch_detail(day["id"])
                if not detail:
                    continue

                row = {"date": detail["date"], "year": year, "month": month}
                for col in detail.get("allergenCollections", []):
                    cat = col["allergen"]["category"]
                    row[cat] = row.get(cat, 0) + (col["value"] or 0)
                rows.append(row)

                time.sleep(SLEEP_BETWEEN)

            if not days:
                time.sleep(0.5)

    if not rows:
        print("No data retrieved.")
        return

    daily = pd.DataFrame(rows)
    for cat in ["TREE", "WEED", "GRASS", "MOLD"]:
        if cat not in daily.columns:
            daily[cat] = 0

    daily = daily.sort_values("date").reset_index(drop=True)
    daily.to_csv(OUTPUT_DAILY, index=False)
    print(f"\nSaved daily: {OUTPUT_DAILY} ({len(daily):,} rows)")
    print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")

    monthly = (
        daily.groupby(["year", "month"])
        .agg(
            n_days=("date", "count"),
            TREE_avg=("TREE", "mean"),
            TREE_max=("TREE", "max"),
            WEED_avg=("WEED", "mean"),
            WEED_max=("WEED", "max"),
            GRASS_avg=("GRASS", "mean"),
            GRASS_max=("GRASS", "max"),
            MOLD_avg=("MOLD", "mean"),
            MOLD_max=("MOLD", "max"),
        )
        .reset_index()
    )
    monthly.to_csv(OUTPUT_MONTHLY, index=False)
    print(f"Saved monthly: {OUTPUT_MONTHLY} ({len(monthly)} rows)")


if __name__ == "__main__":
    main()
