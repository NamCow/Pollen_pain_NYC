"""
Downloads NYC Street Tree Census (2015) from NYC Open Data via Socrata API.

Source: https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/uvpi-gqnh
683,788 trees with NTA codes, species, DBH (diameter), health status.

Output is the full tree-level dataset. Aggregation to canopy density
score per NTA happens in the feature engineering step.
"""

import requests
import pandas as pd
from pathlib import Path

SOCRATA_URL = "https://data.cityofnewyork.us/resource/uvpi-gqnh.json"
OUTPUT_CSV = Path("./data/raw/nyc_street_tree_census_2015.csv")

COLUMNS = [
    "tree_id", "tree_dbh", "status", "health",
    "spc_latin", "spc_common",
    "nta", "nta_name", "boroname",
    "zipcode", "latitude", "longitude",
]

PAGE_SIZE = 50000


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    offset = 0

    while True:
        print(f"Fetching rows {offset:,}–{offset + PAGE_SIZE:,}...")
        params = {
            "$select": ",".join(COLUMNS),
            "$limit": PAGE_SIZE,
            "$offset": offset,
            "$order": "tree_id",
        }
        resp = requests.get(SOCRATA_URL, params=params, timeout=120)
        resp.raise_for_status()
        records = resp.json()

        if not records:
            break

        all_dfs.append(pd.DataFrame(records))
        offset += len(records)

        if len(records) < PAGE_SIZE:
            break

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(df):,} rows)")
    print(f"  NTAs: {df['nta'].nunique()}")
    print(f"  Species: {df['spc_common'].nunique()}")
    print(f"  Health distribution:\n{df['health'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
