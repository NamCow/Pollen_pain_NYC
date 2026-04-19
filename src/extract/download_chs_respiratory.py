"""
Downloads Community Health Survey (CHS) respiratory data from NYC DOHMH EHDP-data.

Source: https://github.com/nychealth/EHDP-data (production branch)

Indicators fetched:
  18  — Adults with a recent asthma attack (UHF42, 2003-2022)
         MeasureID 30  = estimated number
         MeasureID 31  = age-adjusted percent
         MeasureID 363 = unadjusted percent

This is used as a zone-level respiratory baseline control, NOT
a neighborhood-level feature. UHF42 values are applied uniformly
to all NTAs within each UHF zone.
"""

import requests
import pandas as pd
from pathlib import Path

EHDP_BASE = "https://raw.githubusercontent.com/nychealth/EHDP-data/production/indicators/data"
OUTPUT_CSV = Path("./data/raw/dohmh_chs_respiratory_annual_uhf42.csv")

TIMEPERIOD_MAP = {
    7: "2003", 8: "2004", 9: "2005", 10: "2006", 11: "2007", 12: "2008",
    13: "2009", 14: "2010", 38: "2011", 39: "2012", 40: "2013", 41: "2014",
    44: "2015", 45: "2016", 46: "2017", 47: "2018", 48: "2019", 49: "2020",
    289: "2021", 296: "2022", 298: "2023",
}

INDICATORS = [
    {
        "id": 18,
        "label": "adults_recent_asthma_attack",
        "measures": {30: "estimated_number", 31: "age_adjusted_pct", 363: "unadjusted_pct"},
        "geo_filter": "UHF42",
    },
]


def fetch_indicator(indicator_id: int) -> pd.DataFrame:
    url = f"{EHDP_BASE}/{indicator_id}.json"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    for ind in INDICATORS:
        print(f"Fetching indicator {ind['id']} ({ind['label']})...")
        try:
            df = fetch_indicator(ind["id"])
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        df["indicator_id"] = ind["id"]
        df["indicator_label"] = ind["label"]
        df["measure_label"] = df["MeasureID"].map(ind["measures"])
        df["year"] = df["TimePeriodID"].map(TIMEPERIOD_MAP)

        filtered = df[df["GeoType"] == ind["geo_filter"]].copy()
        if len(filtered):
            all_dfs.append(filtered)
            print(f"  {ind['geo_filter']}: {len(filtered):,} rows")
            print(f"  Years: {sorted(filtered['year'].dropna().unique())}")
        else:
            print(f"  [WARN] No {ind['geo_filter']} rows found")

    if not all_dfs:
        print("\nNo data fetched.")
        return

    out = pd.concat(all_dfs, ignore_index=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(out):,} rows, {out['GeoID'].nunique()} UHF areas)")
    print(f"  Years: {sorted(out['year'].dropna().unique())}")
    print(f"  Measures: {out['measure_label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
