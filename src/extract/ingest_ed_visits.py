"""
Downloads asthma ED visit data from the NYC DOHMH Environment & Health Data Portal.

Source: https://github.com/nychealth/EHDP-data (production branch)
Format: Columnar JSON with keys MeasureID, GeoID, GeoType, TimePeriodID, Value, CI, Note, DisplayValue.

Indicators fetched:
  2379 — Asthma ED visits (children 0-17)
         MeasureID 1195 = count, 1196 = age-adjusted rate per 10,000
  2380 — Asthma ED visits (adults 18+)
         MeasureID 1197 = count, 1198 = age-adjusted rate per 10,000, 1199 = number (estimated)
  2388 — Asthma ED visits (all ages, NTA2020 geography)
         MeasureID 1224 = age-adjusted rate per 10,000

Geography levels saved:
  UHF42 — 42 United Hospital Fund neighborhoods (primary modeling geography, 2005-2023 annual)
  NTA2020 — Neighborhood Tabulation Areas 2020 (indicator 2388 only, 2017 and 2023)
"""

import requests
import pandas as pd
from pathlib import Path

EHDP_BASE = "https://raw.githubusercontent.com/nychealth/EHDP-data/production/indicators/data"

OUTPUT_DIR = Path("./data/raw")
OUTPUT_UHF = OUTPUT_DIR / "dohmh_asthma_ed_annual_uhf42.csv" 
OUTPUT_NTA = OUTPUT_DIR / "dohmh_asthma_ed_annual_nta2020.csv"

TIMEPERIOD_MAP = {
    9: "2005", 10: "2006", 11: "2007", 12: "2008", 13: "2009", 14: "2010",
    38: "2011", 39: "2012", 40: "2013", 41: "2014", 44: "2015", 45: "2016",
    46: "2017", 47: "2018", 48: "2019", 49: "2020", 289: "2021", 296: "2022",
    298: "2023",
}

INDICATORS = [
    {
        "id": 2379,
        "label": "asthma_ed_children",
        "measures": {1195: "count", 1196: "rate_per_10k"},
    },
    {
        "id": 2380,
        "label": "asthma_ed_adults",
        "measures": {1197: "count", 1198: "rate_per_10k", 1199: "estimated_count"},
    },
    {
        "id": 2388,
        "label": "asthma_ed_all_ages",
        "measures": {1224: "rate_per_10k"},
    },
]


def fetch_indicator(indicator_id: int) -> pd.DataFrame:
    url = f"{EHDP_BASE}/{indicator_id}.json"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    uhf_dfs = []
    nta_dfs = []

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

        uhf = df[df["GeoType"] == "UHF42"].copy()
        if len(uhf):
            uhf_dfs.append(uhf)
            print(f"  UHF42: {len(uhf):,} rows, years {sorted(uhf['year'].dropna().unique())}")

        nta = df[df["GeoType"] == "NTA2020"].copy()
        if len(nta):
            nta_dfs.append(nta)
            print(f"  NTA2020: {len(nta):,} rows, years {sorted(nta['year'].dropna().unique())}")

        other_geos = df["GeoType"].value_counts().to_dict()
        print(f"  All geographies: {other_geos}")

    if uhf_dfs:
        uhf_out = pd.concat(uhf_dfs, ignore_index=True)
        uhf_out.to_csv(OUTPUT_UHF, index=False)
        print(f"\nSaved: {OUTPUT_UHF}  ({len(uhf_out):,} rows, {uhf_out['GeoID'].nunique()} UHF areas)")
        print(f"  Years: {sorted(uhf_out['year'].dropna().unique())}")
        print(f"  Measures: {uhf_out['measure_label'].value_counts().to_dict()}")
    else:
        print("\n[WARN] No UHF42 data found.")

    if nta_dfs:
        nta_out = pd.concat(nta_dfs, ignore_index=True)
        nta_out.to_csv(OUTPUT_NTA, index=False)
        print(f"\nSaved: {OUTPUT_NTA}  ({len(nta_out):,} rows, {nta_out['GeoID'].nunique()} NTAs)")
        print(f"  Years: {sorted(nta_out['year'].dropna().unique())}")
    else:
        print("\n[WARN] No NTA2020 data found.")

    print("\nNote: UHF42 data requires UHF-to-NTA crosswalk in ETL step before modeling.")
    print("Note: NTA2020 data (indicator 2388) only has 2017 and 2023 — cross-sectional only.")


if __name__ == "__main__":
    main()
