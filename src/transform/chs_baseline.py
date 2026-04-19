"""
Compute CHS respiratory baseline per NTA.

Maps UHF42-level asthma prevalence to NTAs via borough.
Uses the most recent 3-year average for stability.

This is a zone-level control — the same UHF value applies to all NTAs
within that borough. It captures baseline asthma burden independent
of seasonal/environmental triggers.

Input:  data/raw/dohmh_chs_respiratory_annual_uhf42.csv
        data/Input/nynta2020.shp
Output: data/processed/chs_baseline_by_nta.csv
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

CHS_CSV = Path("./data/raw/dohmh_chs_respiratory_annual_uhf42.csv")
NTA_SHP = Path("./data/Input/nynta2020.shp")
OUTPUT = Path("./data/processed/chs_baseline_by_nta.csv")

UHF_BORO_MAP = {
    range(100, 200): "Bronx",
    range(200, 300): "Brooklyn",
    range(300, 400): "Manhattan",
    range(400, 500): "Queens",
    range(500, 600): "Staten Island",
}


def uhf_to_borough(geo_id: int) -> str:
    for rng, boro in UHF_BORO_MAP.items():
        if geo_id in rng:
            return boro
    return "Unknown"


def main():
    chs = pd.read_csv(CHS_CSV)

    recent = chs[
        (chs["measure_label"] == "age_adjusted_pct")
        & (chs["year"] >= 2019)
    ].copy()

    boro_avg = recent.copy()
    boro_avg["borough"] = boro_avg["GeoID"].apply(uhf_to_borough)

    boro_baseline = (
        boro_avg.groupby("borough", as_index=False)["Value"]
        .mean()
        .rename(columns={"Value": "chs_asthma_pct"})
    )

    gdf = gpd.read_file(NTA_SHP)
    nta_boro = gdf[gdf["NTAType"] == "0"][["NTA2020", "NTAName", "BoroName"]].copy()
    nta_boro = nta_boro.rename(columns={"BoroName": "borough", "NTA2020": "nta_code"})

    result = nta_boro.merge(boro_baseline, on="borough", how="left")
    result = result[["nta_code", "NTAName", "borough", "chs_asthma_pct"]].sort_values("nta_code")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"Saved: {OUTPUT} ({len(result)} NTAs)")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
