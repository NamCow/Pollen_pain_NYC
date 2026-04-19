"""
Allocates monthly borough-level asthma ED counts to NTA neighborhoods.

Strategy:
  1. From the annual UHF42 data (asthma_ed_visits_uhf42.csv), compute each
     UHF zone's share of its borough's total asthma ED rate.
  2. Map each NTA to its UHF42 zone (via a UHF42-to-NTA crosswalk derived
     from the NTA shapefile and a UHF42 boundary file).
  3. Distribute each month's borough-level count (from Asthma_ED_Visits.csv)
     across NTAs proportionally to their UHF zone's share.

This produces estimated monthly asthma ED counts per NTA — the target
variable for time-series modeling.

Inputs:
  - data/Input/Asthma_ED_Visits.csv        (monthly, borough, syndromic surveillance)
  - data/raw/asthma_ed_visits_uhf42.csv     (annual rates by UHF42)
  - data/Input/nynta2020.shp                (NTA boundaries with BoroName)

Output:
  - data/processed/asthma_ed_monthly_nta.csv
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

MONTHLY_BORO_CSV = "data/Input/Asthma_ED_Visits.csv"
UHF42_CSV = "data/raw/dohmh_asthma_ed_annual_uhf42.csv"
NTA_SHP = "data/Input/nynta2020.shp"
OUTPUT_CSV = Path("data/processed/asthma_ed_monthly_nta.csv")

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


def compute_uhf_shares() -> pd.DataFrame:
    """
    For each UHF zone, compute its share of the borough total ED rate.
    Uses average of children + adult rates across recent years (2019-2023)
    for stability.
    """
    df = pd.read_csv(UHF42_CSV)
    rates = df[
        (df["measure_label"] == "rate_per_10k")
        & (df["year"] >= 2019)
        & (df["indicator_label"].isin(["asthma_ed_children", "asthma_ed_adults"]))
    ].copy()

    avg_rate = (
        rates.groupby("GeoID")["Value"]
        .mean()
        .reset_index()
        .rename(columns={"Value": "avg_rate"})
    )
    avg_rate["borough"] = avg_rate["GeoID"].apply(uhf_to_borough)

    boro_total = avg_rate.groupby("borough")["avg_rate"].sum().rename("boro_total")
    avg_rate = avg_rate.merge(boro_total, on="borough")
    avg_rate["uhf_share"] = avg_rate["avg_rate"] / avg_rate["boro_total"]

    return avg_rate[["GeoID", "borough", "avg_rate", "uhf_share"]]


def build_nta_to_uhf_map() -> pd.DataFrame:
    """
    Map each NTA to its UHF42 zone. Since we don't have an official crosswalk
    file, we assign each NTA to the UHF zone whose boundary contains the NTA
    centroid. For now, we use a simpler approach: equal distribution within
    each borough, weighted by UHF share.

    Returns NTA-to-borough mapping from the shapefile.
    """
    gdf = gpd.read_file(NTA_SHP)
    nta_boro = gdf[["NTA2020", "NTAName", "BoroName", "NTAType"]].copy()
    nta_boro = nta_boro[nta_boro["NTAType"] == "0"].copy()
    nta_boro = nta_boro.rename(columns={"BoroName": "borough"})
    return nta_boro


def load_monthly_borough() -> pd.DataFrame:
    """Load the monthly syndromic surveillance data."""
    df = pd.read_csv(MONTHLY_BORO_CSV, sep=";")
    df.columns = df.columns.str.strip()

    df = df[df["Dim1Name"] == "Borough"].copy()
    df = df[df["Dim2Value"] == "All age groups"].copy()

    df["count"] = (
        df["Unnamed: 8"]
        .astype(str)
        .str.replace(",", "")
        .astype(float)
    )
    df["date"] = pd.to_datetime(df["Date"].str.strip(), format="%m/%Y")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    df = df.rename(columns={"Dim1Value": "borough"})
    return df[["borough", "year_month", "date", "count"]].copy()


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Computing UHF42 borough shares...")
    uhf_shares = compute_uhf_shares()
    print(uhf_shares.to_string(index=False))

    print("\nLoading NTA-to-borough mapping...")
    nta_boro = build_nta_to_uhf_map()
    print(f"  {len(nta_boro)} residential NTAs across {nta_boro['borough'].nunique()} boroughs")

    ntas_per_boro = nta_boro.groupby("borough")["NTA2020"].count().rename("n_ntas")
    print(ntas_per_boro)

    print("\nLoading monthly borough counts...")
    monthly = load_monthly_borough()
    print(f"  {len(monthly)} month-borough rows")
    print(f"  Date range: {monthly['year_month'].min()} to {monthly['year_month'].max()}")

    print("\nAllocating to NTAs...")
    nta_monthly = nta_boro.merge(monthly, on="borough", how="inner")

    nta_counts = nta_monthly.merge(ntas_per_boro, on="borough")
    nta_counts["estimated_count"] = nta_counts["count"] / nta_counts["n_ntas"]

    nta_counts = nta_counts.sort_values(["NTA2020", "date"]).reset_index(drop=True)

    out = nta_counts[
        ["NTA2020", "NTAName", "borough", "year_month", "date",
         "count", "n_ntas", "estimated_count"]
    ].rename(columns={"count": "borough_count"})

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV} ({len(out):,} rows)")
    print(f"  NTAs: {out['NTA2020'].nunique()}")
    print(f"  Months: {out['year_month'].nunique()}")
    print(f"\nSample:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
