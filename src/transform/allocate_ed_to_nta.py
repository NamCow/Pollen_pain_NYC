"""
Allocates monthly borough-level asthma ED counts to NTA neighborhoods.

Strategy:
  1. Load NTA-level asthma ED rates from the 2023 NTA2020 ground truth data.
  2. Map NTA2020 numeric GeoIDs to alphanumeric NTA codes (e.g. 50101 → BX0101).
  3. Within each borough, compute each NTA's proportional share of the borough
     total rate.
  4. Distribute each month's borough-level count (from syndromic surveillance)
     across NTAs proportionally to their rate share — NOT equally.

This produces estimated monthly asthma ED counts per NTA — the target
variable for time-series modeling.

Inputs:
  - data/Input/Asthma_ED_Visits.csv        (monthly, borough, syndromic surveillance)
  - data/raw/dohmh_asthma_ed_annual_nta2020.csv  (2023 NTA-level rates — allocation weights)
  - data/Input/nynta2020.shp                (NTA boundaries with BoroName)

Output:
  - data/processed/asthma_ed_monthly_nta.csv
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

MONTHLY_BORO_CSV = "data/Input/Asthma_ED_Visits.csv"
NTA2020_RATES_CSV = "data/raw/dohmh_asthma_ed_annual_nta2020.csv"
NTA_SHP = "data/Input/nynta2020.shp"
OUTPUT_CSV = Path("data/processed/asthma_ed_monthly_nta.csv")

# FIPS county code prefix → NTA borough prefix
FIPS_TO_PREFIX = {
    "5": "BX",   # Bronx (FIPS 36005)
    "47": "BK",  # Brooklyn (FIPS 36047)
    "61": "MN",  # Manhattan (FIPS 36061)
    "81": "QN",  # Queens (FIPS 36081)
    "85": "SI",  # Staten Island (FIPS 36085)
}


def geoid_to_nta_code(geoid: int) -> str:
    """Convert EHDP numeric GeoID (e.g. 50101) to NTA2020 code (e.g. BX0101)."""
    s = str(geoid)
    for fips, prefix in FIPS_TO_PREFIX.items():
        if s.startswith(fips):
            return prefix + s[len(fips):]
    raise ValueError(f"Unknown GeoID prefix: {geoid}")


def compute_nta_shares() -> pd.DataFrame:
    """
    For each NTA, compute its proportional share of borough-level ED visits
    using the 2023 NTA2020 ground-truth asthma ED rates as weights.

    Returns DataFrame with columns: nta_code, nta_name, borough, rate, nta_share
    """
    # Load NTA2020 rates
    nta2020 = pd.read_csv(NTA2020_RATES_CSV)
    nta2020["nta_code"] = nta2020["GeoID"].apply(geoid_to_nta_code)
    nta2020 = nta2020[["nta_code", "Value"]].rename(columns={"Value": "rate"})

    # Load NTA names and boroughs from shapefile
    gdf = gpd.read_file(NTA_SHP)
    nta_info = gdf[gdf["NTAType"] == "0"][["NTA2020", "NTAName", "BoroName"]].copy()
    nta_info = nta_info.rename(columns={"NTA2020": "nta_code", "BoroName": "borough"})

    # Merge rates with NTA info
    merged = nta_info.merge(nta2020, on="nta_code", how="left")
    assert merged["rate"].isna().sum() == 0, "Some NTAs have no rate data!"

    # Compute proportional share within each borough
    boro_total = merged.groupby("borough")["rate"].sum().rename("boro_total")
    merged = merged.merge(boro_total, on="borough")
    merged["nta_share"] = merged["rate"] / merged["boro_total"]

    return merged[["nta_code", "NTAName", "borough", "rate", "nta_share"]]


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

    print("Computing NTA allocation shares from 2023 ground-truth rates...")
    nta_shares = compute_nta_shares()
    print(f"  {len(nta_shares)} NTAs across {nta_shares['borough'].nunique()} boroughs")

    print("\n  Borough share ranges:")
    for boro in sorted(nta_shares["borough"].unique()):
        b = nta_shares[nta_shares["borough"] == boro]
        print(
            f"    {boro:15s}: {len(b):2d} NTAs, "
            f"share range [{b['nta_share'].min():.3f} – {b['nta_share'].max():.3f}], "
            f"sum = {b['nta_share'].sum():.3f}"
        )

    print("\nLoading monthly borough counts...")
    monthly = load_monthly_borough()
    print(f"  {len(monthly)} month-borough rows")
    print(f"  Date range: {monthly['year_month'].min()} to {monthly['year_month'].max()}")

    print("\nAllocating to NTAs (proportional by rate)...")
    # Cross join: each NTA × each month for its borough
    nta_monthly = nta_shares.merge(monthly, on="borough", how="inner")

    # Proportional allocation: borough_count × nta_share
    nta_monthly["estimated_count"] = nta_monthly["count"] * nta_monthly["nta_share"]

    # Add n_ntas for reference
    ntas_per_boro = nta_shares.groupby("borough")["nta_code"].count().rename("n_ntas")
    nta_monthly = nta_monthly.merge(ntas_per_boro, on="borough")

    nta_monthly = nta_monthly.sort_values(["nta_code", "date"]).reset_index(drop=True)

    out = nta_monthly[
        ["nta_code", "NTAName", "borough", "year_month", "date",
         "count", "n_ntas", "estimated_count"]
    ].rename(columns={"count": "borough_count"})

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV} ({len(out):,} rows)")
    print(f"  NTAs: {out['nta_code'].nunique()}")
    print(f"  Months: {out['year_month'].nunique()}")

    # Show variation within a borough
    print("\n  Sample: Brooklyn, 2023-07")
    sample = out[(out["borough"] == "Brooklyn") & (out["year_month"] == "2023-07")]
    sample_sorted = sample.sort_values("estimated_count", ascending=False)
    print(f"    Borough total: {sample['borough_count'].iloc[0]:.0f}")
    print(f"    NTA range: {sample['estimated_count'].min():.1f} to {sample['estimated_count'].max():.1f}")
    print(f"    Top 3:")
    for _, row in sample_sorted.head(3).iterrows():
        print(f"      {row['NTAName']:30s} {row['estimated_count']:.1f}")
    print(f"    Bottom 3:")
    for _, row in sample_sorted.tail(3).iterrows():
        print(f"      {row['NTAName']:30s} {row['estimated_count']:.1f}")


if __name__ == "__main__":
    main()
