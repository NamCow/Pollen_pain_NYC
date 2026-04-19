"""
Aggregate street tree census → canopy density score per NTA2020.

The tree census uses NTA2010 codes which don't map to NTA2020.
We use a spatial join (tree lat/lon → NTA2020 polygon) to assign
each tree to its NTA2020 neighborhood, then aggregate.

Input:  data/raw/nyc_street_tree_census_2015.csv
        data/Input/nynta2020.shp
Output: data/processed/tree_canopy_by_nta.csv
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

INPUT = Path("./data/raw/nyc_street_tree_census_2015.csv")
NTA_SHP = Path("./data/Input/nynta2020.shp")
OUTPUT = Path("./data/processed/tree_canopy_by_nta.csv")


def main():
    print("Loading trees...")
    df = pd.read_csv(INPUT, usecols=["tree_dbh", "status", "health", "latitude", "longitude"])
    alive = df[df["status"] == "Alive"].copy()
    alive["tree_dbh"] = pd.to_numeric(alive["tree_dbh"], errors="coerce").fillna(0)
    alive = alive.dropna(subset=["latitude", "longitude"])
    print(f"  {len(alive):,} alive trees with coordinates")

    print("Spatial join to NTA2020...")
    nta = gpd.read_file(NTA_SHP).to_crs(epsg=4326)
    nta = nta[nta["NTAType"] == "0"][["NTA2020", "geometry"]].copy()

    trees_gdf = gpd.GeoDataFrame(
        alive,
        geometry=gpd.points_from_xy(alive["longitude"], alive["latitude"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(trees_gdf, nta, how="inner", predicate="within")
    print(f"  {len(joined):,} trees matched to NTA2020 ({len(alive) - len(joined):,} outside boundaries)")

    print("Aggregating...")
    canopy = (
        joined.groupby("NTA2020", as_index=False)
        .agg(
            tree_count=("tree_dbh", "count"),
            total_dbh=("tree_dbh", "sum"),
            mean_dbh=("tree_dbh", "mean"),
            pct_good_health=("health", lambda x: (x == "Good").mean()),
        )
        .rename(columns={"NTA2020": "nta_code"})
        .sort_values("nta_code")
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canopy.to_csv(OUTPUT, index=False)
    print(f"Saved: {OUTPUT} ({len(canopy)} NTAs)")
    print(canopy.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
