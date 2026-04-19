"""
Spatial alignment utilities: UHF42-to-borough mapping and NTA shapefile helpers.

Shared by allocate_ed_to_nta.py and chs_baseline.py to avoid duplication.
"""

import geopandas as gpd
from pathlib import Path

NTA_SHP = Path("./data/Input/nynta2020.shp")

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


def load_residential_ntas(shp_path: Path = NTA_SHP) -> gpd.GeoDataFrame:
    """Load NTA2020 shapefile, keeping only residential NTAs (NTAType == '0')."""
    gdf = gpd.read_file(shp_path)
    return gdf[gdf["NTAType"] == "0"].copy()


def nta_borough_table(shp_path: Path = NTA_SHP):
    """Return DataFrame with nta_code, NTAName, borough for all residential NTAs."""
    gdf = load_residential_ntas(shp_path)
    df = gdf[["NTA2020", "NTAName", "BoroName"]].copy()
    return df.rename(columns={"NTA2020": "nta_code", "BoroName": "borough"})
