"""
Backfill 2025 weather data for NTAs that only have 2019-2024.
Appends to existing raw weather CSV.
"""

import asyncio
import httpx
import pandas as pd
import geopandas as gpd
from pathlib import Path

NTA_BOUNDARY_PATH = "./data/Input/nynta2020.shp"
OUTPUT_CSV = "./data/raw/openmeteo_weather_daily_by_nta.csv"
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
SLEEP_BETWEEN = 1.5
MAX_RETRIES = 5


def get_nta_centroids() -> pd.DataFrame:
    gdf = gpd.read_file(NTA_BOUNDARY_PATH).to_crs(epsg=4326)
    col_map = {c.lower(): c for c in gdf.columns}
    code_col = col_map.get("nta2020")
    gdf_proj = gdf.to_crs(epsg=2263)
    centroids = gdf.copy()
    centroids["lat"] = gdf_proj.geometry.centroid.to_crs(epsg=4326).y
    centroids["lon"] = gdf_proj.geometry.centroid.to_crs(epsg=4326).x
    return centroids[[code_col, "lat", "lon"]].rename(columns={code_col: "nta_code"})


async def fetch_nta_weather(client, nta_code, lat, lon):
    params = {
        "latitude": round(lat, 5), "longitude": round(lon, 5),
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": DAILY_VARS, "timezone": "America/New_York",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.get(OPEN_METEO_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            df = pd.DataFrame(resp.json()["daily"])
            df.insert(0, "nta_code", nta_code)
            return df
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"    429 rate limit — waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries for {nta_code}")


async def main():
    centroids = get_nta_centroids()

    existing = pd.read_csv(OUTPUT_CSV, usecols=["nta_code", "time"])
    nta_max = existing.groupby("nta_code")["time"].max()
    need_2025 = nta_max[nta_max < "2025-01-01"].index.tolist()
    print(f"{len(need_2025)} NTAs need 2025 weather backfill")

    targets = centroids[centroids["nta_code"].isin(need_2025)]
    all_dfs = []
    failed = []

    async with httpx.AsyncClient() as client:
        for i, (_, row) in enumerate(targets.iterrows(), 1):
            try:
                df = await fetch_nta_weather(client, row["nta_code"], row["lat"], row["lon"])
                all_dfs.append(df)
                print(f"  [{i}/{len(targets)}] OK {row['nta_code']}")
                await asyncio.sleep(SLEEP_BETWEEN)
            except Exception as e:
                print(f"  [ERROR] {row['nta_code']}: {e}")
                failed.append(row["nta_code"])

    if all_dfs:
        new_df = pd.concat(all_dfs, ignore_index=True)
        new_df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
        total = pd.read_csv(OUTPUT_CSV)
        print(f"\nAppended {len(new_df):,} rows. Total: {len(total):,} rows, {total['nta_code'].nunique()} NTAs")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    asyncio.run(main())
