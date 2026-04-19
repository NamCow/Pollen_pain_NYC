import asyncio
import httpx
import pandas as pd
import geopandas as gpd
from pathlib import Path

NTA_BOUNDARY_PATH = "./data/Input/nynta2020.shp"
OUTPUT_CSV = "./data/raw/openmeteo_weather_daily_by_nta.csv"
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
SLEEP_BETWEEN = 1.5   # seconds between requests (free tier limit ~10 req/min)
MAX_RETRIES = 5


def get_nta_centroids() -> pd.DataFrame:
    gdf = gpd.read_file(NTA_BOUNDARY_PATH).to_crs(epsg=4326)
    code_candidates = ["NTA2020", "NTACode", "NTA", "ntacode", "ntacode20"]
    col_map = {c.lower(): c for c in gdf.columns}
    code_col = next((col_map[c.lower()] for c in code_candidates if c.lower() in col_map), None)
    if not code_col:
        raise ValueError(f"No NTA code column found. Columns: {list(gdf.columns)}")
    gdf_proj = gdf.to_crs(epsg=2263)  # NYC state plane for accurate centroids
    centroids = gdf.copy()
    centroids["lat"] = gdf_proj.geometry.centroid.to_crs(epsg=4326).y
    centroids["lon"] = gdf_proj.geometry.centroid.to_crs(epsg=4326).x
    return centroids[[code_col, "lat", "lon"]].rename(columns={code_col: "nta_code"})


async def fetch_nta_weather(
    client: httpx.AsyncClient, nta_code: str, lat: float, lon: float
) -> pd.DataFrame:
    params = {
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": DAILY_VARS,
        "timezone": "America/New_York",
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

    # Resume: skip NTAs already in output file
    done = set()
    out_path = Path(OUTPUT_CSV)
    if out_path.exists():
        existing = pd.read_csv(out_path, usecols=["nta_code"])
        done = set(existing["nta_code"].unique())
        print(f"Resuming — {len(done)} NTAs already fetched, {len(centroids) - len(done)} remaining.")

    remaining = centroids[~centroids["nta_code"].isin(done)]
    print(f"Fetching weather for {len(remaining)} NTAs ({START_DATE} to {END_DATE})...")

    all_dfs = []
    failed = []
    async with httpx.AsyncClient() as client:
        for i, (_, row) in enumerate(remaining.iterrows(), 1):
            try:
                df = await fetch_nta_weather(client, row["nta_code"], row["lat"], row["lon"])
                all_dfs.append(df)
                print(f"  [{i}/{len(remaining)}] OK {row['nta_code']}")
                await asyncio.sleep(SLEEP_BETWEEN)
            except Exception as e:
                print(f"  [ERROR] {row['nta_code']}: {e}")
                failed.append(row["nta_code"])

    if not all_dfs:
        print("No new data fetched.")
        return

    new_df = pd.concat(all_dfs, ignore_index=True)

    # Append to existing file if present
    if out_path.exists() and done:
        new_df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(out_path, index=False)

    total = pd.read_csv(out_path)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(total):,} rows, {total['nta_code'].nunique()} NTAs total)")
    if failed:
        print(f"Failed NTAs ({len(failed)}): {failed}")


if __name__ == "__main__":
    asyncio.run(main())
