import asyncio
from pathlib import Path
from typing import Optional
import ast
import geopandas as gpd
import pandas as pd
from pyiqvia import Client


# =========================
# CONFIG
# =========================
# NTA_BOUNDARY_PATH = "./data/input/nynta2020.shp"
# OUTPUT_CSV = "./data/processed/nta_pollen_daily.csv"

# # Reference ZIPs discovered from empirical testing
# METRO_REF_ZIP = "10019"
# COASTAL_REF_ZIP = "11691"



# # Rockaways NTA codes from official NYC materials
# ROCKAWAY_NTA_CODES = {"QN10", "QN12", "QN15"}

# # Fallback name-based matching if codes differ in your downloaded file
# ROCKAWAY_NAME_KEYWORDS = [
#     "Breezy Point",
#     "Belle Harbor",
#     "Rockaway Park",
#     "Broad Channel",
#     "Hammels",
#     "Arverne",
#     "Edgemere",
#     "Far Rockaway",
#     "Bayswater",
# ]


# # =========================
# # API FETCH
# # =========================
# async def fetch_one_zip(zip_code: str) -> dict:
#     client = Client(str(zip_code).strip())
#     return await client.allergens.historic()


# def normalize_historic_response(raw: dict, zip_code: str) -> pd.DataFrame:
#     """
#     Convert pyiqvia historic() response into a tidy DataFrame with:
#     zip_code, period, date, pollen_index
#     """
#     empty = pd.DataFrame(columns=["zip_code", "period", "date", "pollen_index"])

#     if raw is None or not isinstance(raw, dict):
#         return empty

#     periods = None

#     # Case 1: periods at the top level
#     if isinstance(raw.get("periods"), list):
#         periods = raw["periods"]

#     # Case 2/3: periods nested in Location
#     elif "Location" in raw:
#         location = raw["Location"]

#         if isinstance(location, str):
#             try:
#                 location = ast.literal_eval(location)
#             except Exception:
#                 location = None

#         if isinstance(location, dict) and isinstance(location.get("periods"), list):
#             periods = location["periods"]

#     if not periods:
#         print(f"[DEBUG] No periods found for ZIP {zip_code}. Raw keys: {list(raw.keys())}")
#         return empty

#     df = pd.DataFrame(periods).copy()

#     if "Period" not in df.columns or "Index" not in df.columns:
#         print(f"[DEBUG] Unexpected periods format for ZIP {zip_code}: columns={list(df.columns)}")
#         return empty

#     df = df.rename(columns={"Period": "period", "Index": "pollen_index"})
#     df["zip_code"] = zip_code
#     df["date"] = pd.to_datetime(df["period"]).dt.date.astype(str)

#     return df[["zip_code", "period", "date", "pollen_index"]]


# async def fetch_cluster_series() -> tuple[pd.DataFrame, pd.DataFrame]:
#     metro_raw = await fetch_one_zip(METRO_REF_ZIP)
#     coastal_raw = await fetch_one_zip(COASTAL_REF_ZIP)

#     metro_df = normalize_historic_response(metro_raw, METRO_REF_ZIP)
#     coastal_df = normalize_historic_response(coastal_raw, COASTAL_REF_ZIP)

#     if metro_df.empty:
#         raise RuntimeError(f"Metro reference ZIP {METRO_REF_ZIP} returned no periods.")
#     if coastal_df.empty:
#         raise RuntimeError(f"Coastal reference ZIP {COASTAL_REF_ZIP} returned no periods.")

#     metro_df["pollen_cluster"] = "metro"
#     coastal_df["pollen_cluster"] = "coastal_rockaways"

#     return metro_df, coastal_df


# # =========================
# # NTA HELPERS
# # =========================
# def pick_first_existing(columns: list[str], candidates: list[str]) -> Optional[str]:
#     lower_map = {c.lower(): c for c in columns}
#     for cand in candidates:
#         if cand.lower() in lower_map:
#             return lower_map[cand.lower()]
#     return None


# def load_nta_boundaries(path: str) -> gpd.GeoDataFrame:
#     gdf = gpd.read_file(path)

#     code_col = pick_first_existing(
#         list(gdf.columns),
#         ["NTA2020", "NTACode", "NTA", "ntacode", "ntacode20"]
#     )
#     name_col = pick_first_existing(
#         list(gdf.columns),
#         ["NTAName", "NTA_NAME", "ntaname", "name"]
#     )

#     if code_col is None or name_col is None:
#         raise ValueError(
#             f"Could not find NTA code/name columns in {path}. "
#             f"Columns found: {list(gdf.columns)}"
#         )

#     out = gdf.copy()
#     out = out.rename(columns={code_col: "nta_code", name_col: "nta_name"})

#     # Keep only the two columns we need plus geometry for optional QA
#     keep_cols = ["nta_code", "nta_name", "geometry"]
#     out = out[keep_cols].copy()

#     return out


# def assign_pollen_cluster_to_ntas(nta_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
#     out = nta_gdf.copy()

#     # First try code-based assignment
#     out["pollen_cluster"] = "metro"
#     out.loc[out["nta_code"].isin(ROCKAWAY_NTA_CODES), "pollen_cluster"] = "coastal_rockaways"

#     # Fallback name-based assignment in case your file uses different code format
#     mask_name = out["nta_name"].fillna("").apply(
#         lambda x: any(keyword.lower() in x.lower() for keyword in ROCKAWAY_NAME_KEYWORDS)
#     )
#     out.loc[mask_name, "pollen_cluster"] = "coastal_rockaways"

#     return out


# # =========================
# # BUILD FINAL TABLE
# # =========================
def build_nta_pollen_daily(
    nta_gdf: gpd.GeoDataFrame,
    metro_df: pd.DataFrame,
    coastal_df: pd.DataFrame
) -> pd.DataFrame:
    cluster_series = pd.concat([metro_df, coastal_df], ignore_index=True)
    cluster_series = cluster_series[["pollen_cluster", "date", "period", "pollen_index"]].copy()

    # Join each NTA to its cluster's daily series
    nta_lookup = nta_gdf[["nta_code", "nta_name", "pollen_cluster"]].copy()
    final_df = nta_lookup.merge(cluster_series, on="pollen_cluster", how="left")

    final_df = final_df.sort_values(["nta_code", "date"]).reset_index(drop=True)
    return final_df


# =========================
# CONFIG
# =========================
NTA_BOUNDARY_PATH = "./data/input/nynta2020.shp"
OUTPUT_CSV = "./data/processed/nta_pollen_daily.csv"

METRO_REF_ZIP = "10019"
COASTAL_REF_ZIP = "11691"

# Preferred NTA 2020 Rockaways codes
ROCKAWAY_NTA2020_CODES = {
"QN1401",
"QN1402",
"QN1403",
"QN8492",  
}


# Keep old short-form codes as fallback
ROCKAWAY_LEGACY_CODES = {"QN10", "QN12", "QN15"}

ROCKAWAY_NAME_KEYWORDS = [
    "Breezy Point",
    "Belle Harbor",
    "Rockaway Park",
    "Broad Channel",
    "Hammels",
    "Arverne",
    "Edgemere",
    "Far Rockaway",
    "Bayswater",
]


async def fetch_one_zip(zip_code: str) -> dict:
    client = Client(str(zip_code).strip())
    return await client.allergens.historic()


def normalize_historic_response(raw: dict, zip_code: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["zip_code", "period", "date", "pollen_index"])

    if raw is None or not isinstance(raw, dict):
        return empty

    periods = None

    if isinstance(raw.get("periods"), list):
        periods = raw["periods"]
    elif "Location" in raw:
        location = raw["Location"]

        if isinstance(location, str):
            try:
                location = ast.literal_eval(location)
            except Exception:
                location = None

        if isinstance(location, dict) and isinstance(location.get("periods"), list):
            periods = location["periods"]

    if not periods:
        print(f"[DEBUG] No periods found for ZIP {zip_code}. Raw keys: {list(raw.keys())}")
        return empty

    df = pd.DataFrame(periods).copy()

    if "Period" not in df.columns or "Index" not in df.columns:
        print(f"[DEBUG] Unexpected periods format for ZIP {zip_code}: columns={list(df.columns)}")
        return empty

    df = df.rename(columns={"Period": "period", "Index": "pollen_index"})
    df["zip_code"] = zip_code
    df["date"] = pd.to_datetime(df["period"]).dt.date.astype(str)

    return df[["zip_code", "period", "date", "pollen_index"]]


async def fetch_cluster_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    metro_raw = await fetch_one_zip(METRO_REF_ZIP)
    coastal_raw = await fetch_one_zip(COASTAL_REF_ZIP)

    metro_df = normalize_historic_response(metro_raw, METRO_REF_ZIP)
    coastal_df = normalize_historic_response(coastal_raw, COASTAL_REF_ZIP)

    if metro_df.empty:
        raise RuntimeError(f"Metro reference ZIP {METRO_REF_ZIP} returned no periods.")
    if coastal_df.empty:
        raise RuntimeError(f"Coastal reference ZIP {COASTAL_REF_ZIP} returned no periods.")

    metro_df["pollen_cluster"] = "metro"
    coastal_df["pollen_cluster"] = "coastal_rockaways"

    return metro_df, coastal_df


def pick_first_existing(columns: list[str], candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_nta_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def load_nta_boundaries(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)

    code_col = pick_first_existing(
        list(gdf.columns),
        ["NTA2020", "NTACode", "NTA", "ntacode", "ntacode20"]
    )
    name_col = pick_first_existing(
        list(gdf.columns),
        ["NTAName", "NTA_NAME", "ntaname", "name"]
    )

    if code_col is None or name_col is None:
        raise ValueError(
            f"Could not find NTA code/name columns in {path}. "
            f"Columns found: {list(gdf.columns)}"
        )

    out = gdf.copy()
    out = out.rename(columns={code_col: "nta_code", name_col: "nta_name"})
    out["nta_code_norm"] = out["nta_code"].apply(normalize_nta_code)

    keep_cols = ["nta_code", "nta_code_norm", "nta_name", "geometry"]
    out = out[keep_cols].copy()

    return out


def assign_pollen_cluster_to_ntas(nta_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = nta_gdf.copy()
    out["pollen_cluster"] = "metro"

    # Preferred NTA 2020 exact match
    mask_nta2020 = out["nta_code_norm"].isin(ROCKAWAY_NTA2020_CODES)
    out.loc[mask_nta2020, "pollen_cluster"] = "coastal_rockaways"

    # Legacy fallback
    mask_legacy = out["nta_code_norm"].isin(ROCKAWAY_LEGACY_CODES)
    out.loc[mask_legacy, "pollen_cluster"] = "coastal_rockaways"

    # Name fallback
    mask_name = out["nta_name"].fillna("").apply(
        lambda x: any(keyword.lower() in x.lower() for keyword in ROCKAWAY_NAME_KEYWORDS)
    )
    out.loc[mask_name, "pollen_cluster"] = "coastal_rockaways"

    return out

async def main():
    nta_gdf = load_nta_boundaries(NTA_BOUNDARY_PATH)
    print("Columns:", nta_gdf.columns.tolist())
    print("\nSample rows:")
    print(nta_gdf[["nta_code", "nta_name"]].head(10).to_string(index=False))

    print("\nUnique nta_code samples:")
    print(nta_gdf["nta_code"].unique()[:20])
    nta_gdf = assign_pollen_cluster_to_ntas(nta_gdf)

    print("\nMatched coastal NTAs:")
    print(
        nta_gdf.loc[
            nta_gdf["pollen_cluster"] == "coastal_rockaways",
            ["nta_code", "nta_code_norm", "nta_name"]
        ].to_string(index=False)
    )
    metro_df, coastal_df = await fetch_cluster_series()


    final_df = build_nta_pollen_daily(nta_gdf, metro_df, coastal_df)

    out_path = Path(OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)

    print("\n=== Final output ===")
    print(f"Saved: {out_path}")
    print(f"Rows: {len(final_df):,}")
    print(f"Unique NTAs: {final_df['nta_code'].nunique()}")
    print(f"Date range: {final_df['date'].min()} -> {final_df['date'].max()}")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
