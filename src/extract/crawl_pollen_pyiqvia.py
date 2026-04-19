import asyncio
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from pyiqvia import Client

INPUT_CSV =  "./data/Input/nta_locations.csv"
OUTPUT_CSV =  "./data/raw/pollen_daily_by_zip.csv"


async def fetch_one_zip(zip_code: str):
    client = Client(str(zip_code))
    data = await client.allergens.historic()
    return data


def normalize_historic_response(raw, zip_code: str) -> pd.DataFrame:
    """
    
    """
    if raw is None:
        return pd.DataFrame()

    if isinstance(raw, dict):
        for key in ["data", "results", "history", "periods"]:
            if key in raw and isinstance(raw[key], list):
                df = pd.DataFrame(raw[key])
                df["zip_code"] = zip_code
                return df

        return pd.DataFrame([raw]).assign(zip_code=zip_code)

    # List-records case
    if isinstance(raw, list):
        df = pd.DataFrame(raw)
        df["zip_code"] = zip_code
        return df

    # Fallback
    return pd.DataFrame([{"zip_code": zip_code, "raw_response": str(raw)}])


async def main():
    base_df = pd.read_csv(INPUT_CSV, header=None, names=["zip_code"], dtype={"zip_code": str})
    zip_list = sorted(base_df["zip_code"].dropna().astype(str).unique())

    all_rows = []

    for zip_code in tqdm(zip_list, desc="Fetching pollen"):
        try:
            zip_code = str(zip_code).strip()
            raw = await fetch_one_zip(zip_code)
            one_df = normalize_historic_response(raw, zip_code)
            all_rows.append(one_df)
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"[ERROR] ZIP {zip_code}: {e}")

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved: {OUTPUT_CSV} ({len(out):,} rows)")
    else:
        print("No data saved.")

if __name__ == "__main__":
    asyncio.run(main())
