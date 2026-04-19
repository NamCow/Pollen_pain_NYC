

import pandas as pd
import requests
from datetime import datetime, timedelta

NYC_PREFIXES = ('360050', '360470', '360610', '360810', '360850')

PARAM_MAP = {
    'PM2.5-24hr': 'pm25_mean',
    'NO2':        'no2_mean',
    'OZONE-8HR':  'ozone_mean',
    'OZONE':      'ozone_mean',
}

def fetch_one_day(date: datetime) -> pd.DataFrame:
    yyyymmdd = date.strftime('%Y%m%d')
    yyyy     = date.strftime('%Y')
    url = f"https://files.airnowtech.org/airnow/{yyyy}/{yyyymmdd}/daily_data_v2.dat"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200 or len(r.content) < 50:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    rows = []
    for line in r.text.strip().split('\n'):
        parts = line.strip().split('|')
        if len(parts) < 6:
            continue
        date_str, aqsid, _, param, unit, value = (
            parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
        )
        if not any(aqsid.startswith(p) for p in NYC_PREFIXES):
            continue
        if param not in PARAM_MAP:
            continue
        try:
            rows.append({
                'date':  date_str,
                'param': PARAM_MAP[param],
                'unit':  unit,
                'value': float(value)
            })
        except ValueError:
            continue
    return pd.DataFrame(rows)


def get_airnow_nyc_monthly_2025() -> pd.DataFrame:
    start   = datetime(2025, 1, 1)
    end     = datetime(2025, 12, 31)
    current = start
    all_rows = []
    ok, fail = 0, 0

    print(f"Downloading: {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}")
    print(f"Total days : {(end - start).days + 1}\n")

    while current <= end:
        df_day = fetch_one_day(current)
        if not df_day.empty:
            all_rows.append(df_day)
            ok += 1
        else:
            fail += 1
        if current.day == 1:
            print(f"  Processing {current.strftime('%Y-%m')} ...")
        current += timedelta(days=1)

    print(f"\n✓ OK: {ok} days | ✗ Failed: {fail} days")

    if not all_rows:
        print("No data retrieved!")
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)

    # Print units for verification.
    print("\nUnits found in AirNow data:")
    print(df.groupby(['param', 'unit']).size().to_string())

    # Parse date: AirNow format MM/DD/YY
    df['month'] = (
        pd.to_datetime(df['date'], format='%m/%d/%y', errors='coerce')
        .dt.to_period('M')
        .astype(str)
    )
    df = df.dropna(subset=['month'])

    monthly = (
        df.groupby(['month', 'param'])['value']
        .mean()
        .reset_index()
        .pivot(index='month', columns='param', values='value')
        .reset_index()
    )
    monthly.columns.name = None

    for col in ['pm25_mean', 'no2_mean', 'ozone_mean']:
        if col not in monthly.columns:
            monthly[col] = None

    monthly = monthly[['month', 'pm25_mean', 'no2_mean', 'ozone_mean']]
    monthly = monthly.sort_values('month').reset_index(drop=True)
    return monthly


if __name__ == "__main__":
    print("=" * 50)
    print("AirNow NYC 2025 — Monthly")
    print("=" * 50)

    result = get_airnow_nyc_monthly_2025()

    if result.empty:
        print("No data!")
    else:
        out = 'airnow_nyc_monthly_2025.csv'
        result.to_csv(out, index=False)
        print(f"\n✅ Saved: {out}")
        print(f"   Range: {result['month'].min()} → {result['month'].max()}")
        print(f"   Rows : {len(result)}\n")
        print(result.to_string(index=False))
