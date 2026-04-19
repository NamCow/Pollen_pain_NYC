import pandas as pd
import requests
import zipfile
import io

def get_epa_nyc_monthly(pollutant_code, years=[2022, 2023, 2024]):
    nyc_counties = ['New York', 'Bronx', 'Kings', 'Queens', 'Richmond']
    all_data = []

    for year in years:
        url = f"https://aqs.epa.gov/aqsweb/airdata/daily_{pollutant_code}_{year}.zip"
        print(f"Downloading {pollutant_code} - {year}...")

        response = requests.get(url)
        response.raise_for_status()

        # Mở ZIP, tìm file .csv bên trong (bỏ qua folder)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f)

        # Filter NYC only
        print(df[df['State Name'] == 'New York']['County Name'].unique())
        nyc = df[df['County Name'].isin(nyc_counties)].copy()

        # Aggregate daily → monthly (mean across all monitors)
        nyc['month'] = pd.to_datetime(nyc['Date Local']).dt.to_period('M')
        monthly = (
            nyc.groupby('month')['Arithmetic Mean']
            .mean()
            .reset_index()
            .rename(columns={'Arithmetic Mean': f'{pollutant_code}_mean'})
        )
        monthly['year'] = year
        all_data.append(monthly)
        print(f"  ✓ {len(monthly)} months collected")

    result = pd.concat(all_data, ignore_index=True)
    result['month'] = result['month'].astype(str)
    return result


if __name__ == "__main__":
    # PM2.5 
    pm25 = get_epa_nyc_monthly('88101')
    pm25.to_csv('pm25_nyc_monthly.csv', index=False)
    print("\nPM2.5 sample:")
    print(pm25.head())

    # NO₂
    no2 = get_epa_nyc_monthly('42602')
    no2.to_csv('no2_nyc_monthly.csv', index=False)
    print("\nNO2 sample:")
    print(no2.head())

    # Ozone
    ozone = get_epa_nyc_monthly('44201')
    ozone.to_csv('ozone_nyc_monthly.csv', index=False)
    print("\nOzone sample:")
    print(ozone.head())

    # Merge 
    merged = pm25.merge(no2, on=['month', 'year'], how='outer') \
                 .merge(ozone, on=['month', 'year'], how='outer')
    merged = merged.sort_values('month').reset_index(drop=True)
    merged.to_csv('air_quality_nyc_monthly.csv', index=False)
    print("\n Done! Saved air_quality_nyc_monthly.csv")
    print(merged)