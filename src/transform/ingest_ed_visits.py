import pandas as pd

df = pd.read_csv("./data/Input/Asthma_ED_Visits.csv", sep=';')

df.columns = df.columns.str.strip()

df['Date'] = pd.to_datetime(df['Date'], format='%m/%Y')

df = df[
    (df['Date'].dt.year.between(2022, 2025)) &
    (df['Date'].dt.month.between(3, 10))
]

df = df[df['Dim2Value'] == 'All age groups']

df['number_of_visits'] = (
    df['Unnamed: 8']
    .astype(str)
    .str.replace(',', '', regex=False)
    .astype(int)
)

df = df.drop(columns=['Data note 1', 'Unnamed: 8', 'Select Metric'])

df.to_csv("./data/processed/filtered_asthma_data.csv", sep=';', index=False)