import numpy as np
import pandas as pd
import os

def read_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    xls = pd.ExcelFile(path)
    sheet_names = xls.sheet_names
    print(sheet_names)
    sheets = ['Total PV production ', 'Consumer 1']

    pv_df = pd.read_excel(path, sheet_name=sheets[0])[['Data', 'Producer 1 (kW)']]
    pv_df.rename(columns={'Data': 'datetime', 'Producer 1 (kW)': 'pv_generation_kW'}, inplace=True)

    consumer_df = pd.read_excel(path, sheet_name=sheets[1])[['Data', 'Consumption [kWh]']]
    consumer_df.rename(columns={'Data': 'datetime', 'Consumption [kWh]': 'load_consumption_kW'}, inplace=True)

    # Parse 'datetime' column into datetime objects
    pv_df['datetime'] = pd.to_datetime(pv_df['datetime'], format='%d/%m/%Y %H:%M:%S')
    consumer_df['datetime'] = pd.to_datetime(consumer_df['datetime'], format='%d/%m/%Y %H:%M:%S')

    merged_df = pd.merge_asof(
        pv_df.sort_values('datetime'),
        consumer_df.sort_values('datetime'),
        on='datetime',
        direction='nearest',  # Use 'nearest' if timestamps are not exactly matching
        tolerance=pd.Timedelta('1min')  # Adjust tolerance as needed
    )
    # Drop any rows with missing values
    merged_df.dropna(inplace=True)

    # Reset index
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df

merged_df = read_data('/Users/danialzendehdel/Documents/PhD/Codes /RL-BMS/Data/Data_PV and consumptions.xlsx')
print(pv)