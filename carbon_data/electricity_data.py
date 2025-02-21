import requests
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

api_key = os.getenv("EIA_API_KEY")

api_url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"

params = {
    "api_key": api_key,
    "frequency": "hourly",
    "data[0]": "value",
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
    "offset": 0,
    "length": 1000,  # Smaller for testing
    "facets[respondent][]": "ERCO"  # ERCOT (Texas)
}

response = requests.get(api_url, params=params)

if response.status_code == 200:
    data = response.json()
    
    df = pd.DataFrame(data.get('response', {}).get('data', []))
    
    df['period'] = pd.to_datetime(df['period'])
    
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    
    pivot_df = df.pivot_table(
        index='period', 
        columns='fueltype', 
        values='value',
        aggfunc='sum'
    )
    
    pivot_df = pivot_df.fillna(0)
    
    # Calculate total generation for each time period
    pivot_df['total'] = pivot_df.sum(axis=1)
    
    # Define carbon intensity factors (gCO2/kWh)
    carbon_factors = {
        'coal': 995,
        'natural gas': 465,
        'petroleum': 820,
        'nuclear': 0,
        'wind': 0,
        'solar': 0,
        'hydro': 0,
        'biomass': 230,
        'geothermal': 38,
        'other': 700
    }
    
    # Map EIA fuel types to our carbon factor categories (simplified mapping)
    fuel_mapping = {
        'COL': 'coal',
        'NG': 'natural gas',
        'PEL': 'petroleum',
        'NUC': 'nuclear',
        'WND': 'wind',
        'SUN': 'solar',
        'WAT': 'hydro',
        'OTH': 'other',
        'GEO': 'geothermal',
        'DPV': 'solar',
        'OIL': 'petroleum'
    }
    
    # Calculate carbon intensity
    carbon_intensity = []
    
    for idx, row in pivot_df.iterrows():
        total_gen = row['total']
        if total_gen > 0:
            intensity = 0
            for fuel in pivot_df.columns:
                if fuel != 'total' and fuel in fuel_mapping:
                    # Get percentage of this fuel in the mix
                    fuel_pct = row[fuel] / total_gen
                    # Add its contribution to carbon intensity
                    intensity += fuel_pct * carbon_factors[fuel_mapping[fuel]]
            carbon_intensity.append({'period': idx, 'carbon_intensity': intensity})
    
    carbon_df = pd.DataFrame(carbon_intensity)
    
    pivot_df.to_csv('texas_generation_mix.csv')
    carbon_df.to_csv('texas_carbon_intensity.csv')
    
    print(f"Data processed and saved. Latest carbon intensity: {carbon_df['carbon_intensity'].iloc[0]:.1f} gCO2/kWh")
    
    plt.figure(figsize=(12, 6))
    plt.plot(carbon_df['period'], carbon_df['carbon_intensity'])
    plt.title('Carbon Intensity of Electricity in Texas (ERCOT)')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Carbon Intensity (gCO2/kWh)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('carbon_intensity_trend.png')
    plt.show()
    
else:
    print(f"Error: {response.status_code}")
    print(response.text)