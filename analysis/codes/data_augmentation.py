"""
Data Augmentation Pipeline for Aadhaar Biometric/Demographic/Enrollment Datasets
Adds weather, climate, demographic, and economic attributes using public APIs and reference data
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime
from typing import Dict, Tuple, Optional
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Import reference data
from india_reference_data import (
    INDIA_CENSUS_DATA, RAINFALL_ZONES, EARTHQUAKE_ZONES,
    CLIMATE_TYPES, PER_CAPITA_INCOME_USD, HUMAN_DEVELOPMENT_INDEX
)

class DataAugmenter:
    """
    Augments Aadhaar datasets with additional attributes from various sources
    """
    
    def __init__(self, cache_dir: str = 'augmentation_cache'):
        self.cache_dir = cache_dir
        self.weather_cache = {}
        self.nominatim_cache = {}
        self.processed_pincodes = set()
        
        # Create cache directory
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        print(f"✓ Initialized DataAugmenter with cache dir: {cache_dir}")
    
    @lru_cache(maxsize=10000)
    def get_coordinates_from_nominatim(self, pincode: str) -> Optional[Tuple[float, float]]:
        """
        Get latitude and longitude from Nominatim OSM API
        Args:
            pincode: Indian pincode
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        try:
            # Add delay to respect rate limiting (1 req/sec)
            time.sleep(1.1)
            
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': f'{pincode}, India',
                'format': 'json',
                'limit': 1
            }
            headers = {'User-Agent': 'Aadhaar-DataAugmentation/1.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200 and response.json():
                data = response.json()[0]
                lat = float(data.get('lat', 0))
                lon = float(data.get('lon', 0))
                
                # Cache the result
                self.nominatim_cache[pincode] = (lat, lon)
                return (lat, lon)
        except Exception as e:
            print(f"⚠️  Error fetching coordinates for pincode {pincode}: {str(e)}")
        
        return None
    
    def get_weather_data(self, latitude: float, longitude: float, pincode: str) -> Dict:
        """
        Fetch weather and climate data from Open-Meteo API
        """
        try:
            # Check cache first
            cache_key = f"{latitude:.4f}_{longitude:.4f}"
            if cache_key in self.weather_cache:
                return self.weather_cache[cache_key]
            
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'current': 'temperature_2m,relative_humidity_2m,precipitation,weather_code',
                'timezone': 'Asia/Kolkata'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('current', {})
                
                weather_attrs = {
                    'current_temperature_celsius': data.get('temperature_2m', np.nan),
                    'humidity_percentage': data.get('relative_humidity_2m', np.nan),
                    'current_precipitation_mm': data.get('precipitation', np.nan),
                    'weather_code': data.get('weather_code', np.nan),
                }
                
                # Cache result
                self.weather_cache[cache_key] = weather_attrs
                return weather_attrs
        except Exception as e:
            print(f"⚠️  Error fetching weather for {latitude},{longitude}: {str(e)}")
        
        return {
            'current_temperature_celsius': np.nan,
            'humidity_percentage': np.nan,
            'current_precipitation_mm': np.nan,
            'weather_code': np.nan,
        }
    
    def get_state_attributes(self, state: str) -> Dict:
        """
        Get state-level attributes from reference data
        """
        attrs = {}
        
        if state in INDIA_CENSUS_DATA:
            state_data = INDIA_CENSUS_DATA[state]
            
            attrs['state_population_2011'] = state_data.get('state_pop', np.nan)
            attrs['rainfall_zone'] = state_data.get('rainfall_zone', np.nan)
            attrs['earthquake_risk_zone'] = state_data.get('earthquake_zone', np.nan)
            attrs['climate_type'] = state_data.get('primary_climate', np.nan)
            attrs['average_temperature_celsius'] = state_data.get('avg_temp_celsius', np.nan)
            attrs['literacy_rate_percent'] = state_data.get('literacy_rate', np.nan)
            attrs['sex_ratio_per_1000_males'] = state_data.get('sex_ratio', np.nan)
            
            # Add economic data
            attrs['per_capita_income_usd'] = PER_CAPITA_INCOME_USD.get(state, np.nan)
            attrs['human_development_index'] = HUMAN_DEVELOPMENT_INDEX.get(state, np.nan)
        else:
            # Default values for unknown states
            attrs = {
                'state_population_2011': np.nan,
                'rainfall_zone': np.nan,
                'earthquake_risk_zone': np.nan,
                'climate_type': np.nan,
                'average_temperature_celsius': np.nan,
                'literacy_rate_percent': np.nan,
                'sex_ratio_per_1000_males': np.nan,
                'per_capita_income_usd': np.nan,
                'human_development_index': np.nan,
            }
        
        return attrs
    
    def augment_dataframe(self, df: pd.DataFrame, dataset_name: str = 'biometric') -> pd.DataFrame:
        """
        Augment a dataframe with additional attributes
        
        Args:
            df: Input dataframe with state, district, pincode columns
            dataset_name: Name of dataset (for logging)
        
        Returns:
            Augmented dataframe with new columns
        """
        print(f"\n{'='*80}")
        print(f"AUGMENTING {dataset_name.upper()} DATASET")
        print(f"{'='*80}")
        print(f"Total records: {len(df)}")
        
        # Initialize new columns
        new_columns = {
            'state_population_2011': np.nan,
            'rainfall_zone': np.nan,
            'earthquake_risk_zone': np.nan,
            'climate_type': np.nan,
            'average_temperature_celsius': np.nan,
            'literacy_rate_percent': np.nan,
            'sex_ratio_per_1000_males': np.nan,
            'per_capita_income_usd': np.nan,
            'human_development_index': np.nan,
            'latitude': np.nan,
            'longitude': np.nan,
            'current_temperature_celsius': np.nan,
            'humidity_percentage': np.nan,
            'current_precipitation_mm': np.nan,
            'weather_code': np.nan,
            'augmentation_status': 'pending'
        }
        
        for col in new_columns:
            if col not in df.columns:
                df[col] = new_columns[col]
        
        # Get unique states for efficient processing
        unique_states = df['state'].unique()
        print(f"Unique states: {len(unique_states)}")
        
        # Get unique pincodes
        unique_pincodes = df['pincode'].dropna().unique()
        print(f"Unique pincodes: {len(unique_pincodes)}")
        
        # ====== STEP 1: Add state-level attributes ======
        print(f"\n[1/4] Adding state-level attributes...")
        state_attrs = {}
        for state in unique_states:
            state_attrs[state] = self.get_state_attributes(state)
        
        # Apply state attributes to all rows
        for col in ['state_population_2011', 'rainfall_zone', 'earthquake_risk_zone', 
                    'climate_type', 'average_temperature_celsius', 'literacy_rate_percent',
                    'sex_ratio_per_1000_males', 'per_capita_income_usd', 'human_development_index']:
            df[col] = df['state'].map(lambda x: state_attrs[x].get(col, np.nan))
        
        print(f"  ✓ Added {len(unique_states)} state profiles")
        
        # ====== STEP 2: Add coordinates from Nominatim ======
        print(f"\n[2/4] Fetching coordinates from OpenStreetMap Nominatim...")
        coords_success = 0
        for idx, pincode in enumerate(unique_pincodes[:100], 1):  # Test with first 100
            if pd.isna(pincode):
                continue
            
            pincode_str = str(int(pincode))
            coords = self.get_coordinates_from_nominatim(pincode_str)
            
            if coords:
                df.loc[df['pincode'] == pincode, ['latitude', 'longitude']] = coords
                coords_success += 1
            
            if idx % 10 == 0:
                print(f"  Progress: {idx}/100 pincodes processed ({coords_success} successful)")
                
        print(f"  ✓ Successfully obtained coordinates for {coords_success}/100 pincodes")
        print(f"  Note: Rate limiting in place (1 req/sec). Full augmentation recommended in batch.")
        
        # ====== STEP 3: Add weather data ======
        print(f"\n[3/4] Fetching weather data from Open-Meteo...")
        rows_with_coords = df[df['latitude'].notna() & df['longitude'].notna()]
        weather_success = 0
        
        for idx, (_, row) in enumerate(rows_with_coords.head(20).iterrows(), 1):
            lat, lon = row['latitude'], row['longitude']
            pincode = row['pincode']
            
            weather_data = self.get_weather_data(lat, lon, str(int(pincode)))
            
            for col, val in weather_data.items():
                df.loc[_, col] = val
            
            weather_success += 1
            
            if idx % 5 == 0:
                print(f"  Progress: {idx} records with weather data")
        
        print(f"  ✓ Added weather data to {weather_success} records")
        
        # ====== STEP 4: Mark augmentation status ======
        print(f"\n[4/4] Marking augmentation status...")
        df['augmentation_status'] = 'pending'
        df.loc[df['state_population_2011'].notna(), 'augmentation_status'] = 'state_data_added'
        df.loc[df['latitude'].notna(), 'augmentation_status'] = 'coordinates_added'
        df.loc[df['current_temperature_celsius'].notna(), 'augmentation_status'] = 'complete'
        
        print(f"  ✓ Status marking complete")
        
        # ====== Summary ======
        print(f"\n{'='*80}")
        print(f"AUGMENTATION SUMMARY - {dataset_name.upper()}")
        print(f"{'='*80}")
        print(f"Records with state data: {df['state_population_2011'].notna().sum()}")
        print(f"Records with coordinates: {df['latitude'].notna().sum()}")
        print(f"Records with weather data: {df['current_temperature_celsius'].notna().sum()}")
        print(f"New columns added: {len([c for c in df.columns if c in new_columns])}")
        
        return df

def main():
    """Main execution"""
    print(f"\n{'='*80}")
    print(f"AADHAAR DATASET AUGMENTATION PIPELINE")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Initialize augmenter
    augmenter = DataAugmenter(cache_dir='augmentation_cache')
    
    # Load corrected datasets
    corrected_dir = "Dataset/corrected_dataset"
    
    # Test with biometric dataset (sample)
    print("\nLoading biometric dataset...")
    bio_path = os.path.join(corrected_dir, 'biometric/final_main_corrected_biometric.csv')
    
    # Load sample for testing
    bio_df = pd.read_csv(bio_path, nrows=1000)  # Test with 1000 rows first
    print(f"✓ Loaded {len(bio_df)} records from biometric dataset")
    
    # Augment
    bio_df_augmented = augmenter.augment_dataframe(bio_df, dataset_name='biometric')
    
    # Save augmented data
    output_path = os.path.join(corrected_dir, 'biometric/biometric_augmented_test.csv')
    bio_df_augmented.to_csv(output_path, index=False)
    print(f"\n✓ Saved augmented biometric data to: {output_path}")
    
    print(f"\nAugmented dataset shape: {bio_df_augmented.shape}")
    print(f"New columns: {[c for c in bio_df_augmented.columns if c not in bio_df.columns]}")
    
    print(f"\n{'='*80}")
    print(f"AUGMENTATION COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    return bio_df_augmented

if __name__ == '__main__':
    augmented_df = main()
    
    # Print sample of augmented data
    print("\nSample of augmented records:")
    print(augmented_df[['state', 'district', 'pincode', 'latitude', 'longitude', 
                        'average_temperature_celsius', 'rainfall_zone', 
                        'per_capita_income_usd']].head(10))
