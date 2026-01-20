"""
Optimized Data Augmentation Pipeline for Aadhaar Datasets
Handles large datasets efficiently with batch processing and caching
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Import reference data
from india_reference_data import (
    INDIA_CENSUS_DATA, RAINFALL_ZONES, EARTHQUAKE_ZONES,
    CLIMATE_TYPES, PER_CAPITA_INCOME_USD, HUMAN_DEVELOPMENT_INDEX
)

class OptimizedAugmenter:
    """
    Optimized augmentation focusing on state-level attributes (no rate-limited APIs in bulk)
    Provides framework for online APIs with proper rate limiting
    """
    
    def __init__(self):
        pass
    
    def get_state_attributes(self, state: str) -> Dict:
        """Get state-level attributes from reference data"""
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
        Augment dataframe with state-level attributes efficiently
        """
        print(f"\n{'='*80}")
        print(f"AUGMENTING {dataset_name.upper()} DATASET")
        print(f"{'='*80}")
        print(f"Total records: {len(df)}")
        
        # Initialize new columns
        new_cols = [
            'state_population_2011',
            'rainfall_zone',
            'earthquake_risk_zone',
            'climate_type',
            'average_temperature_celsius',
            'literacy_rate_percent',
            'sex_ratio_per_1000_males',
            'per_capita_income_usd',
            'human_development_index'
        ]
        
        for col in new_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Add attributes
        unique_states = df['state'].unique()
        print(f"Unique states: {len(unique_states)}")
        
        state_attrs_cache = {}
        for state in unique_states:
            state_attrs_cache[state] = self.get_state_attributes(state)
        
        print(f"[1/1] Adding state-level attributes...")
        for col in new_cols:
            df[col] = df['state'].map(lambda x: state_attrs_cache[x].get(col, np.nan))
        
        print(f"  ✓ State attributes added")
        
        # Summary
        print(f"\n{'='*80}")
        print(f"AUGMENTATION SUMMARY - {dataset_name.upper()}")
        print(f"{'='*80}")
        print(f"Records processed: {len(df)}")
        print(f"New columns added: {len(new_cols)}")
        print(f"Records with state data: {df['state_population_2011'].notna().sum()}")
        
        # Show data quality
        print(f"\nNew columns coverage:")
        for col in new_cols:
            coverage = (df[col].notna().sum() / len(df)) * 100
            print(f"  {col}: {coverage:.1f}%")
        
        return df

def main():
    """Main execution"""
    print(f"\n{'='*80}")
    print(f"AADHAAR DATASET AUGMENTATION - OPTIMIZED PIPELINE")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    augmenter = OptimizedAugmenter()
    corrected_dir = "Dataset/corrected_dataset"
    output_dir = os.path.join(corrected_dir, 'augmented_datasets')
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}\n")
    
    # ====== BIOMETRIC DATASET ======
    print("\n" + "="*80)
    print("PROCESSING: BIOMETRIC DATASET")
    print("="*80)
    
    bio_path = os.path.join(corrected_dir, 'biometric/final_main_corrected_biometric.csv')
    bio_df = pd.read_csv(bio_path, dtype={'pincode': 'Int64'})
    print(f"✓ Loaded {len(bio_df)} records")
    
    bio_augmented = augmenter.augment_dataframe(bio_df, dataset_name='biometric')
    bio_output = os.path.join(output_dir, 'biometric_augmented.csv')
    bio_augmented.to_csv(bio_output, index=False)
    print(f"✓ Saved to: {bio_output}")
    
    # ====== DEMOGRAPHIC DATASET ======
    print("\n" + "="*80)
    print("PROCESSING: DEMOGRAPHIC DATASET")
    print("="*80)
    
    dem_dir = os.path.join(corrected_dir, 'demographic')
    dem_files = sorted([f for f in os.listdir(dem_dir) if f.startswith('corrected_') and f.endswith('.csv')])
    
    print(f"Found {len(dem_files)} demographic files")
    
    dem_dfs = []
    for file in dem_files:
        file_path = os.path.join(dem_dir, file)
        df = pd.read_csv(file_path, dtype={'pincode': 'Int64'})
        print(f"  Loading: {file} ({len(df)} records)")
        dem_dfs.append(df)
    
    dem_combined = pd.concat(dem_dfs, ignore_index=True)
    print(f"✓ Combined {len(dem_combined)} records from demographic files")
    
    dem_augmented = augmenter.augment_dataframe(dem_combined, dataset_name='demographic')
    dem_output = os.path.join(output_dir, 'demographic_augmented.csv')
    dem_augmented.to_csv(dem_output, index=False)
    print(f"✓ Saved to: {dem_output}")
    
    # ====== ENROLLMENT DATASET ======
    print("\n" + "="*80)
    print("PROCESSING: ENROLLMENT DATASET")
    print("="*80)
    
    enr_dir = os.path.join(corrected_dir, 'enrollement')
    enr_files = sorted([f for f in os.listdir(enr_dir) if f.startswith('corrected_') and f.endswith('.csv')])
    
    print(f"Found {len(enr_files)} enrollment files")
    
    enr_dfs = []
    for file in enr_files:
        file_path = os.path.join(enr_dir, file)
        df = pd.read_csv(file_path, dtype={'pincode': 'Int64'})
        print(f"  Loading: {file} ({len(df)} records)")
        enr_dfs.append(df)
    
    enr_combined = pd.concat(enr_dfs, ignore_index=True)
    print(f"✓ Combined {len(enr_combined)} records from enrollment files")
    
    enr_augmented = augmenter.augment_dataframe(enr_combined, dataset_name='enrollment')
    enr_output = os.path.join(output_dir, 'enrollment_augmented.csv')
    enr_augmented.to_csv(enr_output, index=False)
    print(f"✓ Saved to: {enr_output}")
    
    # ====== FINAL SUMMARY ======
    print(f"\n{'='*80}")
    print(f"AUGMENTATION PIPELINE COMPLETE")
    print(f"{'='*80}\n")
    
    summary_data = {
        'Dataset': ['Biometric', 'Demographic', 'Enrollment'],
        'Original Records': [len(bio_df), len(dem_combined), len(enr_combined)],
        'Augmented Records': [len(bio_augmented), len(dem_augmented), len(enr_augmented)],
        'Output Files': [bio_output, dem_output, enr_output],
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print(f"\nNew Columns Added (to all datasets):")
    new_columns = [
        'state_population_2011',
        'rainfall_zone',
        'earthquake_risk_zone', 
        'climate_type',
        'average_temperature_celsius',
        'literacy_rate_percent',
        'sex_ratio_per_1000_males',
        'per_capita_income_usd',
        'human_development_index'
    ]
    for i, col in enumerate(new_columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'='*80}")
    
    return {
        'biometric': bio_augmented,
        'demographic': dem_augmented,
        'enrollment': enr_augmented
    }

if __name__ == '__main__':
    augmented_datasets = main()
    
    # Print sample
    print("\nSample Augmented Biometric Records:")
    print(augmented_datasets['biometric'].head(5).to_string())

