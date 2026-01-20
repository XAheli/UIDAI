"""
Census API Client
Author: Shuvam Banerji Seal's Team
UIDAI Hackathon - Aadhaar Data Analysis

Provides census and demographic data:
- Population statistics
- Literacy rates
- Sex ratio data
- Note: Uses local data files as India doesn't have a public Census API
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class CensusClient:
    """
    Client for census data
    Uses local data files since India doesn't have a public Census API
    """
    
    # Census 2011 data (official)
    CENSUS_2011_DATA = {
        'ANDHRA PRADESH': {'population': 49577103, 'literacy_rate': 67.02, 'sex_ratio': 993, 'area_km2': 160205},
        'ARUNACHAL PRADESH': {'population': 1383727, 'literacy_rate': 65.38, 'sex_ratio': 938, 'area_km2': 83743},
        'ASSAM': {'population': 31205576, 'literacy_rate': 72.19, 'sex_ratio': 958, 'area_km2': 78438},
        'BIHAR': {'population': 104099452, 'literacy_rate': 61.80, 'sex_ratio': 918, 'area_km2': 94163},
        'CHHATTISGARH': {'population': 25545198, 'literacy_rate': 70.28, 'sex_ratio': 991, 'area_km2': 135192},
        'GOA': {'population': 1458545, 'literacy_rate': 87.40, 'sex_ratio': 973, 'area_km2': 3702},
        'GUJARAT': {'population': 60439692, 'literacy_rate': 78.03, 'sex_ratio': 919, 'area_km2': 196244},
        'HARYANA': {'population': 25351462, 'literacy_rate': 75.55, 'sex_ratio': 879, 'area_km2': 44212},
        'HIMACHAL PRADESH': {'population': 6864602, 'literacy_rate': 82.80, 'sex_ratio': 972, 'area_km2': 55673},
        'JHARKHAND': {'population': 32988134, 'literacy_rate': 66.41, 'sex_ratio': 949, 'area_km2': 79716},
        'KARNATAKA': {'population': 61095297, 'literacy_rate': 75.36, 'sex_ratio': 973, 'area_km2': 191791},
        'KERALA': {'population': 33406061, 'literacy_rate': 93.91, 'sex_ratio': 1084, 'area_km2': 38863},
        'MADHYA PRADESH': {'population': 72626809, 'literacy_rate': 69.32, 'sex_ratio': 931, 'area_km2': 308252},
        'MAHARASHTRA': {'population': 112374333, 'literacy_rate': 82.34, 'sex_ratio': 929, 'area_km2': 307713},
        'MANIPUR': {'population': 2855794, 'literacy_rate': 79.21, 'sex_ratio': 985, 'area_km2': 22327},
        'MEGHALAYA': {'population': 2966889, 'literacy_rate': 74.43, 'sex_ratio': 989, 'area_km2': 22429},
        'MIZORAM': {'population': 1097206, 'literacy_rate': 91.33, 'sex_ratio': 976, 'area_km2': 21081},
        'NAGALAND': {'population': 1978502, 'literacy_rate': 79.55, 'sex_ratio': 931, 'area_km2': 16579},
        'ODISHA': {'population': 41974218, 'literacy_rate': 72.87, 'sex_ratio': 979, 'area_km2': 155707},
        'PUNJAB': {'population': 27743338, 'literacy_rate': 75.84, 'sex_ratio': 895, 'area_km2': 50362},
        'RAJASTHAN': {'population': 68548437, 'literacy_rate': 66.11, 'sex_ratio': 928, 'area_km2': 342239},
        'SIKKIM': {'population': 610577, 'literacy_rate': 81.42, 'sex_ratio': 890, 'area_km2': 7096},
        'TAMIL NADU': {'population': 72147030, 'literacy_rate': 80.09, 'sex_ratio': 996, 'area_km2': 130058},
        'TELANGANA': {'population': 35003674, 'literacy_rate': 66.54, 'sex_ratio': 988, 'area_km2': 112077},
        'TRIPURA': {'population': 3673917, 'literacy_rate': 87.22, 'sex_ratio': 960, 'area_km2': 10486},
        'UTTAR PRADESH': {'population': 199812341, 'literacy_rate': 67.68, 'sex_ratio': 912, 'area_km2': 240928},
        'UTTARAKHAND': {'population': 10086292, 'literacy_rate': 78.82, 'sex_ratio': 963, 'area_km2': 53483},
        'WEST BENGAL': {'population': 91276115, 'literacy_rate': 76.26, 'sex_ratio': 950, 'area_km2': 88752},
        'DELHI': {'population': 16787941, 'literacy_rate': 86.21, 'sex_ratio': 868, 'area_km2': 1484},
        'JAMMU AND KASHMIR': {'population': 12541302, 'literacy_rate': 67.16, 'sex_ratio': 889, 'area_km2': 222236},
        'LADAKH': {'population': 274289, 'literacy_rate': 77.20, 'sex_ratio': 685, 'area_km2': 96701},
        'PUDUCHERRY': {'population': 1247953, 'literacy_rate': 85.85, 'sex_ratio': 1037, 'area_km2': 490},
        'CHANDIGARH': {'population': 1055450, 'literacy_rate': 86.05, 'sex_ratio': 818, 'area_km2': 114},
        'ANDAMAN AND NICOBAR ISLANDS': {'population': 380581, 'literacy_rate': 86.27, 'sex_ratio': 876, 'area_km2': 8249},
        'DADRA AND NAGAR HAVELI AND DAMAN AND DIU': {'population': 585764, 'literacy_rate': 76.24, 'sex_ratio': 774, 'area_km2': 603},
        'LAKSHADWEEP': {'population': 64473, 'literacy_rate': 91.85, 'sex_ratio': 946, 'area_km2': 32}
    }
    
    # District-level data (subset of major districts)
    DISTRICT_DATA = {
        'MUMBAI': {'state': 'MAHARASHTRA', 'population': 12478447, 'literacy_rate': 89.21},
        'DELHI': {'state': 'DELHI', 'population': 11034555, 'literacy_rate': 86.34},
        'BANGALORE URBAN': {'state': 'KARNATAKA', 'population': 9621551, 'literacy_rate': 87.67},
        'HYDERABAD': {'state': 'TELANGANA', 'population': 6809970, 'literacy_rate': 83.25},
        'CHENNAI': {'state': 'TAMIL NADU', 'population': 4681087, 'literacy_rate': 90.18},
        'KOLKATA': {'state': 'WEST BENGAL', 'population': 4486679, 'literacy_rate': 86.31},
        'AHMEDABAD': {'state': 'GUJARAT', 'population': 7208200, 'literacy_rate': 85.31},
        'PUNE': {'state': 'MAHARASHTRA', 'population': 9426959, 'literacy_rate': 86.15},
        'JAIPUR': {'state': 'RAJASTHAN', 'population': 6626178, 'literacy_rate': 75.51},
        'LUCKNOW': {'state': 'UTTAR PRADESH', 'population': 4589838, 'literacy_rate': 77.29},
    }
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize census client
        
        Args:
            data_dir: Directory for additional census data files
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self._load_additional_data()
    
    def _load_additional_data(self):
        """Load additional census data from files if available"""
        if self.data_dir and self.data_dir.exists():
            # Try to load district data
            district_file = self.data_dir / 'districts.json'
            if district_file.exists():
                try:
                    with open(district_file, 'r') as f:
                        additional_districts = json.load(f)
                        self.DISTRICT_DATA.update(additional_districts)
                except Exception as e:
                    logger.warning(f"Failed to load additional district data: {str(e)}")
    
    def get_state_data(self, state: str) -> Optional[Dict[str, Any]]:
        """
        Get census data for a state
        
        Args:
            state: State name
            
        Returns:
            Dict with census data
        """
        state_upper = state.upper().strip()
        
        # Handle common variations
        variations = {
            'JAMMU & KASHMIR': 'JAMMU AND KASHMIR',
            'J&K': 'JAMMU AND KASHMIR',
            'A&N ISLANDS': 'ANDAMAN AND NICOBAR ISLANDS',
            'DNH': 'DADRA AND NAGAR HAVELI AND DAMAN AND DIU',
        }
        
        state_upper = variations.get(state_upper, state_upper)
        
        if state_upper in self.CENSUS_2011_DATA:
            data = self.CENSUS_2011_DATA[state_upper].copy()
            data['state'] = state_upper
            data['density'] = round(data['population'] / data['area_km2'], 2)
            data['source'] = 'Census 2011'
            return data
        
        return None
    
    def get_all_states_data(self) -> pd.DataFrame:
        """
        Get census data for all states
        
        Returns:
            DataFrame with all states' census data
        """
        records = []
        for state, data in self.CENSUS_2011_DATA.items():
            record = {
                'state': state,
                **data,
                'density': round(data['population'] / data['area_km2'], 2)
            }
            records.append(record)
        
        return pd.DataFrame(records).sort_values('population', ascending=False)
    
    def get_district_data(self, district: str) -> Optional[Dict[str, Any]]:
        """
        Get census data for a district
        
        Args:
            district: District name
            
        Returns:
            Dict with census data
        """
        district_upper = district.upper().strip()
        
        if district_upper in self.DISTRICT_DATA:
            data = self.DISTRICT_DATA[district_upper].copy()
            data['district'] = district_upper
            data['source'] = 'Census 2011'
            return data
        
        return None
    
    def compare_states(
        self,
        states: List[str],
        metric: str = 'population'
    ) -> pd.DataFrame:
        """
        Compare multiple states on a metric
        
        Args:
            states: List of state names
            metric: Metric to compare
            
        Returns:
            DataFrame with comparison
        """
        records = []
        for state in states:
            data = self.get_state_data(state)
            if data:
                records.append({
                    'state': state.upper(),
                    metric: data.get(metric),
                    'rank': None  # Will be filled
                })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(metric, ascending=False)
            df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_national_statistics(self) -> Dict[str, Any]:
        """
        Get national-level statistics
        
        Returns:
            Dict with national statistics
        """
        df = self.get_all_states_data()
        
        total_pop = df['population'].sum()
        total_area = df['area_km2'].sum()
        
        return {
            'total_population': int(total_pop),
            'total_area_km2': int(total_area),
            'population_density': round(total_pop / total_area, 2),
            'average_literacy_rate': round(df['literacy_rate'].mean(), 2),
            'weighted_literacy_rate': round(
                (df['population'] * df['literacy_rate']).sum() / total_pop, 2
            ),
            'average_sex_ratio': round(df['sex_ratio'].mean(), 0),
            'weighted_sex_ratio': round(
                (df['population'] * df['sex_ratio']).sum() / total_pop, 0
            ),
            'most_populous_state': df.iloc[0]['state'],
            'highest_literacy_state': df.loc[df['literacy_rate'].idxmax(), 'state'],
            'highest_sex_ratio_state': df.loc[df['sex_ratio'].idxmax(), 'state'],
            'num_states': len(df),
            'source': 'Census 2011'
        }
    
    def get_region_statistics(
        self,
        region: str
    ) -> Dict[str, Any]:
        """
        Get statistics for a region
        
        Args:
            region: Region name (North, South, East, West, Central, Northeast)
            
        Returns:
            Dict with regional statistics
        """
        regions = {
            'NORTH': ['DELHI', 'HARYANA', 'HIMACHAL PRADESH', 'JAMMU AND KASHMIR', 'LADAKH', 'PUNJAB', 'RAJASTHAN', 'UTTARAKHAND', 'CHANDIGARH'],
            'SOUTH': ['ANDHRA PRADESH', 'KARNATAKA', 'KERALA', 'TAMIL NADU', 'TELANGANA', 'PUDUCHERRY', 'LAKSHADWEEP', 'ANDAMAN AND NICOBAR ISLANDS'],
            'EAST': ['BIHAR', 'JHARKHAND', 'ODISHA', 'WEST BENGAL'],
            'WEST': ['GOA', 'GUJARAT', 'MAHARASHTRA', 'DADRA AND NAGAR HAVELI AND DAMAN AND DIU'],
            'CENTRAL': ['CHHATTISGARH', 'MADHYA PRADESH', 'UTTAR PRADESH'],
            'NORTHEAST': ['ARUNACHAL PRADESH', 'ASSAM', 'MANIPUR', 'MEGHALAYA', 'MIZORAM', 'NAGALAND', 'SIKKIM', 'TRIPURA']
        }
        
        region_upper = region.upper()
        if region_upper not in regions:
            return {'error': f'Unknown region: {region}'}
        
        states = regions[region_upper]
        records = []
        for state in states:
            data = self.get_state_data(state)
            if data:
                records.append(data)
        
        if not records:
            return {'error': 'No data found for region'}
        
        df = pd.DataFrame(records)
        total_pop = df['population'].sum()
        total_area = df['area_km2'].sum()
        
        return {
            'region': region_upper,
            'states': states,
            'total_population': int(total_pop),
            'total_area_km2': int(total_area),
            'population_density': round(total_pop / total_area, 2),
            'average_literacy_rate': round(df['literacy_rate'].mean(), 2),
            'weighted_literacy_rate': round(
                (df['population'] * df['literacy_rate']).sum() / total_pop, 2
            ),
            'average_sex_ratio': round(df['sex_ratio'].mean(), 0)
        }
    
    def export_to_json(self, output_path: Union[str, Path]) -> str:
        """
        Export all census data to JSON
        
        Args:
            output_path: Output file path
            
        Returns:
            Path to created file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'states': self.CENSUS_2011_DATA,
            'districts': self.DISTRICT_DATA,
            'national': self.get_national_statistics(),
            'regions': {
                region: self.get_region_statistics(region)
                for region in ['NORTH', 'SOUTH', 'EAST', 'WEST', 'CENTRAL', 'NORTHEAST']
            },
            'metadata': {
                'source': 'Census of India 2011',
                'generated_by': "Shuvam Banerji Seal's Team"
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(output_path)


__all__ = ['CensusClient']
