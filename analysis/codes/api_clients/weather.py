"""
Weather API Client
Author: Shuvam Banerji Seal's Team
UIDAI Hackathon - Aadhaar Data Analysis

Provides weather data from Open-Meteo API:
- Historical weather data
- Climate analysis
- Weather patterns for enrollment correlation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .base import BaseAPIClient

logger = logging.getLogger(__name__)


class WeatherClient(BaseAPIClient):
    """
    Client for Open-Meteo Weather API
    Free, no API key required
    """
    
    def __init__(
        self,
        rate_limit: float = 10.0,
        cache_ttl: int = 86400  # 24 hours
    ):
        """
        Initialize weather client
        
        Args:
            rate_limit: Requests per second
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__(
            base_url='https://archive-api.open-meteo.com/v1',
            api_key=None,  # No API key required
            rate_limit=rate_limit,
            cache_ttl=cache_ttl
        )
        
        # Alternative endpoint for recent data
        self.forecast_url = 'https://api.open-meteo.com/v1'
    
    def health_check(self) -> bool:
        """Check if API is available"""
        try:
            # Make a minimal request
            result = self._make_request(
                'archive',
                params={
                    'latitude': 28.6139,  # Delhi
                    'longitude': 77.2090,
                    'start_date': '2024-01-01',
                    'end_date': '2024-01-01',
                    'daily': 'temperature_2m_mean'
                }
            )
            return 'daily' in result
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def get_historical_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get historical weather data
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: Weather variables to fetch
            
        Returns:
            DataFrame with weather data
        """
        if variables is None:
            variables = [
                'temperature_2m_mean',
                'temperature_2m_max',
                'temperature_2m_min',
                'precipitation_sum',
                'rain_sum',
                'wind_speed_10m_max'
            ]
        
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date,
            'end_date': end_date,
            'daily': ','.join(variables),
            'timezone': 'Asia/Kolkata'
        }
        
        result = self._make_request('archive', params=params)
        
        if 'daily' not in result:
            raise ValueError(f"No weather data returned: {result}")
        
        # Convert to DataFrame
        daily = result['daily']
        df = pd.DataFrame({
            'date': pd.to_datetime(daily['time']),
            **{var: daily.get(var, [None] * len(daily['time'])) for var in variables}
        })
        
        df['latitude'] = latitude
        df['longitude'] = longitude
        
        return df
    
    def get_weather_for_locations(
        self,
        locations: List[Tuple[float, float, str]],
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get weather data for multiple locations
        
        Args:
            locations: List of (latitude, longitude, name) tuples
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: Weather variables to fetch
            
        Returns:
            Combined DataFrame with weather data
        """
        all_data = []
        
        for lat, lon, name in locations:
            try:
                df = self.get_historical_weather(lat, lon, start_date, end_date, variables)
                df['location_name'] = name
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to get weather for {name}: {str(e)}")
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def get_india_state_weather(
        self,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get weather data for major Indian state capitals
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: Weather variables to fetch
            
        Returns:
            DataFrame with weather data for all states
        """
        # Indian state capitals with coordinates
        state_capitals = [
            (28.6139, 77.2090, 'DELHI'),
            (19.0760, 72.8777, 'MAHARASHTRA'),
            (22.5726, 88.3639, 'WEST BENGAL'),
            (13.0827, 80.2707, 'TAMIL NADU'),
            (12.9716, 77.5946, 'KARNATAKA'),
            (17.3850, 78.4867, 'TELANGANA'),
            (23.0225, 72.5714, 'GUJARAT'),
            (26.9124, 75.7873, 'RAJASTHAN'),
            (30.7333, 76.7794, 'PUNJAB'),
            (26.8467, 80.9462, 'UTTAR PRADESH'),
            (25.5941, 85.1376, 'BIHAR'),
            (22.2587, 71.1924, 'MADHYA PRADESH'),
            (21.2514, 81.6296, 'CHHATTISGARH'),
            (20.2961, 85.8245, 'ODISHA'),
            (23.2599, 77.4126, 'JHARKHAND'),
            (10.8505, 76.2711, 'KERALA'),
            (17.6868, 83.2185, 'ANDHRA PRADESH'),
            (26.1445, 91.7362, 'ASSAM'),
            (15.2993, 74.1240, 'GOA'),
            (31.1048, 77.1734, 'HIMACHAL PRADESH'),
            (34.0837, 74.7973, 'JAMMU AND KASHMIR'),
            (23.1645, 79.9864, 'UTTARAKHAND'),
            (27.5330, 88.5122, 'SIKKIM'),
            (25.4670, 91.3662, 'MEGHALAYA'),
            (24.8170, 93.9368, 'MANIPUR'),
            (25.6747, 94.1086, 'NAGALAND'),
            (23.7271, 92.7176, 'MIZORAM'),
            (27.1024, 93.6166, 'ARUNACHAL PRADESH'),
            (23.8315, 91.2868, 'TRIPURA'),
        ]
        
        return self.get_weather_for_locations(
            state_capitals, start_date, end_date, variables
        )
    
    def analyze_weather_patterns(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze weather patterns from collected data
        
        Args:
            df: Weather DataFrame
            
        Returns:
            Analysis results
        """
        if df.empty:
            return {'error': 'No data to analyze'}
        
        analysis = {
            'summary': {},
            'by_location': {},
            'seasonal_patterns': {}
        }
        
        # Overall summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        analysis['summary'] = {
            col: {
                'mean': float(df[col].mean()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'std': float(df[col].std())
            }
            for col in numeric_cols if col not in ['latitude', 'longitude']
        }
        
        # By location
        if 'location_name' in df.columns:
            for location in df['location_name'].unique():
                loc_df = df[df['location_name'] == location]
                analysis['by_location'][location] = {
                    col: {
                        'mean': float(loc_df[col].mean()),
                        'min': float(loc_df[col].min()),
                        'max': float(loc_df[col].max())
                    }
                    for col in numeric_cols if col not in ['latitude', 'longitude']
                }
        
        # Seasonal patterns
        if 'date' in df.columns:
            df['month'] = pd.to_datetime(df['date']).dt.month
            monthly_avg = df.groupby('month')[numeric_cols].mean()
            
            for col in numeric_cols:
                if col not in ['latitude', 'longitude', 'month']:
                    analysis['seasonal_patterns'][col] = {
                        int(month): float(val)
                        for month, val in monthly_avg[col].items()
                    }
        
        return analysis


# Indian city coordinates for reference
INDIAN_CITIES = {
    'Delhi': (28.6139, 77.2090),
    'Mumbai': (19.0760, 72.8777),
    'Kolkata': (22.5726, 88.3639),
    'Chennai': (13.0827, 80.2707),
    'Bangalore': (12.9716, 77.5946),
    'Hyderabad': (17.3850, 78.4867),
    'Ahmedabad': (23.0225, 72.5714),
    'Pune': (18.5204, 73.8567),
    'Jaipur': (26.9124, 75.7873),
    'Lucknow': (26.8467, 80.9462),
    'Patna': (25.5941, 85.1376),
    'Bhopal': (23.2599, 77.4126),
    'Chandigarh': (30.7333, 76.7794),
    'Guwahati': (26.1445, 91.7362),
    'Thiruvananthapuram': (8.5241, 76.9366)
}


__all__ = ['WeatherClient', 'INDIAN_CITIES']
