"""
Geographic API Client
Author: Shuvam Banerji Seal's Team
UIDAI Hackathon - Aadhaar Data Analysis

Provides geographic data from OpenStreetMap Nominatim:
- Geocoding (address to coordinates)
- Reverse geocoding (coordinates to address)
- Location search
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import logging
import time

from .base import BaseAPIClient

logger = logging.getLogger(__name__)


class GeoClient(BaseAPIClient):
    """
    Client for OpenStreetMap Nominatim API
    Free, no API key required
    """
    
    def __init__(
        self,
        rate_limit: float = 1.0,  # Nominatim limit: 1 req/sec
        cache_ttl: int = 604800  # 1 week
    ):
        """
        Initialize geographic client
        
        Args:
            rate_limit: Requests per second (Nominatim limit is 1/sec)
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__(
            base_url='https://nominatim.openstreetmap.org',
            api_key=None,
            rate_limit=rate_limit,
            cache_ttl=cache_ttl
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Override headers for Nominatim"""
        return {
            'User-Agent': 'UIDAI-Analysis/1.0 (Shuvam Banerji Seal Team - Academic Research)',
            'Accept': 'application/json'
        }
    
    def health_check(self) -> bool:
        """Check if API is available"""
        try:
            result = self._make_request(
                'search',
                params={'q': 'Delhi, India', 'format': 'json', 'limit': 1}
            )
            return isinstance(result, list) and len(result) > 0
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def geocode(
        self,
        query: str,
        country: str = 'India'
    ) -> Optional[Dict[str, Any]]:
        """
        Convert address to coordinates
        
        Args:
            query: Address or place name
            country: Country to search in
            
        Returns:
            Dict with lat, lon, and display_name
        """
        params = {
            'q': f"{query}, {country}",
            'format': 'json',
            'limit': 1,
            'countrycodes': 'in'
        }
        
        result = self._make_request('search', params=params)
        
        if not result:
            return None
        
        location = result[0]
        return {
            'latitude': float(location['lat']),
            'longitude': float(location['lon']),
            'display_name': location['display_name'],
            'type': location.get('type', ''),
            'importance': float(location.get('importance', 0))
        }
    
    def reverse_geocode(
        self,
        latitude: float,
        longitude: float
    ) -> Optional[Dict[str, Any]]:
        """
        Convert coordinates to address
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            
        Returns:
            Dict with address details
        """
        params = {
            'lat': latitude,
            'lon': longitude,
            'format': 'json'
        }
        
        result = self._make_request('reverse', params=params)
        
        if 'error' in result:
            return None
        
        address = result.get('address', {})
        return {
            'display_name': result.get('display_name', ''),
            'city': address.get('city', address.get('town', address.get('village', ''))),
            'district': address.get('state_district', address.get('county', '')),
            'state': address.get('state', ''),
            'country': address.get('country', ''),
            'postcode': address.get('postcode', '')
        }
    
    def geocode_batch(
        self,
        queries: List[str],
        country: str = 'India',
        delay: float = 1.1  # Ensure we respect rate limit
    ) -> pd.DataFrame:
        """
        Geocode multiple locations
        
        Args:
            queries: List of addresses/places
            country: Country to search in
            delay: Delay between requests
            
        Returns:
            DataFrame with geocoded results
        """
        results = []
        
        for i, query in enumerate(queries):
            try:
                result = self.geocode(query, country)
                if result:
                    results.append({
                        'query': query,
                        **result
                    })
                else:
                    results.append({
                        'query': query,
                        'latitude': None,
                        'longitude': None,
                        'display_name': None
                    })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Geocoded {i + 1}/{len(queries)} locations")
                
                # Respect rate limit
                time.sleep(delay)
                
            except Exception as e:
                logger.warning(f"Failed to geocode '{query}': {str(e)}")
                results.append({
                    'query': query,
                    'latitude': None,
                    'longitude': None,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def search_places(
        self,
        query: str,
        limit: int = 10,
        country: str = 'India'
    ) -> List[Dict[str, Any]]:
        """
        Search for places matching query
        
        Args:
            query: Search query
            limit: Maximum results
            country: Country to search in
            
        Returns:
            List of matching places
        """
        params = {
            'q': f"{query}, {country}",
            'format': 'json',
            'limit': limit,
            'countrycodes': 'in'
        }
        
        result = self._make_request('search', params=params)
        
        return [
            {
                'name': item.get('display_name', ''),
                'latitude': float(item['lat']),
                'longitude': float(item['lon']),
                'type': item.get('type', ''),
                'importance': float(item.get('importance', 0))
            }
            for item in result
        ]
    
    def get_state_boundaries(
        self,
        state_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get boundary information for an Indian state
        
        Args:
            state_name: Name of the state
            
        Returns:
            Dict with boundary info
        """
        params = {
            'q': f"{state_name}, India",
            'format': 'json',
            'limit': 1,
            'polygon_geojson': 1
        }
        
        result = self._make_request('search', params=params)
        
        if not result:
            return None
        
        location = result[0]
        boundingbox = location.get('boundingbox', [])
        
        return {
            'name': state_name,
            'display_name': location.get('display_name', ''),
            'latitude': float(location['lat']),
            'longitude': float(location['lon']),
            'boundingbox': {
                'south': float(boundingbox[0]) if len(boundingbox) > 0 else None,
                'north': float(boundingbox[1]) if len(boundingbox) > 1 else None,
                'west': float(boundingbox[2]) if len(boundingbox) > 2 else None,
                'east': float(boundingbox[3]) if len(boundingbox) > 3 else None
            },
            'geojson': location.get('geojson')
        }
    
    def calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate distance between two coordinates using Haversine formula
        
        Args:
            lat1, lon1: First coordinate
            lat2, lon2: Second coordinate
            
        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c


# Indian state coordinates (approximate centroids)
INDIAN_STATE_COORDS = {
    'ANDHRA PRADESH': (15.9129, 79.7400),
    'ARUNACHAL PRADESH': (28.2180, 94.7278),
    'ASSAM': (26.2006, 92.9376),
    'BIHAR': (25.0961, 85.3131),
    'CHHATTISGARH': (21.2787, 81.8661),
    'GOA': (15.2993, 74.1240),
    'GUJARAT': (22.2587, 71.1924),
    'HARYANA': (29.0588, 76.0856),
    'HIMACHAL PRADESH': (31.1048, 77.1734),
    'JHARKHAND': (23.6102, 85.2799),
    'KARNATAKA': (15.3173, 75.7139),
    'KERALA': (10.8505, 76.2711),
    'MADHYA PRADESH': (22.9734, 78.6569),
    'MAHARASHTRA': (19.7515, 75.7139),
    'MANIPUR': (24.6637, 93.9063),
    'MEGHALAYA': (25.4670, 91.3662),
    'MIZORAM': (23.1645, 92.9376),
    'NAGALAND': (26.1584, 94.5624),
    'ODISHA': (20.9517, 85.0985),
    'PUNJAB': (31.1471, 75.3412),
    'RAJASTHAN': (27.0238, 74.2179),
    'SIKKIM': (27.5330, 88.5122),
    'TAMIL NADU': (11.1271, 78.6569),
    'TELANGANA': (18.1124, 79.0193),
    'TRIPURA': (23.9408, 91.9882),
    'UTTAR PRADESH': (26.8467, 80.9462),
    'UTTARAKHAND': (30.0668, 79.0193),
    'WEST BENGAL': (22.9868, 87.8550),
    'DELHI': (28.7041, 77.1025),
    'JAMMU AND KASHMIR': (33.7782, 76.5762),
    'LADAKH': (34.1526, 77.5770),
    'PUDUCHERRY': (11.9416, 79.8083),
    'CHANDIGARH': (30.7333, 76.7794),
    'ANDAMAN AND NICOBAR ISLANDS': (11.7401, 92.6586),
    'DADRA AND NAGAR HAVELI AND DAMAN AND DIU': (20.1809, 73.0169),
    'LAKSHADWEEP': (10.5667, 72.6417)
}


__all__ = ['GeoClient', 'INDIAN_STATE_COORDS']
