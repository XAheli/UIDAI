"""
External API Clients Package
Author: Shuvam Banerji Seal's Team
UIDAI Hackathon - Aadhaar Data Analysis

Provides clients for external data sources:
- Open-Meteo (Weather data)
- OpenStreetMap (Geographic data)
- India Post (Pincode data)
- Census API (Population data)
"""

from .weather import WeatherClient
from .geographic import GeoClient
from .pincode import PincodeClient
from .census import CensusClient
from .base import BaseAPIClient, RateLimiter, APICache

__all__ = [
    'WeatherClient',
    'GeoClient',
    'PincodeClient',
    'CensusClient',
    'BaseAPIClient',
    'RateLimiter',
    'APICache'
]
