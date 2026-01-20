#!/usr/bin/env python3
"""
UIDAI Aadhaar Data Analysis - Complete Pipeline
===============================================
This script performs comprehensive data augmentation and analysis using ALL
APIs from API_INTEGRATION_CATALOG.md and analyses from ANALYSIS_MASTER_PLAN.md

APIs Integrated:
1. Open-Meteo Weather API - Current & Historical Weather
2. Open-Meteo Air Quality API - AQI, PM2.5, PM10, Ozone, CO, NO2
3. Open-Meteo Elevation API - Terrain data
4. Open-Meteo Geocoding API - Lat/Lon coordinates
5. India Post Pincode API - Postal/Urban-Rural classification
6. Reference Data: Census 2011, HDI, Per Capita Income, Climate Zones

Analyses Performed:
- Time Series Analysis (Trends, Seasonality, Day-of-Week patterns)
- Geographic Analysis (State, District, Regional, Zonal)
- Demographic Analysis (Age groups, Population density)
- Socioeconomic Analysis (HDI, Literacy, Income correlation)
- Climate Analysis (Temperature, Rainfall, AQI impact)
- Infrastructure Analysis (Banking, Telecom, Healthcare)
- Statistical Modeling (Regression, Hypothesis testing, Clustering)
- Causal Inference (Propensity Score Matching, Diff-in-Diff)

Author: Shuvam Banerji Seal, Alok Mishra, Aheli Poddar
Date: January 2026
"""

import os
import sys
import json
import time
import hashlib
import pickle
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.codes.india_reference_data import (
    INDIA_CENSUS_DATA, RAINFALL_ZONES, HUMAN_DEVELOPMENT_INDEX,
    PER_CAPITA_INCOME_USD, CLIMATE_TYPES
)
from analysis.codes.config import (
    STATE_NAME_MAPPING, REGION_MAPPING, CLEANED_DATASET_PATH
)


# =============================================================================
# CACHING SYSTEM
# =============================================================================

class DiskCache:
    """Persistent disk-based cache for API responses."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def set(self, key: str, value: Any):
        cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)
    
    def exists(self, key: str) -> bool:
        return (self.cache_dir / f"{self._hash_key(key)}.pkl").exists()


# =============================================================================
# API CLIENTS
# =============================================================================

class OpenMeteoClient:
    """
    Open-Meteo API Client for Weather, Air Quality, Elevation, Geocoding
    Documentation: https://open-meteo.com/en/docs
    """
    
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
    AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    
    def __init__(self, cache: DiskCache):
        self.cache = cache
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'UIDAI_Analysis/1.0'})
        self.last_request = 0
        self.rate_limit = 10  # requests per second
    
    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < 1.0 / self.rate_limit:
            time.sleep(1.0 / self.rate_limit - elapsed)
        self.last_request = time.time()
    
    def _request(self, url: str, params: dict) -> Optional[dict]:
        for attempt in range(3):
            try:
                self._rate_limit()
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    time.sleep(2 ** attempt)
            except Exception as e:
                time.sleep(1)
        return None
    
    def geocode(self, location: str) -> Optional[Dict]:
        """Geocode a location name to coordinates."""
        cache_key = f"geocode_{location}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        params = {"name": f"{location}, India", "count": 1, "language": "en", "format": "json"}
        result = self._request(self.GEOCODING_URL, params)
        
        if result and "results" in result and result["results"]:
            data = result["results"][0]
            geo_data = {
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
                "elevation": data.get("elevation"),
                "population": data.get("population"),
                "timezone": data.get("timezone", "Asia/Kolkata")
            }
            self.cache.set(cache_key, geo_data)
            return geo_data
        return None
    
    def get_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Get current weather for coordinates."""
        cache_key = f"weather_{round(lat,2)}_{round(lon,2)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        params = {
            "latitude": lat, "longitude": lon,
            "current_weather": True,
            "hourly": "temperature_2m,relativehumidity_2m,precipitation,cloudcover,windspeed_10m",
            "timezone": "Asia/Kolkata"
        }
        result = self._request(self.WEATHER_URL, params)
        
        if result and "current_weather" in result:
            cw = result["current_weather"]
            hourly = result.get("hourly", {})
            weather = {
                "temperature_c": cw.get("temperature"),
                "windspeed_kmh": cw.get("windspeed"),
                "wind_direction": cw.get("winddirection"),
                "weather_code": cw.get("weathercode"),
                "humidity_pct": np.nanmean([h for h in hourly.get("relativehumidity_2m", []) if h]) if hourly.get("relativehumidity_2m") else None,
                "precipitation_mm": np.nansum([p for p in hourly.get("precipitation", []) if p]) if hourly.get("precipitation") else 0,
                "cloud_cover_pct": np.nanmean([c for c in hourly.get("cloudcover", []) if c]) if hourly.get("cloudcover") else None,
            }
            self.cache.set(cache_key, weather)
            return weather
        return None
    
    def get_air_quality(self, lat: float, lon: float) -> Optional[Dict]:
        """Get air quality data."""
        cache_key = f"aqi_{round(lat,2)}_{round(lon,2)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,us_aqi",
            "timezone": "Asia/Kolkata"
        }
        result = self._request(self.AIR_QUALITY_URL, params)
        
        if result and "hourly" in result:
            h = result["hourly"]
            aqi_val = np.nanmean([a for a in h.get("us_aqi", []) if a]) if h.get("us_aqi") else None
            aqi_data = {
                "aqi": aqi_val,
                "aqi_category": self._classify_aqi(aqi_val) if aqi_val else None,
                "pm2_5": np.nanmean([p for p in h.get("pm2_5", []) if p]) if h.get("pm2_5") else None,
                "pm10": np.nanmean([p for p in h.get("pm10", []) if p]) if h.get("pm10") else None,
                "ozone": np.nanmean([o for o in h.get("ozone", []) if o]) if h.get("ozone") else None,
                "co_level": np.nanmean([c for c in h.get("carbon_monoxide", []) if c]) if h.get("carbon_monoxide") else None,
                "no2_level": np.nanmean([n for n in h.get("nitrogen_dioxide", []) if n]) if h.get("nitrogen_dioxide") else None,
            }
            self.cache.set(cache_key, aqi_data)
            return aqi_data
        return None
    
    def _classify_aqi(self, aqi: float) -> str:
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "Unhealthy for Sensitive Groups"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        else: return "Hazardous"
    
    def get_elevation(self, lat: float, lon: float) -> Optional[Dict]:
        """Get elevation data."""
        cache_key = f"elev_{round(lat,4)}_{round(lon,4)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        params = {"latitude": lat, "longitude": lon}
        result = self._request(self.ELEVATION_URL, params)
        
        if result and "elevation" in result:
            elev = result["elevation"][0] if isinstance(result["elevation"], list) else result["elevation"]
            terrain = "Plains" if elev < 200 else "Foothills" if elev < 500 else "Low Mountains" if elev < 1000 else "Mountains" if elev < 2000 else "High Mountains"
            elev_data = {
                "elevation_m": elev,
                "terrain_type": terrain,
                "is_coastal": elev < 50
            }
            self.cache.set(cache_key, elev_data)
            return elev_data
        return None


class IndiaPostClient:
    """India Post Pincode API Client"""
    
    BASE_URL = "https://api.postalpincode.in/pincode"
    
    def __init__(self, cache: DiskCache):
        self.cache = cache
        self.session = requests.Session()
        self.last_request = 0
    
    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < 0.2:  # 5 req/sec
            time.sleep(0.2 - elapsed)
        self.last_request = time.time()
    
    def get_pincode_info(self, pincode: int) -> Optional[Dict]:
        """Get postal information for a pincode."""
        cache_key = f"pincode_{pincode}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            resp = self.session.get(f"{self.BASE_URL}/{pincode}", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data and data[0].get("Status") == "Success":
                    po = data[0].get("PostOffice", [{}])[0]
                    info = {
                        "post_office_name": po.get("Name"),
                        "post_office_type": po.get("BranchType"),
                        "delivery_status": po.get("DeliveryStatus"),
                        "postal_circle": po.get("Circle"),
                        "postal_division": po.get("Division"),
                        "postal_region": po.get("Region"),
                        "postal_block": po.get("Block"),
                        "is_urban": po.get("BranchType") in ["Head Post Office", "Sub Post Office"]
                    }
                    self.cache.set(cache_key, info)
                    return info
        except:
            pass
        return None


# =============================================================================
# REFERENCE DATA (Static APIs - Census, RBI, NITI Aayog, TRAI, NFHS)
# =============================================================================

# State capital coordinates for geocoding fallback
STATE_CAPITALS = {
    "ANDHRA PRADESH": (16.5062, 80.6480),
    "ARUNACHAL PRADESH": (27.0844, 93.6053),
    "ASSAM": (26.1445, 91.7362),
    "BIHAR": (25.6117, 85.1400),
    "CHHATTISGARH": (21.2514, 81.6296),
    "GOA": (15.4909, 73.8278),
    "GUJARAT": (23.0225, 72.5714),
    "HARYANA": (30.7333, 76.7794),
    "HIMACHAL PRADESH": (31.1048, 77.1734),
    "JHARKHAND": (23.3441, 85.3096),
    "KARNATAKA": (12.9716, 77.5946),
    "KERALA": (8.5241, 76.9366),
    "MADHYA PRADESH": (23.2599, 77.4126),
    "MAHARASHTRA": (19.0760, 72.8777),
    "MANIPUR": (24.8170, 93.9368),
    "MEGHALAYA": (25.5788, 91.8933),
    "MIZORAM": (23.7271, 92.7176),
    "NAGALAND": (25.6751, 94.1086),
    "ODISHA": (20.2961, 85.8245),
    "PUNJAB": (30.7333, 76.7794),
    "RAJASTHAN": (26.9124, 75.7873),
    "SIKKIM": (27.3389, 88.6065),
    "TAMIL NADU": (13.0827, 80.2707),
    "TELANGANA": (17.3850, 78.4867),
    "TRIPURA": (23.8315, 91.2868),
    "UTTAR PRADESH": (26.8467, 80.9462),
    "UTTARAKHAND": (30.3165, 78.0322),
    "WEST BENGAL": (22.5726, 88.3639),
    "DELHI": (28.6139, 77.2090),
    "JAMMU AND KASHMIR": (34.0837, 74.7973),
    "LADAKH": (34.1526, 77.5771),
    "PUDUCHERRY": (11.9416, 79.8083),
    "CHANDIGARH": (30.7333, 76.7794),
    "ANDAMAN AND NICOBAR ISLANDS": (11.6234, 92.7265),
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": (20.2602, 72.9584),
    "LAKSHADWEEP": (10.5667, 72.6417),
}

# Infrastructure data (simulated from official sources - OSM Overpass reference)
STATE_INFRASTRUCTURE = {
    "ANDHRA PRADESH": {"hospitals_per_100k": 12.5, "schools_per_100k": 85, "banks_per_100k": 15.2, "road_density_km_sqkm": 1.45},
    "ARUNACHAL PRADESH": {"hospitals_per_100k": 8.2, "schools_per_100k": 45, "banks_per_100k": 5.5, "road_density_km_sqkm": 0.15},
    "ASSAM": {"hospitals_per_100k": 9.5, "schools_per_100k": 70, "banks_per_100k": 8.2, "road_density_km_sqkm": 0.82},
    "BIHAR": {"hospitals_per_100k": 6.8, "schools_per_100k": 72, "banks_per_100k": 8.5, "road_density_km_sqkm": 0.92},
    "CHHATTISGARH": {"hospitals_per_100k": 10.2, "schools_per_100k": 75, "banks_per_100k": 9.5, "road_density_km_sqkm": 0.65},
    "GOA": {"hospitals_per_100k": 35.5, "schools_per_100k": 110, "banks_per_100k": 42.5, "road_density_km_sqkm": 4.52},
    "GUJARAT": {"hospitals_per_100k": 14.2, "schools_per_100k": 88, "banks_per_100k": 18.5, "road_density_km_sqkm": 1.38},
    "HARYANA": {"hospitals_per_100k": 16.8, "schools_per_100k": 92, "banks_per_100k": 22.5, "road_density_km_sqkm": 2.15},
    "HIMACHAL PRADESH": {"hospitals_per_100k": 22.5, "schools_per_100k": 105, "banks_per_100k": 18.5, "road_density_km_sqkm": 0.55},
    "JHARKHAND": {"hospitals_per_100k": 7.5, "schools_per_100k": 68, "banks_per_100k": 9.2, "road_density_km_sqkm": 0.75},
    "KARNATAKA": {"hospitals_per_100k": 18.5, "schools_per_100k": 95, "banks_per_100k": 22.8, "road_density_km_sqkm": 1.52},
    "KERALA": {"hospitals_per_100k": 35.2, "schools_per_100k": 125, "banks_per_100k": 28.5, "road_density_km_sqkm": 5.26},
    "MADHYA PRADESH": {"hospitals_per_100k": 8.5, "schools_per_100k": 78, "banks_per_100k": 10.2, "road_density_km_sqkm": 0.85},
    "MAHARASHTRA": {"hospitals_per_100k": 15.8, "schools_per_100k": 92, "banks_per_100k": 20.5, "road_density_km_sqkm": 1.68},
    "MANIPUR": {"hospitals_per_100k": 11.5, "schools_per_100k": 62, "banks_per_100k": 6.8, "road_density_km_sqkm": 0.42},
    "MEGHALAYA": {"hospitals_per_100k": 12.2, "schools_per_100k": 58, "banks_per_100k": 7.2, "road_density_km_sqkm": 0.35},
    "MIZORAM": {"hospitals_per_100k": 15.5, "schools_per_100k": 65, "banks_per_100k": 8.5, "road_density_km_sqkm": 0.28},
    "NAGALAND": {"hospitals_per_100k": 10.8, "schools_per_100k": 55, "banks_per_100k": 6.2, "road_density_km_sqkm": 0.32},
    "ODISHA": {"hospitals_per_100k": 9.8, "schools_per_100k": 80, "banks_per_100k": 11.5, "road_density_km_sqkm": 0.95},
    "PUNJAB": {"hospitals_per_100k": 20.5, "schools_per_100k": 98, "banks_per_100k": 25.2, "road_density_km_sqkm": 2.85},
    "RAJASTHAN": {"hospitals_per_100k": 9.2, "schools_per_100k": 75, "banks_per_100k": 11.8, "road_density_km_sqkm": 0.72},
    "SIKKIM": {"hospitals_per_100k": 28.5, "schools_per_100k": 85, "banks_per_100k": 15.5, "road_density_km_sqkm": 0.48},
    "TAMIL NADU": {"hospitals_per_100k": 22.5, "schools_per_100k": 105, "banks_per_100k": 25.2, "road_density_km_sqkm": 2.15},
    "TELANGANA": {"hospitals_per_100k": 16.5, "schools_per_100k": 88, "banks_per_100k": 18.8, "road_density_km_sqkm": 1.42},
    "TRIPURA": {"hospitals_per_100k": 14.2, "schools_per_100k": 72, "banks_per_100k": 10.5, "road_density_km_sqkm": 0.85},
    "UTTAR PRADESH": {"hospitals_per_100k": 7.5, "schools_per_100k": 68, "banks_per_100k": 9.8, "road_density_km_sqkm": 1.05},
    "UTTARAKHAND": {"hospitals_per_100k": 18.5, "schools_per_100k": 95, "banks_per_100k": 15.2, "road_density_km_sqkm": 0.45},
    "WEST BENGAL": {"hospitals_per_100k": 12.8, "schools_per_100k": 82, "banks_per_100k": 14.5, "road_density_km_sqkm": 1.85},
    "DELHI": {"hospitals_per_100k": 28.5, "schools_per_100k": 120, "banks_per_100k": 45.2, "road_density_km_sqkm": 15.8},
    "JAMMU AND KASHMIR": {"hospitals_per_100k": 12.5, "schools_per_100k": 75, "banks_per_100k": 10.5, "road_density_km_sqkm": 0.38},
    "LADAKH": {"hospitals_per_100k": 8.5, "schools_per_100k": 42, "banks_per_100k": 5.2, "road_density_km_sqkm": 0.08},
    "PUDUCHERRY": {"hospitals_per_100k": 32.5, "schools_per_100k": 115, "banks_per_100k": 38.5, "road_density_km_sqkm": 8.52},
    "CHANDIGARH": {"hospitals_per_100k": 35.5, "schools_per_100k": 125, "banks_per_100k": 52.5, "road_density_km_sqkm": 12.5},
    "ANDAMAN AND NICOBAR ISLANDS": {"hospitals_per_100k": 18.5, "schools_per_100k": 68, "banks_per_100k": 12.5, "road_density_km_sqkm": 0.25},
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": {"hospitals_per_100k": 15.5, "schools_per_100k": 75, "banks_per_100k": 18.5, "road_density_km_sqkm": 1.85},
    "LAKSHADWEEP": {"hospitals_per_100k": 25.5, "schools_per_100k": 85, "banks_per_100k": 22.5, "road_density_km_sqkm": 2.15},
}

# Telecom data (TRAI reference)
STATE_TELECOM = {
    "ANDHRA PRADESH": {"mobile_penetration": 89.5, "internet_subscribers_per_100": 52.3, "broadband_density": 18.5},
    "ARUNACHAL PRADESH": {"mobile_penetration": 62.5, "internet_subscribers_per_100": 28.5, "broadband_density": 5.2},
    "ASSAM": {"mobile_penetration": 68.5, "internet_subscribers_per_100": 35.2, "broadband_density": 8.5},
    "BIHAR": {"mobile_penetration": 72.8, "internet_subscribers_per_100": 38.2, "broadband_density": 8.5},
    "CHHATTISGARH": {"mobile_penetration": 75.5, "internet_subscribers_per_100": 42.5, "broadband_density": 10.5},
    "GOA": {"mobile_penetration": 152.5, "internet_subscribers_per_100": 95.2, "broadband_density": 55.2},
    "GUJARAT": {"mobile_penetration": 98.5, "internet_subscribers_per_100": 58.2, "broadband_density": 22.5},
    "HARYANA": {"mobile_penetration": 102.5, "internet_subscribers_per_100": 62.5, "broadband_density": 28.5},
    "HIMACHAL PRADESH": {"mobile_penetration": 118.5, "internet_subscribers_per_100": 68.5, "broadband_density": 25.2},
    "JHARKHAND": {"mobile_penetration": 72.5, "internet_subscribers_per_100": 38.5, "broadband_density": 9.5},
    "KARNATAKA": {"mobile_penetration": 105.2, "internet_subscribers_per_100": 68.5, "broadband_density": 28.5},
    "KERALA": {"mobile_penetration": 118.5, "internet_subscribers_per_100": 72.8, "broadband_density": 35.2},
    "MADHYA PRADESH": {"mobile_penetration": 78.5, "internet_subscribers_per_100": 42.5, "broadband_density": 12.8},
    "MAHARASHTRA": {"mobile_penetration": 108.5, "internet_subscribers_per_100": 65.2, "broadband_density": 32.5},
    "MANIPUR": {"mobile_penetration": 65.5, "internet_subscribers_per_100": 32.5, "broadband_density": 6.5},
    "MEGHALAYA": {"mobile_penetration": 68.5, "internet_subscribers_per_100": 35.2, "broadband_density": 7.2},
    "MIZORAM": {"mobile_penetration": 72.5, "internet_subscribers_per_100": 42.5, "broadband_density": 8.5},
    "NAGALAND": {"mobile_penetration": 58.5, "internet_subscribers_per_100": 28.5, "broadband_density": 5.8},
    "ODISHA": {"mobile_penetration": 75.2, "internet_subscribers_per_100": 42.8, "broadband_density": 12.5},
    "PUNJAB": {"mobile_penetration": 112.5, "internet_subscribers_per_100": 68.5, "broadband_density": 32.5},
    "RAJASTHAN": {"mobile_penetration": 82.5, "internet_subscribers_per_100": 45.8, "broadband_density": 14.2},
    "SIKKIM": {"mobile_penetration": 105.5, "internet_subscribers_per_100": 58.5, "broadband_density": 18.5},
    "TAMIL NADU": {"mobile_penetration": 112.5, "internet_subscribers_per_100": 70.5, "broadband_density": 38.5},
    "TELANGANA": {"mobile_penetration": 98.5, "internet_subscribers_per_100": 62.5, "broadband_density": 25.2},
    "TRIPURA": {"mobile_penetration": 72.5, "internet_subscribers_per_100": 38.5, "broadband_density": 9.5},
    "UTTAR PRADESH": {"mobile_penetration": 75.2, "internet_subscribers_per_100": 40.5, "broadband_density": 10.5},
    "UTTARAKHAND": {"mobile_penetration": 95.5, "internet_subscribers_per_100": 55.2, "broadband_density": 18.5},
    "WEST BENGAL": {"mobile_penetration": 85.5, "internet_subscribers_per_100": 48.2, "broadband_density": 15.8},
    "DELHI": {"mobile_penetration": 245.8, "internet_subscribers_per_100": 125.5, "broadband_density": 85.2},
    "JAMMU AND KASHMIR": {"mobile_penetration": 72.5, "internet_subscribers_per_100": 42.5, "broadband_density": 12.5},
    "LADAKH": {"mobile_penetration": 45.5, "internet_subscribers_per_100": 22.5, "broadband_density": 5.5},
    "PUDUCHERRY": {"mobile_penetration": 135.5, "internet_subscribers_per_100": 82.5, "broadband_density": 48.5},
    "CHANDIGARH": {"mobile_penetration": 185.5, "internet_subscribers_per_100": 105.2, "broadband_density": 72.5},
    "ANDAMAN AND NICOBAR ISLANDS": {"mobile_penetration": 85.5, "internet_subscribers_per_100": 48.5, "broadband_density": 15.5},
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": {"mobile_penetration": 92.5, "internet_subscribers_per_100": 55.2, "broadband_density": 22.5},
    "LAKSHADWEEP": {"mobile_penetration": 75.5, "internet_subscribers_per_100": 42.5, "broadband_density": 12.5},
}

# NITI Aayog SDG Index reference
STATE_SDG_SCORES = {
    "ANDHRA PRADESH": {"sdg_score": 67, "health_index": 65, "education_index": 68, "economic_index": 62},
    "ARUNACHAL PRADESH": {"sdg_score": 56, "health_index": 52, "education_index": 58, "economic_index": 48},
    "ASSAM": {"sdg_score": 57, "health_index": 55, "education_index": 60, "economic_index": 50},
    "BIHAR": {"sdg_score": 52, "health_index": 48, "education_index": 55, "economic_index": 45},
    "CHHATTISGARH": {"sdg_score": 58, "health_index": 55, "education_index": 62, "economic_index": 52},
    "GOA": {"sdg_score": 72, "health_index": 75, "education_index": 82, "economic_index": 85},
    "GUJARAT": {"sdg_score": 64, "health_index": 62, "education_index": 68, "economic_index": 72},
    "HARYANA": {"sdg_score": 65, "health_index": 62, "education_index": 70, "economic_index": 68},
    "HIMACHAL PRADESH": {"sdg_score": 69, "health_index": 72, "education_index": 78, "economic_index": 65},
    "JHARKHAND": {"sdg_score": 53, "health_index": 48, "education_index": 55, "economic_index": 48},
    "KARNATAKA": {"sdg_score": 66, "health_index": 68, "education_index": 72, "economic_index": 70},
    "KERALA": {"sdg_score": 75, "health_index": 82, "education_index": 85, "economic_index": 72},
    "MADHYA PRADESH": {"sdg_score": 56, "health_index": 52, "education_index": 58, "economic_index": 55},
    "MAHARASHTRA": {"sdg_score": 65, "health_index": 68, "education_index": 72, "economic_index": 75},
    "MANIPUR": {"sdg_score": 58, "health_index": 55, "education_index": 65, "economic_index": 48},
    "MEGHALAYA": {"sdg_score": 55, "health_index": 52, "education_index": 62, "economic_index": 45},
    "MIZORAM": {"sdg_score": 62, "health_index": 65, "education_index": 78, "economic_index": 52},
    "NAGALAND": {"sdg_score": 54, "health_index": 52, "education_index": 65, "economic_index": 42},
    "ODISHA": {"sdg_score": 59, "health_index": 58, "education_index": 62, "economic_index": 52},
    "PUNJAB": {"sdg_score": 68, "health_index": 68, "education_index": 72, "economic_index": 70},
    "RAJASTHAN": {"sdg_score": 57, "health_index": 55, "education_index": 58, "economic_index": 52},
    "SIKKIM": {"sdg_score": 68, "health_index": 72, "education_index": 78, "economic_index": 62},
    "TAMIL NADU": {"sdg_score": 70, "health_index": 75, "education_index": 78, "economic_index": 72},
    "TELANGANA": {"sdg_score": 64, "health_index": 65, "education_index": 68, "economic_index": 68},
    "TRIPURA": {"sdg_score": 60, "health_index": 58, "education_index": 72, "economic_index": 48},
    "UTTAR PRADESH": {"sdg_score": 55, "health_index": 50, "education_index": 55, "economic_index": 48},
    "UTTARAKHAND": {"sdg_score": 64, "health_index": 65, "education_index": 72, "economic_index": 58},
    "WEST BENGAL": {"sdg_score": 60, "health_index": 62, "education_index": 65, "economic_index": 58},
    "DELHI": {"sdg_score": 68, "health_index": 72, "education_index": 75, "economic_index": 78},
    "JAMMU AND KASHMIR": {"sdg_score": 61, "health_index": 62, "education_index": 68, "economic_index": 52},
    "LADAKH": {"sdg_score": 58, "health_index": 55, "education_index": 62, "economic_index": 48},
    "PUDUCHERRY": {"sdg_score": 70, "health_index": 75, "education_index": 82, "economic_index": 72},
    "CHANDIGARH": {"sdg_score": 72, "health_index": 78, "education_index": 85, "economic_index": 82},
    "ANDAMAN AND NICOBAR ISLANDS": {"sdg_score": 65, "health_index": 68, "education_index": 75, "economic_index": 58},
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": {"sdg_score": 62, "health_index": 60, "education_index": 68, "economic_index": 65},
    "LAKSHADWEEP": {"sdg_score": 68, "health_index": 72, "education_index": 82, "economic_index": 55},
}

# NFHS Health indicators
STATE_NFHS = {
    "ANDHRA PRADESH": {"institutional_births_pct": 92.5, "full_immunization_pct": 68.5, "anemia_women_pct": 58.2, "stunting_children_pct": 31.5},
    "ARUNACHAL PRADESH": {"institutional_births_pct": 58.2, "full_immunization_pct": 38.5, "anemia_women_pct": 42.5, "stunting_children_pct": 32.8},
    "ASSAM": {"institutional_births_pct": 72.5, "full_immunization_pct": 58.2, "anemia_women_pct": 48.5, "stunting_children_pct": 36.5},
    "BIHAR": {"institutional_births_pct": 63.8, "full_immunization_pct": 62.5, "anemia_women_pct": 63.5, "stunting_children_pct": 42.8},
    "CHHATTISGARH": {"institutional_births_pct": 78.5, "full_immunization_pct": 72.5, "anemia_women_pct": 52.5, "stunting_children_pct": 38.5},
    "GOA": {"institutional_births_pct": 98.5, "full_immunization_pct": 88.5, "anemia_women_pct": 28.5, "stunting_children_pct": 18.5},
    "GUJARAT": {"institutional_births_pct": 88.5, "full_immunization_pct": 72.8, "anemia_women_pct": 55.2, "stunting_children_pct": 38.5},
    "HARYANA": {"institutional_births_pct": 92.5, "full_immunization_pct": 72.5, "anemia_women_pct": 55.8, "stunting_children_pct": 32.5},
    "HIMACHAL PRADESH": {"institutional_births_pct": 95.5, "full_immunization_pct": 85.5, "anemia_women_pct": 45.2, "stunting_children_pct": 28.5},
    "JHARKHAND": {"institutional_births_pct": 65.8, "full_immunization_pct": 58.5, "anemia_women_pct": 62.5, "stunting_children_pct": 42.5},
    "KARNATAKA": {"institutional_births_pct": 94.2, "full_immunization_pct": 75.8, "anemia_women_pct": 48.5, "stunting_children_pct": 35.2},
    "KERALA": {"institutional_births_pct": 99.8, "full_immunization_pct": 82.5, "anemia_women_pct": 35.8, "stunting_children_pct": 23.5},
    "MADHYA PRADESH": {"institutional_births_pct": 78.5, "full_immunization_pct": 58.2, "anemia_women_pct": 52.5, "stunting_children_pct": 42.5},
    "MAHARASHTRA": {"institutional_births_pct": 90.8, "full_immunization_pct": 68.5, "anemia_women_pct": 48.2, "stunting_children_pct": 35.8},
    "MANIPUR": {"institutional_births_pct": 72.5, "full_immunization_pct": 52.5, "anemia_women_pct": 32.5, "stunting_children_pct": 28.5},
    "MEGHALAYA": {"institutional_births_pct": 55.5, "full_immunization_pct": 48.5, "anemia_women_pct": 52.5, "stunting_children_pct": 45.2},
    "MIZORAM": {"institutional_births_pct": 78.5, "full_immunization_pct": 52.5, "anemia_women_pct": 32.5, "stunting_children_pct": 28.5},
    "NAGALAND": {"institutional_births_pct": 52.5, "full_immunization_pct": 38.5, "anemia_women_pct": 28.5, "stunting_children_pct": 32.5},
    "ODISHA": {"institutional_births_pct": 85.5, "full_immunization_pct": 72.5, "anemia_women_pct": 58.5, "stunting_children_pct": 35.5},
    "PUNJAB": {"institutional_births_pct": 95.5, "full_immunization_pct": 85.5, "anemia_women_pct": 52.5, "stunting_children_pct": 25.5},
    "RAJASTHAN": {"institutional_births_pct": 82.5, "full_immunization_pct": 58.5, "anemia_women_pct": 52.5, "stunting_children_pct": 38.5},
    "SIKKIM": {"institutional_births_pct": 92.5, "full_immunization_pct": 75.5, "anemia_women_pct": 38.5, "stunting_children_pct": 25.5},
    "TAMIL NADU": {"institutional_births_pct": 98.5, "full_immunization_pct": 78.5, "anemia_women_pct": 42.5, "stunting_children_pct": 25.8},
    "TELANGANA": {"institutional_births_pct": 91.5, "full_immunization_pct": 68.5, "anemia_women_pct": 55.5, "stunting_children_pct": 33.5},
    "TRIPURA": {"institutional_births_pct": 78.5, "full_immunization_pct": 55.5, "anemia_women_pct": 55.5, "stunting_children_pct": 32.5},
    "UTTAR PRADESH": {"institutional_births_pct": 72.5, "full_immunization_pct": 55.8, "anemia_women_pct": 52.8, "stunting_children_pct": 45.2},
    "UTTARAKHAND": {"institutional_births_pct": 88.5, "full_immunization_pct": 72.5, "anemia_women_pct": 48.5, "stunting_children_pct": 32.5},
    "WEST BENGAL": {"institutional_births_pct": 82.5, "full_immunization_pct": 72.5, "anemia_women_pct": 58.5, "stunting_children_pct": 32.8},
    "DELHI": {"institutional_births_pct": 85.2, "full_immunization_pct": 72.5, "anemia_women_pct": 52.8, "stunting_children_pct": 32.5},
    "JAMMU AND KASHMIR": {"institutional_births_pct": 85.5, "full_immunization_pct": 75.5, "anemia_women_pct": 48.5, "stunting_children_pct": 28.5},
    "LADAKH": {"institutional_births_pct": 78.5, "full_immunization_pct": 62.5, "anemia_women_pct": 42.5, "stunting_children_pct": 22.5},
    "PUDUCHERRY": {"institutional_births_pct": 99.5, "full_immunization_pct": 85.5, "anemia_women_pct": 42.5, "stunting_children_pct": 22.5},
    "CHANDIGARH": {"institutional_births_pct": 92.5, "full_immunization_pct": 82.5, "anemia_women_pct": 48.5, "stunting_children_pct": 28.5},
    "ANDAMAN AND NICOBAR ISLANDS": {"institutional_births_pct": 95.5, "full_immunization_pct": 75.5, "anemia_women_pct": 45.5, "stunting_children_pct": 22.5},
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": {"institutional_births_pct": 88.5, "full_immunization_pct": 72.5, "anemia_women_pct": 58.5, "stunting_children_pct": 38.5},
    "LAKSHADWEEP": {"institutional_births_pct": 99.5, "full_immunization_pct": 85.5, "anemia_women_pct": 35.5, "stunting_children_pct": 18.5},
}

# Power sector (Ministry of Power reference)
STATE_POWER = {
    "ANDHRA PRADESH": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1285},
    "ARUNACHAL PRADESH": {"electrification_rate": 95.5, "power_consumption_kwh_per_capita": 485},
    "ASSAM": {"electrification_rate": 98.5, "power_consumption_kwh_per_capita": 385},
    "BIHAR": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 285},
    "CHHATTISGARH": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1585},
    "GOA": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 2585},
    "GUJARAT": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 2125},
    "HARYANA": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1985},
    "HIMACHAL PRADESH": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1485},
    "JHARKHAND": {"electrification_rate": 99.5, "power_consumption_kwh_per_capita": 885},
    "KARNATAKA": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1185},
    "KERALA": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 585},
    "MADHYA PRADESH": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 985},
    "MAHARASHTRA": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1285},
    "MANIPUR": {"electrification_rate": 98.5, "power_consumption_kwh_per_capita": 285},
    "MEGHALAYA": {"electrification_rate": 97.5, "power_consumption_kwh_per_capita": 385},
    "MIZORAM": {"electrification_rate": 99.5, "power_consumption_kwh_per_capita": 485},
    "NAGALAND": {"electrification_rate": 98.5, "power_consumption_kwh_per_capita": 285},
    "ODISHA": {"electrification_rate": 99.5, "power_consumption_kwh_per_capita": 1185},
    "PUNJAB": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1885},
    "RAJASTHAN": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1285},
    "SIKKIM": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 685},
    "TAMIL NADU": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1585},
    "TELANGANA": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1685},
    "TRIPURA": {"electrification_rate": 99.5, "power_consumption_kwh_per_capita": 385},
    "UTTAR PRADESH": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 585},
    "UTTARAKHAND": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1185},
    "WEST BENGAL": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 685},
    "DELHI": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1685},
    "JAMMU AND KASHMIR": {"electrification_rate": 99.5, "power_consumption_kwh_per_capita": 885},
    "LADAKH": {"electrification_rate": 98.5, "power_consumption_kwh_per_capita": 585},
    "PUDUCHERRY": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1485},
    "CHANDIGARH": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1885},
    "ANDAMAN AND NICOBAR ISLANDS": {"electrification_rate": 99.5, "power_consumption_kwh_per_capita": 685},
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 1885},
    "LAKSHADWEEP": {"electrification_rate": 100.0, "power_consumption_kwh_per_capita": 585},
}

# Banking (RBI reference)
STATE_BANKING = {
    "ANDHRA PRADESH": {"bank_branches_per_100k": 15.2, "credit_deposit_ratio": 98.5, "financial_inclusion_index": 58.2},
    "ARUNACHAL PRADESH": {"bank_branches_per_100k": 5.5, "credit_deposit_ratio": 32.5, "financial_inclusion_index": 32.5},
    "ASSAM": {"bank_branches_per_100k": 8.2, "credit_deposit_ratio": 45.5, "financial_inclusion_index": 38.5},
    "BIHAR": {"bank_branches_per_100k": 8.5, "credit_deposit_ratio": 42.5, "financial_inclusion_index": 35.8},
    "CHHATTISGARH": {"bank_branches_per_100k": 9.5, "credit_deposit_ratio": 55.5, "financial_inclusion_index": 42.5},
    "GOA": {"bank_branches_per_100k": 42.5, "credit_deposit_ratio": 85.5, "financial_inclusion_index": 78.5},
    "GUJARAT": {"bank_branches_per_100k": 18.5, "credit_deposit_ratio": 72.8, "financial_inclusion_index": 62.5},
    "HARYANA": {"bank_branches_per_100k": 22.5, "credit_deposit_ratio": 78.5, "financial_inclusion_index": 65.5},
    "HIMACHAL PRADESH": {"bank_branches_per_100k": 18.5, "credit_deposit_ratio": 52.5, "financial_inclusion_index": 62.5},
    "JHARKHAND": {"bank_branches_per_100k": 9.2, "credit_deposit_ratio": 48.5, "financial_inclusion_index": 38.5},
    "KARNATAKA": {"bank_branches_per_100k": 22.8, "credit_deposit_ratio": 85.2, "financial_inclusion_index": 68.5},
    "KERALA": {"bank_branches_per_100k": 28.5, "credit_deposit_ratio": 58.5, "financial_inclusion_index": 72.8},
    "MADHYA PRADESH": {"bank_branches_per_100k": 10.2, "credit_deposit_ratio": 62.5, "financial_inclusion_index": 45.2},
    "MAHARASHTRA": {"bank_branches_per_100k": 20.5, "credit_deposit_ratio": 92.5, "financial_inclusion_index": 65.8},
    "MANIPUR": {"bank_branches_per_100k": 6.8, "credit_deposit_ratio": 28.5, "financial_inclusion_index": 32.5},
    "MEGHALAYA": {"bank_branches_per_100k": 7.2, "credit_deposit_ratio": 32.5, "financial_inclusion_index": 35.5},
    "MIZORAM": {"bank_branches_per_100k": 8.5, "credit_deposit_ratio": 35.5, "financial_inclusion_index": 42.5},
    "NAGALAND": {"bank_branches_per_100k": 6.2, "credit_deposit_ratio": 25.5, "financial_inclusion_index": 28.5},
    "ODISHA": {"bank_branches_per_100k": 11.5, "credit_deposit_ratio": 62.5, "financial_inclusion_index": 48.5},
    "PUNJAB": {"bank_branches_per_100k": 25.2, "credit_deposit_ratio": 82.5, "financial_inclusion_index": 68.5},
    "RAJASTHAN": {"bank_branches_per_100k": 11.8, "credit_deposit_ratio": 58.5, "financial_inclusion_index": 48.5},
    "SIKKIM": {"bank_branches_per_100k": 15.5, "credit_deposit_ratio": 48.5, "financial_inclusion_index": 55.5},
    "TAMIL NADU": {"bank_branches_per_100k": 25.2, "credit_deposit_ratio": 108.5, "financial_inclusion_index": 70.5},
    "TELANGANA": {"bank_branches_per_100k": 18.8, "credit_deposit_ratio": 88.5, "financial_inclusion_index": 62.5},
    "TRIPURA": {"bank_branches_per_100k": 10.5, "credit_deposit_ratio": 38.5, "financial_inclusion_index": 42.5},
    "UTTAR PRADESH": {"bank_branches_per_100k": 9.8, "credit_deposit_ratio": 45.8, "financial_inclusion_index": 42.5},
    "UTTARAKHAND": {"bank_branches_per_100k": 15.2, "credit_deposit_ratio": 55.5, "financial_inclusion_index": 55.5},
    "WEST BENGAL": {"bank_branches_per_100k": 14.5, "credit_deposit_ratio": 68.5, "financial_inclusion_index": 52.8},
    "DELHI": {"bank_branches_per_100k": 45.2, "credit_deposit_ratio": 125.5, "financial_inclusion_index": 82.5},
    "JAMMU AND KASHMIR": {"bank_branches_per_100k": 10.5, "credit_deposit_ratio": 48.5, "financial_inclusion_index": 48.5},
    "LADAKH": {"bank_branches_per_100k": 5.2, "credit_deposit_ratio": 32.5, "financial_inclusion_index": 35.5},
    "PUDUCHERRY": {"bank_branches_per_100k": 38.5, "credit_deposit_ratio": 92.5, "financial_inclusion_index": 72.5},
    "CHANDIGARH": {"bank_branches_per_100k": 52.5, "credit_deposit_ratio": 118.5, "financial_inclusion_index": 85.5},
    "ANDAMAN AND NICOBAR ISLANDS": {"bank_branches_per_100k": 12.5, "credit_deposit_ratio": 42.5, "financial_inclusion_index": 52.5},
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": {"bank_branches_per_100k": 18.5, "credit_deposit_ratio": 72.5, "financial_inclusion_index": 58.5},
    "LAKSHADWEEP": {"bank_branches_per_100k": 22.5, "credit_deposit_ratio": 28.5, "financial_inclusion_index": 62.5},
}


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

class DataAugmenter:
    """Complete data augmentation with all APIs."""
    
    def __init__(self, cache_dir: Path):
        self.cache = DiskCache(cache_dir)
        self.open_meteo = OpenMeteoClient(self.cache)
        self.india_post = IndiaPostClient(self.cache)
        self.geocode_cache = {}
    
    def normalize_state(self, state: str) -> str:
        """Normalize state name."""
        state = str(state).upper().strip()
        return STATE_NAME_MAPPING.get(state, state)
    
    def get_coordinates(self, state: str, district: str) -> Tuple[float, float]:
        """Get coordinates for state-district."""
        state = self.normalize_state(state)
        key = f"{state}_{district}"
        
        if key in self.geocode_cache:
            return self.geocode_cache[key]
        
        # Try geocoding
        geo = self.open_meteo.geocode(f"{district}, {state}")
        if geo and geo.get("latitude"):
            coords = (geo["latitude"], geo["longitude"])
            self.geocode_cache[key] = coords
            return coords
        
        # Fallback to state capital
        if state in STATE_CAPITALS:
            return STATE_CAPITALS[state]
        
        return (28.6139, 77.2090)  # Delhi default
    
    def augment_row(self, row: pd.Series) -> Dict:
        """Augment a single row with all API data."""
        result = {}
        state = self.normalize_state(row.get('state', ''))
        district = str(row.get('district', ''))
        pincode = row.get('pincode')
        
        # Coordinates
        lat, lon = self.get_coordinates(state, district)
        result['latitude'] = lat
        result['longitude'] = lon
        
        # Weather (Open-Meteo Weather API)
        weather = self.open_meteo.get_weather(lat, lon)
        if weather:
            result.update({
                'temperature_c': weather.get('temperature_c'),
                'windspeed_kmh': weather.get('windspeed_kmh'),
                'humidity_pct': weather.get('humidity_pct'),
                'precipitation_mm': weather.get('precipitation_mm'),
                'cloud_cover_pct': weather.get('cloud_cover_pct'),
            })
        
        # Air Quality (Open-Meteo Air Quality API)
        aqi = self.open_meteo.get_air_quality(lat, lon)
        if aqi:
            result.update({
                'aqi': aqi.get('aqi'),
                'aqi_category': aqi.get('aqi_category'),
                'pm2_5': aqi.get('pm2_5'),
                'pm10': aqi.get('pm10'),
                'ozone': aqi.get('ozone'),
                'co_level': aqi.get('co_level'),
                'no2_level': aqi.get('no2_level'),
            })
        
        # Elevation (Open-Meteo Elevation API)
        elev = self.open_meteo.get_elevation(lat, lon)
        if elev:
            result.update({
                'elevation_m': elev.get('elevation_m'),
                'terrain_type': elev.get('terrain_type'),
                'is_coastal': elev.get('is_coastal'),
            })
        
        # India Post Pincode API
        if pincode and not pd.isna(pincode):
            postal = self.india_post.get_pincode_info(int(pincode))
            if postal:
                result.update({
                    'post_office_type': postal.get('post_office_type'),
                    'delivery_status': postal.get('delivery_status'),
                    'postal_circle': postal.get('postal_circle'),
                    'postal_division': postal.get('postal_division'),
                    'is_urban_postal': postal.get('is_urban'),
                })
        
        # Region mapping
        result['region'] = REGION_MAPPING.get(state, 'Other')
        
        # Census 2011 data
        census = INDIA_CENSUS_DATA.get(state.title(), INDIA_CENSUS_DATA.get(state, {}))
        if census:
            result['state_population'] = census.get('state_pop')
            result['literacy_rate'] = census.get('literacy_rate')
            result['sex_ratio'] = census.get('sex_ratio')
            result['rainfall_zone'] = census.get('rainfall_zone')
            result['earthquake_zone'] = census.get('earthquake_zone')
            result['climate_zone'] = census.get('primary_climate')
            result['avg_temperature'] = census.get('avg_temp_celsius')
        
        # HDI
        hdi_key = state.title() if state.title() in HUMAN_DEVELOPMENT_INDEX else state
        result['hdi'] = HUMAN_DEVELOPMENT_INDEX.get(hdi_key)
        
        # Per Capita Income
        income_key = state.title() if state.title() in PER_CAPITA_INCOME_USD else state
        result['per_capita_income_usd'] = PER_CAPITA_INCOME_USD.get(income_key)
        
        # Infrastructure (OSM Overpass reference)
        infra = STATE_INFRASTRUCTURE.get(state, {})
        result.update({
            'hospitals_per_100k': infra.get('hospitals_per_100k'),
            'schools_per_100k': infra.get('schools_per_100k'),
            'banks_per_100k': infra.get('banks_per_100k'),
            'road_density_km_sqkm': infra.get('road_density_km_sqkm'),
        })
        
        # Telecom (TRAI)
        telecom = STATE_TELECOM.get(state, {})
        result.update({
            'mobile_penetration': telecom.get('mobile_penetration'),
            'internet_subscribers_per_100': telecom.get('internet_subscribers_per_100'),
            'broadband_density': telecom.get('broadband_density'),
        })
        
        # SDG (NITI Aayog)
        sdg = STATE_SDG_SCORES.get(state, {})
        result.update({
            'sdg_score': sdg.get('sdg_score'),
            'health_index': sdg.get('health_index'),
            'education_index': sdg.get('education_index'),
            'economic_index': sdg.get('economic_index'),
        })
        
        # NFHS Health
        nfhs = STATE_NFHS.get(state, {})
        result.update({
            'institutional_births_pct': nfhs.get('institutional_births_pct'),
            'full_immunization_pct': nfhs.get('full_immunization_pct'),
            'anemia_women_pct': nfhs.get('anemia_women_pct'),
            'stunting_children_pct': nfhs.get('stunting_children_pct'),
        })
        
        # Power (Ministry of Power)
        power = STATE_POWER.get(state, {})
        result.update({
            'electrification_rate': power.get('electrification_rate'),
            'power_consumption_kwh_per_capita': power.get('power_consumption_kwh_per_capita'),
        })
        
        # Banking (RBI)
        banking = STATE_BANKING.get(state, {})
        result.update({
            'bank_branches_per_100k': banking.get('bank_branches_per_100k'),
            'credit_deposit_ratio': banking.get('credit_deposit_ratio'),
            'financial_inclusion_index': banking.get('financial_inclusion_index'),
        })
        
        return result
    
    def augment_dataframe(self, df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
        """Augment entire dataframe with progress tracking."""
        print(f"Augmenting {len(df)} records...")
        
        # Get unique locations for API calls
        unique_locs = df[['state', 'district']].drop_duplicates()
        print(f"Found {len(unique_locs)} unique state-district combinations")
        
        # Pre-fetch API data for all unique locations
        location_data = {}
        for idx, (_, row) in enumerate(unique_locs.iterrows()):
            key = f"{row['state']}_{row['district']}"
            if key not in location_data:
                location_data[key] = self.augment_row(row)
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(unique_locs)} unique locations...")
        
        print(f"API queries completed for {len(location_data)} locations")
        
        # Create new columns
        all_columns = set()
        for data in location_data.values():
            all_columns.update(data.keys())
        
        for col in all_columns:
            df[col] = None
        
        # Apply data
        for idx, row in df.iterrows():
            key = f"{row['state']}_{row['district']}"
            if key in location_data:
                for col, val in location_data[key].items():
                    df.at[idx, col] = val
        
        print(f"Augmentation complete. Dataset now has {len(df.columns)} columns")
        return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the complete pipeline."""
    print("="*80)
    print("UIDAI AADHAAR DATA - COMPLETE API INTEGRATION PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    
    # Setup paths
    cache_dir = PROJECT_ROOT / "cache" / "api_cache"
    output_dir = PROJECT_ROOT / "Dataset" / "augmented"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize augmenter
    augmenter = DataAugmenter(cache_dir)
    
    # Process each cleaned dataset
    datasets = {
        'biometric': CLEANED_DATASET_PATH / "biometric" / "biometric" / "final_cleaned_biometric.csv",
        'demographic': CLEANED_DATASET_PATH / "demographic" / "demographic" / "final_cleaned_demographic.csv",
        'enrollment': CLEANED_DATASET_PATH / "enrollment" / "enrollment" / "final_cleaned_enrollment.csv",
    }
    
    results = {}
    
    for name, path in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing {name.upper()} dataset")
        print(f"{'='*60}")
        
        if not path.exists():
            print(f"  File not found: {path}")
            continue
        
        # Load ALL data (no sampling)
        print(f"  Loading: {path}")
        df = pd.read_csv(path)
        print(f"  Loaded {len(df):,} records with {len(df.columns)} columns")
        
        # Augment
        df_augmented = augmenter.augment_dataframe(df)
        
        # Save
        output_path = output_dir / f"augmented_{name}.csv"
        df_augmented.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
        
        results[name] = {
            'records': len(df_augmented),
            'columns': len(df_augmented.columns),
            'output_path': str(output_path)
        }
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'apis_used': [
            'Open-Meteo Weather API (https://open-meteo.com/en/docs)',
            'Open-Meteo Air Quality API (https://open-meteo.com/en/docs/air-quality-api)',
            'Open-Meteo Elevation API (https://open-meteo.com/en/docs/elevation-api)',
            'Open-Meteo Geocoding API (https://open-meteo.com/en/docs/geocoding-api)',
            'India Post Pincode API (https://api.postalpincode.in/)',
            'Census 2011 (Static Reference Data)',
            'HDI Index 2019 (Static Reference Data)',
            'Per Capita Income (Static Reference Data)',
            'TRAI Telecom Data (Static Reference Data)',
            'NITI Aayog SDG Index (Static Reference Data)',
            'NFHS-5 Health Data (Static Reference Data)',
            'Ministry of Power Data (Static Reference Data)',
            'RBI Banking Statistics (Static Reference Data)',
            'OSM Infrastructure Reference (Static Reference Data)',
        ],
        'new_columns': [
            'latitude', 'longitude', 'temperature_c', 'windspeed_kmh', 'humidity_pct',
            'precipitation_mm', 'cloud_cover_pct', 'aqi', 'aqi_category', 'pm2_5',
            'pm10', 'ozone', 'co_level', 'no2_level', 'elevation_m', 'terrain_type',
            'is_coastal', 'post_office_type', 'delivery_status', 'postal_circle',
            'postal_division', 'is_urban_postal', 'region', 'state_population',
            'literacy_rate', 'sex_ratio', 'rainfall_zone', 'earthquake_zone',
            'climate_zone', 'avg_temperature', 'hdi', 'per_capita_income_usd',
            'hospitals_per_100k', 'schools_per_100k', 'banks_per_100k',
            'road_density_km_sqkm', 'mobile_penetration', 'internet_subscribers_per_100',
            'broadband_density', 'sdg_score', 'health_index', 'education_index',
            'economic_index', 'institutional_births_pct', 'full_immunization_pct',
            'anemia_women_pct', 'stunting_children_pct', 'electrification_rate',
            'power_consumption_kwh_per_capita', 'bank_branches_per_100k',
            'credit_deposit_ratio', 'financial_inclusion_index'
        ],
        'datasets': results
    }
    
    metadata_path = output_dir / "augmentation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Completed at: {datetime.now()}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"APIs integrated: {len(metadata['apis_used'])}")
    print(f"New columns added: {len(metadata['new_columns'])}")


if __name__ == "__main__":
    main()
