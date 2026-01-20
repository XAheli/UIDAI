# API Integration Catalog
**UIDAI Aadhaar Data Augmentation**  
**Version:** 1.0  
**Date:** January 20, 2026

---

## 📋 API Integration Status

| # | API Name | Status | Rate Limit | Auth Required | Priority |
|---|----------|--------|------------|---------------|----------|
| 1 | Open-Meteo Weather | ✅ Documented | 100/sec | ❌ No | HIGH |
| 2 | Open-Meteo Historical | ✅ Documented | 10/sec | ❌ No | HIGH |
| 3 | Open-Meteo Air Quality | ✅ Documented | 100/sec | ❌ No | HIGH |
| 4 | Open-Meteo Elevation | ✅ Documented | 100/sec | ❌ No | MEDIUM |
| 5 | Open-Meteo Geocoding | ✅ Documented | 10/sec | ❌ No | HIGH |
| 6 | OpenStreetMap Nominatim | 📝 Planned | 1/sec | ❌ No | HIGH |
| 7 | OSM Overpass API | 📝 Planned | Variable | ❌ No | MEDIUM |
| 8 | India Post Pincode | 📝 Planned | 5/sec | ❌ No | HIGH |
| 9 | World Bank Open Data | 📝 Planned | 500/min | ❌ No | MEDIUM |
| 10 | RBI Database | 📝 Planned | Unknown | ❌ No | MEDIUM |
| 11 | NITI Aayog Portal | 📝 Planned | Manual | ❌ No | LOW |
| 12 | Census API | 📝 Planned | Unknown | ❌ No | MEDIUM |
| 13 | TRAI Data Portal | 📝 Planned | Manual | ❌ No | LOW |
| 14 | NFHS Data | 📝 Planned | Manual | ❌ No | LOW |

---

## 🌐 TIER 1: Free, No Auth, High Rate Limit

### 1. Open-Meteo Weather API
**Status:** ✅ Ready to integrate  
**Documentation:** https://open-meteo.com/en/docs

#### Endpoints
```
GET https://api.open-meteo.com/v1/forecast
```

#### Parameters
```python
{
    "latitude": 28.6139,    # Required
    "longitude": 77.2090,   # Required
    "current_weather": true,
    "hourly": [
        "temperature_2m",
        "precipitation",
        "rain",
        "weathercode",
        "cloudcover",
        "windspeed_10m",
        "humidity"
    ],
    "timezone": "Asia/Kolkata"
}
```

#### Response Columns
- `temperature_c`: Current temperature
- `precipitation_mm`: Precipitation amount
- `humidity_pct`: Relative humidity
- `wind_speed_kmh`: Wind speed
- `cloud_cover_pct`: Cloud coverage
- `weather_code`: WMO weather code

#### Integration Plan
```python
def fetch_current_weather(latitude, longitude):
    """Fetch current weather for a location."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True,
        "timezone": "Asia/Kolkata"
    }
    # Rate limit: 100 requests/second
    # Multiprocessing: Use ThreadPoolExecutor
    return requests.get(url, params=params).json()
```

#### Data Augmentation Mapping
```
Dataset columns: pincode → (lat, lon) via geocoding
API call: fetch_current_weather(lat, lon)
New columns: 
  - current_temperature_c
  - current_precipitation_mm
  - current_humidity_pct
  - current_wind_speed_kmh
```

---

### 2. Open-Meteo Historical Weather API
**Status:** ✅ Ready to integrate  
**Documentation:** https://open-meteo.com/en/docs/historical-weather-api

#### Endpoint
```
GET https://archive-api.open-meteo.com/v1/archive
```

#### Parameters
```python
{
    "latitude": 28.6139,
    "longitude": 77.2090,
    "start_date": "2025-11-01",  # Dataset date
    "end_date": "2025-11-01",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "windspeed_10m_max"
    ],
    "timezone": "Asia/Kolkata"
}
```

#### Response Columns
- `temp_max_c`: Daily maximum temperature
- `temp_min_c`: Daily minimum temperature
- `temp_mean_c`: Daily mean temperature
- `precipitation_total_mm`: Total precipitation
- `rain_mm`: Rainfall
- `wind_max_kmh`: Maximum wind speed

#### Use Case
Match enrollment dates with historical weather to analyze:
- Did rainfall affect enrollment?
- Temperature extremes impact?
- Seasonal patterns?

---

### 3. Open-Meteo Air Quality API
**Status:** ✅ Ready to integrate  
**Documentation:** https://open-meteo.com/en/docs/air-quality-api

#### Endpoint
```
GET https://air-quality-api.open-meteo.com/v1/air-quality
```

#### Parameters
```python
{
    "latitude": 28.6139,
    "longitude": 77.2090,
    "hourly": [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "ozone",
        "us_aqi"
    ],
    "timezone": "Asia/Kolkata"
}
```

#### Response Columns
- `aqi`: US Air Quality Index
- `pm2_5`: PM2.5 particles
- `pm10`: PM10 particles
- `ozone`: Ozone concentration
- `co`: Carbon monoxide
- `no2`: Nitrogen dioxide

#### Use Case
Health indicators correlation:
- High AQI cities: Lower enrollment?
- Pollution hotspots identification
- Environmental justice analysis

---

### 4. Open-Meteo Elevation API
**Status:** ✅ Ready to integrate  
**Documentation:** https://open-meteo.com/en/docs/elevation-api

#### Endpoint
```
GET https://api.open-meteo.com/v1/elevation
```

#### Parameters
```python
{
    "latitude": [28.6139, 28.7041],  # Array of latitudes
    "longitude": [77.2090, 77.1025]  # Array of longitudes
}
```

#### Response
```json
{
    "elevation": [216.0, 225.0]  # Meters above sea level
}
```

#### Use Case
Terrain analysis:
- Mountain vs plain enrollment patterns
- Accessibility challenges
- Infrastructure difficulty correlation

---

### 5. Open-Meteo Geocoding API
**Status:** ✅ Ready to integrate  
**Documentation:** https://open-meteo.com/en/docs/geocoding-api

#### Endpoint
```
GET https://geocoding-api.open-meteo.com/v1/search
```

#### Parameters
```python
{
    "name": "Mumbai",
    "count": 10,
    "language": "en",
    "format": "json"
}
```

#### Response
```json
{
    "results": [
        {
            "id": 1275339,
            "name": "Mumbai",
            "latitude": 19.07283,
            "longitude": 72.88261,
            "elevation": 14.0,
            "timezone": "Asia/Kolkata",
            "population": 12691836,
            "country": "India",
            "admin1": "Maharashtra"
        }
    ]
}
```

#### Use Case
- Convert pincode/district to lat/lon
- Population validation
- Administrative hierarchy

---

## 🌐 TIER 2: Free, No Auth, Moderate Rate Limit

### 6. OpenStreetMap Nominatim
**Status:** 📝 To be integrated  
**Documentation:** https://nominatim.org/release-docs/latest/api/Search/

#### Endpoint
```
GET https://nominatim.openstreetmap.org/search
```

#### Parameters
```python
{
    "q": "110001, Delhi, India",  # Query string
    "format": "json",
    "limit": 1,
    "addressdetails": 1
}
```

#### Response
```json
{
    "lat": "28.6517178",
    "lon": "77.2219388",
    "display_name": "New Delhi, Delhi, 110001, India",
    "address": {
        "postcode": "110001",
        "city": "New Delhi",
        "state": "Delhi",
        "country": "India"
    },
    "boundingbox": ["28.6117178", "28.6917178", "77.1819388", "77.2619388"]
}
```

#### Rate Limit
- 1 request/second
- Must include User-Agent header

#### Integration Strategy
```python
import time
import requests

class NominatimClient:
    def __init__(self):
        self.last_request = 0
        self.user_agent = "UIDAI_Analysis/1.0"
    
    def geocode(self, pincode, district, state):
        # Rate limit: 1 req/sec
        time.sleep(max(0, 1 - (time.time() - self.last_request)))
        
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": f"{pincode}, {district}, {state}, India",
            "format": "json",
            "limit": 1
        }
        headers = {"User-Agent": self.user_agent}
        
        response = requests.get(url, params=params, headers=headers)
        self.last_request = time.time()
        return response.json()
```

#### New Columns
- `latitude`
- `longitude`
- `bounding_box`
- `osm_place_id`

---

### 7. OSM Overpass API
**Status:** 📝 To be integrated  
**Documentation:** https://wiki.openstreetmap.org/wiki/Overpass_API

#### Endpoint
```
POST https://overpass-api.de/api/interpreter
```

#### Query Example (Amenities Count)
```python
query = """
[out:json];
area["name"="Mumbai"]->.a;
(
  node["amenity"="hospital"](area.a);
  node["amenity"="school"](area.a);
  node["amenity"="bank"](area.a);
);
out count;
"""
```

#### Use Cases
- Count hospitals per district
- Count schools per district
- Count banks per district
- Road network density
- Public transport availability

#### New Columns
- `hospitals_count`
- `schools_count`
- `banks_count`
- `atm_count`
- `post_offices_count`
- `police_stations_count`
- `roads_km`

#### Rate Limit
- Variable, depends on query complexity
- Use batch queries per district
- Cache results

---

### 8. India Post Pincode API
**Status:** 📝 To be integrated  
**Documentation:** https://api.postalpincode.in/

#### Endpoint
```
GET https://api.postalpincode.in/pincode/{pincode}
```

#### Example
```
GET https://api.postalpincode.in/pincode/110001
```

#### Response
```json
[
    {
        "Status": "Success",
        "PostOffice": [
            {
                "Name": "Connaught Place",
                "Description": null,
                "BranchType": "Sub Post Office",
                "DeliveryStatus": "Delivery",
                "Circle": "Delhi",
                "District": "Central Delhi",
                "Division": "New Delhi Central",
                "Region": "Delhi",
                "Block": "New Delhi",
                "State": "Delhi",
                "Country": "India",
                "Pincode": "110001"
            }
        ]
    }
]
```

#### New Columns
- `post_office_name`
- `post_office_type` (Sub Post Office, Head Post Office)
- `delivery_status`
- `circle`
- `division`
- `block`
- `pincode_validated`

#### Use Case
- Validate pincodes
- Urban vs rural classification (delivery status)
- Administrative hierarchy verification

---

## 🌐 TIER 3: Free, No Auth, Manual/Low Rate Limit

### 9. World Bank Open Data API
**Status:** 📝 To be integrated  
**Documentation:** https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

#### Endpoint
```
GET https://api.worldbank.org/v2/country/IND/indicator/{indicator}
```

#### Indicators (India-level, can map to states)
- `NY.GDP.PCAP.CD`: GDP per capita
- `SI.POV.GINI`: Gini index
- `SP.POP.TOTL`: Total population
- `SL.UEM.TOTL.ZS`: Unemployment rate
- `SE.ADT.LITR.ZS`: Adult literacy rate

#### Example
```
GET https://api.worldbank.org/v2/country/IND/indicator/NY.GDP.PCAP.CD?format=json&date=2020:2025
```

#### Use Case
- National economic indicators
- Time series of economic growth
- Compare with enrollment trends

---

### 10. Reserve Bank of India (RBI)
**Status:** 📝 To be integrated  
**Documentation:** https://rbi.org.in/Scripts/Statistics.aspx

#### Data Available (Manual Download)
- State-wise banking statistics
- Credit-deposit ratio
- Financial inclusion metrics
- Branch network data

#### Integration Method
- Download CSV files manually
- Parse and merge with dataset
- Update quarterly/annually

#### New Columns
- `bank_branches_per_100k`
- `credit_deposit_ratio`
- `financial_inclusion_index`

---

### 11. NITI Aayog Data Portal
**Status:** 📝 To be integrated  
**Documentation:** https://niti.gov.in/

#### Data Available
- SDG India Index (state-wise scores)
- Aspirational Districts Programme data
- Development indicators

#### Integration Method
- Manual download
- Excel/CSV parsing
- Annual updates

#### New Columns
- `sdg_composite_score`
- `sdg_health_score`
- `sdg_education_score`
- `aspirational_district` (boolean)

---

### 12. Census API (2011)
**Status:** 📝 To be integrated  
**Documentation:** https://censusindia.gov.in/

#### Data Available
- District-level demographics
- Tehsil/Taluk level data
- Urban/rural split

#### Integration Method
- Download district profiles
- Parse PDF/Excel files
- Merge with dataset

#### New Columns
- `urban_population_pct`
- `rural_population_pct`
- `working_population_pct`
- `sc_population_pct`
- `st_population_pct`
- `child_population_0_6`

---

### 13. TRAI (Telecom Regulatory Authority of India)
**Status:** 📝 To be integrated  
**Documentation:** https://www.trai.gov.in/

#### Data Available
- State-wise telecom subscribers
- Internet penetration
- Broadband subscribers

#### Integration Method
- Download quarterly reports
- Excel parsing
- State-level aggregation

#### New Columns
- `mobile_subscribers_per_100`
- `internet_subscribers_per_100`
- `broadband_density`
- `telecom_coverage_pct`

---

### 14. NFHS (National Family Health Survey)
**Status:** 📝 To be integrated  
**Documentation:** http://rchiips.org/nfhs/

#### Data Available
- District-level health indicators
- Institutional births
- Vaccination coverage
- Nutrition indicators

#### Integration Method
- Download district fact sheets
- PDF/Excel parsing
- Merge on district names

#### New Columns
- `institutional_births_pct`
- `full_immunization_pct`
- `anemia_women_pct`
- `stunting_children_pct`
- `underweight_children_pct`

---

## 🔧 Implementation Guidelines

### General API Client Structure
```python
from typing import Dict, Any, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

class BaseAPIClient:
    """Base class for all API clients."""
    
    def __init__(self, rate_limit: float, max_workers: int = 10):
        self.rate_limit = rate_limit  # Requests per second
        self.max_workers = max_workers
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = requests.Session()
        self.last_request_time = 0
        
    def _rate_limit_wait(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        wait_time = (1 / self.rate_limit) - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=10))
    def _make_request(self, url: str, params: Dict[str, Any]) -> Dict:
        """Make HTTP request with retry logic."""
        self._rate_limit_wait()
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def batch_fetch(self, requests_list: list) -> list:
        """Fetch data in parallel with rate limiting."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._fetch_one, requests_list))
        return results
```

### Caching Strategy
```python
import pickle
from pathlib import Path
from functools import wraps
import hashlib

class PersistentCache:
    """Disk-based cache for API responses."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve from cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, key: str, value: Any):
        """Save to cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)

def cached_api_call(cache_dir: Path):
    """Decorator for caching API calls."""
    cache = PersistentCache(cache_dir)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = cache._get_key(*args, **kwargs)
            result = cache.get(key)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key, result)
            return result
        return wrapper
    return decorator
```

### Multiprocessing Data Augmentation
```python
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm

def augment_chunk(chunk: pd.DataFrame, api_client: BaseAPIClient) -> pd.DataFrame:
    """Augment a chunk of data with API calls."""
    results = []
    for idx, row in chunk.iterrows():
        api_data = api_client.fetch(
            latitude=row['latitude'],
            longitude=row['longitude'],
            date=row['date']
        )
        results.append(api_data)
    
    # Merge API data with chunk
    chunk_augmented = chunk.copy()
    for col, values in zip(api_data.keys(), zip(*results)):
        chunk_augmented[col] = values
    
    return chunk_augmented

def parallel_augment(df: pd.DataFrame, api_client: BaseAPIClient, 
                    chunk_size: int = 10000, workers: int = 12) -> pd.DataFrame:
    """Augment dataframe using multiprocessing."""
    
    # Split into chunks
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(augment_chunk, chunks, [api_client]*len(chunks)),
            total=len(chunks),
            desc="Augmenting data"
        ))
    
    # Combine results
    return pd.concat(results, ignore_index=True)
```

---

## 📊 Data Augmentation Pipeline

### Step 1: Geocoding (If needed)
```python
# Convert pincode/district to lat/lon
geocoding_client = OpenMeteoGeocodingClient()
df['latitude'], df['longitude'] = geocoding_client.batch_geocode(
    df['pincode'], df['district'], df['state']
)
```

### Step 2: Weather Data
```python
# Fetch historical weather for each date
weather_client = OpenMeteoHistoricalClient(rate_limit=10)
weather_data = parallel_augment(df, weather_client)
```

### Step 3: Air Quality
```python
# Fetch air quality data
air_quality_client = OpenMeteoAirQualityClient(rate_limit=100)
aqi_data = parallel_augment(df, air_quality_client)
```

### Step 4: Infrastructure (OSM)
```python
# Fetch infrastructure counts per district (batch by district)
osm_client = OverpassClient(rate_limit=1)
infrastructure_data = osm_client.fetch_district_amenities(df['district'].unique())
df = df.merge(infrastructure_data, on='district')
```

### Step 5: Manual Data Integration
```python
# Load manually downloaded datasets
census_data = pd.read_csv('external/census_2011_detailed.csv')
niti_data = pd.read_excel('external/sdg_india_index.xlsx')
trai_data = pd.read_excel('external/trai_telecom_2025.xlsx')

# Merge with main dataset
df = df.merge(census_data, on=['state', 'district'], how='left')
df = df.merge(niti_data, on='state', how='left')
df = df.merge(trai_data, on='state', how='left')
```

---

## 🎯 Expected Outcome

### Final Dataset Schema (50+ columns)

#### Original Columns (7-8)
- date, state, district, pincode
- age_group_1, age_group_2, age_group_3

#### Augmented Columns (25+)
- **Census:** population, literacy, sex_ratio, HDI, etc.
- **Geographic:** region, zone, climate, earthquake_zone, etc.
- **Economic:** per_capita_income, gdp_growth, etc.

#### New API Columns (20-25)
- **Weather:** temp_max, temp_min, precipitation, humidity
- **Air Quality:** aqi, pm2.5, pm10, ozone
- **Geospatial:** latitude, longitude, elevation
- **Infrastructure:** hospitals_count, schools_count, banks_count
- **Connectivity:** mobile_penetration, internet_subscribers
- **Health:** institutional_births, immunization_rate
- **Development:** sdg_score, financial_inclusion_index

**Total:** ~55-60 columns per record

---

## ⏱️ Estimated Processing Time

### With Multiprocessing (12 cores)

| Task | Records | API Rate | Time Estimate |
|------|---------|----------|---------------|
| Geocoding | 6.1M unique pincodes | 10/sec | ~17 hours |
| Historical Weather | 6.1M | 10/sec | ~17 hours |
| Air Quality | 6.1M | 100/sec | ~1.7 hours |
| Elevation | 6.1M | 100/sec | ~1.7 hours |
| OSM Amenities | ~960 districts | 1/sec | ~16 minutes |
| Manual Data | - | - | ~1 hour |

**Total:** ~40-45 hours for complete augmentation

### Optimization Strategies
1. **Pincode-level caching:** Reuse data for same pincode
2. **District-level aggregation:** For OSM data
3. **Batch requests:** Where API supports
4. **Parallel API calls:** Different APIs simultaneously
5. **Resume capability:** Save progress, resume on failure

**Optimized Time:** ~15-20 hours

---

## 📝 Next Steps

1. **Review API catalog** - Approve APIs to integrate
2. **Priority ranking** - Which APIs first?
3. **Create API clients** - Implement one by one
4. **Test on sample** - 10K records first
5. **Full augmentation** - Run on complete dataset
6. **Validation** - Verify data quality
7. **Analysis** - Begin comprehensive analysis

---

**Document Status:** Draft for Review  
**Last Updated:** January 20, 2026
