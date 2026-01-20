# API Reference Guide

This document provides detailed information about all the APIs used in the data augmentation pipeline.

---

## Table of Contents

1. [Open-Meteo Weather API](#open-meteo-weather-api)
2. [Open-Meteo Historical Weather API](#open-meteo-historical-weather-api)
3. [Open-Meteo Air Quality API](#open-meteo-air-quality-api)
4. [Open-Meteo Elevation API](#open-meteo-elevation-api)
5. [Open-Meteo Geocoding API](#open-meteo-geocoding-api)
6. [Reference Data Sources](#reference-data-sources)

---

## Open-Meteo Weather API

### Overview
Provides current and forecast weather data for any location worldwide.

### Endpoint
```
https://api.open-meteo.com/v1/forecast
```

### Rate Limits
- **Free tier**: 10,000 calls/day
- **Rate**: ~100 requests/second

### Parameters Used

| Parameter | Type | Description |
|-----------|------|-------------|
| `latitude` | float | WGS84 latitude (-90 to 90) |
| `longitude` | float | WGS84 longitude (-180 to 180) |
| `current` | string | Comma-separated list of variables |
| `timezone` | string | Timezone for timestamps |

### Variables Retrieved

| Variable | Unit | Description |
|----------|------|-------------|
| `temperature_2m` | °C | Air temperature at 2m height |
| `relative_humidity_2m` | % | Relative humidity at 2m |
| `apparent_temperature` | °C | Feels-like temperature |
| `precipitation` | mm | Precipitation amount |
| `rain` | mm | Rain amount |
| `weather_code` | WMO code | Weather condition code |
| `cloud_cover` | % | Total cloud cover |
| `wind_speed_10m` | km/h | Wind speed at 10m |
| `wind_direction_10m` | ° | Wind direction at 10m |

### Example Request
```python
import requests

params = {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
    "timezone": "Asia/Kolkata"
}
response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
```

### Example Response
```json
{
    "latitude": 28.625,
    "longitude": 77.21875,
    "timezone": "Asia/Kolkata",
    "current": {
        "time": "2025-01-20T14:00",
        "temperature_2m": 22.5,
        "relative_humidity_2m": 45,
        "precipitation": 0.0,
        "wind_speed_10m": 12.3
    }
}
```

### WMO Weather Codes

| Code | Description |
|------|-------------|
| 0 | Clear sky |
| 1, 2, 3 | Mainly clear, partly cloudy, overcast |
| 45, 48 | Fog |
| 51, 53, 55 | Drizzle (light, moderate, dense) |
| 61, 63, 65 | Rain (slight, moderate, heavy) |
| 71, 73, 75 | Snow (slight, moderate, heavy) |
| 80, 81, 82 | Rain showers (slight, moderate, violent) |
| 95 | Thunderstorm |
| 96, 99 | Thunderstorm with hail |

---

## Open-Meteo Historical Weather API

### Overview
Access historical weather data from 1940 to present using ERA5 reanalysis.

### Endpoint
```
https://archive-api.open-meteo.com/v1/archive
```

### Rate Limits
- **Free tier**: 10,000 calls/day
- **Rate**: ~10 requests/second

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `latitude` | float | Yes | WGS84 latitude |
| `longitude` | float | Yes | WGS84 longitude |
| `start_date` | string | Yes | Start date (YYYY-MM-DD) |
| `end_date` | string | Yes | End date (YYYY-MM-DD) |
| `daily` | string | No | Daily variables to retrieve |
| `hourly` | string | No | Hourly variables to retrieve |

### Daily Variables

| Variable | Unit | Description |
|----------|------|-------------|
| `temperature_2m_max` | °C | Maximum daily temperature |
| `temperature_2m_min` | °C | Minimum daily temperature |
| `temperature_2m_mean` | °C | Mean daily temperature |
| `precipitation_sum` | mm | Daily precipitation sum |
| `rain_sum` | mm | Daily rain sum |
| `snowfall_sum` | cm | Daily snowfall sum |
| `windspeed_10m_max` | km/h | Maximum daily wind speed |
| `sunrise` | ISO8601 | Sunrise time |
| `sunset` | ISO8601 | Sunset time |

### Example Request
```python
params = {
    "latitude": 19.0760,
    "longitude": 72.8777,
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
    "timezone": "Asia/Kolkata"
}
response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
```

### Data Sources

| Model | Resolution | Coverage | Update |
|-------|------------|----------|--------|
| ECMWF IFS | 9 km | 2017-present | Daily with 2-day delay |
| ERA5 | 25 km | 1940-present | Daily with 5-day delay |
| ERA5-Land | 11 km | 1950-present | Daily with 5-day delay |

---

## Open-Meteo Air Quality API

### Overview
Air quality forecasts with pollutant concentrations and air quality indices.

### Endpoint
```
https://air-quality-api.open-meteo.com/v1/air-quality
```

### Rate Limits
- **Rate**: ~100 requests/second

### Variables Retrieved

| Variable | Unit | Description |
|----------|------|-------------|
| `european_aqi` | Index | European Air Quality Index (0-100+) |
| `us_aqi` | Index | US Air Quality Index (0-500) |
| `pm10` | μg/m³ | Particulate Matter < 10μm |
| `pm2_5` | μg/m³ | Particulate Matter < 2.5μm |
| `carbon_monoxide` | μg/m³ | CO concentration |
| `nitrogen_dioxide` | μg/m³ | NO2 concentration |
| `sulphur_dioxide` | μg/m³ | SO2 concentration |
| `ozone` | μg/m³ | O3 concentration |
| `dust` | μg/m³ | Dust particles |
| `uv_index` | Index | UV radiation index |

### AQI Ranges

**European AQI:**
| Range | Level |
|-------|-------|
| 0-20 | Good |
| 20-40 | Fair |
| 40-60 | Moderate |
| 60-80 | Poor |
| 80-100 | Very Poor |
| >100 | Extremely Poor |

**US AQI:**
| Range | Level |
|-------|-------|
| 0-50 | Good |
| 51-100 | Moderate |
| 101-150 | Unhealthy for Sensitive Groups |
| 151-200 | Unhealthy |
| 201-300 | Very Unhealthy |
| 301-500 | Hazardous |

### Example Request
```python
params = {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "current": "european_aqi,us_aqi,pm10,pm2_5,uv_index",
    "timezone": "Asia/Kolkata"
}
response = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=params)
```

---

## Open-Meteo Elevation API

### Overview
Get terrain elevation for any coordinates using Copernicus DEM 90m dataset.

### Endpoint
```
https://api.open-meteo.com/v1/elevation
```

### Rate Limits
- **Rate**: ~100 requests/second

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `latitude` | float/array | Yes | WGS84 latitude(s) |
| `longitude` | float/array | Yes | WGS84 longitude(s) |

### Response

| Field | Type | Description |
|-------|------|-------------|
| `elevation` | float[] | Elevation in meters above sea level |

### Example Request
```python
params = {
    "latitude": 27.1751,  # Jaipur
    "longitude": 78.0421
}
response = requests.get("https://api.open-meteo.com/v1/elevation", params=params)
# Response: {"elevation": [258.0]}
```

### Batch Requests
Request multiple locations at once:
```python
params = {
    "latitude": "28.6139,19.0760,12.9716",  # Delhi, Mumbai, Bangalore
    "longitude": "77.2090,72.8777,77.5946"
}
# Returns elevations for all three locations
```

---

## Open-Meteo Geocoding API

### Overview
Search for cities, districts, and get coordinates with population data.

### Endpoint
```
https://geocoding-api.open-meteo.com/v1/search
```

### Rate Limits
- **Rate**: ~10 requests/second

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Search term (location name) |
| `count` | int | No | Number of results (max 100) |
| `language` | string | No | Result language (en, hi, etc.) |
| `format` | string | No | Response format (json, protobuf) |
| `countryCode` | string | No | ISO country code filter (IN) |

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique location ID |
| `name` | string | Location name |
| `latitude` | float | WGS84 latitude |
| `longitude` | float | WGS84 longitude |
| `elevation` | float | Elevation in meters |
| `feature_code` | string | GeoNames feature code |
| `country_code` | string | ISO 2-letter country code |
| `population` | int | Population count |
| `admin1` | string | State/Province |
| `admin2` | string | District/County |
| `admin3` | string | Sub-district |
| `timezone` | string | Timezone identifier |

### Feature Codes

| Code | Description |
|------|-------------|
| PPLC | Capital of a political entity |
| PPLA | Seat of first-order admin division |
| PPLA2 | Seat of second-order admin division |
| PPL | Populated place |

### Example Request
```python
params = {
    "name": "Gorakhpur, Uttar Pradesh",
    "count": 5,
    "language": "en",
    "format": "json",
    "countryCode": "IN"
}
response = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=params)
```

### Example Response
```json
{
    "results": [{
        "id": 1270583,
        "name": "Gorakhpur",
        "latitude": 26.7606,
        "longitude": 83.3732,
        "elevation": 84.0,
        "feature_code": "PPLA2",
        "country_code": "IN",
        "country": "India",
        "population": 673446,
        "admin1": "Uttar Pradesh",
        "admin2": "Gorakhpur",
        "timezone": "Asia/Kolkata"
    }]
}
```

---

## Reference Data Sources

### Census of India 2011

Source: https://censusindia.gov.in/

**State-Level Data:**
- Population
- Area (sq km)
- Density (per sq km)
- Literacy rate
- Sex ratio (females per 1000 males)

### Human Development Index (HDI)

Source: UNDP India Human Development Report

**HDI Ranges:**
| Range | Level |
|-------|-------|
| 0.80+ | Very High |
| 0.70-0.79 | High |
| 0.55-0.69 | Medium |
| <0.55 | Low |

### Per Capita Income

Source: RBI/MOSPI State Economic Statistics

- Net State Domestic Product (NSDP) per capita
- Annual figures in INR

### Seismic Zones

Source: Bureau of Indian Standards (IS 1893)

| Zone | Intensity | Risk Level |
|------|-----------|------------|
| I | Very Low | Negligible |
| II | Low | Low |
| III | Moderate | Moderate |
| IV | Severe | High |
| V | Very Severe | Very High |

### Rainfall Zones

Based on India Meteorological Department (IMD) classifications:

| Zone | Annual Rainfall | Regions |
|------|-----------------|---------|
| Very Low | <500mm | Western Rajasthan, Ladakh |
| Low | 500-1000mm | Gujarat, Haryana, Punjab |
| Medium | 1000-2000mm | Most of India |
| High | 2000-3000mm | Western Ghats, Northeast |
| Very High | >3000mm | Cherrapunji, Kerala |

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad Request | Check parameters |
| 429 | Rate Limited | Wait and retry |
| 500 | Server Error | Retry with backoff |
| 503 | Service Unavailable | Retry later |

### Retry Strategy

```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=2, min=1, max=60),
    stop=stop_after_attempt(5)
)
def make_api_request(url, params):
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()
```

---

## Attribution Requirements

All Open-Meteo APIs are free for non-commercial use. Required attribution:

> Weather data by [Open-Meteo.com](https://open-meteo.com/)

For air quality data:
> Air quality data from CAMS (Copernicus Atmosphere Monitoring Service)

For elevation data:
> Elevation data from Copernicus DEM (ESA)

---

## Useful Links

- [Open-Meteo Documentation](https://open-meteo.com/en/docs)
- [Open-Meteo GitHub](https://github.com/open-meteo/open-meteo)
- [Census of India](https://censusindia.gov.in/)
- [India Pincode Directory](https://data.gov.in/catalog/all-india-pincode-directory)
