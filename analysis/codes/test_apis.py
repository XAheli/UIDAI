"""
Research and Analysis of Available Public APIs for Indian Data Augmentation
Based on Pincode, District, and State Information
"""

import requests
import json
from typing import Dict, List, Tuple
import time

print("=" * 80)
print("EVALUATING PUBLIC APIS FOR INDIAN PINCODE/LOCATION DATA")
print("=" * 80)

# Test data
test_pincode = "110001"  # New Delhi
test_district = "North Delhi"
test_state = "Delhi"

api_results = {}

# ============================================================================
# 1. INDIA POSTAL CODE API (Pincode to Location)
# ============================================================================
print("\n1. INDIA POSTAL CODE APIs")
print("-" * 80)

# API 1: Postalpincode.in API
print("\n1a. postalpincode.in API:")
print("     Details: Free, no authentication required")
print("     URL Pattern: https://api.postalpincode.in/postoffice/{pincode}")
try:
    response = requests.get(f"https://api.postalpincode.in/postoffice/{test_pincode}", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print("     ✓ Status: WORKING")
        print(f"     Sample Response: {json.dumps(data, indent=2)[:500]}")
        print("     Available Fields: State, District, PostOffices, Longitude, Latitude, etc.")
        api_results['postalpincode'] = 'WORKING'
    else:
        print(f"     ✗ Status: {response.status_code}")
        api_results['postalpincode'] = f'ERROR: {response.status_code}'
except Exception as e:
    print(f"     ✗ Error: {str(e)}")
    api_results['postalpincode'] = f'ERROR: {str(e)}'

# API 2: Getpostcode.com API
print("\n1b. Getpostcode.com API:")
print("     URL Pattern: https://www.getpostcode.com/postcode/{pincode}")
try:
    response = requests.get(f"https://www.getpostcode.com/postcode/{test_pincode}", timeout=5)
    if response.status_code == 200:
        print("     ✓ Status: WORKING")
        print(f"     Sample Response: {response.text[:300]}")
        api_results['getpostcode'] = 'WORKING'
    else:
        print(f"     ✗ Status: {response.status_code}")
        api_results['getpostcode'] = f'ERROR: {response.status_code}'
except Exception as e:
    print(f"     ✗ Error: {str(e)}")
    api_results['getpostcode'] = f'ERROR: {str(e)}'

# API 3: Pincode.in API
print("\n1c. Pincode.in API:")
print("     URL Pattern: https://pincode.in/api/pincode/{pincode}")
try:
    response = requests.get(f"https://pincode.in/api/pincode/{test_pincode}", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print("     ✓ Status: WORKING")
        print(f"     Response: {json.dumps(data, indent=2)[:500]}")
        api_results['pincode_in'] = 'WORKING'
    else:
        print(f"     ✗ Status: {response.status_code}")
        api_results['pincode_in'] = f'ERROR: {response.status_code}'
except Exception as e:
    print(f"     ✗ Error: {str(e)}")
    api_results['pincode_in'] = f'ERROR: {str(e)}'

# ============================================================================
# 2. WEATHER DATA APIS
# ============================================================================
print("\n\n2. WEATHER & CLIMATE DATA APIs")
print("-" * 80)

# API: Open-Meteo Weather API (Free, no key required)
print("\n2a. Open-Meteo Weather API:")
print("     Details: Free, no authentication, historical data available")
print("     URL Pattern: https://api.open-meteo.com/v1/forecast")
try:
    # Delhi coordinates: 28.7041, 77.1025
    params = {
        'latitude': 28.7041,
        'longitude': 77.1025,
        'current': 'temperature_2m,relative_humidity_2m,precipitation,weather_code'
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=5)
    if response.status_code == 200:
        data = response.json()
        print("     ✓ Status: WORKING")
        print(f"     Available Data: Current temp, humidity, precipitation, weather code")
        print(f"     Sample: {json.dumps(data.get('current'), indent=2)[:300]}")
        api_results['openmeteo'] = 'WORKING'
    else:
        print(f"     ✗ Status: {response.status_code}")
        api_results['openmeteo'] = f'ERROR: {response.status_code}'
except Exception as e:
    print(f"     ✗ Error: {str(e)}")
    api_results['openmeteo'] = f'ERROR: {str(e)}'

# ============================================================================
# 3. LOCATION DATA APIS
# ============================================================================
print("\n\n3. LOCATION & GEOMETRY APIs")
print("-" * 80)

# API: OpenCage Geocoding (has free tier)
print("\n3a. OpenCage Geocoding API:")
print("     Details: Free tier available (2,500 requests/day)")
print("     Note: Requires API key")

# API: OpenStreetMap Nominatim (Geocoding)
print("\n3b. OpenStreetMap Nominatim API:")
print("     Details: Free, no key required, rate limited")
print("     URL Pattern: https://nominatim.openstreetmap.org/search")
try:
    params = {
        'q': f'{test_pincode}, India',
        'format': 'json'
    }
    headers = {'User-Agent': 'MyApp/1.0'}
    response = requests.get("https://nominatim.openstreetmap.org/search", params=params, 
                           headers=headers, timeout=5)
    if response.status_code == 200:
        data = response.json()
        print("     ✓ Status: WORKING")
        print(f"     Available: Latitude, Longitude, Address, Bounding Box")
        if data:
            print(f"     Sample: {json.dumps(data[0], indent=2)[:400]}")
        api_results['nominatim'] = 'WORKING'
    else:
        print(f"     ✗ Status: {response.status_code}")
        api_results['nominatim'] = f'ERROR: {response.status_code}'
except Exception as e:
    print(f"     ✗ Error: {str(e)}")
    api_results['nominatim'] = f'ERROR: {str(e)}'

# ============================================================================
# 4. POPULATION & DEMOGRAPHIC DATA
# ============================================================================
print("\n\n4. POPULATION & DEMOGRAPHIC DATA")
print("-" * 80)

print("\n4a. India Census Data:")
print("     Note: 2011 Census data available via various sources")
print("     - Census India Official Portal (web scraping required)")
print("     - State/District level aggregates")
print("     - Population, literacy rate, sex ratio, etc.")

print("\n4b. India Data Portal (data.gov.in):")
print("     Details: Government open data portal")
print("     Available: Census data, health stats, education data")
print("     Format: CSV/JSON downloads (batch processing recommended)")

# ============================================================================
# 5. ECONOMIC & DEVELOPMENT DATA
# ============================================================================
print("\n\n5. ECONOMIC & DEVELOPMENT DATA")
print("-" * 80)

print("\n5a. World Bank Open Data API:")
print("     Details: Free API for development indicators")
print("     URL: https://api.worldbank.org/v2/")
try:
    response = requests.get("https://api.worldbank.org/v2/country/IND?format=json", timeout=5)
    if response.status_code == 200:
        print("     ✓ Status: WORKING")
        print("     Available: GDP, development indicators, macro data")
        api_results['worldbank'] = 'WORKING'
    else:
        print(f"     ✗ Status: {response.status_code}")
        api_results['worldbank'] = f'ERROR: {response.status_code}'
except Exception as e:
    print(f"     ✗ Error: {str(e)}")

print("\n5b. India GDP and Economic Data:")
print("     - RBI Data Portal (www.rbi.org.in)")
print("     - MOSPI (Ministry of Statistics) - bit.ly/MOSPI-data")
print("     - Per capita income, growth rates available at state level")

# ============================================================================
# 6. NATURAL HAZARDS & CLIMATE RISK
# ============================================================================
print("\n\n6. NATURAL HAZARDS & CLIMATE RISK DATA")
print("-" * 80)

print("\n6a. Natural Hazards APIs:")
print("     - USGS Earthquake Hazards Program (api.usgs.gov)")
print("     - Flood Risk: FloodBase API (requires account)")
print("     - Air Quality: WAQI.info Air Quality API (free with key)")

print("\n6b. Climate Risk Data:")
print("     - India Meteorological Department (IMD)")
print("     - Rainfall, temperature historical data")
print("     - Regional climate classifications")

# ============================================================================
# 7. INFRASTRUCTURE & SERVICES
# ============================================================================
print("\n\n7. INFRASTRUCTURE & SERVICES DATA")
print("-" * 80)

print("\n7a. OpenStreetMap API:")
print("     - Healthcare facilities, schools, markets")
print("     - Overpass API for complex queries")
print("     URL: https://overpass-api.de/api/")

print("\n7b. Google Places API (Freemium):")
print("     - Requires API key")
print("     - Nearby places, reviews, ratings")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("API AVAILABILITY SUMMARY")
print("=" * 80)

working_apis = [k for k, v in api_results.items() if v == 'WORKING']
failed_apis = [k for k, v in api_results.items() if v != 'WORKING']

print(f"\n✓ WORKING APIs ({len(working_apis)}):")
for api in working_apis:
    print(f"  - {api}")

if failed_apis:
    print(f"\n✗ FAILED APIs ({len(failed_apis)}):")
    for api in failed_apis:
        print(f"  - {api}: {api_results[api]}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR DATA AUGMENTATION")
print("=" * 80)

recommendations = """
TIER 1: PRIORITY APIs (Free, No Key, Reliable)
-------
1. postalpincode.in - Pincode to Location mapping
   └─ Returns: State, District, PostOffices, Coordinates
   └─ Rate: Unlimited (appears to be)
   
2. pincode.in - Additional pincode data
   └─ Returns: District, State, coordinates
   
3. Open-Meteo Weather - Climate data
   └─ Returns: Temperature, humidity, precipitation
   └─ Rate: Unlimited free
   
4. OpenStreetMap Nominatim - Reverse geocoding
   └─ Returns: Coordinates, addresses, geometry
   └─ Rate: Limited (1 request/sec) - need delays

TIER 2: SECONDARY APIs (Requires Setup or Key)
-------
1. World Bank Open Data - Development indicators
   └─ Free, no key required
   └─ State/National level data
   
2. WAQI Air Quality - Air pollution data
   └─ Free tier with key
   └─ Hourly updates available

TIER 3: BATCH/OFFLINE DATA (Data Download)
-------
1. India Census 2011 Data
   └─ Download from data.gov.in
   └─ State and District level statistics
   └─ Population, literacy, sex ratio, etc.
   
2. India Meteorological Department
   └─ Historical rainfall, temperature
   └─ Regional climate zone classifications

SUGGESTED AUGMENTATION PIPELINE:
-----------
1. Use postalpincode.in for each pincode
   └─ Get exact state/district/coordinates
   
2. Fetch weather data via Open-Meteo (using coordinates)
   └─ Add: Temperature, Humidity, Precipitation patterns
   
3. Add pre-computed census data
   └─ Add: Population density, literacy rate, sex ratio
   
4. Add economic indicators
   └─ Add: Per-capita income, development index
   
5. Add climate/geography data
   └─ Add: Rainfall zone, temperature zone, natural hazards

ESTIMATED NEW COLUMNS:
-----------
From Pincode APIs:
  - Exact latitude/longitude
  - Pin office name
  - Pin area name
  
From Weather APIs:
  - Average annual temperature
  - Annual rainfall
  - Humidity levels
  - Climate classification
  
From Census Data:
  - Population (district level)
  - Population density
  - Literacy rate
  - Sex ratio
  - Urban percentage
  
From Economic Data:
  - Per-capita income
  - Development index
  - Primary industry
  
From Climate/Natural:
  - Rainfall zone classification
  - Temperature zone classification
  - Earthquake risk level
  - Flood risk level
"""

print(recommendations)

print("\n" + "=" * 80)
