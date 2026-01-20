#!/usr/bin/env python3
"""
Comprehensive Data Augmentation Pipeline
=========================================

This script augments Aadhaar datasets with additional information from multiple APIs:

APIs Used (All Free, No API Key Required):
1. Open-Meteo Weather API - Current weather and forecasts
2. Open-Meteo Historical Weather API - Historical climate data
3. Open-Meteo Air Quality API - Air quality indices and pollutants
4. Open-Meteo Elevation API - Terrain elevation data
5. Open-Meteo Geocoding API - Location details and coordinates
6. Reference Census Data - Population, literacy, HDI, etc.

Features:
- Multiprocessing for parallel execution
- Retry strategies with exponential backoff
- Progress tracking with tqdm
- Comprehensive logging
- Caching to avoid redundant API calls
- Graceful error handling

Usage:
    python data_augmenter.py --input /path/to/cleaned/data.csv --output /path/to/augmented/
    python data_augmenter.py --all  # Process all cleaned datasets
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from multiprocessing import Manager, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import (
        API_ENDPOINTS, API_RATE_LIMITS, RETRY_CONFIG,
        CLEANED_DATASET_PATH, AUGMENTED_DATASET_PATH,
        INDIA_CENSUS_DATA, PINCODE_REGION_MAP, CPU_COUNT,
        PROJECT_ROOT
    )
    from logger import setup_logger, get_timestamped_log_file, ProgressLogger
except ImportError:
    from .config import (
        API_ENDPOINTS, API_RATE_LIMITS, RETRY_CONFIG,
        CLEANED_DATASET_PATH, AUGMENTED_DATASET_PATH,
        INDIA_CENSUS_DATA, PINCODE_REGION_MAP, CPU_COUNT,
        PROJECT_ROOT
    )
    from .logger import setup_logger, get_timestamped_log_file, ProgressLogger

import logging

# Set up logging
log_file = get_timestamped_log_file("data_augmentation")
logger = setup_logger(__name__, log_file)


# =============================================================================
# CACHING SYSTEM
# =============================================================================

CACHE_DIR = PROJECT_ROOT / "augmentation_cache"
CACHE_DIR.mkdir(exist_ok=True)


class PersistentCache:
    """
    Persistent cache for API responses using pickle files.
    Thread-safe with file locking.
    """
    
    def __init__(self, cache_name: str, max_size: int = 100000):
        self.cache_file = CACHE_DIR / f"{cache_name}_cache.pkl"
        self.max_size = max_size
        self._cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk if exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                logger.info(f"Loaded {len(self._cache)} cached entries from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}
    
    def save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
            logger.info(f"Saved {len(self._cache)} entries to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        if len(self._cache) >= self.max_size:
            # Remove oldest 10% of entries
            keys_to_remove = list(self._cache.keys())[:int(self.max_size * 0.1)]
            for k in keys_to_remove:
                del self._cache[k]
        self._cache[key] = value
    
    def __contains__(self, key: str) -> bool:
        return key in self._cache


# Initialize caches
state_cache = PersistentCache("state_data")
pincode_cache = PersistentCache("pincode_data")
geocoding_cache = PersistentCache("geocoding")
elevation_cache = PersistentCache("elevation")


# =============================================================================
# API CLIENTS WITH RETRY LOGIC
# =============================================================================

@retry(
    stop=stop_after_attempt(RETRY_CONFIG['max_attempts']),
    wait=wait_exponential(
        multiplier=RETRY_CONFIG['wait_multiplier'],
        min=RETRY_CONFIG['wait_min'],
        max=RETRY_CONFIG['wait_max']
    ),
    retry=retry_if_exception_type((requests.RequestException, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def make_api_request(url: str, params: Dict[str, Any], timeout: int = 30) -> Dict:
    """
    Make an API request with retry logic.
    
    Args:
        url: API endpoint URL
        params: Query parameters
        timeout: Request timeout in seconds
    
    Returns:
        JSON response as dictionary
    """
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


class OpenMeteoClient:
    """
    Client for Open-Meteo APIs with rate limiting and caching.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.last_request_time = {}
    
    def _rate_limit(self, api_name: str):
        """Apply rate limiting for an API."""
        rate = API_RATE_LIMITS.get(api_name, 10)
        min_interval = 1.0 / rate
        
        if api_name in self.last_request_time:
            elapsed = time.time() - self.last_request_time[api_name]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        
        self.last_request_time[api_name] = time.time()
    
    def get_geocoding(self, location: str, country_code: str = "IN") -> Optional[Dict]:
        """
        Get location coordinates and details using geocoding API.
        
        Args:
            location: City or district name
            country_code: ISO country code
        
        Returns:
            Location details including coordinates
        """
        cache_key = f"geo_{location}_{country_code}"
        if cache_key in geocoding_cache:
            return geocoding_cache.get(cache_key)
        
        try:
            self._rate_limit("geocoding")
            params = {
                "name": location,
                "count": 5,
                "language": "en",
                "format": "json",
                "countryCode": country_code
            }
            response = make_api_request(API_ENDPOINTS["geocoding"], params)
            
            if response.get("results"):
                result = response["results"][0]
                geocoding_cache.set(cache_key, result)
                return result
        except Exception as e:
            logger.debug(f"Geocoding failed for {location}: {e}")
        
        return None
    
    def get_elevation(self, latitude: float, longitude: float) -> Optional[float]:
        """
        Get terrain elevation for coordinates.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
        
        Returns:
            Elevation in meters
        """
        cache_key = f"elev_{latitude:.4f}_{longitude:.4f}"
        if cache_key in elevation_cache:
            return elevation_cache.get(cache_key)
        
        try:
            self._rate_limit("elevation")
            params = {"latitude": latitude, "longitude": longitude}
            response = make_api_request(API_ENDPOINTS["elevation"], params)
            
            if "elevation" in response:
                elevation = response["elevation"][0]
                elevation_cache.set(cache_key, elevation)
                return elevation
        except Exception as e:
            logger.debug(f"Elevation lookup failed: {e}")
        
        return None
    
    def get_current_weather(self, latitude: float, longitude: float) -> Optional[Dict]:
        """
        Get current weather conditions.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
        
        Returns:
            Current weather data
        """
        try:
            self._rate_limit("weather_forecast")
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                          "precipitation,rain,weather_code,cloud_cover,wind_speed_10m,"
                          "wind_direction_10m",
                "timezone": "Asia/Kolkata"
            }
            response = make_api_request(API_ENDPOINTS["weather_forecast"], params)
            return response.get("current")
        except Exception as e:
            logger.debug(f"Weather lookup failed: {e}")
        
        return None
    
    def get_air_quality(self, latitude: float, longitude: float) -> Optional[Dict]:
        """
        Get current air quality data.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
        
        Returns:
            Air quality data including AQI
        """
        try:
            self._rate_limit("air_quality")
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "european_aqi,us_aqi,pm10,pm2_5,carbon_monoxide,"
                          "nitrogen_dioxide,sulphur_dioxide,ozone,dust,uv_index",
                "timezone": "Asia/Kolkata"
            }
            response = make_api_request(API_ENDPOINTS["air_quality"], params)
            return response.get("current")
        except Exception as e:
            logger.debug(f"Air quality lookup failed: {e}")
        
        return None
    
    def get_historical_climate(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str
    ) -> Optional[Dict]:
        """
        Get historical climate averages.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Historical climate data
        """
        try:
            self._rate_limit("historical_weather")
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                        "precipitation_sum,rain_sum,windspeed_10m_max",
                "timezone": "Asia/Kolkata"
            }
            response = make_api_request(API_ENDPOINTS["historical_weather"], params)
            return response.get("daily")
        except Exception as e:
            logger.debug(f"Historical weather lookup failed: {e}")
        
        return None


# =============================================================================
# DATA AUGMENTATION ENGINE
# =============================================================================

@dataclass
class AugmentedRecord:
    """
    Data class for augmented record with all additional fields.
    """
    # Census data (state-level)
    population_2011: int = 0
    area_sq_km: int = 0
    density_per_sq_km: int = 0
    literacy_rate: float = 0.0
    sex_ratio: int = 0
    capital: str = ""
    region: str = ""
    zone: str = ""
    rainfall_zone: str = ""
    earthquake_zone: str = ""
    climate_type: str = ""
    per_capita_income_inr: int = 0
    hdi: float = 0.0
    major_language: str = ""
    coastal: bool = False
    
    # Geographic data
    latitude: float = 0.0
    longitude: float = 0.0
    elevation_m: float = 0.0
    
    # Current weather
    current_temp_c: float = 0.0
    current_humidity: float = 0.0
    current_precipitation_mm: float = 0.0
    current_wind_speed_kmh: float = 0.0
    
    # Air quality
    aqi_european: int = 0
    aqi_us: int = 0
    pm2_5: float = 0.0
    pm10: float = 0.0
    uv_index: float = 0.0
    
    # Historical climate averages
    avg_temp_max_c: float = 0.0
    avg_temp_min_c: float = 0.0
    avg_rainfall_mm: float = 0.0


class DataAugmenter:
    """
    Main data augmentation engine.
    Processes records and adds comprehensive data from multiple sources.
    """
    
    def __init__(self, use_live_apis: bool = True):
        """
        Initialize the augmenter.
        
        Args:
            use_live_apis: Whether to use live API calls (vs cache only)
        """
        self.use_live_apis = use_live_apis
        self.api_client = OpenMeteoClient() if use_live_apis else None
        self.stats = {
            "records_processed": 0,
            "census_data_added": 0,
            "geocoding_success": 0,
            "weather_data_added": 0,
            "air_quality_added": 0,
            "elevation_added": 0,
            "api_errors": 0,
            "cache_hits": 0,
        }
    
    def get_state_data(self, state: str) -> Dict[str, Any]:
        """
        Get census and reference data for a state.
        
        Args:
            state: State name
        
        Returns:
            State census data dictionary
        """
        # Check cache first
        cache_key = f"state_{state}"
        if cache_key in state_cache:
            self.stats['cache_hits'] += 1
            return state_cache.get(cache_key)
        
        # Look up in census data
        if state in INDIA_CENSUS_DATA:
            data = INDIA_CENSUS_DATA[state].copy()
            state_cache.set(cache_key, data)
            return data
        
        return {}
    
    def augment_record(
        self,
        record: Dict[str, Any],
        fetch_weather: bool = False,
        fetch_air_quality: bool = False
    ) -> Dict[str, Any]:
        """
        Augment a single record with additional data.
        
        Args:
            record: Original record dictionary
            fetch_weather: Whether to fetch live weather data
            fetch_air_quality: Whether to fetch live air quality data
        
        Returns:
            Augmented record dictionary
        """
        augmented = record.copy()
        state = record.get('state', '')
        district = record.get('district', '')
        pincode = record.get('pincode', '')
        
        # Add census/reference data (always available)
        state_data = self.get_state_data(state)
        if state_data:
            augmented.update({
                'population_2011': state_data.get('population_2011', 0),
                'area_sq_km': state_data.get('area_sq_km', 0),
                'density_per_sq_km': state_data.get('density_per_sq_km', 0),
                'literacy_rate': state_data.get('literacy_rate', 0.0),
                'sex_ratio': state_data.get('sex_ratio', 0),
                'capital': state_data.get('capital', ''),
                'region': state_data.get('region', ''),
                'zone': state_data.get('zone', ''),
                'rainfall_zone': state_data.get('rainfall_zone', ''),
                'earthquake_zone': state_data.get('earthquake_zone', ''),
                'climate_type': state_data.get('climate_type', ''),
                'avg_temp_min_c': state_data.get('avg_temp_min_c', 0),
                'avg_temp_max_c': state_data.get('avg_temp_max_c', 0),
                'avg_rainfall_mm': state_data.get('avg_rainfall_mm', 0),
                'per_capita_income_inr': state_data.get('per_capita_income_inr', 0),
                'hdi': state_data.get('hdi', 0.0),
                'major_language': state_data.get('major_language', ''),
                'coastal': state_data.get('coastal', False),
            })
            self.stats['census_data_added'] += 1
        
        # Add pincode region info
        if pincode and len(str(pincode)) >= 2:
            prefix = str(pincode)[:2]
            if prefix in PINCODE_REGION_MAP:
                region_info = PINCODE_REGION_MAP[prefix]
                augmented['pincode_region'] = region_info.get('region', '')
        
        # Get coordinates via geocoding (if live APIs enabled)
        lat, lon = None, None
        if self.use_live_apis and self.api_client and district:
            geo_data = self.api_client.get_geocoding(f"{district}, {state}")
            if geo_data:
                lat = geo_data.get('latitude')
                lon = geo_data.get('longitude')
                augmented['latitude'] = lat
                augmented['longitude'] = lon
                augmented['geo_population'] = geo_data.get('population', 0)
                self.stats['geocoding_success'] += 1
                
                # Get elevation
                if lat and lon:
                    elevation = self.api_client.get_elevation(lat, lon)
                    if elevation:
                        augmented['elevation_m'] = elevation
                        self.stats['elevation_added'] += 1
                
                # Get live weather (if requested)
                if fetch_weather and lat and lon:
                    weather = self.api_client.get_current_weather(lat, lon)
                    if weather:
                        augmented.update({
                            'current_temp_c': weather.get('temperature_2m', 0),
                            'current_humidity': weather.get('relative_humidity_2m', 0),
                            'current_precipitation_mm': weather.get('precipitation', 0),
                            'current_wind_speed_kmh': weather.get('wind_speed_10m', 0),
                            'weather_code': weather.get('weather_code', 0),
                        })
                        self.stats['weather_data_added'] += 1
                
                # Get air quality (if requested)
                if fetch_air_quality and lat and lon:
                    aqi = self.api_client.get_air_quality(lat, lon)
                    if aqi:
                        augmented.update({
                            'aqi_european': aqi.get('european_aqi', 0),
                            'aqi_us': aqi.get('us_aqi', 0),
                            'pm2_5': aqi.get('pm2_5', 0),
                            'pm10': aqi.get('pm10', 0),
                            'uv_index': aqi.get('uv_index', 0),
                        })
                        self.stats['air_quality_added'] += 1
        
        self.stats['records_processed'] += 1
        return augmented
    
    def augment_dataframe(
        self,
        df: pd.DataFrame,
        fetch_weather: bool = False,
        fetch_air_quality: bool = False,
        progress_bar: bool = True
    ) -> pd.DataFrame:
        """
        Augment an entire DataFrame.
        
        Args:
            df: Input DataFrame
            fetch_weather: Whether to fetch live weather
            fetch_air_quality: Whether to fetch live air quality
            progress_bar: Whether to show progress bar
        
        Returns:
            Augmented DataFrame
        """
        records = df.to_dict('records')
        augmented_records = []
        
        iterator = tqdm(records, desc="Augmenting records") if progress_bar else records
        
        for record in iterator:
            try:
                augmented = self.augment_record(
                    record,
                    fetch_weather=fetch_weather,
                    fetch_air_quality=fetch_air_quality
                )
                augmented_records.append(augmented)
            except Exception as e:
                self.stats['api_errors'] += 1
                augmented_records.append(record)  # Keep original on error
        
        return pd.DataFrame(augmented_records)


def augment_chunk(args: Tuple) -> pd.DataFrame:
    """
    Augment a chunk of data (for multiprocessing).
    
    Args:
        args: Tuple of (chunk_df, chunk_id, use_live_apis)
    
    Returns:
        Augmented chunk DataFrame
    """
    chunk_df, chunk_id, use_live_apis = args
    
    augmenter = DataAugmenter(use_live_apis=use_live_apis)
    result = augmenter.augment_dataframe(
        chunk_df,
        fetch_weather=False,  # Too slow for batch processing
        fetch_air_quality=False,
        progress_bar=False
    )
    
    return result


def process_dataset_multiprocessing(
    input_file: Path,
    output_file: Path,
    use_live_apis: bool = False,
    chunk_size: int = 50000,
    num_workers: int = None
) -> Dict[str, int]:
    """
    Process a dataset using multiprocessing.
    
    Args:
        input_file: Path to input CSV
        output_file: Path to output CSV
        use_live_apis: Whether to use live API calls
        chunk_size: Size of chunks for parallel processing
        num_workers: Number of worker processes
    
    Returns:
        Processing statistics
    """
    if num_workers is None:
        num_workers = CPU_COUNT
    
    logger.info(f"Processing {input_file} with {num_workers} workers")
    
    # Read the input file
    df = pd.read_csv(input_file, dtype=str, low_memory=False)
    total_records = len(df)
    
    logger.info(f"Loaded {total_records:,} records")
    
    # Split into chunks
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    chunk_args = [(chunk, i, use_live_apis) for i, chunk in enumerate(chunks)]
    
    logger.info(f"Split into {len(chunks)} chunks of ~{chunk_size:,} records")
    
    # Process chunks in parallel
    augmented_chunks = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(augment_chunk, args) for args in chunk_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            try:
                result = future.result()
                augmented_chunks.append(result)
            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
    
    # Combine results
    logger.info("Combining results...")
    final_df = pd.concat(augmented_chunks, ignore_index=True)
    
    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    
    logger.info(f"Saved augmented data to {output_file} ({len(final_df):,} records)")
    
    # Save caches
    state_cache.save_cache()
    pincode_cache.save_cache()
    geocoding_cache.save_cache()
    elevation_cache.save_cache()
    
    return {
        "input_records": total_records,
        "output_records": len(final_df),
        "chunks_processed": len(chunks),
    }


def main():
    """Main entry point for the data augmentation script."""
    parser = argparse.ArgumentParser(
        description="Augment Aadhaar datasets with additional data from APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python data_augmenter.py --all
    python data_augmenter.py --input ./cleaned/biometric.csv --output ./augmented/biometric_aug.csv
    python data_augmenter.py --all --live-apis  # Include live weather/air quality (slower)
        """
    )
    
    parser.add_argument(
        '--all', action='store_true',
        help='Process all cleaned datasets'
    )
    parser.add_argument(
        '--input', type=Path,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output', type=Path,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--live-apis', action='store_true',
        help='Use live APIs for geocoding (slower but more accurate)'
    )
    parser.add_argument(
        '--workers', type=int, default=CPU_COUNT,
        help=f'Number of worker processes (default: {CPU_COUNT})'
    )
    parser.add_argument(
        '--chunk-size', type=int, default=50000,
        help='Chunk size for parallel processing (default: 50000)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting Data Augmentation Pipeline")
    logger.info(f"Workers: {args.workers}, Live APIs: {args.live_apis}")
    logger.info("=" * 60)
    
    all_stats = {}
    
    if args.all:
        # Process all cleaned datasets
        datasets = {
            'biometric': {
                'input': CLEANED_DATASET_PATH / 'biometric' / 'final_cleaned_biometric.csv',
                'output': AUGMENTED_DATASET_PATH / 'biometric_augmented.csv'
            },
            'demographic': {
                'input': CLEANED_DATASET_PATH / 'demographic' / 'final_cleaned_demographic.csv',
                'output': AUGMENTED_DATASET_PATH / 'demographic_augmented.csv'
            },
            'enrollment': {
                'input': CLEANED_DATASET_PATH / 'enrollment' / 'final_cleaned_enrollment.csv',
                'output': AUGMENTED_DATASET_PATH / 'enrollment_augmented.csv'
            }
        }
        
        for name, paths in datasets.items():
            if paths['input'].exists():
                logger.info(f"\nProcessing {name} dataset...")
                stats = process_dataset_multiprocessing(
                    input_file=paths['input'],
                    output_file=paths['output'],
                    use_live_apis=args.live_apis,
                    chunk_size=args.chunk_size,
                    num_workers=args.workers
                )
                all_stats[name] = stats
            else:
                logger.warning(f"Input file not found: {paths['input']}")
    
    elif args.input and args.output:
        stats = process_dataset_multiprocessing(
            input_file=args.input,
            output_file=args.output,
            use_live_apis=args.live_apis,
            chunk_size=args.chunk_size,
            num_workers=args.workers
        )
        all_stats['custom'] = stats
    
    else:
        parser.print_help()
        sys.exit(1)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Augmentation Complete!")
    logger.info("=" * 60)
    
    for name, stats in all_stats.items():
        logger.info(f"\n{name.title()} Dataset:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:,}")


if __name__ == "__main__":
    main()
