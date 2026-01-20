# UIDAI Hackathon - Aadhaar Data Processing Pipeline

## Overview

This project provides a comprehensive, production-grade data processing pipeline for Aadhaar datasets. It includes tools for:

- **CSV Validation & Cleaning**: Fix formatting issues, standardize names, validate data
- **Data Augmentation**: Enrich datasets with census data, weather, air quality, elevation
- **Quality Assurance**: Comprehensive logging, error tracking, and validation reports

## Features

### 🚀 Performance
- **Multiprocessing**: Uses all CPU cores for parallel processing
- **Caching**: Persistent caching to avoid redundant API calls
- **Chunked Processing**: Handles large datasets efficiently

### 🔄 Reliability
- **Retry Strategies**: Exponential backoff with configurable retries
- **Error Handling**: Graceful degradation on API failures
- **Logging**: Detailed logs for debugging and monitoring

### 📊 Data Quality
- **State Name Standardization**: Maps 100+ variations to 36 official names
- **District Name Normalization**: Title case standardization
- **Pincode Validation**: 6-digit Indian pincode format validation

## Project Structure

```
UIDAI_hackathon/
├── analysis/
│   ├── codes/
│   │   ├── __init__.py           # Package initialization
│   │   ├── config.py             # Configuration & constants
│   │   ├── logger.py             # Logging utilities
│   │   ├── csv_cleaner.py        # CSV validation & cleaning
│   │   └── data_augmenter.py     # Data augmentation engine
│   ├── results/                  # Analysis results
│   └── analyze.ipynb             # Jupyter notebook for exploration
├── Dataset/
│   ├── corrected_dataset/        # Input: Corrected raw data
│   │   ├── biometric/
│   │   ├── demographic/
│   │   └── enrollement/
│   ├── cleaned/                  # Output: Cleaned data
│   │   ├── biometric/
│   │   ├── demographic/
│   │   └── enrollment/
│   └── augmented/                # Output: Augmented data
├── docs/                         # Documentation
├── logs/                         # Processing logs
├── augmentation_cache/           # API response cache
├── pyproject.toml                # Project configuration
├── .gitattributes                # Git LFS configuration
└── README.md                     # This file
```

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Clean the Data

```bash
# Clean all datasets
python analysis/codes/csv_cleaner.py --all

# Clean specific dataset
python analysis/codes/csv_cleaner.py --type biometric
```

### 3. Augment the Data

```bash
# Augment all cleaned datasets
python analysis/codes/data_augmenter.py --all

# With live API calls (slower but more accurate)
python analysis/codes/data_augmenter.py --all --live-apis
```

## API Documentation

### Open-Meteo APIs (Free, No API Key)

| API | Endpoint | Rate Limit | Data Provided |
|-----|----------|------------|---------------|
| Weather Forecast | api.open-meteo.com/v1/forecast | 100 req/sec | Temperature, humidity, precipitation, wind |
| Historical Weather | archive-api.open-meteo.com/v1/archive | 10 req/sec | Historical climate data (1940-present) |
| Air Quality | air-quality-api.open-meteo.com/v1/air-quality | 100 req/sec | AQI, PM2.5, PM10, ozone, CO2 |
| Elevation | api.open-meteo.com/v1/elevation | 100 req/sec | Terrain elevation (90m resolution) |
| Geocoding | geocoding-api.open-meteo.com/v1/search | 10 req/sec | Coordinates, population, admin areas |

### Reference Data Sources

| Data | Source | Coverage |
|------|--------|----------|
| Census Data | Census of India 2011 | All 36 States/UTs |
| HDI | UNDP India | State-level |
| Per Capita Income | RBI/MOSPI | State-level |
| Literacy Rate | Census 2011 | State-level |

## Augmented Data Columns

### Census/Reference Data (State-Level)
- `population_2011`: Population from 2011 census
- `area_sq_km`: Geographic area in square kilometers
- `density_per_sq_km`: Population density
- `literacy_rate`: Literacy percentage
- `sex_ratio`: Females per 1000 males
- `capital`: State capital city
- `region`: Geographic region (North, South, etc.)
- `zone`: Administrative zone
- `rainfall_zone`: Rainfall classification (Low/Medium/High/Very High)
- `earthquake_zone`: Seismic zone (I-V)
- `climate_type`: Climate classification
- `per_capita_income_inr`: Annual per capita income
- `hdi`: Human Development Index
- `major_language`: Primary language
- `coastal`: Boolean (has coastline)

### Geographic Data
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `elevation_m`: Elevation in meters
- `pincode_region`: Regional classification based on pincode

### Weather Data (if --live-apis)
- `current_temp_c`: Current temperature
- `current_humidity`: Relative humidity %
- `current_precipitation_mm`: Recent precipitation
- `current_wind_speed_kmh`: Wind speed

### Air Quality Data (if --live-apis)
- `aqi_european`: European Air Quality Index
- `aqi_us`: US Air Quality Index
- `pm2_5`: PM2.5 concentration (μg/m³)
- `pm10`: PM10 concentration (μg/m³)
- `uv_index`: UV Index

## Configuration

### Environment Variables

```bash
# Optional: Override default paths
export UIDAI_DATA_DIR=/custom/data/path
export UIDAI_LOG_LEVEL=DEBUG

# Optional: Set number of workers
export UIDAI_WORKERS=8
```

### Config File

Edit `analysis/codes/config.py` to customize:
- API endpoints and rate limits
- Retry configuration
- State name mappings
- Expected column schemas

## Logging

Logs are stored in `logs/` directory with timestamps:
- `csv_cleaning_YYYYMMDD_HHMMSS.log`
- `data_augmentation_YYYYMMDD_HHMMSS.log`

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Git LFS

Large CSV files are tracked with Git LFS:
```
*.csv filter=lfs diff=lfs merge=lfs -text
```

## Performance Tips

1. **For large datasets (>1M records)**:
   - Use default chunk size (50,000)
   - Disable live APIs for batch processing
   - Run augmentation separately

2. **For accurate geocoding**:
   - Use `--live-apis` flag
   - Expect slower processing (rate limited)
   - Results are cached for subsequent runs

3. **Memory optimization**:
   - Process datasets one at a time
   - Use `--chunk-size 25000` for limited RAM

## Troubleshooting

### Common Issues

1. **"No CSV files found"**
   - Check input directory path
   - Ensure files have .csv extension

2. **"API request failed"**
   - Check internet connectivity
   - Retry will happen automatically
   - Check logs for specific errors

3. **"Unknown state name"**
   - Add mapping to `STATE_NAME_MAPPING` in config.py

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest`
4. Submit a pull request

## Acknowledgments

- [Open-Meteo](https://open-meteo.com/) for free weather APIs
- Census of India for reference data
- UIDAI for the hackathon opportunity
