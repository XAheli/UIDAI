"""
Indian District and State Reference Data
Based on 2011 Census and other official sources
"""

# State and District mapping with population data (2011 Census)
INDIA_CENSUS_DATA = {
    'Andaman and Nicobar Islands': {
        'districts': {
            'Andaman': {
                'population': 380612,
                'area_sq_km': 8249,
                'literacy_rate': 92.36,
                'sex_ratio': 960,
                'urban_population_pct': 27.7
            },
            'Nicobar': {
                'population': 37267,
                'area_sq_km': 1841,
                'literacy_rate': 87.09,
                'sex_ratio': 781,
                'urban_population_pct': 2.8
            }
        },
        'state_pop': 380572,
        'rainfall_zone': 'High',
        'earthquake_zone': 'V (Very High)',
        'primary_climate': 'Tropical'
    },
    'Andhra Pradesh': {
        'state_pop': 84580777,
        'rainfall_zone': 'Moderate to High',
        'earthquake_zone': 'II-III',
        'primary_climate': 'Tropical/Sub-tropical',
        'avg_temp_celsius': 27.5,
        'literacy_rate': 67.67,
        'sex_ratio': 993
    },
    'Arunachal Pradesh': {
        'state_pop': 1382611,
        'rainfall_zone': 'Very High',
        'earthquake_zone': 'IV-V',
        'primary_climate': 'Tropical/Alpine',
        'avg_temp_celsius': 18.5,
        'literacy_rate': 66.95,
        'sex_ratio': 938
    },
    'Assam': {
        'state_pop': 31205144,
        'rainfall_zone': 'High',
        'earthquake_zone': 'IV-V',
        'primary_climate': 'Tropical Monsoon',
        'avg_temp_celsius': 25.0,
        'literacy_rate': 73.18,
        'sex_ratio': 958
    },
    'Bihar': {
        'state_pop': 103804637,
        'rainfall_zone': 'Moderate',
        'earthquake_zone': 'IV-V',
        'primary_climate': 'Sub-tropical',
        'avg_temp_celsius': 26.0,
        'literacy_rate': 63.82,
        'sex_ratio': 917
    },
    'Chandigarh': {
        'state_pop': 1054686,
        'rainfall_zone': 'Low',
        'earthquake_zone': 'IV',
        'primary_climate': 'Temperate',
        'avg_temp_celsius': 24.0,
        'literacy_rate': 86.32,
        'sex_ratio': 818
    },
    'Chhattisgarh': {
        'state_pop': 25545198,
        'rainfall_zone': 'Moderate',
        'earthquake_zone': 'II-III',
        'primary_climate': 'Tropical',
        'avg_temp_celsius': 26.5,
        'literacy_rate': 71.04,
        'sex_ratio': 991
    },
    'Dadra and Nagar Haveli and Daman and Diu': {
        'state_pop': 1458545,
        'rainfall_zone': 'Moderate',
        'earthquake_zone': 'III',
        'primary_climate': 'Tropical/Sub-tropical',
        'avg_temp_celsius': 27.5,
        'literacy_rate': 79.31,
        'sex_ratio': 973
    },
    'Delhi': {
        'state_pop': 16753235,
        'rainfall_zone': 'Low',
        'earthquake_zone': 'IV',
        'primary_climate': 'Semi-arid',
        'avg_temp_celsius': 25.5,
        'literacy_rate': 86.29,
        'sex_ratio': 868
    },
    'Goa': {
        'state_pop': 1347668,
        'rainfall_zone': 'High',
        'earthquake_zone': 'III',
        'primary_climate': 'Tropical',
        'avg_temp_celsius': 27.8,
        'literacy_rate': 87.40,
        'sex_ratio': 973
    },
    'Gujarat': {
        'state_pop': 60439692,
        'rainfall_zone': 'Low to Moderate',
        'earthquake_zone': 'III-IV',
        'primary_climate': 'Semi-arid/Arid',
        'avg_temp_celsius': 27.0,
        'literacy_rate': 79.31,
        'sex_ratio': 919
    },
    'Haryana': {
        'state_pop': 25351462,
        'rainfall_zone': 'Low',
        'earthquake_zone': 'IV',
        'primary_climate': 'Semi-arid',
        'avg_temp_celsius': 25.0,
        'literacy_rate': 75.60,
        'sex_ratio': 879
    },
    'Himachal Pradesh': {
        'state_pop': 6864602,
        'rainfall_zone': 'High',
        'earthquake_zone': 'IV-V',
        'primary_climate': 'Temperate/Alpine',
        'avg_temp_celsius': 14.0,
        'literacy_rate': 83.78,
        'sex_ratio': 972
    },
    'Jammu and Kashmir': {
        'state_pop': 12541302,
        'rainfall_zone': 'Moderate to High',
        'earthquake_zone': 'IV-V',
        'primary_climate': 'Temperate',
        'avg_temp_celsius': 12.5,
        'literacy_rate': 68.74,
        'sex_ratio': 889
    },
    'Jharkhand': {
        'state_pop': 32966134,
        'rainfall_zone': 'Moderate',
        'earthquake_zone': 'III',
        'primary_climate': 'Tropical',
        'avg_temp_celsius': 25.5,
        'literacy_rate': 67.40,
        'sex_ratio': 948
    },
    'Karnataka': {
        'state_pop': 61130704,
        'rainfall_zone': 'Moderate to High',
        'earthquake_zone': 'II-III',
        'primary_climate': 'Tropical/Sub-tropical',
        'avg_temp_celsius': 26.5,
        'literacy_rate': 75.60,
        'sex_ratio': 973
    },
    'Kerala': {
        'state_pop': 33387677,
        'rainfall_zone': 'Very High',
        'earthquake_zone': 'II',
        'primary_climate': 'Tropical Monsoon',
        'avg_temp_celsius': 27.2,
        'literacy_rate': 93.91,
        'sex_ratio': 1084
    },
    'Ladakh': {
        'state_pop': 274289,
        'rainfall_zone': 'Very Low',
        'earthquake_zone': 'IV-V',
        'primary_climate': 'Alpine/Arid',
        'avg_temp_celsius': 7.0,
        'literacy_rate': 74.40,
        'sex_ratio': 938
    },
    'Lakshadweep': {
        'state_pop': 64429,
        'rainfall_zone': 'High',
        'earthquake_zone': 'III',
        'primary_climate': 'Tropical',
        'avg_temp_celsius': 27.8,
        'literacy_rate': 92.28,
        'sex_ratio': 945
    },
    'Madhya Pradesh': {
        'state_pop': 72597565,
        'rainfall_zone': 'Moderate',
        'earthquake_zone': 'II-III',
        'primary_climate': 'Sub-tropical',
        'avg_temp_celsius': 25.5,
        'literacy_rate': 70.63,
        'sex_ratio': 931
    },
    'Maharashtra': {
        'state_pop': 112372972,
        'rainfall_zone': 'Moderate',
        'earthquake_zone': 'II-III',
        'primary_climate': 'Tropical/Sub-tropical',
        'avg_temp_celsius': 26.5,
        'literacy_rate': 82.91,
        'sex_ratio': 929
    },
    'Manipur': {
        'state_pop': 2721756,
        'rainfall_zone': 'Very High',
        'earthquake_zone': 'IV-V',
        'primary_climate': 'Tropical Monsoon',
        'avg_temp_celsius': 22.5,
        'literacy_rate': 79.80,
        'sex_ratio': 989
    },
    'Meghalaya': {
        'state_pop': 2966889,
        'rainfall_zone': 'Extremely High',
        'earthquake_zone': 'IV-V',
        'primary_climate': 'Tropical Monsoon',
        'avg_temp_celsius': 20.0,
        'literacy_rate': 75.48,
        'sex_ratio': 989
    },
    'Mizoram': {
        'state_pop': 1097206,
        'rainfall_zone': 'Very High',
        'earthquake_zone': 'IV',
        'primary_climate': 'Tropical Monsoon',
        'avg_temp_celsius': 19.5,
        'literacy_rate': 91.58,
        'sex_ratio': 976
    },
    'Nagaland': {
        'state_pop': 1978502,
        'rainfall_zone': 'Very High',
        'earthquake_zone': 'IV-V',
        'primary_climate': 'Tropical/Temperate',
        'avg_temp_celsius': 19.0,
        'literacy_rate': 80.11,
        'sex_ratio': 931
    },
    'Odisha': {
        'state_pop': 42,
        'rainfall_zone': 'Moderate to High',
        'earthquake_zone': 'II-III',
        'primary_climate': 'Tropical',
        'avg_temp_celsius': 26.0,
        'literacy_rate': 73.45,
        'sex_ratio': 979
    },
    'Puducherry': {
        'state_pop': 1244464,
        'rainfall_zone': 'Moderate',
        'earthquake_zone': 'II',
        'primary_climate': 'Tropical',
        'avg_temp_celsius': 27.8,
        'literacy_rate': 86.55,
        'sex_ratio': 995
    },
    'Punjab': {
        'state_pop': 27704236,
        'rainfall_zone': 'Low',
        'earthquake_zone': 'IV',
        'primary_climate': 'Temperate',
        'avg_temp_celsius': 24.5,
        'literacy_rate': 75.84,
        'sex_ratio': 895
    },
    'Rajasthan': {
        'state_pop': 68548437,
        'rainfall_zone': 'Low',
        'earthquake_zone': 'III-IV',
        'primary_climate': 'Arid/Semi-arid',
        'avg_temp_celsius': 27.5,
        'literacy_rate': 75.51,
        'sex_ratio': 928
    },
    'Sikkim': {
        'state_pop': 610577,
        'rainfall_zone': 'Very High',
        'earthquake_zone': 'IV',
        'primary_climate': 'Temperate/Alpine',
        'avg_temp_celsius': 13.0,
        'literacy_rate': 89.33,
        'sex_ratio': 889
    },
    'Tamil Nadu': {
        'state_pop': 72138958,
        'rainfall_zone': 'Moderate',
        'earthquake_zone': 'II',
        'primary_climate': 'Tropical',
        'avg_temp_celsius': 28.0,
        'literacy_rate': 80.33,
        'sex_ratio': 995
    },
    'Telangana': {
        'state_pop': 35193978,
        'rainfall_zone': 'Moderate',
        'earthquake_zone': 'II',
        'primary_climate': 'Tropical/Sub-tropical',
        'avg_temp_celsius': 27.5,
        'literacy_rate': 66.50,
        'sex_ratio': 988
    },
    'Tripura': {
        'state_pop': 3671032,
        'rainfall_zone': 'High',
        'earthquake_zone': 'IV',
        'primary_climate': 'Tropical Monsoon',
        'avg_temp_celsius': 24.5,
        'literacy_rate': 87.75,
        'sex_ratio': 960
    },
    'Uttar Pradesh': {
        'state_pop': 199812341,
        'rainfall_zone': 'Low to Moderate',
        'earthquake_zone': 'III-IV',
        'primary_climate': 'Sub-tropical',
        'avg_temp_celsius': 25.5,
        'literacy_rate': 69.72,
        'sex_ratio': 912
    },
    'Uttarakhand': {
        'state_pop': 10086292,
        'rainfall_zone': 'High',
        'earthquake_zone': 'IV-V',
        'primary_climate': 'Temperate',
        'avg_temp_celsius': 15.5,
        'literacy_rate': 79.63,
        'sex_ratio': 963
    },
    'West Bengal': {
        'state_pop': 91276115,
        'rainfall_zone': 'High',
        'earthquake_zone': 'III-IV',
        'primary_climate': 'Tropical/Temperate',
        'avg_temp_celsius': 26.0,
        'literacy_rate': 77.12,
        'sex_ratio': 950
    }
}

# Rainfall zone descriptions
RAINFALL_ZONES = {
    'Very Low': 'Rainfall < 150mm/year',
    'Low': 'Rainfall 150-500mm/year',
    'Moderate': 'Rainfall 500-1500mm/year',
    'High': 'Rainfall 1500-2500mm/year',
    'Very High': 'Rainfall 2500-4000mm/year',
    'Extremely High': 'Rainfall > 4000mm/year'
}

# Earthquake zones
EARTHQUAKE_ZONES = {
    'II': 'Low to Moderate seismic activity',
    'III': 'Moderate to High seismic activity',
    'IV': 'High seismic activity',
    'IV-V': 'Very High seismic activity',
    'V': 'Severe seismic activity'
}

# Climate classifications
CLIMATE_TYPES = {
    'Tropical': 'Hot and humid year-round',
    'Tropical Monsoon': 'Monsoons with high rainfall',
    'Tropical/Sub-tropical': 'Warm year-round',
    'Sub-tropical': 'Warm summers, mild winters',
    'Temperate': 'Moderate temperatures',
    'Temperate/Alpine': 'Cooler with mountain characteristics',
    'Alpine': 'Cold, mountainous',
    'Alpine/Arid': 'Cold and dry',
    'Arid': 'Very dry',
    'Semi-arid': 'Low rainfall',
}

# Per capita income by state (2021 est., in USD)
PER_CAPITA_INCOME_USD = {
    'Andaman and Nicobar Islands': 4500,
    'Andhra Pradesh': 2800,
    'Arunachal Pradesh': 2200,
    'Assam': 1800,
    'Bihar': 1100,
    'Chandigarh': 5500,
    'Chhattisgarh': 2600,
    'Dadra and Nagar Haveli and Daman and Diu': 4200,
    'Delhi': 6200,
    'Goa': 5800,
    'Gujarat': 3200,
    'Haryana': 4200,
    'Himachal Pradesh': 3200,
    'Jammu and Kashmir': 1900,
    'Jharkhand': 1700,
    'Karnataka': 3500,
    'Kerala': 4200,
    'Ladakh': 3800,
    'Lakshadweep': 5200,
    'Madhya Pradesh': 1900,
    'Maharashtra': 3800,
    'Manipur': 1500,
    'Meghalaya': 2000,
    'Mizoram': 2800,
    'Nagaland': 1900,
    'Odisha': 1700,
    'Puducherry': 4500,
    'Punjab': 3200,
    'Rajasthan': 2200,
    'Sikkim': 3200,
    'Tamil Nadu': 3600,
    'Telangana': 3200,
    'Tripura': 2000,
    'Uttar Pradesh': 1700,
    'Uttarakhand': 2800,
    'West Bengal': 2000
}

# HDI (Human Development Index) by state (2019)
HUMAN_DEVELOPMENT_INDEX = {
    'Andaman and Nicobar Islands': 0.743,
    'Andhra Pradesh': 0.593,
    'Arunachal Pradesh': 0.537,
    'Assam': 0.549,
    'Bihar': 0.471,
    'Chandigarh': 0.802,
    'Chhattisgarh': 0.578,
    'Dadra and Nagar Haveli and Daman and Diu': 0.652,
    'Delhi': 0.803,
    'Goa': 0.794,
    'Gujarat': 0.650,
    'Haryana': 0.703,
    'Himachal Pradesh': 0.711,
    'Jammu and Kashmir': 0.556,
    'Jharkhand': 0.515,
    'Karnataka': 0.649,
    'Kerala': 0.784,
    'Ladakh': 0.615,
    'Lakshadweep': 0.750,
    'Madhya Pradesh': 0.539,
    'Maharashtra': 0.689,
    'Manipur': 0.565,
    'Meghalaya': 0.572,
    'Mizoram': 0.684,
    'Nagaland': 0.598,
    'Odisha': 0.568,
    'Puducherry': 0.755,
    'Punjab': 0.695,
    'Rajasthan': 0.599,
    'Sikkim': 0.704,
    'Tamil Nadu': 0.686,
    'Telangana': 0.620,
    'Tripura': 0.639,
    'Uttar Pradesh': 0.528,
    'Uttarakhand': 0.651,
    'West Bengal': 0.603
}
