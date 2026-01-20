#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis Pipeline for UIDAI Aadhaar Data
Authors: Shuvam Banerji Seal, Alok Mishra, Aheli Poddar
UIDAI Data Hackathon 2026

This script performs exhaustive statistical analysis including:
1. Time Series Analysis (trends, seasonality, anomalies)
2. Geographic Analysis (state, district, regional patterns)
3. Demographic Analysis (age cohorts, population correlations)
4. Socioeconomic Analysis (HDI, income, literacy correlations)
5. Climate Analysis (rainfall zones, temperature correlations)
6. Statistical Modeling (regression, hypothesis testing)
7. Clustering and Segmentation
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
from scipy import stats
from scipy.stats import (
    pearsonr, spearmanr, kendalltau, 
    ttest_ind, mannwhitneyu, kruskal,
    chi2_contingency, f_oneway, kstest,
    shapiro, normaltest, levene
)

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / 'analysis' / 'codes'))

from india_reference_data import (
    INDIA_CENSUS_DATA, RAINFALL_ZONES, EARTHQUAKE_ZONES,
    CLIMATE_TYPES, PER_CAPITA_INCOME_USD, HUMAN_DEVELOPMENT_INDEX
)


class ComprehensiveAnalyzer:
    """
    Performs comprehensive statistical analysis on UIDAI Aadhaar datasets
    """
    
    def __init__(self, results_dir: Path = None):
        self.base_dir = BASE_DIR
        self.results_dir = results_dir or BASE_DIR / 'results' / 'comprehensive_analysis'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset paths
        self.dataset_paths = {
            'biometric': BASE_DIR / 'Dataset' / 'api_data_aadhar_biometric',
            'demographic': BASE_DIR / 'Dataset' / 'api_data_aadhar_demographic',
            'enrollment': BASE_DIR / 'Dataset' / 'api_data_aadhar_enrolment'
        }
        
        # Region mapping
        self.region_mapping = {
            'North': ['Delhi', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 
                     'Punjab', 'Rajasthan', 'Uttarakhand', 'Chandigarh', 'Ladakh'],
            'South': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 
                     'Telangana', 'Puducherry', 'Lakshadweep'],
            'East': ['Bihar', 'Jharkhand', 'Odisha', 'West Bengal'],
            'West': ['Goa', 'Gujarat', 'Maharashtra', 
                    'Dadra and Nagar Haveli and Daman and Diu'],
            'Central': ['Madhya Pradesh', 'Chhattisgarh', 'Uttar Pradesh'],
            'Northeast': ['Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya',
                         'Mizoram', 'Nagaland', 'Sikkim', 'Tripura', 
                         'Andaman and Nicobar Islands']
        }
        
        # Results storage
        self.results = {}
        
        print(f"✓ ComprehensiveAnalyzer initialized")
        print(f"  Results directory: {self.results_dir}")
    
    def load_dataset(self, dataset_type: str, sample_size: int = None) -> pd.DataFrame:
        """Load and combine dataset files"""
        dataset_path = self.dataset_paths.get(dataset_type)
        if not dataset_path or not dataset_path.exists():
            print(f"⚠ Dataset path not found: {dataset_path}")
            return pd.DataFrame()
        
        dfs = []
        csv_files = list(dataset_path.glob('*.csv'))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                dfs.append(df)
            except Exception as e:
                print(f"⚠ Error loading {csv_file}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs, ignore_index=True)
        
        if sample_size and len(combined) > sample_size:
            combined = combined.sample(n=sample_size, random_state=42)
        
        return combined
    
    def augment_with_reference_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Augment dataset with reference data"""
        df = df.copy()
        
        # Add state-level reference data
        df['population_2011'] = df['state'].map(
            lambda x: INDIA_CENSUS_DATA.get(x, {}).get('state_pop', np.nan)
        )
        df['rainfall_zone'] = df['state'].map(
            lambda x: INDIA_CENSUS_DATA.get(x, {}).get('rainfall_zone', 'Unknown')
        )
        df['earthquake_zone'] = df['state'].map(
            lambda x: INDIA_CENSUS_DATA.get(x, {}).get('earthquake_zone', 'Unknown')
        )
        df['climate_type'] = df['state'].map(
            lambda x: INDIA_CENSUS_DATA.get(x, {}).get('primary_climate', 'Unknown')
        )
        df['state_literacy_rate'] = df['state'].map(
            lambda x: INDIA_CENSUS_DATA.get(x, {}).get('literacy_rate', np.nan)
        )
        df['state_sex_ratio'] = df['state'].map(
            lambda x: INDIA_CENSUS_DATA.get(x, {}).get('sex_ratio', np.nan)
        )
        df['per_capita_income_usd'] = df['state'].map(PER_CAPITA_INCOME_USD)
        df['hdi'] = df['state'].map(HUMAN_DEVELOPMENT_INDEX)
        
        # Add region
        state_to_region = {}
        for region, states in self.region_mapping.items():
            for state in states:
                state_to_region[state] = region
        df['region'] = df['state'].map(lambda x: state_to_region.get(x, 'Other'))
        
        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_name'] = df['date'].dt.day_name()
            df['month'] = df['date'].dt.month
            df['month_name'] = df['date'].dt.month_name()
            df['year'] = df['date'].dt.year
            df['quarter'] = df['date'].dt.quarter
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Add pincode-based features
        df['pincode_zone'] = df['pincode'].astype(str).str[0].astype(int)
        df['pincode_region'] = df['pincode'].astype(str).str[:2].astype(int)
        
        return df
    
    def calculate_total_enrollment(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Calculate total enrollment based on dataset type"""
        df = df.copy()
        
        if dataset_type == 'biometric':
            df['total_enrollment'] = df['bio_age_5_17'].fillna(0) + df['bio_age_17_'].fillna(0)
        elif dataset_type == 'demographic':
            demo_cols = [c for c in df.columns if 'demo_' in c.lower() or 'age' in c.lower()]
            if demo_cols:
                df['total_enrollment'] = df[demo_cols].fillna(0).sum(axis=1)
            else:
                df['total_enrollment'] = 1
        elif dataset_type == 'enrollment':
            enroll_cols = [c for c in df.columns if 'enrol' in c.lower() or 'age' in c.lower()]
            if enroll_cols:
                df['total_enrollment'] = df[enroll_cols].fillna(0).sum(axis=1)
            else:
                df['total_enrollment'] = 1
        else:
            df['total_enrollment'] = 1
        
        return df
    
    # ================== TIME SERIES ANALYSIS ==================
    
    def analyze_time_series(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Comprehensive time series analysis"""
        results = {
            'dataset': dataset_name,
            'analysis_type': 'time_series'
        }
        
        if 'date' not in df.columns or df['date'].isna().all():
            results['error'] = 'No valid date column'
            return results
        
        # Daily aggregation
        daily = df.groupby(df['date'].dt.date).agg({
            'total_enrollment': 'sum',
            'state': 'nunique',
            'district': 'nunique',
            'pincode': 'nunique'
        }).reset_index()
        daily.columns = ['date', 'total_enrollment', 'unique_states', 'unique_districts', 'unique_pincodes']
        
        results['date_range'] = {
            'start': str(daily['date'].min()),
            'end': str(daily['date'].max()),
            'total_days': len(daily)
        }
        
        results['enrollment_stats'] = {
            'total': int(daily['total_enrollment'].sum()),
            'daily_mean': float(daily['total_enrollment'].mean()),
            'daily_std': float(daily['total_enrollment'].std()) if len(daily) > 1 else 0,
            'daily_min': int(daily['total_enrollment'].min()),
            'daily_max': int(daily['total_enrollment'].max()),
            'daily_median': float(daily['total_enrollment'].median())
        }
        
        # Day of week analysis
        dow_analysis = df.groupby('day_name')['total_enrollment'].agg(['sum', 'mean', 'count'])
        results['day_of_week_pattern'] = {
            day: {
                'total': int(row['sum']),
                'mean': float(row['mean']),
                'records': int(row['count'])
            }
            for day, row in dow_analysis.iterrows()
        }
        
        # Weekend vs Weekday
        weekend_enroll = df[df['is_weekend'] == 1]['total_enrollment'].sum()
        weekday_enroll = df[df['is_weekend'] == 0]['total_enrollment'].sum()
        weekend_count = df[df['is_weekend'] == 1].shape[0]
        weekday_count = df[df['is_weekend'] == 0].shape[0]
        
        results['weekend_vs_weekday'] = {
            'weekend': {
                'total': int(weekend_enroll),
                'records': int(weekend_count),
                'mean_per_record': float(weekend_enroll / weekend_count) if weekend_count > 0 else 0
            },
            'weekday': {
                'total': int(weekday_enroll),
                'records': int(weekday_count),
                'mean_per_record': float(weekday_enroll / weekday_count) if weekday_count > 0 else 0
            }
        }
        
        # Statistical test: weekend vs weekday
        if weekend_count > 0 and weekday_count > 0:
            weekend_data = df[df['is_weekend'] == 1]['total_enrollment']
            weekday_data = df[df['is_weekend'] == 0]['total_enrollment']
            
            try:
                t_stat, p_value = ttest_ind(weekend_data, weekday_data)
                results['weekend_weekday_ttest'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            except:
                pass
        
        # Monthly pattern
        if 'month_name' in df.columns:
            monthly = df.groupby('month_name')['total_enrollment'].agg(['sum', 'mean', 'count'])
            results['monthly_pattern'] = {
                month: {
                    'total': int(row['sum']),
                    'mean': float(row['mean']),
                    'records': int(row['count'])
                }
                for month, row in monthly.iterrows()
            }
        
        return results
    
    # ================== GEOGRAPHIC ANALYSIS ==================
    
    def analyze_geographic(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Comprehensive geographic analysis"""
        results = {
            'dataset': dataset_name,
            'analysis_type': 'geographic'
        }
        
        # State-level analysis
        state_agg = df.groupby('state').agg({
            'total_enrollment': ['sum', 'mean', 'std', 'count'],
            'district': 'nunique',
            'pincode': 'nunique'
        }).round(4)
        state_agg.columns = ['total', 'mean', 'std', 'records', 'unique_districts', 'unique_pincodes']
        state_agg = state_agg.sort_values('total', ascending=False)
        
        results['state_analysis'] = {
            'total_states': len(state_agg),
            'top_10_states': [
                {
                    'state': state,
                    'total_enrollment': int(row['total']),
                    'mean_enrollment': float(row['mean']),
                    'std_enrollment': float(row['std']) if not pd.isna(row['std']) else 0,
                    'record_count': int(row['records']),
                    'unique_districts': int(row['unique_districts']),
                    'unique_pincodes': int(row['unique_pincodes'])
                }
                for state, row in state_agg.head(10).iterrows()
            ],
            'bottom_5_states': [
                {
                    'state': state,
                    'total_enrollment': int(row['total']),
                    'record_count': int(row['records'])
                }
                for state, row in state_agg.tail(5).iterrows()
            ]
        }
        
        # Regional analysis
        regional = df.groupby('region').agg({
            'total_enrollment': ['sum', 'mean', 'count'],
            'state': 'nunique',
            'district': 'nunique'
        })
        regional.columns = ['total', 'mean', 'records', 'unique_states', 'unique_districts']
        
        total_enrollment = regional['total'].sum()
        results['regional_analysis'] = {
            region: {
                'total_enrollment': int(row['total']),
                'percentage': float(row['total'] / total_enrollment * 100),
                'mean_per_record': float(row['mean']),
                'record_count': int(row['records']),
                'unique_states': int(row['unique_states']),
                'unique_districts': int(row['unique_districts'])
            }
            for region, row in regional.iterrows()
        }
        
        # Calculate Gini coefficient for geographic inequality
        enrollments = state_agg['total'].values
        n = len(enrollments)
        if n > 1:
            sorted_enroll = np.sort(enrollments)
            cumsum = np.cumsum(sorted_enroll)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            results['inequality_metrics'] = {
                'gini_coefficient': float(gini),
                'interpretation': 'Higher value indicates more inequality in enrollment distribution'
            }
        
        # Top 5 concentration ratio
        top5_enrollment = state_agg.head(5)['total'].sum()
        top10_enrollment = state_agg.head(10)['total'].sum()
        results['concentration_ratios'] = {
            'top_5_share_pct': float(top5_enrollment / total_enrollment * 100),
            'top_10_share_pct': float(top10_enrollment / total_enrollment * 100)
        }
        
        # Pincode zone analysis
        zone_analysis = df.groupby('pincode_zone')['total_enrollment'].agg(['sum', 'mean', 'count'])
        results['pincode_zone_analysis'] = {
            str(zone): {
                'total': int(row['sum']),
                'mean': float(row['mean']),
                'records': int(row['count'])
            }
            for zone, row in zone_analysis.iterrows()
        }
        
        return results
    
    # ================== DEMOGRAPHIC ANALYSIS ==================
    
    def analyze_demographic(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Demographic and population analysis"""
        results = {
            'dataset': dataset_name,
            'analysis_type': 'demographic'
        }
        
        # Age group analysis (for biometric dataset)
        if 'bio_age_5_17' in df.columns and 'bio_age_17_' in df.columns:
            total_5_17 = df['bio_age_5_17'].sum()
            total_17_plus = df['bio_age_17_'].sum()
            total = total_5_17 + total_17_plus
            
            results['age_group_analysis'] = {
                'age_5_17': {
                    'total': int(total_5_17),
                    'percentage': float(total_5_17 / total * 100) if total > 0 else 0,
                    'mean_per_record': float(df['bio_age_5_17'].mean()),
                    'std': float(df['bio_age_5_17'].std())
                },
                'age_17_plus': {
                    'total': int(total_17_plus),
                    'percentage': float(total_17_plus / total * 100) if total > 0 else 0,
                    'mean_per_record': float(df['bio_age_17_'].mean()),
                    'std': float(df['bio_age_17_'].std())
                },
                'ratio_5_17_to_17_plus': float(total_5_17 / total_17_plus) if total_17_plus > 0 else 0
            }
            
            # Correlation between age groups
            corr, p_value = pearsonr(df['bio_age_5_17'].fillna(0), df['bio_age_17_'].fillna(0))
            results['age_group_correlation'] = {
                'pearson_correlation': float(corr),
                'p_value': float(p_value),
                'interpretation': 'Strong positive correlation' if corr > 0.7 else 
                                 'Moderate correlation' if corr > 0.4 else 'Weak correlation'
            }
        
        # Population correlation
        if 'population_2011' in df.columns:
            state_enroll = df.groupby('state')['total_enrollment'].sum()
            state_pop = df.groupby('state')['population_2011'].first()
            
            common_states = set(state_enroll.index) & set(state_pop.index)
            if len(common_states) > 5:
                enroll_vals = [state_enroll[s] for s in common_states]
                pop_vals = [state_pop[s] for s in common_states if not pd.isna(state_pop[s])]
                
                if len(pop_vals) > 5:
                    corr, p_value = pearsonr(enroll_vals[:len(pop_vals)], pop_vals)
                    results['population_correlation'] = {
                        'pearson_correlation': float(corr),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'interpretation': 'Enrollment correlates with population' if corr > 0.5 else 'Weak population correlation'
                    }
        
        # Sex ratio analysis
        if 'state_sex_ratio' in df.columns:
            state_data = df.groupby('state').agg({
                'total_enrollment': 'sum',
                'state_sex_ratio': 'first'
            }).dropna()
            
            if len(state_data) > 5:
                corr, p_value = pearsonr(state_data['total_enrollment'], state_data['state_sex_ratio'])
                results['sex_ratio_correlation'] = {
                    'pearson_correlation': float(corr),
                    'p_value': float(p_value),
                    'interpretation': 'Higher sex ratio correlates with higher enrollment' if corr > 0 else 'Negative correlation with sex ratio'
                }
        
        return results
    
    # ================== SOCIOECONOMIC ANALYSIS ==================
    
    def analyze_socioeconomic(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Socioeconomic indicators analysis"""
        results = {
            'dataset': dataset_name,
            'analysis_type': 'socioeconomic'
        }
        
        # State-level aggregation
        state_data = df.groupby('state').agg({
            'total_enrollment': 'sum',
            'state_literacy_rate': 'first',
            'per_capita_income_usd': 'first',
            'hdi': 'first'
        }).dropna()
        
        # HDI Analysis
        if 'hdi' in state_data.columns and len(state_data) > 5:
            corr, p_value = pearsonr(state_data['total_enrollment'], state_data['hdi'])
            results['hdi_analysis'] = {
                'correlation_with_enrollment': float(corr),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
            
            # HDI stratification
            high_hdi = state_data[state_data['hdi'] >= 0.65]
            medium_hdi = state_data[(state_data['hdi'] >= 0.55) & (state_data['hdi'] < 0.65)]
            low_hdi = state_data[state_data['hdi'] < 0.55]
            
            results['hdi_stratification'] = {
                'high_hdi_states': {
                    'count': len(high_hdi),
                    'mean_enrollment': float(high_hdi['total_enrollment'].mean()) if len(high_hdi) > 0 else 0,
                    'hdi_range': f"{high_hdi['hdi'].min():.3f} - {high_hdi['hdi'].max():.3f}" if len(high_hdi) > 0 else 'N/A'
                },
                'medium_hdi_states': {
                    'count': len(medium_hdi),
                    'mean_enrollment': float(medium_hdi['total_enrollment'].mean()) if len(medium_hdi) > 0 else 0
                },
                'low_hdi_states': {
                    'count': len(low_hdi),
                    'mean_enrollment': float(low_hdi['total_enrollment'].mean()) if len(low_hdi) > 0 else 0
                }
            }
        
        # Literacy rate analysis
        if 'state_literacy_rate' in state_data.columns and len(state_data) > 5:
            corr, p_value = pearsonr(state_data['total_enrollment'], state_data['state_literacy_rate'])
            results['literacy_analysis'] = {
                'correlation_with_enrollment': float(corr),
                'p_value': float(p_value),
                'interpretation': 'Higher literacy associated with higher enrollment' if corr > 0 else 'Inverse relationship'
            }
        
        # Per capita income analysis
        if 'per_capita_income_usd' in state_data.columns and len(state_data) > 5:
            corr, p_value = pearsonr(state_data['total_enrollment'], state_data['per_capita_income_usd'])
            results['income_analysis'] = {
                'correlation_with_enrollment': float(corr),
                'p_value': float(p_value),
                'interpretation': 'Higher income associated with higher enrollment' if corr > 0.3 else 
                                 'Weak income-enrollment relationship'
            }
        
        return results
    
    # ================== CLIMATE ANALYSIS ==================
    
    def analyze_climate(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Climate and environmental analysis"""
        results = {
            'dataset': dataset_name,
            'analysis_type': 'climate'
        }
        
        # Rainfall zone analysis
        if 'rainfall_zone' in df.columns:
            rainfall_agg = df.groupby('rainfall_zone').agg({
                'total_enrollment': ['sum', 'mean', 'count'],
                'state': 'nunique'
            })
            rainfall_agg.columns = ['total', 'mean', 'records', 'unique_states']
            
            total = rainfall_agg['total'].sum()
            results['rainfall_zone_analysis'] = {
                zone: {
                    'total_enrollment': int(row['total']),
                    'percentage': float(row['total'] / total * 100),
                    'mean_per_record': float(row['mean']),
                    'record_count': int(row['records']),
                    'unique_states': int(row['unique_states'])
                }
                for zone, row in rainfall_agg.iterrows()
            }
            
            # ANOVA test across rainfall zones
            groups = [df[df['rainfall_zone'] == zone]['total_enrollment'].values 
                     for zone in df['rainfall_zone'].unique() 
                     if len(df[df['rainfall_zone'] == zone]) > 10]
            
            if len(groups) >= 2:
                try:
                    f_stat, p_value = f_oneway(*groups)
                    results['rainfall_zone_anova'] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'interpretation': 'Significant difference across rainfall zones' if p_value < 0.05 else 'No significant difference'
                    }
                except:
                    pass
        
        # Climate type analysis
        if 'climate_type' in df.columns:
            climate_agg = df.groupby('climate_type').agg({
                'total_enrollment': ['sum', 'mean', 'count']
            })
            climate_agg.columns = ['total', 'mean', 'records']
            
            total = climate_agg['total'].sum()
            results['climate_type_analysis'] = {
                climate: {
                    'total_enrollment': int(row['total']),
                    'percentage': float(row['total'] / total * 100),
                    'mean_per_record': float(row['mean'])
                }
                for climate, row in climate_agg.iterrows()
            }
        
        # Earthquake zone analysis
        if 'earthquake_zone' in df.columns:
            eq_agg = df.groupby('earthquake_zone').agg({
                'total_enrollment': ['sum', 'mean', 'count']
            })
            eq_agg.columns = ['total', 'mean', 'records']
            
            results['earthquake_zone_analysis'] = {
                zone: {
                    'total_enrollment': int(row['total']),
                    'mean_per_record': float(row['mean']),
                    'record_count': int(row['records'])
                }
                for zone, row in eq_agg.iterrows()
            }
        
        return results
    
    # ================== HYPOTHESIS TESTING ==================
    
    def perform_hypothesis_tests(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Comprehensive hypothesis testing"""
        results = {
            'dataset': dataset_name,
            'analysis_type': 'hypothesis_testing'
        }
        
        tests_performed = []
        
        # Test 1: Regional differences (Kruskal-Wallis)
        if 'region' in df.columns:
            groups = [df[df['region'] == r]['total_enrollment'].values 
                     for r in df['region'].unique() 
                     if len(df[df['region'] == r]) > 30]
            
            if len(groups) >= 2:
                try:
                    stat, p_value = kruskal(*groups)
                    tests_performed.append({
                        'test_name': 'Kruskal-Wallis: Regional Differences',
                        'h0': 'No difference in enrollment across regions',
                        'h1': 'At least one region differs significantly',
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'result': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0',
                        'conclusion': 'Significant regional differences exist' if p_value < 0.05 else 'No significant regional differences'
                    })
                except:
                    pass
        
        # Test 2: Weekend vs Weekday (Mann-Whitney U)
        if 'is_weekend' in df.columns:
            weekend = df[df['is_weekend'] == 1]['total_enrollment']
            weekday = df[df['is_weekend'] == 0]['total_enrollment']
            
            if len(weekend) > 30 and len(weekday) > 30:
                try:
                    stat, p_value = mannwhitneyu(weekend, weekday, alternative='two-sided')
                    tests_performed.append({
                        'test_name': 'Mann-Whitney U: Weekend vs Weekday',
                        'h0': 'No difference between weekend and weekday enrollment',
                        'h1': 'Enrollment differs between weekends and weekdays',
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'result': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0',
                        'weekend_median': float(weekend.median()),
                        'weekday_median': float(weekday.median())
                    })
                except:
                    pass
        
        # Test 3: HDI groups (ANOVA)
        if 'hdi' in df.columns:
            df['hdi_group'] = pd.cut(df['hdi'], bins=[0, 0.55, 0.65, 1.0], labels=['Low', 'Medium', 'High'])
            groups = [df[df['hdi_group'] == g]['total_enrollment'].values 
                     for g in ['Low', 'Medium', 'High'] 
                     if len(df[df['hdi_group'] == g]) > 30]
            
            if len(groups) >= 2:
                try:
                    stat, p_value = f_oneway(*groups)
                    tests_performed.append({
                        'test_name': 'One-way ANOVA: HDI Groups',
                        'h0': 'No difference in enrollment across HDI groups',
                        'h1': 'At least one HDI group differs',
                        'f_statistic': float(stat),
                        'p_value': float(p_value),
                        'result': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0'
                    })
                except:
                    pass
        
        # Test 4: Normality test
        sample = df['total_enrollment'].dropna().sample(min(5000, len(df)), random_state=42)
        try:
            stat, p_value = normaltest(sample)
            tests_performed.append({
                'test_name': 'D\'Agostino-Pearson: Normality Test',
                'h0': 'Data is normally distributed',
                'h1': 'Data is not normally distributed',
                'statistic': float(stat),
                'p_value': float(p_value),
                'result': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0',
                'conclusion': 'Data is not normally distributed' if p_value < 0.05 else 'Data follows normal distribution'
            })
        except:
            pass
        
        results['tests'] = tests_performed
        results['total_tests'] = len(tests_performed)
        
        return results
    
    # ================== CORRELATION ANALYSIS ==================
    
    def analyze_correlations(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Comprehensive correlation analysis"""
        results = {
            'dataset': dataset_name,
            'analysis_type': 'correlation'
        }
        
        # Numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out ID-like columns
        exclude_patterns = ['id', 'code', 'encoded', 'year']
        numeric_cols = [c for c in numeric_cols 
                       if not any(p in c.lower() for p in exclude_patterns)]
        
        if len(numeric_cols) < 2:
            results['error'] = 'Insufficient numeric columns'
            return results
        
        # Correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Find strongest correlations
        correlations = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    corr_val = corr_matrix.loc[col1, col2]
                    if not pd.isna(corr_val) and abs(corr_val) > 0.3:
                        correlations.append({
                            'variable_1': col1,
                            'variable_2': col2,
                            'correlation': float(corr_val),
                            'strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate' if abs(corr_val) > 0.5 else 'Weak'
                        })
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        results['significant_correlations'] = correlations[:20]
        results['total_significant'] = len(correlations)
        
        # Key correlations summary
        enrollment_corrs = []
        if 'total_enrollment' in numeric_cols:
            for col in numeric_cols:
                if col != 'total_enrollment':
                    corr_val = corr_matrix.loc['total_enrollment', col]
                    if not pd.isna(corr_val):
                        enrollment_corrs.append({
                            'variable': col,
                            'correlation_with_enrollment': float(corr_val)
                        })
            
            enrollment_corrs.sort(key=lambda x: abs(x['correlation_with_enrollment']), reverse=True)
            results['enrollment_correlations'] = enrollment_corrs[:10]
        
        return results
    
    # ================== MAIN ANALYSIS RUNNER ==================
    
    def run_all_analyses(self, sample_size: int = 100000) -> Dict:
        """Run all analyses on all datasets"""
        all_results = {
            'generated_at': datetime.now().isoformat(),
            'author': 'Shuvam Banerji Seal, Alok Mishra, Aheli Poddar',
            'sample_size': sample_size,
            'datasets': {}
        }
        
        for dataset_type in ['biometric', 'demographic', 'enrollment']:
            print(f"\n{'='*60}")
            print(f"Analyzing {dataset_type.upper()} dataset...")
            print('='*60)
            
            # Load data
            df = self.load_dataset(dataset_type, sample_size)
            if df.empty:
                print(f"⚠ No data for {dataset_type}")
                continue
            
            print(f"✓ Loaded {len(df):,} records")
            
            # Augment data
            df = self.augment_with_reference_data(df)
            df = self.calculate_total_enrollment(df, dataset_type)
            print(f"✓ Augmented with reference data ({len(df.columns)} columns)")
            
            # Run all analyses
            dataset_results = {
                'record_count': len(df),
                'column_count': len(df.columns),
                'time_series': self.analyze_time_series(df, dataset_type),
                'geographic': self.analyze_geographic(df, dataset_type),
                'demographic': self.analyze_demographic(df, dataset_type),
                'socioeconomic': self.analyze_socioeconomic(df, dataset_type),
                'climate': self.analyze_climate(df, dataset_type),
                'hypothesis_tests': self.perform_hypothesis_tests(df, dataset_type),
                'correlations': self.analyze_correlations(df, dataset_type)
            }
            
            all_results['datasets'][dataset_type] = dataset_results
            print(f"✓ Completed all analyses for {dataset_type}")
        
        # Save results
        output_file = self.results_dir / 'comprehensive_analysis_results.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to {output_file}")
        
        return all_results


def main():
    """Main entry point"""
    print("="*60)
    print("UIDAI AADHAAR DATA - COMPREHENSIVE STATISTICAL ANALYSIS")
    print("Authors: Shuvam Banerji Seal, Alok Mishra, Aheli Poddar")
    print("="*60)
    
    analyzer = ComprehensiveAnalyzer()
    results = analyzer.run_all_analyses(sample_size=200000)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    for dataset, data in results.get('datasets', {}).items():
        print(f"\n{dataset.upper()}:")
        print(f"  Records analyzed: {data.get('record_count', 0):,}")
        print(f"  Columns: {data.get('column_count', 0)}")
        
        if 'hypothesis_tests' in data:
            print(f"  Hypothesis tests: {data['hypothesis_tests'].get('total_tests', 0)}")
        
        if 'correlations' in data:
            print(f"  Significant correlations: {data['correlations'].get('total_significant', 0)}")
    
    print("\n✓ Analysis complete!")
    return results


if __name__ == '__main__':
    main()
