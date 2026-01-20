"""
Time Series Analysis Module
===========================
Author: Shuvam Banerji Seal's Team
Date: January 2026

Comprehensive time series analysis for Aadhaar enrollment data.
Includes trend analysis, seasonality detection, forecasting, and anomaly detection.
All operations use multiprocessing for optimal performance.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.parallel import ParallelProcessor, parallel_apply, CPU_COUNT
from utils.io_utils import load_dataset, save_results, export_to_json, get_project_root
from utils.progress import timed_operation, ProgressTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """
    Time Series Analyzer for Aadhaar enrollment data.
    
    Performs:
    - Trend analysis
    - Seasonality detection
    - Day-of-week effects
    - Month-over-month growth
    - Anomaly detection
    - Forecasting preparation
    
    Author: Shuvam Banerji Seal's Team
    """
    
    def __init__(self, n_workers: int = CPU_COUNT):
        self.n_workers = n_workers
        self.processor = ParallelProcessor(n_workers=n_workers)
        self.results = {}
        logger.info(f"TimeSeriesAnalyzer initialized with {n_workers} workers")
    
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date column to datetime."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['date'])
        return df
    
    def _get_enrollment_columns(self, df: pd.DataFrame) -> List[str]:
        """Get enrollment count columns based on dataset type."""
        possible_cols = [
            'bio_age_5_17', 'bio_age_17_',  # Biometric
            'demo_age_5_17', 'demo_age_17_',  # Demographic
            'age_0_5', 'age_5_17', 'age_18_greater'  # Enrollment
        ]
        return [col for col in possible_cols if col in df.columns]
    
    def daily_trends(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze daily enrollment trends.
        
        Args:
            df: Dataset with date and enrollment columns
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with trend analysis results
        """
        logger.info(f"Analyzing daily trends for {dataset_name}")
        
        df = self._parse_dates(df)
        enrollment_cols = self._get_enrollment_columns(df)
        
        if not enrollment_cols:
            raise ValueError("No enrollment columns found")
        
        # Create total enrollment column
        df['total_enrollment'] = df[enrollment_cols].sum(axis=1)
        
        # Daily aggregation
        daily = df.groupby('date').agg({
            'total_enrollment': 'sum',
            'state': 'nunique',
            'district': 'nunique',
            'pincode': 'nunique'
        }).reset_index()
        
        daily.columns = ['date', 'total_enrollment', 'unique_states', 'unique_districts', 'unique_pincodes']
        daily = daily.sort_values('date')
        
        # Calculate statistics
        results = {
            'dataset_name': dataset_name,
            'date_range': {
                'start': daily['date'].min().strftime('%Y-%m-%d'),
                'end': daily['date'].max().strftime('%Y-%m-%d'),
                'total_days': len(daily)
            },
            'enrollment_stats': {
                'total': int(daily['total_enrollment'].sum()),
                'daily_mean': float(daily['total_enrollment'].mean()),
                'daily_std': float(daily['total_enrollment'].std()),
                'daily_min': int(daily['total_enrollment'].min()),
                'daily_max': int(daily['total_enrollment'].max()),
                'daily_median': float(daily['total_enrollment'].median())
            },
            'coverage': {
                'avg_states_per_day': float(daily['unique_states'].mean()),
                'avg_districts_per_day': float(daily['unique_districts'].mean()),
                'avg_pincodes_per_day': float(daily['unique_pincodes'].mean())
            }
        }
        
        # Moving averages
        daily['ma_7'] = daily['total_enrollment'].rolling(window=7, min_periods=1).mean()
        daily['ma_30'] = daily['total_enrollment'].rolling(window=30, min_periods=1).mean()
        
        # Trend direction (linear regression slope)
        x = np.arange(len(daily))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, daily['total_enrollment'])
        
        results['trend'] = {
            'slope': float(slope),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'direction': 'increasing' if slope > 0 else 'decreasing'
        }
        
        # Daily data for export
        results['daily_data'] = daily.to_dict(orient='records')
        
        return results
    
    def seasonality_analysis(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in enrollment data.
        
        Args:
            df: Dataset
            dataset_name: Name of the dataset
            
        Returns:
            Seasonality analysis results
        """
        logger.info(f"Analyzing seasonality for {dataset_name}")
        
        df = self._parse_dates(df)
        enrollment_cols = self._get_enrollment_columns(df)
        df['total_enrollment'] = df[enrollment_cols].sum(axis=1)
        
        # Add time components
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_name'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.month_name()
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Day of week analysis
        dow_stats = df.groupby(['day_of_week', 'day_name'])['total_enrollment'].agg([
            'sum', 'mean', 'std', 'count'
        ]).reset_index()
        dow_stats.columns = ['day_of_week', 'day_name', 'total', 'mean', 'std', 'count']
        
        # Month analysis
        month_stats = df.groupby(['month', 'month_name'])['total_enrollment'].agg([
            'sum', 'mean', 'std', 'count'
        ]).reset_index()
        month_stats.columns = ['month', 'month_name', 'total', 'mean', 'std', 'count']
        
        # Weekend vs weekday
        weekend_stats = df.groupby('is_weekend')['total_enrollment'].agg(['sum', 'mean', 'count']).reset_index()
        
        # ANOVA test for day of week effects
        dow_groups = [group['total_enrollment'].values for _, group in df.groupby('day_of_week')]
        if len(dow_groups) > 1 and all(len(g) > 0 for g in dow_groups):
            f_stat, p_value = stats.f_oneway(*dow_groups)
            dow_effect_significant = p_value < 0.05
        else:
            f_stat, p_value = None, None
            dow_effect_significant = False
        
        results = {
            'dataset_name': dataset_name,
            'day_of_week': {
                'stats': dow_stats.to_dict(orient='records'),
                'anova_f_stat': float(f_stat) if f_stat else None,
                'anova_p_value': float(p_value) if p_value else None,
                'significant_effect': dow_effect_significant,
                'best_day': dow_stats.loc[dow_stats['mean'].idxmax(), 'day_name'],
                'worst_day': dow_stats.loc[dow_stats['mean'].idxmin(), 'day_name']
            },
            'monthly': {
                'stats': month_stats.to_dict(orient='records'),
                'best_month': month_stats.loc[month_stats['mean'].idxmax(), 'month_name'] if len(month_stats) > 0 else None,
                'worst_month': month_stats.loc[month_stats['mean'].idxmin(), 'month_name'] if len(month_stats) > 0 else None
            },
            'weekend_effect': {
                'weekday_mean': float(weekend_stats[weekend_stats['is_weekend'] == False]['mean'].values[0]) if len(weekend_stats[weekend_stats['is_weekend'] == False]) > 0 else None,
                'weekend_mean': float(weekend_stats[weekend_stats['is_weekend'] == True]['mean'].values[0]) if len(weekend_stats[weekend_stats['is_weekend'] == True]) > 0 else None
            }
        }
        
        return results
    
    def growth_analysis(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze growth rates and patterns.
        
        Args:
            df: Dataset
            dataset_name: Name of the dataset
            
        Returns:
            Growth analysis results
        """
        logger.info(f"Analyzing growth for {dataset_name}")
        
        df = self._parse_dates(df)
        enrollment_cols = self._get_enrollment_columns(df)
        df['total_enrollment'] = df[enrollment_cols].sum(axis=1)
        
        # Daily totals
        daily = df.groupby('date')['total_enrollment'].sum().reset_index()
        daily = daily.sort_values('date')
        
        # Calculate growth rates
        daily['pct_change'] = daily['total_enrollment'].pct_change() * 100
        daily['abs_change'] = daily['total_enrollment'].diff()
        
        # Weekly aggregation
        daily['week'] = daily['date'].dt.isocalendar().week
        daily['year'] = daily['date'].dt.year
        weekly = daily.groupby(['year', 'week'])['total_enrollment'].sum().reset_index()
        weekly['wow_growth'] = weekly['total_enrollment'].pct_change() * 100
        
        # Monthly aggregation
        daily['month'] = daily['date'].dt.to_period('M')
        monthly = daily.groupby('month')['total_enrollment'].sum().reset_index()
        monthly['mom_growth'] = monthly['total_enrollment'].pct_change() * 100
        
        # Cumulative enrollment
        daily['cumulative'] = daily['total_enrollment'].cumsum()
        
        results = {
            'dataset_name': dataset_name,
            'daily_growth': {
                'mean_pct_change': float(daily['pct_change'].mean()) if daily['pct_change'].notna().any() else None,
                'std_pct_change': float(daily['pct_change'].std()) if daily['pct_change'].notna().any() else None,
                'positive_days': int((daily['pct_change'] > 0).sum()),
                'negative_days': int((daily['pct_change'] < 0).sum())
            },
            'weekly_growth': {
                'mean_wow_growth': float(weekly['wow_growth'].mean()) if weekly['wow_growth'].notna().any() else None,
                'data': weekly.to_dict(orient='records')
            },
            'monthly_growth': {
                'mean_mom_growth': float(monthly['mom_growth'].mean()) if monthly['mom_growth'].notna().any() else None,
                'data': [
                    {
                        'month': str(row['month']),
                        'total': int(row['total_enrollment']),
                        'mom_growth': float(row['mom_growth']) if pd.notna(row['mom_growth']) else None
                    }
                    for _, row in monthly.iterrows()
                ]
            },
            'cumulative': {
                'total': int(daily['cumulative'].iloc[-1]) if len(daily) > 0 else 0,
                'data': daily[['date', 'cumulative']].to_dict(orient='records')
            }
        }
        
        return results
    
    def anomaly_detection(self, df: pd.DataFrame, dataset_name: str, threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect anomalies in enrollment patterns.
        
        Args:
            df: Dataset
            dataset_name: Name of the dataset
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            Anomaly detection results
        """
        logger.info(f"Detecting anomalies for {dataset_name}")
        
        df = self._parse_dates(df)
        enrollment_cols = self._get_enrollment_columns(df)
        df['total_enrollment'] = df[enrollment_cols].sum(axis=1)
        
        # Daily aggregation
        daily = df.groupby('date')['total_enrollment'].sum().reset_index()
        daily = daily.sort_values('date')
        
        # Calculate z-scores
        mean = daily['total_enrollment'].mean()
        std = daily['total_enrollment'].std()
        daily['z_score'] = (daily['total_enrollment'] - mean) / std
        
        # Identify anomalies
        daily['is_anomaly'] = daily['z_score'].abs() > threshold
        daily['anomaly_type'] = daily.apply(
            lambda row: 'high' if row['z_score'] > threshold else ('low' if row['z_score'] < -threshold else 'normal'),
            axis=1
        )
        
        anomalies = daily[daily['is_anomaly']].copy()
        
        # IQR method
        Q1 = daily['total_enrollment'].quantile(0.25)
        Q3 = daily['total_enrollment'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        daily['is_iqr_anomaly'] = (daily['total_enrollment'] < lower_bound) | (daily['total_enrollment'] > upper_bound)
        
        results = {
            'dataset_name': dataset_name,
            'method': {
                'zscore_threshold': threshold,
                'mean': float(mean),
                'std': float(std),
                'iqr_lower': float(lower_bound),
                'iqr_upper': float(upper_bound)
            },
            'anomalies': {
                'total_count': int(daily['is_anomaly'].sum()),
                'high_anomalies': int((daily['anomaly_type'] == 'high').sum()),
                'low_anomalies': int((daily['anomaly_type'] == 'low').sum()),
                'iqr_anomalies': int(daily['is_iqr_anomaly'].sum()),
                'details': anomalies[['date', 'total_enrollment', 'z_score', 'anomaly_type']].to_dict(orient='records')
            }
        }
        
        return results
    
    def run_full_analysis(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run complete time series analysis on all datasets.
        
        Args:
            datasets: Dictionary of dataset name to DataFrame
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting full time series analysis")
        
        all_results = {}
        
        with ProgressTracker(len(datasets) * 4, "Time Series Analysis") as tracker:
            for name, df in datasets.items():
                logger.info(f"Processing {name} dataset")
                
                # Daily trends
                tracker.step(f"{name}: Daily trends")
                all_results[f'{name}_daily_trends'] = self.daily_trends(df, name)
                
                # Seasonality
                tracker.step(f"{name}: Seasonality")
                all_results[f'{name}_seasonality'] = self.seasonality_analysis(df, name)
                
                # Growth
                tracker.step(f"{name}: Growth")
                all_results[f'{name}_growth'] = self.growth_analysis(df, name)
                
                # Anomalies
                tracker.step(f"{name}: Anomalies")
                all_results[f'{name}_anomalies'] = self.anomaly_detection(df, name)
        
        # Save results
        logger.info("Saving time series analysis results")
        for key, result in all_results.items():
            save_results(result, key, 'time_series', format='json', include_timestamp=False)
        
        # Export for web
        export_to_json(all_results, 'time_series_analysis')
        
        logger.info("Time series analysis complete")
        
        return all_results


def run_time_series_analysis(nrows: Optional[int] = None) -> Dict[str, Any]:
    """
    Main entry point for time series analysis.
    
    Args:
        nrows: Number of rows to load (None for all)
        
    Returns:
        Analysis results
    """
    logger.info("=" * 60)
    logger.info("UIDAI Time Series Analysis")
    logger.info("Author: Shuvam Banerji Seal's Team")
    logger.info("=" * 60)
    
    # Load datasets
    with timed_operation("Loading datasets"):
        datasets = {}
        for name in ['biometric', 'demographic', 'enrollment']:
            try:
                datasets[name] = load_dataset(name, 'augmented', nrows=nrows)
                logger.info(f"Loaded {name}: {len(datasets[name]):,} records")
            except FileNotFoundError:
                logger.warning(f"Dataset not found: {name}")
    
    if not datasets:
        logger.error("No datasets found!")
        return {}
    
    # Run analysis
    analyzer = TimeSeriesAnalyzer()
    
    with timed_operation("Time series analysis"):
        results = analyzer.run_full_analysis(datasets)
    
    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Time Series Analysis")
    parser.add_argument('--sample', type=int, default=None, help="Number of rows to sample")
    parser.add_argument('--workers', type=int, default=CPU_COUNT, help="Number of workers")
    
    args = parser.parse_args()
    
    results = run_time_series_analysis(nrows=args.sample)
    
    print(f"\nAnalysis complete. Results saved to results/analysis/time_series/")
