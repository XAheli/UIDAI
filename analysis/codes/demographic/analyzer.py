"""
Demographic Analysis Module
===========================
Author: Shuvam Banerji Seal's Team
Date: January 2026

Comprehensive demographic analysis for Aadhaar enrollment data.
Includes age group analysis, population correlation, gender ratios,
and demographic trends.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.parallel import ParallelProcessor, parallel_groupby_apply, CPU_COUNT
from utils.io_utils import load_dataset, save_results, export_to_json
from utils.progress import timed_operation, ProgressTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Age group classifications
AGE_GROUP_MAPPING = {
    'bio': {
        'children': 'bio_age_5_17',
        'adults': 'bio_age_17_'
    },
    'demo': {
        'children': 'demo_age_5_17',
        'adults': 'demo_age_17_'
    },
    'enroll': {
        'infants': 'age_0_5',
        'children': 'age_5_17',
        'adults': 'age_18_greater'
    }
}


class DemographicAnalyzer:
    """
    Demographic Analyzer for Aadhaar enrollment data.
    
    Performs:
    - Age group analysis
    - Population correlation
    - Enrollment rate analysis
    - Age distribution patterns
    - Census data integration
    
    Author: Shuvam Banerji Seal's Team
    """
    
    def __init__(self, n_workers: int = CPU_COUNT):
        self.n_workers = n_workers
        self.processor = ParallelProcessor(n_workers=n_workers)
        logger.info(f"DemographicAnalyzer initialized with {n_workers} workers")
    
    def _detect_age_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect age group columns based on column names."""
        columns = df.columns.tolist()
        
        if 'bio_age_5_17' in columns:
            return AGE_GROUP_MAPPING['bio']
        elif 'demo_age_5_17' in columns:
            return AGE_GROUP_MAPPING['demo']
        elif 'age_0_5' in columns:
            return AGE_GROUP_MAPPING['enroll']
        else:
            return {}
    
    def age_group_analysis(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze enrollment patterns by age groups.
        
        Args:
            df: Dataset
            dataset_name: Name of dataset
            
        Returns:
            Age group analysis results
        """
        logger.info(f"Analyzing age groups for {dataset_name}")
        
        df = df.copy()
        age_cols = self._detect_age_columns(df)
        
        if not age_cols:
            return {'dataset_name': dataset_name, 'error': 'No age columns found'}
        
        # Calculate total enrollment
        age_columns = list(age_cols.values())
        df['total_enrollment'] = df[age_columns].sum(axis=1)
        
        # Age group totals
        age_group_totals = {
            name: int(df[col].sum())
            for name, col in age_cols.items()
        }
        
        total = sum(age_group_totals.values())
        
        # Age group percentages
        age_group_pct = {
            name: round(value / total * 100, 2) if total > 0 else 0
            for name, value in age_group_totals.items()
        }
        
        # Age group by state
        state_age_stats = df.groupby('state')[age_columns].sum().reset_index()
        state_age_stats['total'] = state_age_stats[age_columns].sum(axis=1)
        
        for name, col in age_cols.items():
            state_age_stats[f'{name}_pct'] = (state_age_stats[col] / state_age_stats['total'] * 100).round(2)
        
        # Identify states with unusual age distributions
        age_ratios = {}
        for name, col in age_cols.items():
            if 'children' in name or 'infants' in name:
                age_ratios[name] = state_age_stats[f'{name}_pct'].values
        
        if age_ratios:
            combined_youth_pct = np.zeros(len(state_age_stats))
            for name, values in age_ratios.items():
                combined_youth_pct += values
            state_age_stats['youth_pct'] = combined_youth_pct
            
            # States with highest youth enrollment
            youth_states = state_age_stats.nlargest(10, 'youth_pct')[['state', 'youth_pct']].to_dict(orient='records')
            
            # States with highest adult enrollment
            state_age_stats['adult_pct'] = 100 - state_age_stats['youth_pct']
            adult_states = state_age_stats.nlargest(10, 'adult_pct')[['state', 'adult_pct']].to_dict(orient='records')
        else:
            youth_states = []
            adult_states = []
        
        results = {
            'dataset_name': dataset_name,
            'age_columns': age_cols,
            'totals': {
                'by_age_group': age_group_totals,
                'total': total
            },
            'percentages': age_group_pct,
            'by_state': state_age_stats.to_dict(orient='records'),
            'top_youth_states': youth_states,
            'top_adult_states': adult_states
        }
        
        return results
    
    def population_correlation(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze correlation between enrollment and population data.
        
        Args:
            df: Dataset with census columns
            dataset_name: Name of dataset
            
        Returns:
            Population correlation results
        """
        logger.info(f"Analyzing population correlation for {dataset_name}")
        
        df = df.copy()
        
        # Check for census columns
        census_cols = ['total_population', 'male_population', 'female_population',
                       'population_0_6', 'literacy_rate']
        
        available_census = [col for col in census_cols if col in df.columns]
        
        if not available_census:
            return {
                'dataset_name': dataset_name,
                'error': 'No census columns available'
            }
        
        # Calculate enrollment
        age_cols = self._detect_age_columns(df)
        if age_cols:
            age_columns = list(age_cols.values())
            df['total_enrollment'] = df[age_columns].sum(axis=1)
        else:
            df['total_enrollment'] = 1  # Count records
        
        # Aggregate by state
        agg_dict = {'total_enrollment': 'sum'}
        for col in available_census:
            if col in df.columns:
                agg_dict[col] = 'mean'  # Take mean since it's same for all records in state
        
        state_stats = df.groupby('state').agg(agg_dict).reset_index()
        
        # Calculate correlations
        correlations = {}
        for col in available_census:
            if col in state_stats.columns and state_stats[col].notna().any():
                corr, p_value = stats.pearsonr(
                    state_stats['total_enrollment'].fillna(0),
                    state_stats[col].fillna(0)
                )
                correlations[col] = {
                    'correlation': round(float(corr), 4),
                    'p_value': round(float(p_value), 6),
                    'significant': p_value < 0.05
                }
        
        # Enrollment per capita (if population available)
        if 'total_population' in state_stats.columns:
            state_stats['enrollment_per_capita'] = (
                state_stats['total_enrollment'] / state_stats['total_population'] * 1000
            ).round(2)
            
            per_capita_stats = {
                'mean': float(state_stats['enrollment_per_capita'].mean()),
                'median': float(state_stats['enrollment_per_capita'].median()),
                'max': float(state_stats['enrollment_per_capita'].max()),
                'min': float(state_stats['enrollment_per_capita'].min()),
                'top_states': state_stats.nlargest(5, 'enrollment_per_capita')[
                    ['state', 'enrollment_per_capita']
                ].to_dict(orient='records')
            }
        else:
            per_capita_stats = None
        
        results = {
            'dataset_name': dataset_name,
            'available_census_columns': available_census,
            'correlations': correlations,
            'per_capita_stats': per_capita_stats,
            'state_details': state_stats.to_dict(orient='records')
        }
        
        return results
    
    def literacy_enrollment_analysis(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze relationship between literacy rates and enrollment.
        
        Args:
            df: Dataset with literacy data
            dataset_name: Name of dataset
            
        Returns:
            Literacy-enrollment analysis results
        """
        logger.info(f"Analyzing literacy-enrollment relationship for {dataset_name}")
        
        df = df.copy()
        
        if 'literacy_rate' not in df.columns:
            return {
                'dataset_name': dataset_name,
                'error': 'Literacy rate column not available'
            }
        
        # Calculate enrollment
        age_cols = self._detect_age_columns(df)
        if age_cols:
            age_columns = list(age_cols.values())
            df['total_enrollment'] = df[age_columns].sum(axis=1)
        else:
            df['total_enrollment'] = 1
        
        # Group by literacy rate bins
        df['literacy_bin'] = pd.cut(
            df['literacy_rate'],
            bins=[0, 60, 70, 80, 90, 100],
            labels=['<60%', '60-70%', '70-80%', '80-90%', '90%+']
        )
        
        literacy_stats = df.groupby('literacy_bin', observed=True).agg({
            'total_enrollment': ['sum', 'mean', 'count'],
            'state': 'nunique',
            'district': 'nunique'
        }).reset_index()
        
        literacy_stats.columns = ['literacy_bin', 'total_enrollment', 'mean_enrollment',
                                  'record_count', 'unique_states', 'unique_districts']
        
        # State-level analysis
        state_literacy = df.groupby('state').agg({
            'total_enrollment': 'sum',
            'literacy_rate': 'mean'
        }).reset_index()
        
        # Linear regression
        if state_literacy['literacy_rate'].notna().sum() > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                state_literacy['literacy_rate'].fillna(0),
                state_literacy['total_enrollment']
            )
            regression = {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        else:
            regression = None
        
        results = {
            'dataset_name': dataset_name,
            'by_literacy_bin': literacy_stats.to_dict(orient='records'),
            'state_literacy': state_literacy.to_dict(orient='records'),
            'regression': regression
        }
        
        return results
    
    def sex_ratio_analysis(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze enrollment patterns in relation to sex ratios.
        
        Args:
            df: Dataset with sex ratio data
            dataset_name: Name of dataset
            
        Returns:
            Sex ratio analysis results
        """
        logger.info(f"Analyzing sex ratio patterns for {dataset_name}")
        
        df = df.copy()
        
        if 'sex_ratio' not in df.columns:
            # Calculate from male/female populations if available
            if 'male_population' in df.columns and 'female_population' in df.columns:
                df['sex_ratio'] = (df['female_population'] / df['male_population'] * 1000).round(0)
            else:
                return {
                    'dataset_name': dataset_name,
                    'error': 'Sex ratio data not available'
                }
        
        # Calculate enrollment
        age_cols = self._detect_age_columns(df)
        if age_cols:
            age_columns = list(age_cols.values())
            df['total_enrollment'] = df[age_columns].sum(axis=1)
        else:
            df['total_enrollment'] = 1
        
        # Classify sex ratio
        df['sex_ratio_category'] = pd.cut(
            df['sex_ratio'],
            bins=[0, 900, 950, 1000, 1050, 2000],
            labels=['Very Low (<900)', 'Low (900-950)', 'Near Equal (950-1000)',
                    'Female Surplus (1000-1050)', 'High Female (>1050)']
        )
        
        # Stats by category
        category_stats = df.groupby('sex_ratio_category', observed=True).agg({
            'total_enrollment': ['sum', 'mean'],
            'state': 'nunique',
            'district': 'nunique'
        }).reset_index()
        
        category_stats.columns = ['category', 'total_enrollment', 'mean_enrollment',
                                  'unique_states', 'unique_districts']
        
        # State-level sex ratio
        state_sex_ratio = df.groupby('state').agg({
            'total_enrollment': 'sum',
            'sex_ratio': 'mean'
        }).reset_index()
        state_sex_ratio = state_sex_ratio.sort_values('sex_ratio', ascending=False)
        
        results = {
            'dataset_name': dataset_name,
            'by_category': category_stats.to_dict(orient='records'),
            'state_sex_ratio': state_sex_ratio.to_dict(orient='records'),
            'top_sex_ratio_states': state_sex_ratio.head(10).to_dict(orient='records'),
            'lowest_sex_ratio_states': state_sex_ratio.tail(10).to_dict(orient='records')
        }
        
        return results
    
    def age_distribution_trends(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze age distribution trends over time.
        
        Args:
            df: Dataset
            dataset_name: Name of dataset
            
        Returns:
            Age distribution trends results
        """
        logger.info(f"Analyzing age distribution trends for {dataset_name}")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['date'])
        
        age_cols = self._detect_age_columns(df)
        if not age_cols:
            return {
                'dataset_name': dataset_name,
                'error': 'No age columns found'
            }
        
        age_columns = list(age_cols.values())
        
        # Monthly trends by age group
        df['month'] = df['date'].dt.to_period('M')
        monthly_age = df.groupby('month')[age_columns].sum().reset_index()
        monthly_age['month'] = monthly_age['month'].astype(str)
        
        # Calculate proportions over time
        for col in age_columns:
            monthly_age[f'{col}_pct'] = (
                monthly_age[col] / monthly_age[age_columns].sum(axis=1) * 100
            ).round(2)
        
        # Trend direction for each age group
        trends = {}
        for col in age_columns:
            if len(monthly_age) > 1:
                x = np.arange(len(monthly_age))
                slope, _, r_value, p_value, _ = stats.linregress(x, monthly_age[col])
                trends[col] = {
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'direction': 'increasing' if slope > 0 else 'decreasing',
                    'significant': p_value < 0.05
                }
        
        results = {
            'dataset_name': dataset_name,
            'age_columns': age_cols,
            'monthly_trends': monthly_age.to_dict(orient='records'),
            'trend_directions': trends
        }
        
        return results
    
    def run_full_analysis(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run complete demographic analysis on all datasets.
        
        Args:
            datasets: Dictionary of dataset name to DataFrame
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting full demographic analysis")
        
        all_results = {}
        
        with ProgressTracker(len(datasets) * 5, "Demographic Analysis") as tracker:
            for name, df in datasets.items():
                logger.info(f"Processing {name} dataset")
                
                # Age group analysis
                tracker.step(f"{name}: Age groups")
                all_results[f'{name}_age_groups'] = self.age_group_analysis(df, name)
                
                # Population correlation
                tracker.step(f"{name}: Population correlation")
                all_results[f'{name}_population'] = self.population_correlation(df, name)
                
                # Literacy analysis
                tracker.step(f"{name}: Literacy")
                all_results[f'{name}_literacy'] = self.literacy_enrollment_analysis(df, name)
                
                # Sex ratio analysis
                tracker.step(f"{name}: Sex ratio")
                all_results[f'{name}_sex_ratio'] = self.sex_ratio_analysis(df, name)
                
                # Age trends
                tracker.step(f"{name}: Age trends")
                all_results[f'{name}_age_trends'] = self.age_distribution_trends(df, name)
        
        # Save results
        logger.info("Saving demographic analysis results")
        for key, result in all_results.items():
            save_results(result, key, 'demographic', format='json', include_timestamp=False)
        
        # Export for web
        export_to_json(all_results, 'demographic_analysis')
        
        logger.info("Demographic analysis complete")
        
        return all_results


def run_demographic_analysis(nrows: Optional[int] = None) -> Dict[str, Any]:
    """
    Main entry point for demographic analysis.
    
    Args:
        nrows: Number of rows to load (None for all)
        
    Returns:
        Analysis results
    """
    logger.info("=" * 60)
    logger.info("UIDAI Demographic Analysis")
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
    analyzer = DemographicAnalyzer()
    
    with timed_operation("Demographic analysis"):
        results = analyzer.run_full_analysis(datasets)
    
    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demographic Analysis")
    parser.add_argument('--sample', type=int, default=None, help="Number of rows to sample")
    parser.add_argument('--workers', type=int, default=CPU_COUNT, help="Number of workers")
    
    args = parser.parse_args()
    
    results = run_demographic_analysis(nrows=args.sample)
    
    print(f"\nAnalysis complete. Results saved to results/analysis/demographic/")
