"""
Geographic Analysis Module
==========================
Author: Shuvam Banerji Seal's Team
Date: January 2026

Comprehensive geographic analysis for Aadhaar enrollment data.
Includes state/district analysis, pincode patterns, regional comparisons,
and spatial clustering.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

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
from utils.validators import VALID_STATES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# State codes and regions
STATE_REGIONS = {
    'North': ['Jammu & Kashmir', 'Himachal Pradesh', 'Punjab', 'Chandigarh', 
              'Uttarakhand', 'Haryana', 'NCT of Delhi', 'Uttar Pradesh'],
    'South': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Telangana',
              'Puducherry', 'Lakshadweep', 'Andaman & Nicobar Islands'],
    'East': ['Bihar', 'Jharkhand', 'Odisha', 'West Bengal', 'Sikkim'],
    'West': ['Rajasthan', 'Gujarat', 'Maharashtra', 'Goa', 'Dadra & Nagar Haveli',
             'Daman & Diu'],
    'Central': ['Madhya Pradesh', 'Chhattisgarh'],
    'Northeast': ['Assam', 'Arunachal Pradesh', 'Manipur', 'Meghalaya', 'Mizoram',
                  'Nagaland', 'Tripura']
}

# Flatten for reverse lookup
STATE_TO_REGION = {}
for region, states in STATE_REGIONS.items():
    for state in states:
        STATE_TO_REGION[state] = region


class GeographicAnalyzer:
    """
    Geographic Analyzer for Aadhaar enrollment data.
    
    Performs:
    - State-level analysis
    - District-level analysis
    - Pincode pattern analysis
    - Regional comparisons
    - Urban/rural distribution
    - Geographic clustering
    
    Author: Shuvam Banerji Seal's Team
    """
    
    def __init__(self, n_workers: int = CPU_COUNT):
        self.n_workers = n_workers
        self.processor = ParallelProcessor(n_workers=n_workers)
        logger.info(f"GeographicAnalyzer initialized with {n_workers} workers")
    
    def _get_enrollment_total(self, df: pd.DataFrame) -> pd.Series:
        """Calculate total enrollment from available columns."""
        enrollment_cols = [col for col in df.columns if any(
            x in col for x in ['bio_age', 'demo_age', 'age_0_5', 'age_5_17', 'age_18']
        )]
        if enrollment_cols:
            return df[enrollment_cols].sum(axis=1)
        return pd.Series([0] * len(df))
    
    def _assign_region(self, state: str) -> str:
        """Assign region based on state name."""
        return STATE_TO_REGION.get(state, 'Unknown')
    
    def state_analysis(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze enrollment patterns by state.
        
        Args:
            df: Dataset
            dataset_name: Name of dataset
            
        Returns:
            State-level analysis results
        """
        logger.info(f"Analyzing state patterns for {dataset_name}")
        
        df = df.copy()
        df['total_enrollment'] = self._get_enrollment_total(df)
        df['region'] = df['state'].apply(self._assign_region)
        
        # State aggregation
        state_stats = df.groupby('state').agg({
            'total_enrollment': ['sum', 'mean', 'std', 'count'],
            'district': 'nunique',
            'pincode': 'nunique'
        }).reset_index()
        
        state_stats.columns = ['state', 'total_enrollment', 'mean_enrollment', 
                               'std_enrollment', 'record_count', 'unique_districts', 
                               'unique_pincodes']
        
        state_stats['region'] = state_stats['state'].apply(self._assign_region)
        state_stats = state_stats.sort_values('total_enrollment', ascending=False)
        
        # Top and bottom states
        top_10 = state_stats.head(10).to_dict(orient='records')
        bottom_10 = state_stats.tail(10).to_dict(orient='records')
        
        # State ranking
        state_stats['rank'] = range(1, len(state_stats) + 1)
        
        # Calculate market share
        total = state_stats['total_enrollment'].sum()
        state_stats['market_share_pct'] = (state_stats['total_enrollment'] / total * 100).round(2)
        
        # Concentration metrics
        top_5_share = state_stats.head(5)['market_share_pct'].sum()
        top_10_share = state_stats.head(10)['market_share_pct'].sum()
        
        # Gini coefficient for inequality
        sorted_shares = np.sort(state_stats['market_share_pct'].values)
        n = len(sorted_shares)
        cumsum = np.cumsum(sorted_shares)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        results = {
            'dataset_name': dataset_name,
            'summary': {
                'total_states': int(state_stats['state'].nunique()),
                'total_enrollment': int(total),
                'mean_per_state': float(state_stats['total_enrollment'].mean()),
                'std_per_state': float(state_stats['total_enrollment'].std()),
                'top_5_concentration': float(top_5_share),
                'top_10_concentration': float(top_10_share),
                'gini_coefficient': float(gini)
            },
            'top_10_states': top_10,
            'bottom_10_states': bottom_10,
            'state_details': state_stats.to_dict(orient='records')
        }
        
        return results
    
    def regional_analysis(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze enrollment patterns by geographic region.
        
        Args:
            df: Dataset
            dataset_name: Name of dataset
            
        Returns:
            Regional analysis results
        """
        logger.info(f"Analyzing regional patterns for {dataset_name}")
        
        df = df.copy()
        df['total_enrollment'] = self._get_enrollment_total(df)
        df['region'] = df['state'].apply(self._assign_region)
        
        # Regional aggregation
        region_stats = df.groupby('region').agg({
            'total_enrollment': ['sum', 'mean', 'count'],
            'state': 'nunique',
            'district': 'nunique',
            'pincode': 'nunique'
        }).reset_index()
        
        region_stats.columns = ['region', 'total_enrollment', 'mean_enrollment',
                                'record_count', 'unique_states', 'unique_districts',
                                'unique_pincodes']
        
        # Calculate shares
        total = region_stats['total_enrollment'].sum()
        region_stats['share_pct'] = (region_stats['total_enrollment'] / total * 100).round(2)
        region_stats = region_stats.sort_values('total_enrollment', ascending=False)
        
        # Regional comparisons
        results = {
            'dataset_name': dataset_name,
            'regional_stats': region_stats.to_dict(orient='records'),
            'region_ranking': region_stats['region'].tolist(),
            'dominant_region': region_stats.iloc[0]['region'] if len(region_stats) > 0 else None,
            'regions_by_state': {
                region: states for region, states in STATE_REGIONS.items()
            }
        }
        
        return results
    
    def district_analysis(self, df: pd.DataFrame, dataset_name: str, top_n: int = 100) -> Dict[str, Any]:
        """
        Analyze enrollment patterns by district.
        
        Args:
            df: Dataset
            dataset_name: Name of dataset
            top_n: Number of top districts to return
            
        Returns:
            District-level analysis results
        """
        logger.info(f"Analyzing district patterns for {dataset_name}")
        
        df = df.copy()
        df['total_enrollment'] = self._get_enrollment_total(df)
        df['region'] = df['state'].apply(self._assign_region)
        
        # District aggregation
        district_stats = df.groupby(['state', 'district']).agg({
            'total_enrollment': ['sum', 'mean', 'count'],
            'pincode': 'nunique'
        }).reset_index()
        
        district_stats.columns = ['state', 'district', 'total_enrollment', 
                                  'mean_enrollment', 'record_count', 'unique_pincodes']
        
        district_stats['region'] = district_stats['state'].apply(self._assign_region)
        district_stats = district_stats.sort_values('total_enrollment', ascending=False)
        
        # Top and bottom districts
        top_districts = district_stats.head(top_n).to_dict(orient='records')
        bottom_districts = district_stats.tail(top_n).to_dict(orient='records')
        
        # District distribution by state
        districts_per_state = df.groupby('state')['district'].nunique().reset_index()
        districts_per_state.columns = ['state', 'district_count']
        
        # Summary statistics
        results = {
            'dataset_name': dataset_name,
            'summary': {
                'total_districts': int(district_stats['district'].nunique()),
                'avg_districts_per_state': float(districts_per_state['district_count'].mean()),
                'max_districts_in_state': int(districts_per_state['district_count'].max()),
                'min_districts_in_state': int(districts_per_state['district_count'].min())
            },
            'top_districts': top_districts,
            'bottom_districts': bottom_districts,
            'districts_per_state': districts_per_state.to_dict(orient='records')
        }
        
        return results
    
    def pincode_analysis(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze pincode patterns and distribution.
        
        Args:
            df: Dataset
            dataset_name: Name of dataset
            
        Returns:
            Pincode analysis results
        """
        logger.info(f"Analyzing pincode patterns for {dataset_name}")
        
        df = df.copy()
        df['total_enrollment'] = self._get_enrollment_total(df)
        
        # Extract pincode prefix (first 2 digits = postal circle)
        df['pincode_str'] = df['pincode'].astype(str).str.zfill(6)
        df['postal_circle'] = df['pincode_str'].str[:2]
        df['region_code'] = df['pincode_str'].str[0]
        
        # Pincode distribution
        pincode_stats = df.groupby('pincode').agg({
            'total_enrollment': ['sum', 'count'],
            'state': 'first',
            'district': 'first'
        }).reset_index()
        pincode_stats.columns = ['pincode', 'total_enrollment', 'record_count', 'state', 'district']
        pincode_stats = pincode_stats.sort_values('total_enrollment', ascending=False)
        
        # Top pincodes
        top_pincodes = pincode_stats.head(50).to_dict(orient='records')
        
        # Postal circle analysis (first 2 digits)
        circle_stats = df.groupby('postal_circle').agg({
            'total_enrollment': 'sum',
            'pincode': 'nunique',
            'state': 'nunique'
        }).reset_index()
        circle_stats.columns = ['postal_circle', 'total_enrollment', 'unique_pincodes', 'unique_states']
        circle_stats = circle_stats.sort_values('total_enrollment', ascending=False)
        
        # Region code analysis (first digit)
        region_stats = df.groupby('region_code').agg({
            'total_enrollment': 'sum',
            'pincode': 'nunique'
        }).reset_index()
        region_stats.columns = ['region_code', 'total_enrollment', 'unique_pincodes']
        
        # Pincode density
        pincodes_per_district = df.groupby(['state', 'district'])['pincode'].nunique().reset_index()
        pincodes_per_district.columns = ['state', 'district', 'pincode_count']
        
        results = {
            'dataset_name': dataset_name,
            'summary': {
                'total_unique_pincodes': int(df['pincode'].nunique()),
                'avg_enrollment_per_pincode': float(pincode_stats['total_enrollment'].mean()),
                'median_enrollment_per_pincode': float(pincode_stats['total_enrollment'].median()),
                'max_enrollment_pincode': int(pincode_stats['total_enrollment'].max()),
                'min_enrollment_pincode': int(pincode_stats['total_enrollment'].min())
            },
            'top_50_pincodes': top_pincodes,
            'postal_circles': circle_stats.to_dict(orient='records'),
            'region_codes': region_stats.to_dict(orient='records'),
            'pincode_density': {
                'mean_per_district': float(pincodes_per_district['pincode_count'].mean()),
                'max_per_district': int(pincodes_per_district['pincode_count'].max()),
                'distribution': pincodes_per_district.to_dict(orient='records')
            }
        }
        
        return results
    
    def geographic_clustering(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Identify geographic clusters based on enrollment patterns.
        
        Args:
            df: Dataset
            dataset_name: Name of dataset
            
        Returns:
            Clustering analysis results
        """
        logger.info(f"Analyzing geographic clustering for {dataset_name}")
        
        df = df.copy()
        df['total_enrollment'] = self._get_enrollment_total(df)
        df['region'] = df['state'].apply(self._assign_region)
        
        # State-level clustering based on enrollment volume
        state_stats = df.groupby('state')['total_enrollment'].agg(['sum', 'mean', 'count']).reset_index()
        state_stats.columns = ['state', 'total', 'mean', 'count']
        
        # K-means-like clustering using percentiles
        percentiles = [0, 25, 50, 75, 100]
        thresholds = np.percentile(state_stats['total'], percentiles)
        
        def assign_cluster(total):
            if total >= thresholds[3]:
                return 'Very High'
            elif total >= thresholds[2]:
                return 'High'
            elif total >= thresholds[1]:
                return 'Medium'
            else:
                return 'Low'
        
        state_stats['cluster'] = state_stats['total'].apply(assign_cluster)
        
        # Cluster summary
        cluster_summary = state_stats.groupby('cluster').agg({
            'state': list,
            'total': ['sum', 'mean']
        }).reset_index()
        cluster_summary.columns = ['cluster', 'states', 'total_enrollment', 'mean_enrollment']
        
        # Geographic spread analysis
        state_count_by_region = df.groupby('region')['state'].nunique().reset_index()
        state_count_by_region.columns = ['region', 'state_count']
        
        results = {
            'dataset_name': dataset_name,
            'clusters': {
                row['cluster']: {
                    'states': row['states'],
                    'total_enrollment': int(row['total_enrollment']),
                    'mean_enrollment': float(row['mean_enrollment'])
                }
                for _, row in cluster_summary.iterrows()
            },
            'state_clusters': state_stats[['state', 'cluster', 'total']].to_dict(orient='records'),
            'thresholds': {
                'percentiles': percentiles,
                'values': thresholds.tolist()
            }
        }
        
        return results
    
    def run_full_analysis(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run complete geographic analysis on all datasets.
        
        Args:
            datasets: Dictionary of dataset name to DataFrame
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting full geographic analysis")
        
        all_results = {}
        
        with ProgressTracker(len(datasets) * 5, "Geographic Analysis") as tracker:
            for name, df in datasets.items():
                logger.info(f"Processing {name} dataset")
                
                # State analysis
                tracker.step(f"{name}: State analysis")
                all_results[f'{name}_state'] = self.state_analysis(df, name)
                
                # Regional analysis
                tracker.step(f"{name}: Regional analysis")
                all_results[f'{name}_regional'] = self.regional_analysis(df, name)
                
                # District analysis
                tracker.step(f"{name}: District analysis")
                all_results[f'{name}_district'] = self.district_analysis(df, name)
                
                # Pincode analysis
                tracker.step(f"{name}: Pincode analysis")
                all_results[f'{name}_pincode'] = self.pincode_analysis(df, name)
                
                # Clustering
                tracker.step(f"{name}: Clustering")
                all_results[f'{name}_clustering'] = self.geographic_clustering(df, name)
        
        # Save results
        logger.info("Saving geographic analysis results")
        for key, result in all_results.items():
            save_results(result, key, 'geographic', format='json', include_timestamp=False)
        
        # Export for web
        export_to_json(all_results, 'geographic_analysis')
        
        logger.info("Geographic analysis complete")
        
        return all_results


def run_geographic_analysis(nrows: Optional[int] = None) -> Dict[str, Any]:
    """
    Main entry point for geographic analysis.
    
    Args:
        nrows: Number of rows to load (None for all)
        
    Returns:
        Analysis results
    """
    logger.info("=" * 60)
    logger.info("UIDAI Geographic Analysis")
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
    analyzer = GeographicAnalyzer()
    
    with timed_operation("Geographic analysis"):
        results = analyzer.run_full_analysis(datasets)
    
    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Geographic Analysis")
    parser.add_argument('--sample', type=int, default=None, help="Number of rows to sample")
    parser.add_argument('--workers', type=int, default=CPU_COUNT, help="Number of workers")
    
    args = parser.parse_args()
    
    results = run_geographic_analysis(nrows=args.sample)
    
    print(f"\nAnalysis complete. Results saved to results/analysis/geographic/")
