"""
Statistical Analysis Module
============================
Author: Shuvam Banerji Seal's Team
Date: January 2026

Comprehensive statistical analysis for Aadhaar enrollment data.
Includes descriptive statistics, hypothesis testing, distribution analysis,
and advanced statistical measures.
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

from utils.parallel import ParallelProcessor, CPU_COUNT
from utils.io_utils import load_dataset, save_results, export_to_json
from utils.progress import timed_operation, ProgressTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Statistical Analyzer for Aadhaar enrollment data.
    
    Performs:
    - Descriptive statistics
    - Distribution analysis
    - Hypothesis testing
    - Correlation analysis
    - Variance analysis
    - Outlier detection
    
    Author: Shuvam Banerji Seal's Team
    """
    
    def __init__(self, n_workers: int = CPU_COUNT):
        self.n_workers = n_workers
        self.processor = ParallelProcessor(n_workers=n_workers)
        logger.info(f"StatisticalAnalyzer initialized with {n_workers} workers")
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get all numeric columns."""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def _get_enrollment_columns(self, df: pd.DataFrame) -> List[str]:
        """Get enrollment-related columns."""
        possible_cols = ['bio_age_5_17', 'bio_age_17_', 'demo_age_5_17', 'demo_age_17_',
                         'age_0_5', 'age_5_17', 'age_18_greater']
        return [col for col in possible_cols if col in df.columns]
    
    def descriptive_statistics(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics.
        
        Args:
            df: Dataset
            dataset_name: Name of dataset
            
        Returns:
            Descriptive statistics results
        """
        logger.info(f"Calculating descriptive statistics for {dataset_name}")
        
        numeric_cols = self._get_numeric_columns(df)
        
        results = {
            'dataset_name': dataset_name,
            'record_count': len(df),
            'column_count': len(df.columns),
            'numeric_columns': numeric_cols,
            'statistics': {}
        }
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            stats_dict = {
                'count': int(len(col_data)),
                'missing': int(df[col].isna().sum()),
                'missing_pct': round(df[col].isna().sum() / len(df) * 100, 2),
                'mean': round(float(col_data.mean()), 4),
                'std': round(float(col_data.std()), 4),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'median': float(col_data.median()),
                'q1': float(col_data.quantile(0.25)),
                'q3': float(col_data.quantile(0.75)),
                'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                'skewness': round(float(col_data.skew()), 4),
                'kurtosis': round(float(col_data.kurtosis()), 4),
                'variance': round(float(col_data.var()), 4),
                'range': float(col_data.max() - col_data.min())
            }
            
            # Coefficient of variation
            if stats_dict['mean'] != 0:
                stats_dict['cv'] = round(stats_dict['std'] / stats_dict['mean'] * 100, 2)
            else:
                stats_dict['cv'] = None
            
            results['statistics'][col] = stats_dict
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        results['categorical_columns'] = {}
        
        for col in cat_cols:
            results['categorical_columns'][col] = {
                'unique_count': int(df[col].nunique()),
                'top_values': df[col].value_counts().head(10).to_dict()
            }
        
        return results
    
    def distribution_analysis(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze data distributions.
        
        Args:
            df: Dataset
            dataset_name: Name of dataset
            
        Returns:
            Distribution analysis results
        """
        logger.info(f"Analyzing distributions for {dataset_name}")
        
        enrollment_cols = self._get_enrollment_columns(df)
        
        if enrollment_cols:
            df['total_enrollment'] = df[enrollment_cols].sum(axis=1)
            analysis_col = 'total_enrollment'
        else:
            # Use first numeric column
            numeric_cols = self._get_numeric_columns(df)
            if numeric_cols:
                analysis_col = numeric_cols[0]
            else:
                return {'dataset_name': dataset_name, 'error': 'No suitable columns'}
        
        data = df[analysis_col].dropna()
        
        # Normality tests
        # Shapiro-Wilk (for small samples)
        sample_size = min(5000, len(data))
        sample = data.sample(sample_size, random_state=42) if len(data) > sample_size else data
        
        shapiro_stat, shapiro_p = stats.shapiro(sample)
        
        # D'Agostino-Pearson (for larger samples)
        dagostino_stat, dagostino_p = stats.normaltest(data)
        
        # Kolmogorov-Smirnov
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        
        # Distribution fitting
        # Try different distributions
        distributions = ['norm', 'lognorm', 'expon', 'gamma']
        best_fit = None
        best_aic = float('inf')
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                
                # Calculate log-likelihood
                log_likelihood = np.sum(dist.logpdf(data, *params))
                k = len(params)
                aic = 2 * k - 2 * log_likelihood
                
                if aic < best_aic:
                    best_aic = aic
                    best_fit = {
                        'distribution': dist_name,
                        'parameters': params,
                        'aic': aic,
                        'log_likelihood': log_likelihood
                    }
            except Exception:
                continue
        
        # Percentile distribution
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = {
            f'p{p}': float(np.percentile(data, p))
            for p in percentiles
        }
        
        # Histogram data (for visualization)
        hist, bin_edges = np.histogram(data, bins=50)
        histogram = {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
        
        results = {
            'dataset_name': dataset_name,
            'analysis_column': analysis_col,
            'sample_size': len(data),
            'normality_tests': {
                'shapiro_wilk': {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                },
                'dagostino_pearson': {
                    'statistic': float(dagostino_stat),
                    'p_value': float(dagostino_p),
                    'is_normal': dagostino_p > 0.05
                },
                'kolmogorov_smirnov': {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_p),
                    'is_normal': ks_p > 0.05
                }
            },
            'best_fit_distribution': best_fit,
            'percentiles': percentile_values,
            'histogram': histogram
        }
        
        return results
    
    def correlation_analysis(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze correlations between variables.
        
        Args:
            df: Dataset
            dataset_name: Name of dataset
            
        Returns:
            Correlation analysis results
        """
        logger.info(f"Analyzing correlations for {dataset_name}")
        
        numeric_cols = self._get_numeric_columns(df)
        
        if len(numeric_cols) < 2:
            return {
                'dataset_name': dataset_name,
                'error': 'Not enough numeric columns for correlation'
            }
        
        # Limit columns to prevent memory issues
        if len(numeric_cols) > 20:
            # Prioritize enrollment and census columns
            priority_cols = [col for col in numeric_cols if any(
                x in col.lower() for x in ['age', 'population', 'literacy', 'ratio', 'enrollment']
            )]
            if len(priority_cols) < 20:
                priority_cols.extend([c for c in numeric_cols if c not in priority_cols][:20-len(priority_cols)])
            numeric_cols = priority_cols[:20]
        
        # Pearson correlation matrix
        corr_matrix = df[numeric_cols].corr(method='pearson')
        
        # Find strong correlations (|r| > 0.5)
        strong_correlations = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Upper triangle only
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.5:
                        strong_correlations.append({
                            'var1': col1,
                            'var2': col2,
                            'correlation': round(float(corr), 4),
                            'strength': 'strong' if abs(corr) > 0.7 else 'moderate'
                        })
        
        # Sort by absolute correlation
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Spearman correlation for non-linear relationships
        spearman_matrix = df[numeric_cols].corr(method='spearman')
        
        results = {
            'dataset_name': dataset_name,
            'columns_analyzed': numeric_cols,
            'pearson_correlation': corr_matrix.to_dict(),
            'spearman_correlation': spearman_matrix.to_dict(),
            'strong_correlations': strong_correlations[:20],
            'total_strong_correlations': len(strong_correlations)
        }
        
        return results
    
    def hypothesis_testing(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Perform various hypothesis tests.
        
        Args:
            df: Dataset
            dataset_name: Name of dataset
            
        Returns:
            Hypothesis testing results
        """
        logger.info(f"Performing hypothesis tests for {dataset_name}")
        
        results = {
            'dataset_name': dataset_name,
            'tests': {}
        }
        
        # Get enrollment total
        enrollment_cols = self._get_enrollment_columns(df)
        if enrollment_cols:
            df['total_enrollment'] = df[enrollment_cols].sum(axis=1)
        else:
            return results
        
        # Test 1: Regional differences (ANOVA)
        if 'state' in df.columns:
            state_groups = [group['total_enrollment'].values 
                          for _, group in df.groupby('state') 
                          if len(group) > 10]
            
            if len(state_groups) > 2:
                f_stat, p_value = stats.f_oneway(*state_groups)
                results['tests']['state_anova'] = {
                    'description': 'One-way ANOVA for state differences',
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'interpretation': 'Significant state-level differences' if p_value < 0.05 
                                    else 'No significant state-level differences'
                }
        
        # Test 2: High vs Low population correlation
        if 'total_population' in df.columns:
            median_pop = df['total_population'].median()
            high_pop = df[df['total_population'] >= median_pop]['total_enrollment']
            low_pop = df[df['total_population'] < median_pop]['total_enrollment']
            
            if len(high_pop) > 0 and len(low_pop) > 0:
                t_stat, p_value = stats.ttest_ind(high_pop, low_pop)
                results['tests']['population_ttest'] = {
                    'description': 'T-test: High vs Low population areas',
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'high_pop_mean': float(high_pop.mean()),
                    'low_pop_mean': float(low_pop.mean())
                }
        
        # Test 3: Literacy correlation
        if 'literacy_rate' in df.columns:
            valid_data = df[['literacy_rate', 'total_enrollment']].dropna()
            if len(valid_data) > 10:
                corr, p_value = stats.pearsonr(
                    valid_data['literacy_rate'],
                    valid_data['total_enrollment']
                )
                results['tests']['literacy_correlation'] = {
                    'description': 'Pearson correlation: Literacy vs Enrollment',
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
        
        # Test 4: Chi-square for categorical associations
        if 'state' in df.columns and 'district' in df.columns:
            # Create contingency table (state vs enrollment quartile)
            try:
                df['enrollment_quartile'] = pd.qcut(
                    df['total_enrollment'], 4, 
                    labels=['Q1', 'Q2', 'Q3', 'Q4'], 
                    duplicates='drop'
                )
                contingency = pd.crosstab(df['state'], df['enrollment_quartile'])
                
                if contingency.size > 0 and contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                    results['tests']['chi_square'] = {
                        'description': 'Chi-square: State vs Enrollment Quartile',
                        'chi2_statistic': float(chi2),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        'significant': p_value < 0.05
                    }
            except Exception as e:
                logger.warning(f"Chi-square test failed: {e}")
        
        # Test 5: Kruskal-Wallis (non-parametric ANOVA)
        if 'state' in df.columns:
            state_groups = [group['total_enrollment'].values 
                          for _, group in df.groupby('state') 
                          if len(group) > 10]
            
            if len(state_groups) > 2:
                h_stat, p_value = stats.kruskal(*state_groups)
                results['tests']['kruskal_wallis'] = {
                    'description': 'Kruskal-Wallis: Non-parametric state comparison',
                    'h_statistic': float(h_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
        
        return results
    
    def outlier_analysis(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Detect and analyze outliers.
        
        Args:
            df: Dataset
            dataset_name: Name of dataset
            
        Returns:
            Outlier analysis results
        """
        logger.info(f"Analyzing outliers for {dataset_name}")
        
        enrollment_cols = self._get_enrollment_columns(df)
        if enrollment_cols:
            df['total_enrollment'] = df[enrollment_cols].sum(axis=1)
            analysis_col = 'total_enrollment'
        else:
            numeric_cols = self._get_numeric_columns(df)
            if numeric_cols:
                analysis_col = numeric_cols[0]
            else:
                return {'dataset_name': dataset_name, 'error': 'No suitable columns'}
        
        data = df[analysis_col].dropna()
        
        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        z_outliers = data[z_scores > 3]
        
        # Modified Z-score (using median)
        median = data.median()
        mad = np.median(np.abs(data - median))
        modified_z = 0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)
        mad_outliers = data[np.abs(modified_z) > 3.5]
        
        # Percentile method
        p1, p99 = data.quantile([0.01, 0.99])
        percentile_outliers = data[(data < p1) | (data > p99)]
        
        results = {
            'dataset_name': dataset_name,
            'analysis_column': analysis_col,
            'total_records': len(data),
            'methods': {
                'iqr': {
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_count': len(iqr_outliers),
                    'outlier_pct': round(len(iqr_outliers) / len(data) * 100, 2)
                },
                'zscore': {
                    'threshold': 3,
                    'outlier_count': len(z_outliers),
                    'outlier_pct': round(len(z_outliers) / len(data) * 100, 2)
                },
                'modified_zscore': {
                    'threshold': 3.5,
                    'outlier_count': len(mad_outliers),
                    'outlier_pct': round(len(mad_outliers) / len(data) * 100, 2)
                },
                'percentile': {
                    'lower_pct': 1,
                    'upper_pct': 99,
                    'outlier_count': len(percentile_outliers),
                    'outlier_pct': round(len(percentile_outliers) / len(data) * 100, 2)
                }
            }
        }
        
        # Analyze where outliers occur
        if 'state' in df.columns:
            df_with_outliers = df.copy()
            df_with_outliers['is_outlier'] = df_with_outliers[analysis_col].apply(
                lambda x: (x < lower_bound) | (x > upper_bound)
            )
            
            outlier_by_state = df_with_outliers.groupby('state')['is_outlier'].agg(['sum', 'count']).reset_index()
            outlier_by_state.columns = ['state', 'outlier_count', 'total_count']
            outlier_by_state['outlier_pct'] = (outlier_by_state['outlier_count'] / outlier_by_state['total_count'] * 100).round(2)
            outlier_by_state = outlier_by_state.sort_values('outlier_pct', ascending=False)
            
            results['outliers_by_state'] = outlier_by_state.head(20).to_dict(orient='records')
        
        return results
    
    def variance_analysis(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze variance across different dimensions.
        
        Args:
            df: Dataset
            dataset_name: Name of dataset
            
        Returns:
            Variance analysis results
        """
        logger.info(f"Analyzing variance for {dataset_name}")
        
        enrollment_cols = self._get_enrollment_columns(df)
        if enrollment_cols:
            df['total_enrollment'] = df[enrollment_cols].sum(axis=1)
        else:
            return {'dataset_name': dataset_name, 'error': 'No enrollment columns'}
        
        results = {
            'dataset_name': dataset_name,
            'overall_variance': float(df['total_enrollment'].var()),
            'overall_std': float(df['total_enrollment'].std()),
            'overall_cv': round(df['total_enrollment'].std() / df['total_enrollment'].mean() * 100, 2)
        }
        
        # Within-state variance
        if 'state' in df.columns:
            state_variance = df.groupby('state')['total_enrollment'].agg(['var', 'std', 'mean']).reset_index()
            state_variance.columns = ['state', 'variance', 'std', 'mean']
            state_variance['cv'] = (state_variance['std'] / state_variance['mean'] * 100).round(2)
            state_variance = state_variance.sort_values('cv', ascending=False)
            
            results['state_variance'] = {
                'highest_cv_states': state_variance.head(10).to_dict(orient='records'),
                'lowest_cv_states': state_variance.tail(10).to_dict(orient='records'),
                'mean_cv': float(state_variance['cv'].mean())
            }
            
            # Between-state vs within-state variance
            between_state_var = df.groupby('state')['total_enrollment'].mean().var()
            within_state_var = df.groupby('state')['total_enrollment'].var().mean()
            
            results['variance_decomposition'] = {
                'between_state_variance': float(between_state_var),
                'within_state_variance': float(within_state_var),
                'variance_ratio': float(between_state_var / within_state_var) if within_state_var > 0 else None
            }
        
        # District-level variance
        if 'district' in df.columns:
            district_variance = df.groupby(['state', 'district'])['total_enrollment'].agg(['var', 'std', 'mean']).reset_index()
            district_variance.columns = ['state', 'district', 'variance', 'std', 'mean']
            district_variance['cv'] = (district_variance['std'] / district_variance['mean'] * 100).round(2)
            
            results['district_variance'] = {
                'highest_cv_districts': district_variance.nlargest(10, 'cv').to_dict(orient='records'),
                'mean_cv': float(district_variance['cv'].mean())
            }
        
        return results
    
    def run_full_analysis(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run complete statistical analysis on all datasets.
        
        Args:
            datasets: Dictionary of dataset name to DataFrame
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting full statistical analysis")
        
        all_results = {}
        
        with ProgressTracker(len(datasets) * 6, "Statistical Analysis") as tracker:
            for name, df in datasets.items():
                logger.info(f"Processing {name} dataset")
                
                # Descriptive statistics
                tracker.step(f"{name}: Descriptive stats")
                all_results[f'{name}_descriptive'] = self.descriptive_statistics(df, name)
                
                # Distribution analysis
                tracker.step(f"{name}: Distribution")
                all_results[f'{name}_distribution'] = self.distribution_analysis(df, name)
                
                # Correlation analysis
                tracker.step(f"{name}: Correlation")
                all_results[f'{name}_correlation'] = self.correlation_analysis(df, name)
                
                # Hypothesis testing
                tracker.step(f"{name}: Hypothesis tests")
                all_results[f'{name}_hypothesis'] = self.hypothesis_testing(df, name)
                
                # Outlier analysis
                tracker.step(f"{name}: Outliers")
                all_results[f'{name}_outliers'] = self.outlier_analysis(df, name)
                
                # Variance analysis
                tracker.step(f"{name}: Variance")
                all_results[f'{name}_variance'] = self.variance_analysis(df, name)
        
        # Save results
        logger.info("Saving statistical analysis results")
        for key, result in all_results.items():
            save_results(result, key, 'statistical', format='json', include_timestamp=False)
        
        # Export for web
        export_to_json(all_results, 'statistical_analysis')
        
        logger.info("Statistical analysis complete")
        
        return all_results


def run_statistical_analysis(nrows: Optional[int] = None) -> Dict[str, Any]:
    """
    Main entry point for statistical analysis.
    
    Args:
        nrows: Number of rows to load (None for all)
        
    Returns:
        Analysis results
    """
    logger.info("=" * 60)
    logger.info("UIDAI Statistical Analysis")
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
    analyzer = StatisticalAnalyzer()
    
    with timed_operation("Statistical analysis"):
        results = analyzer.run_full_analysis(datasets)
    
    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical Analysis")
    parser.add_argument('--sample', type=int, default=None, help="Number of rows to sample")
    parser.add_argument('--workers', type=int, default=CPU_COUNT, help="Number of workers")
    
    args = parser.parse_args()
    
    results = run_statistical_analysis(nrows=args.sample)
    
    print(f"\nAnalysis complete. Results saved to results/analysis/statistical/")
