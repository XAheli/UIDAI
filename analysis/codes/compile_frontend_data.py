#!/usr/bin/env python3
"""
Compile Frontend Data
=====================
Author: Shuvam Banerji Seal's Team
Date: January 2026

Compiles all analysis results into frontend-compatible JSON files.
Handles NaN/Infinity values and ensures proper data structure.
"""

import os
import sys
import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd
import shutil

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def clean_value(val: Any, depth: int = 0, max_depth: int = 100) -> Any:
    """Clean a value for JSON serialization, handling NaN, Infinity, and complex types."""
    if depth > max_depth:
        return str(val)
    
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, np.floating):
        float_val = float(val)
        if math.isnan(float_val) or math.isinf(float_val):
            return None
        return float_val
    if isinstance(val, np.ndarray):
        return [clean_value(v, depth + 1, max_depth) for v in val.tolist()]
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    if isinstance(val, pd.Series):
        return clean_value(val.to_dict(), depth + 1, max_depth)
    if isinstance(val, pd.DataFrame):
        return clean_value(val.to_dict('records'), depth + 1, max_depth)
    if isinstance(val, dict):
        return {str(k): clean_value(v, depth + 1, max_depth) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [clean_value(v, depth + 1, max_depth) for v in val]
    
    # Check for pd.NA and other pandas null types
    try:
        if pd.isna(val):
            return None
    except (ValueError, TypeError):
        pass
    
    return str(val)


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load a JSON file and clean its values."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return clean_value(data)
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return {}


def compile_time_series_data(results_dir: Path) -> Dict[str, Any]:
    """Compile time series analysis results."""
    ts_dir = results_dir / "analysis" / "time_series"
    compiled = {}
    
    for dataset in ['biometric', 'demographic', 'enrollment']:
        # Load daily trends
        daily_file = ts_dir / f"{dataset}_daily_trends.json"
        if daily_file.exists():
            data = load_json_file(daily_file)
            compiled[f"{dataset}_daily_trends"] = data
        
        # Load seasonality
        season_file = ts_dir / f"{dataset}_seasonality.json"
        if season_file.exists():
            data = load_json_file(season_file)
            compiled[f"{dataset}_seasonality"] = data
        
        # Load anomalies
        anomaly_file = ts_dir / f"{dataset}_anomalies.json"
        if anomaly_file.exists():
            data = load_json_file(anomaly_file)
            compiled[f"{dataset}_anomalies"] = data
        
        # Load growth metrics
        growth_file = ts_dir / f"{dataset}_growth.json"
        if growth_file.exists():
            data = load_json_file(growth_file)
            compiled[f"{dataset}_growth"] = data
    
    # Also load any other JSON files in the directory
    if ts_dir.exists():
        for filepath in ts_dir.glob("*.json"):
            key = filepath.stem
            if key not in compiled:
                data = load_json_file(filepath)
                compiled[key] = data
    
    # Add metadata
    compiled['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'author': "Shuvam Banerji Seal's Team",
        'description': 'Time series analysis including trends, seasonality, anomalies, and growth metrics'
    }
    
    # Add interpretations
    compiled['interpretations'] = {
        'daily_trends': {
            'title': 'Daily Enrollment Trends',
            'description': 'This chart shows the daily enrollment counts across all Aadhaar enrollment centers. The moving averages (7-day and 30-day) help smooth out daily fluctuations to reveal underlying trends.',
            'insights': [
                'Higher enrollments typically occur on weekdays',
                'Seasonal patterns may reflect school enrollment drives',
                'Anomalies often correspond to policy changes or technical issues'
            ]
        },
        'seasonality': {
            'title': 'Seasonality Patterns',
            'description': 'Analysis of recurring patterns in enrollment data based on day of week and month.',
            'insights': [
                'Weekend enrollments are typically lower due to office hours',
                'End-of-month peaks may indicate deadline-driven enrollments',
                'Festival periods often show reduced activity'
            ]
        },
        'anomalies': {
            'title': 'Anomaly Detection',
            'description': 'Statistical identification of unusual enrollment patterns using z-score and IQR methods.',
            'insights': [
                'High anomalies may indicate successful enrollment drives',
                'Low anomalies could signal system outages or holidays',
                'Consistent anomalies warrant investigation'
            ]
        }
    }
    
    return compiled


def compile_statistical_data(results_dir: Path) -> Dict[str, Any]:
    """Compile statistical analysis results."""
    stat_dir = results_dir / "analysis" / "statistical"
    compiled = {}
    
    for dataset in ['biometric', 'demographic', 'enrollment']:
        # Load all statistical files
        for analysis_type in ['descriptive', 'distribution', 'correlation', 'hypothesis', 'outliers', 'variance']:
            filepath = stat_dir / f"{dataset}_{analysis_type}.json"
            if filepath.exists():
                data = load_json_file(filepath)
                compiled[f"{dataset}_{analysis_type}"] = data
    
    # Add metadata
    compiled['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'author': "Shuvam Banerji Seal's Team",
        'description': 'Comprehensive statistical analysis including descriptive statistics, distributions, correlations, and hypothesis testing'
    }
    
    # Add interpretations
    compiled['interpretations'] = {
        'descriptive': {
            'title': 'Descriptive Statistics',
            'description': 'Summary statistics for all numeric variables including central tendency, dispersion, and shape metrics.',
            'metrics_explained': {
                'mean': 'Average value across all records',
                'median': 'Middle value when sorted (50th percentile)',
                'std': 'Standard deviation - measure of spread',
                'skewness': 'Measure of asymmetry (0 = symmetric, positive = right tail, negative = left tail)',
                'kurtosis': 'Measure of tail heaviness (3 = normal, >3 = heavy tails)',
                'cv': 'Coefficient of variation - relative variability (std/mean * 100)'
            }
        },
        'distribution': {
            'title': 'Distribution Analysis',
            'description': 'Analysis of data distribution shape, normality testing, and best-fit distribution identification.',
            'insights': [
                'Enrollment data typically follows a right-skewed distribution',
                'Non-normal distributions may require non-parametric tests',
                'Log transformation often normalizes enrollment counts'
            ]
        },
        'correlation': {
            'title': 'Correlation Analysis',
            'description': 'Pearson and Spearman correlations between variables to identify relationships.',
            'interpretation_guide': {
                '0.9 to 1.0': 'Very strong positive correlation',
                '0.7 to 0.9': 'Strong positive correlation',
                '0.5 to 0.7': 'Moderate positive correlation',
                '0.3 to 0.5': 'Weak positive correlation',
                '0 to 0.3': 'Negligible correlation'
            }
        },
        'hypothesis': {
            'title': 'Hypothesis Testing',
            'description': 'Statistical tests to validate assumptions about enrollment patterns.',
            'test_explanations': {
                't-test': 'Compares means between two groups',
                'ANOVA': 'Compares means across multiple groups',
                'chi-square': 'Tests independence between categorical variables',
                'normality': 'Tests if data follows normal distribution'
            },
            'p_value_interpretation': 'p < 0.05 indicates statistically significant results'
        },
        'outliers': {
            'title': 'Outlier Detection',
            'description': 'Identification of unusual values using IQR and Z-score methods.',
            'methods': {
                'IQR': 'Values below Q1-1.5*IQR or above Q3+1.5*IQR',
                'Z-score': 'Values more than 3 standard deviations from mean'
            }
        },
        'variance': {
            'title': 'Variance Analysis',
            'description': 'Analysis of variability between states and within states.',
            'insights': [
                'High between-state variance indicates regional disparities',
                'High within-state variance suggests local inconsistencies',
                'CV helps compare variability across different scales'
            ]
        }
    }
    
    return compiled


def compile_geographic_data(results_dir: Path) -> Dict[str, Any]:
    """Compile geographic analysis results."""
    geo_dir = results_dir / "analysis" / "geographic"
    compiled = {}
    
    for dataset in ['biometric', 'demographic', 'enrollment']:
        # Load all geographic files with various naming patterns
        for analysis_type in ['state_analysis', 'regional_analysis', 'district_analysis', 
                              'urban_rural', 'zone_analysis', 'coverage',
                              'state', 'regional', 'district', 'pincode', 'clustering']:
            filepath = geo_dir / f"{dataset}_{analysis_type}.json"
            if filepath.exists():
                data = load_json_file(filepath)
                compiled[f"{dataset}_{analysis_type}"] = data
    
    # Also load any other JSON files in the directory
    if geo_dir.exists():
        for filepath in geo_dir.glob("*.json"):
            key = filepath.stem
            if key not in compiled:
                data = load_json_file(filepath)
                compiled[key] = data
    
    # Add metadata
    compiled['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'author': "Shuvam Banerji Seal's Team",
        'description': 'Geographic analysis including state-wise, regional, and district-level patterns'
    }
    
    # Add interpretations
    compiled['interpretations'] = {
        'state_analysis': {
            'title': 'State-wise Analysis',
            'description': 'Comparison of Aadhaar enrollment across all Indian states and union territories.',
            'key_findings': [
                'Enrollment correlates strongly with population',
                'Per capita enrollment rates reveal true coverage',
                'Smaller states may have higher coverage rates'
            ]
        },
        'regional_analysis': {
            'title': 'Regional Analysis',
            'description': 'Comparison across regions: North, South, East, West, Central, and Northeast.',
            'insights': [
                'Southern states generally show higher enrollment rates',
                'Northeast region faces unique challenges',
                'Urban concentration affects regional patterns'
            ]
        },
        'urban_rural': {
            'title': 'Urban vs Rural Analysis',
            'description': 'Comparison of enrollment patterns between urban and rural areas.',
            'insights': [
                'Urban areas have higher enrollment density',
                'Rural areas have larger absolute numbers',
                'Accessibility affects enrollment in rural regions'
            ]
        },
        'zone_analysis': {
            'title': 'Zone Analysis',
            'description': 'Analysis by climate and earthquake zones.',
            'insights': [
                'Climate zones affect enrollment seasonality',
                'Earthquake-prone areas may have documentation drives',
                'Coastal vs inland patterns differ'
            ]
        }
    }
    
    return compiled


def compile_demographic_data(results_dir: Path) -> Dict[str, Any]:
    """Compile demographic analysis results."""
    demo_dir = results_dir / "analysis" / "demographic"
    compiled = {}
    
    for dataset in ['biometric', 'demographic', 'enrollment']:
        # Load all demographic files with various naming patterns
        for analysis_type in ['age_distribution', 'gender_analysis', 'population_correlation',
                              'literacy_correlation', 'socioeconomic',
                              'age_groups', 'age_trends', 'literacy', 'population', 'sex_ratio']:
            filepath = demo_dir / f"{dataset}_{analysis_type}.json"
            if filepath.exists():
                data = load_json_file(filepath)
                compiled[f"{dataset}_{analysis_type}"] = data
    
    # Also load any other JSON files in the directory
    if demo_dir.exists():
        for filepath in demo_dir.glob("*.json"):
            key = filepath.stem
            if key not in compiled:
                data = load_json_file(filepath)
                compiled[key] = data
    
    # Add metadata
    compiled['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'author': "Shuvam Banerji Seal's Team",
        'description': 'Demographic analysis including age, gender, population, and socioeconomic factors'
    }
    
    # Add interpretations
    compiled['interpretations'] = {
        'age_distribution': {
            'title': 'Age Distribution Analysis',
            'description': 'Analysis of enrollment patterns across different age groups.',
            'age_groups': {
                '0-5': 'Children requiring registration',
                '5-17': 'School-age children',
                '17+': 'Adults'
            },
            'insights': [
                'Adult enrollments dominate the dataset',
                'Child enrollment shows different patterns',
                'Age group proportions vary by state'
            ]
        },
        'gender_analysis': {
            'title': 'Gender Analysis',
            'description': 'Analysis of enrollment by gender using sex ratio as proxy.',
            'insights': [
                'States with higher sex ratios show different patterns',
                'Gender disparity in enrollment needs monitoring',
                'Urban areas tend toward gender parity'
            ]
        },
        'population_correlation': {
            'title': 'Population Correlation',
            'description': 'Relationship between population and enrollment.',
            'insights': [
                'Strong positive correlation expected',
                'Per capita rates reveal true coverage',
                'Population density affects enrollment access'
            ]
        },
        'socioeconomic': {
            'title': 'Socioeconomic Analysis',
            'description': 'Impact of income, HDI, and literacy on enrollment.',
            'insights': [
                'Higher HDI correlates with better coverage',
                'Literacy strongly predicts enrollment success',
                'Income shows moderate correlation'
            ]
        }
    }
    
    return compiled


def compile_ml_results(results_dir: Path) -> Dict[str, Any]:
    """Compile ML model results."""
    ml_dir = results_dir / "models"
    compiled = {}
    
    # Load any model result files
    for filepath in ml_dir.glob("*.json"):
        data = load_json_file(filepath)
        key = filepath.stem
        compiled[key] = data
    
    # Load predictions if available
    pred_dir = results_dir / "analysis" / "ml_predictions"
    if pred_dir.exists():
        for filepath in pred_dir.glob("*.json"):
            data = load_json_file(filepath)
            compiled[filepath.stem] = data
    
    # Add metadata and explanations
    compiled['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'author': "Shuvam Banerji Seal's Team",
        'description': 'Machine learning model results including classification, regression, and time series forecasting'
    }
    
    compiled['model_explanations'] = {
        'logistic_regression': {
            'name': 'Logistic Regression',
            'description': 'Linear model for classification, predicts probability of class membership.',
            'strengths': ['Fast training', 'Interpretable coefficients', 'Works well for linearly separable data'],
            'limitations': ['Assumes linear decision boundary', 'May underperform with complex patterns']
        },
        'random_forest': {
            'name': 'Random Forest',
            'description': 'Ensemble of decision trees that reduces overfitting through averaging.',
            'strengths': ['Handles non-linear relationships', 'Feature importance', 'Robust to outliers'],
            'limitations': ['Less interpretable', 'Can be slow for large datasets']
        },
        'xgboost': {
            'name': 'XGBoost',
            'description': 'Gradient boosting algorithm that builds trees sequentially to correct errors.',
            'strengths': ['Often best performance', 'Handles missing values', 'Regularization built-in'],
            'limitations': ['Many hyperparameters', 'Risk of overfitting']
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting',
            'description': 'Ensemble method that builds models sequentially to minimize loss.',
            'strengths': ['High accuracy', 'Handles mixed data types', 'Flexible loss functions'],
            'limitations': ['Sensitive to outliers', 'Longer training time']
        },
        'svm': {
            'name': 'Support Vector Machine',
            'description': 'Finds optimal hyperplane to separate classes with maximum margin.',
            'strengths': ['Effective in high dimensions', 'Kernel trick for non-linear', 'Memory efficient'],
            'limitations': ['Slow for large datasets', 'Sensitive to feature scaling']
        },
        'knn': {
            'name': 'K-Nearest Neighbors',
            'description': 'Classifies based on majority vote of k nearest training examples.',
            'strengths': ['Simple and intuitive', 'No training phase', 'Naturally handles multi-class'],
            'limitations': ['Slow prediction', 'Sensitive to irrelevant features']
        },
        'time_series_arima': {
            'name': 'ARIMA',
            'description': 'AutoRegressive Integrated Moving Average for time series forecasting.',
            'strengths': ['Good for stationary series', 'Well-understood theory', 'Confidence intervals'],
            'limitations': ['Requires stationarity', 'Linear only', 'Single series']
        }
    }
    
    compiled['evaluation_metrics'] = {
        'accuracy': 'Proportion of correct predictions',
        'precision': 'True positives / (True positives + False positives)',
        'recall': 'True positives / (True positives + False negatives)',
        'f1_score': 'Harmonic mean of precision and recall',
        'roc_auc': 'Area under the ROC curve - measures discrimination ability',
        'mae': 'Mean Absolute Error - average absolute difference',
        'rmse': 'Root Mean Square Error - penalizes large errors',
        'r2': 'R-squared - proportion of variance explained'
    }
    
    return compiled


def save_compiled_data(data: Dict[str, Any], name: str, output_dir: Path) -> Path:
    """Save compiled data to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{name}.json"
    
    # Clean the data before saving
    cleaned_data = clean_value(data)
    
    with open(filepath, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    logger.info(f"Saved {name}.json ({filepath.stat().st_size / 1024:.1f} KB)")
    return filepath


def main():
    """Main function to compile all frontend data."""
    project_root = get_project_root()
    results_dir = project_root / "results"
    frontend_data_dir = project_root / "web" / "frontend" / "public" / "data"
    
    logger.info("=" * 60)
    logger.info("Compiling Frontend Data")
    logger.info("=" * 60)
    
    # Compile each category
    categories = {
        'time_series': compile_time_series_data,
        'statistical': compile_statistical_data,
        'geographic': compile_geographic_data,
        'demographic': compile_demographic_data,
        'ml_results': compile_ml_results
    }
    
    for name, compile_func in categories.items():
        logger.info(f"\nCompiling {name}...")
        try:
            data = compile_func(results_dir)
            if data:
                save_compiled_data(data, name, frontend_data_dir)
            else:
                logger.warning(f"No data compiled for {name}")
        except Exception as e:
            logger.error(f"Failed to compile {name}: {e}")
    
    # Create summary/index file
    logger.info("\nCreating index.json...")
    index_data = {
        'generated_at': datetime.now().isoformat(),
        'author': "Shuvam Banerji Seal's Team",
        'project': 'UIDAI Aadhaar Data Analysis',
        'datasets': {
            'biometric': {
                'description': 'Biometric authentication data',
                'records': 'See analysis files'
            },
            'demographic': {
                'description': 'Demographic authentication data',
                'records': 'See analysis files'
            },
            'enrollment': {
                'description': 'Enrollment data',
                'records': 'See analysis files'
            }
        },
        'available_analyses': list(categories.keys()),
        'interpretations_included': True
    }
    save_compiled_data(index_data, 'index', frontend_data_dir)
    
    # Create analysis summary
    logger.info("\nCreating analysis_summary.json...")
    summary_data = {
        'generated_at': datetime.now().isoformat(),
        'author': "Shuvam Banerji Seal's Team",
        'total_analyses': len(categories),
        'analysis_types': {
            'time_series': ['Daily trends', 'Seasonality', 'Anomalies', 'Growth metrics'],
            'statistical': ['Descriptive stats', 'Distribution', 'Correlation', 'Hypothesis tests', 'Outliers', 'Variance'],
            'geographic': ['State analysis', 'Regional analysis', 'District analysis', 'Urban/Rural', 'Zone analysis'],
            'demographic': ['Age distribution', 'Gender analysis', 'Population correlation', 'Socioeconomic'],
            'ml_models': ['Classification', 'Regression', 'Time series forecasting', 'Anomaly detection']
        }
    }
    save_compiled_data(summary_data, 'analysis_summary', frontend_data_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("Frontend data compilation complete!")
    logger.info(f"Files saved to: {frontend_data_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
