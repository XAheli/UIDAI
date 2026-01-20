"""
Data Validators
===============
Author: Shuvam Banerji Seal's Team
Date: January 2026

Validation utilities for data quality checks.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
import re

logger = logging.getLogger(__name__)

# Valid Indian states and UTs (36 total)
VALID_STATES = {
    'Andaman And Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh',
    'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh',
    'Dadra And Nagar Haveli And Daman And Diu', 'Delhi', 'Goa', 'Gujarat',
    'Haryana', 'Himachal Pradesh', 'Jammu And Kashmir', 'Jharkhand',
    'Karnataka', 'Kerala', 'Ladakh', 'Lakshadweep', 'Madhya Pradesh',
    'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
    'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
    'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
}

# Required columns for each dataset
REQUIRED_COLUMNS = {
    'biometric': ['date', 'state', 'district', 'pincode', 'bio_age_5_17', 'bio_age_17_'],
    'demographic': ['date', 'state', 'district', 'pincode', 'demo_age_5_17', 'demo_age_17_'],
    'enrollment': ['date', 'state', 'district', 'pincode', 'age_0_5', 'age_5_17', 'age_18_greater']
}


def validate_dataframe(
    df: pd.DataFrame,
    dataset_type: Optional[str] = None,
    raise_errors: bool = False
) -> Dict[str, Any]:
    """
    Validate a dataframe for data quality issues.
    
    Args:
        df: Dataframe to validate
        dataset_type: Type of dataset (biometric, demographic, enrollment)
        raise_errors: Raise exception on critical errors
        
    Returns:
        Validation report
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'issues': [],
        'warnings': [],
        'valid': True
    }
    
    # Check for required columns
    if dataset_type and dataset_type in REQUIRED_COLUMNS:
        required = REQUIRED_COLUMNS[dataset_type]
        missing = [col for col in required if col not in df.columns]
        if missing:
            report['issues'].append(f"Missing required columns: {missing}")
            report['valid'] = False
    
    # Check for null values
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        report['null_values'] = null_cols.to_dict()
        report['warnings'].append(f"Found null values in {len(null_cols)} columns")
    
    # Check for duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        report['duplicates'] = int(dup_count)
        report['warnings'].append(f"Found {dup_count} duplicate rows")
    
    # Check date column
    if 'date' in df.columns:
        try:
            dates = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
            invalid_dates = dates.isna().sum()
            if invalid_dates > 0:
                report['warnings'].append(f"Found {invalid_dates} invalid dates")
        except Exception as e:
            report['warnings'].append(f"Could not validate dates: {e}")
    
    # Data type summary
    report['dtypes'] = df.dtypes.astype(str).to_dict()
    
    # Memory usage
    report['memory_mb'] = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    if raise_errors and not report['valid']:
        raise ValueError(f"Validation failed: {report['issues']}")
    
    return report


def validate_state_names(
    df: pd.DataFrame,
    column: str = 'state'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and report on state names.
    
    Args:
        df: Dataframe with state column
        column: Name of state column
        
    Returns:
        Tuple of (dataframe with validation column, report)
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in dataframe")
    
    # Check against valid states
    df = df.copy()
    df['_state_valid'] = df[column].isin(VALID_STATES)
    
    invalid_states = df[~df['_state_valid']][column].unique()
    
    report = {
        'total_unique_states': df[column].nunique(),
        'valid_states': len(df[df['_state_valid']][column].unique()),
        'invalid_states': list(invalid_states),
        'invalid_count': int((~df['_state_valid']).sum())
    }
    
    return df, report


def validate_pincodes(
    df: pd.DataFrame,
    column: str = 'pincode'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate Indian pincodes.
    
    Args:
        df: Dataframe with pincode column
        column: Name of pincode column
        
    Returns:
        Tuple of (dataframe with validation column, report)
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in dataframe")
    
    df = df.copy()
    
    # Convert to string
    df['_pincode_str'] = df[column].astype(str).str.strip()
    
    # Valid pincode: 6 digits, first digit 1-9
    pincode_pattern = r'^[1-9]\d{5}$'
    df['_pincode_valid'] = df['_pincode_str'].str.match(pincode_pattern, na=False)
    
    invalid_pincodes = df[~df['_pincode_valid']]['_pincode_str'].unique()
    
    # Pincode ranges for validation
    pincode_ranges = {
        'North': (1, 2),
        'West': (3, 4),
        'South': (5, 6),
        'East': (7, 8),
        'Army': (9, 9)
    }
    
    def get_pincode_region(pincode):
        if pd.isna(pincode) or len(str(pincode)) < 1:
            return 'Unknown'
        first_digit = int(str(pincode)[0])
        for region, (start, end) in pincode_ranges.items():
            if start <= first_digit <= end:
                return region
        return 'Unknown'
    
    df['_pincode_region'] = df['_pincode_str'].apply(get_pincode_region)
    
    report = {
        'total_pincodes': df[column].nunique(),
        'valid_pincodes': int(df['_pincode_valid'].sum()),
        'invalid_count': int((~df['_pincode_valid']).sum()),
        'invalid_samples': list(invalid_pincodes[:10]),
        'region_distribution': df['_pincode_region'].value_counts().to_dict()
    }
    
    return df, report


def validate_age_columns(
    df: pd.DataFrame,
    dataset_type: str
) -> Dict[str, Any]:
    """
    Validate age-related columns.
    
    Args:
        df: Dataframe
        dataset_type: Type of dataset
        
    Returns:
        Validation report
    """
    age_cols = {
        'biometric': ['bio_age_5_17', 'bio_age_17_'],
        'demographic': ['demo_age_5_17', 'demo_age_17_'],
        'enrollment': ['age_0_5', 'age_5_17', 'age_18_greater']
    }
    
    if dataset_type not in age_cols:
        return {'error': f'Unknown dataset type: {dataset_type}'}
    
    cols = age_cols[dataset_type]
    report = {}
    
    for col in cols:
        if col not in df.columns:
            report[col] = {'error': 'Column not found'}
            continue
        
        values = pd.to_numeric(df[col], errors='coerce')
        
        report[col] = {
            'total': int(len(values)),
            'non_null': int(values.notna().sum()),
            'min': float(values.min()) if values.notna().any() else None,
            'max': float(values.max()) if values.notna().any() else None,
            'mean': float(values.mean()) if values.notna().any() else None,
            'negative_count': int((values < 0).sum()),
            'zero_count': int((values == 0).sum())
        }
    
    return report


def generate_validation_report(
    df: pd.DataFrame,
    dataset_name: str,
    dataset_type: str
) -> Dict[str, Any]:
    """
    Generate comprehensive validation report.
    
    Args:
        df: Dataframe to validate
        dataset_name: Name for the report
        dataset_type: Type of dataset
        
    Returns:
        Complete validation report
    """
    report = {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'generated_at': pd.Timestamp.now().isoformat(),
        'summary': validate_dataframe(df, dataset_type),
        'state_validation': validate_state_names(df)[1] if 'state' in df.columns else None,
        'pincode_validation': validate_pincodes(df)[1] if 'pincode' in df.columns else None,
        'age_validation': validate_age_columns(df, dataset_type)
    }
    
    # Overall quality score
    issues = report['summary']['issues']
    warnings = report['summary']['warnings']
    
    score = 100
    score -= len(issues) * 20
    score -= len(warnings) * 5
    
    if report['state_validation'] and report['state_validation']['invalid_count'] > 0:
        score -= 10
    
    if report['pincode_validation'] and report['pincode_validation']['invalid_count'] > 0:
        score -= 10
    
    report['quality_score'] = max(0, score)
    
    return report


if __name__ == "__main__":
    # Test validators
    test_df = pd.DataFrame({
        'date': ['01-01-2025', '02-01-2025'],
        'state': ['Maharashtra', 'Unknown State'],
        'district': ['Mumbai', 'Test'],
        'pincode': ['400001', '123'],
        'bio_age_5_17': [10, -5],
        'bio_age_17_': [20, 30]
    })
    
    report = generate_validation_report(test_df, 'test', 'biometric')
    print(f"Quality Score: {report['quality_score']}")
    print(f"Issues: {report['summary']['issues']}")
    print(f"Warnings: {report['summary']['warnings']}")
