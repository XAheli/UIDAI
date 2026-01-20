#!/usr/bin/env python3
"""Test script for JSON serialization"""

import sys
import json
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

def clean_for_json(obj, depth=0):
    """Clean data for JSON serialization."""
    if depth > 100:
        return str(obj)
    
    if obj is None:
        return None
    if isinstance(obj, (bool,)):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.floating):
        float_val = float(obj)
        if np.isnan(float_val) or np.isinf(float_val):
            return None
        return float_val
    if isinstance(obj, np.ndarray):
        return [clean_for_json(v, depth+1) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.Series,)):
        return clean_for_json(obj.to_dict(), depth+1)
    if isinstance(obj, pd.DataFrame):
        return clean_for_json(obj.to_dict('records'), depth+1)
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v, depth+1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v, depth+1) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except:
        pass
    return str(obj)


if __name__ == "__main__":
    # Test time series
    print("Testing time_series...")
    from time_series.analyzer import run_time_series_analysis
    results = run_time_series_analysis(nrows=1000)
    cleaned = clean_for_json(results)
    json_str = json.dumps(cleaned)
    print(f"  time_series JSON length: {len(json_str)}")
    
    # Test geographic
    print("Testing geographic...")
    from geographic.analyzer import run_geographic_analysis
    results = run_geographic_analysis(nrows=1000)
    cleaned = clean_for_json(results)
    json_str = json.dumps(cleaned)
    print(f"  geographic JSON length: {len(json_str)}")
    
    # Test demographic
    print("Testing demographic...")
    from demographic.analyzer import run_demographic_analysis
    results = run_demographic_analysis(nrows=1000)
    cleaned = clean_for_json(results)
    json_str = json.dumps(cleaned)
    print(f"  demographic JSON length: {len(json_str)}")
    
    # Test statistical
    print("Testing statistical...")
    from statistical.analyzer import run_statistical_analysis
    results = run_statistical_analysis(nrows=1000)
    cleaned = clean_for_json(results)
    json_str = json.dumps(cleaned)
    print(f"  statistical JSON length: {len(json_str)}")
    
    print("\nAll tests passed!")
