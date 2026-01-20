"""
I/O Utilities for Data Analysis
===============================
Author: Shuvam Banerji Seal's Team
Date: January 2026

File I/O utilities for loading, saving, and exporting data.
Supports CSV, JSON, and Parquet formats with compression.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    # Navigate up from utils/ -> codes/ -> analysis/ -> project_root/
    return current.parent.parent.parent.parent


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(
    name: str,
    dataset_type: str = "augmented",
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    parse_dates: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load a dataset by name and type.
    
    Args:
        name: Dataset name (biometric, demographic, enrollment)
        dataset_type: Type (augmented, cleaned, raw)
        nrows: Number of rows to load (None for all)
        usecols: Columns to load (None for all)
        dtype: Column dtypes
        parse_dates: Columns to parse as dates
        
    Returns:
        Loaded dataframe
    """
    project_root = get_project_root()
    
    # Map dataset names to files
    dataset_paths = {
        "augmented": {
            "biometric": project_root / "Dataset/augmented/biometric_augmented.csv",
            "demographic": project_root / "Dataset/augmented/demographic_augmented.csv",
            "enrollment": project_root / "Dataset/augmented/enrollment_augmented.csv",
        },
        "cleaned": {
            "biometric": project_root / "Dataset/cleaned/biometric/final_cleaned_biometric.csv",
            "demographic": project_root / "Dataset/cleaned/demographic/final_cleaned_demographic.csv",
            "enrollment": project_root / "Dataset/cleaned/enrollment/final_cleaned_enrollment.csv",
        }
    }
    
    if dataset_type not in dataset_paths:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    if name not in dataset_paths[dataset_type]:
        raise ValueError(f"Unknown dataset name: {name}")
    
    filepath = dataset_paths[dataset_type][name]
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    logger.info(f"Loading {name} ({dataset_type}) from {filepath}")
    
    df = pd.read_csv(
        filepath,
        nrows=nrows,
        usecols=usecols,
        dtype=dtype,
        parse_dates=parse_dates,
        low_memory=False
    )
    
    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
    
    return df


def load_all_datasets(
    dataset_type: str = "augmented",
    nrows: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load all datasets.
    
    Args:
        dataset_type: Type (augmented, cleaned)
        nrows: Number of rows per dataset
        
    Returns:
        Dictionary of dataframes
    """
    datasets = {}
    for name in ["biometric", "demographic", "enrollment"]:
        try:
            datasets[name] = load_dataset(name, dataset_type, nrows)
        except FileNotFoundError:
            logger.warning(f"Dataset not found: {name}")
    
    return datasets


def save_results(
    data: Union[pd.DataFrame, Dict[str, Any]],
    name: str,
    category: str,
    format: str = "csv",
    include_timestamp: bool = True
) -> Path:
    """
    Save analysis results to file.
    
    Args:
        data: Data to save (DataFrame or dict)
        name: File name (without extension)
        category: Category folder (time_series, geographic, etc.)
        format: Output format (csv, json, parquet)
        include_timestamp: Add timestamp to filename
        
    Returns:
        Path to saved file
    """
    project_root = get_project_root()
    output_dir = project_root / "results" / "analysis" / category
    ensure_dir(output_dir)
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}"
    else:
        filename = name
    
    if format == "csv":
        filepath = output_dir / f"{filename}.csv"
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            pd.DataFrame(data).to_csv(filepath, index=False)
    
    elif format == "json":
        filepath = output_dir / f"{filename}.json"
        if isinstance(data, pd.DataFrame):
            data.to_json(filepath, orient="records", indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
    
    elif format == "parquet":
        filepath = output_dir / f"{filename}.parquet"
        if isinstance(data, pd.DataFrame):
            data.to_parquet(filepath, index=False)
        else:
            pd.DataFrame(data).to_parquet(filepath, index=False)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Saved results to {filepath}")
    return filepath


def export_to_json(
    data: Union[pd.DataFrame, Dict[str, Any], List[Any]],
    name: str,
    minify: bool = False
) -> Path:
    """
    Export data to JSON for web consumption.
    
    Args:
        data: Data to export
        name: File name (without extension)
        minify: Minify JSON output
        
    Returns:
        Path to exported file
    """
    project_root = get_project_root()
    output_dir = project_root / "results" / "exports" / "json"
    ensure_dir(output_dir)
    
    filepath = output_dir / f"{name}.json"
    
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to JSON-serializable format
        json_data = data.to_dict(orient="records")
    elif isinstance(data, np.ndarray):
        json_data = data.tolist()
    else:
        json_data = data
    
    # Handle special types and clean NaN/Infinity values with circular reference protection
    def clean_value(val, depth=0, max_depth=100):
        """Clean a single value for JSON serialization."""
        if depth > max_depth:
            return str(val)
        
        if val is None:
            return None
        if isinstance(val, (bool,)):
            return val
        if isinstance(val, str):
            return val
        if isinstance(val, float):
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, np.floating):
            float_val = float(val)
            if np.isnan(float_val) or np.isinf(float_val):
                return None
            return float_val
        if isinstance(val, np.ndarray):
            return [clean_value(v, depth+1, max_depth) for v in val.tolist()]
        if isinstance(val, pd.Timestamp):
            return val.isoformat()
        if isinstance(val, (pd.Series,)):
            return clean_value(val.to_dict(), depth+1, max_depth)
        if isinstance(val, pd.DataFrame):
            return clean_value(val.to_dict('records'), depth+1, max_depth)
        if isinstance(val, dict):
            return {str(k): clean_value(v, depth+1, max_depth) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [clean_value(v, depth+1, max_depth) for v in val]
        try:
            if pd.isna(val):
                return None
        except (ValueError, TypeError):
            pass
        return str(val)
    
    def json_serializer(obj):
        cleaned = clean_value(obj, depth=0)
        if cleaned is not None:
            return cleaned
        return str(obj)
    
    # Deep clean the data before serialization
    json_data = clean_value(json_data, depth=0)
    
    with open(filepath, 'w') as f:
        if minify:
            json.dump(json_data, f, default=json_serializer, separators=(',', ':'))
        else:
            json.dump(json_data, f, default=json_serializer, indent=2)
    
    logger.info(f"Exported JSON to {filepath}")
    return filepath


def export_to_csv(
    data: pd.DataFrame,
    name: str,
    compression: Optional[str] = None
) -> Path:
    """
    Export data to CSV for web consumption.
    
    Args:
        data: DataFrame to export
        name: File name (without extension)
        compression: Compression type (gzip, zip, etc.)
        
    Returns:
        Path to exported file
    """
    project_root = get_project_root()
    output_dir = project_root / "results" / "exports" / "csv"
    ensure_dir(output_dir)
    
    if compression:
        filepath = output_dir / f"{name}.csv.{compression}"
    else:
        filepath = output_dir / f"{name}.csv"
    
    data.to_csv(filepath, index=False, compression=compression)
    
    logger.info(f"Exported CSV to {filepath}")
    return filepath


def load_result(
    name: str,
    category: str,
    format: str = "csv"
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Load a previously saved result.
    
    Args:
        name: File name (without extension and timestamp)
        category: Category folder
        format: File format
        
    Returns:
        Loaded data
    """
    project_root = get_project_root()
    result_dir = project_root / "results" / "analysis" / category
    
    # Find matching files (may have timestamp)
    pattern = f"{name}*.{format}"
    matching_files = list(result_dir.glob(pattern))
    
    if not matching_files:
        raise FileNotFoundError(f"No results found matching: {pattern}")
    
    # Use most recent file
    filepath = max(matching_files, key=lambda p: p.stat().st_mtime)
    
    if format == "csv":
        return pd.read_csv(filepath)
    elif format == "json":
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == "parquet":
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


class ResultsManager:
    """
    Manager for organizing and accessing analysis results.
    
    Author: Shuvam Banerji Seal's Team
    """
    
    def __init__(self):
        self.project_root = get_project_root()
        self.results_dir = self.project_root / "results"
        self.exports_dir = self.results_dir / "exports"
    
    def list_results(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available results."""
        results = []
        
        if category:
            search_dir = self.results_dir / "analysis" / category
        else:
            search_dir = self.results_dir / "analysis"
        
        if not search_dir.exists():
            return results
        
        for filepath in search_dir.rglob("*"):
            if filepath.is_file() and filepath.suffix in ['.csv', '.json', '.parquet']:
                results.append({
                    'name': filepath.stem,
                    'category': filepath.parent.name,
                    'format': filepath.suffix[1:],
                    'size_mb': filepath.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(filepath.stat().st_mtime)
                })
        
        return sorted(results, key=lambda x: x['modified'], reverse=True)
    
    def get_export_manifest(self) -> Dict[str, List[str]]:
        """Get manifest of all exported files for web."""
        manifest = {'csv': [], 'json': []}
        
        for format_type in ['csv', 'json']:
            export_dir = self.exports_dir / format_type
            if export_dir.exists():
                manifest[format_type] = [
                    f.name for f in export_dir.iterdir() if f.is_file()
                ]
        
        return manifest
    
    def cleanup_old_results(self, days: int = 7):
        """Remove results older than specified days."""
        import shutil
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0
        
        for filepath in self.results_dir.rglob("*"):
            if filepath.is_file():
                modified = datetime.fromtimestamp(filepath.stat().st_mtime)
                if modified < cutoff:
                    filepath.unlink()
                    removed += 1
        
        logger.info(f"Removed {removed} old result files")
        return removed


if __name__ == "__main__":
    # Test utilities
    print(f"Project root: {get_project_root()}")
    
    # Test loading dataset
    try:
        df = load_dataset("biometric", nrows=100)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
