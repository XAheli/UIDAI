"""
UIDAI Aadhaar Data Analysis - Utility Package
==============================================
Author: Shuvam Banerji Seal's Team
Date: January 2026

This package contains utility functions and classes used across all analysis modules.
"""

from .parallel import ParallelProcessor, parallel_apply, parallel_map
from .io_utils import (
    load_dataset, save_results, export_to_json, export_to_csv,
    ensure_dir, get_project_root
)
from .validators import validate_dataframe, validate_state_names, validate_pincodes
from .progress import ProgressTracker, timed_operation

__all__ = [
    'ParallelProcessor', 'parallel_apply', 'parallel_map',
    'load_dataset', 'save_results', 'export_to_json', 'export_to_csv',
    'ensure_dir', 'get_project_root',
    'validate_dataframe', 'validate_state_names', 'validate_pincodes',
    'ProgressTracker', 'timed_operation'
]
