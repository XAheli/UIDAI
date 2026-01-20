#!/usr/bin/env python3
"""
CSV Validation and Cleaning Script
===================================

This script validates and cleans Aadhaar datasets by:
1. Checking for null, empty, and invalid values
2. Standardizing state and district names
3. Validating pincode format
4. Fixing encoding issues
5. Removing duplicate records

Uses multiprocessing for performance on large datasets.

Usage:
    python csv_cleaner.py --input-dir /path/to/corrected --output-dir /path/to/cleaned
    python csv_cleaner.py --all  # Process all dataset types
"""

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import chardet
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import (
        BIOMETRIC_PATH, DEMOGRAPHIC_PATH, ENROLLMENT_PATH,
        CLEANED_DATASET_PATH, INDIA_STATES, STATE_NAME_MAPPING,
        EXPECTED_COLUMNS, NON_NULL_FIELDS, INVALID_CHARS, CPU_COUNT
    )
    from logger import setup_logger, get_timestamped_log_file, ProgressLogger
except ImportError:
    from .config import (
        BIOMETRIC_PATH, DEMOGRAPHIC_PATH, ENROLLMENT_PATH,
        CLEANED_DATASET_PATH, INDIA_STATES, STATE_NAME_MAPPING,
        EXPECTED_COLUMNS, NON_NULL_FIELDS, INVALID_CHARS, CPU_COUNT
    )
    from .logger import setup_logger, get_timestamped_log_file, ProgressLogger


# Set up logging
log_file = get_timestamped_log_file("csv_cleaning")
logger = setup_logger(__name__, log_file)


class CSVCleaner:
    """
    Comprehensive CSV validation and cleaning utility.
    Uses multiprocessing for parallel file processing.
    """
    
    def __init__(self, input_dir: Path, output_dir: Path, dataset_type: str):
        """
        Initialize the CSV cleaner.
        
        Args:
            input_dir: Directory containing input CSV files
            output_dir: Directory to write cleaned CSV files
            dataset_type: Type of dataset (biometric, demographic, enrollment)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dataset_type = dataset_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "files_processed": 0,
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "null_values_fixed": 0,
            "state_names_fixed": 0,
            "district_names_fixed": 0,
            "pincode_issues": 0,
            "duplicates_removed": 0,
            "encoding_issues": 0,
        }
    
    def detect_encoding(self, file_path: Path) -> str:
        """
        Detect the encoding of a file using chardet with UTF-8 fallback.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Detected encoding string
        """
        # Try common encodings first
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    # Read a sample to verify encoding works
                    f.read(10000)
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # Fall back to chardet detection
        with open(file_path, 'rb') as f:
            raw_data = f.read(100000)
            result = chardet.detect(raw_data)
            detected = result['encoding']
            
        return detected if detected else 'utf-8'
    
    def standardize_state_name(self, state: str) -> Tuple[str, bool]:
        """
        Standardize a state name to the official format.
        
        Args:
            state: Input state name
        
        Returns:
            Tuple of (standardized name, whether it was changed)
        """
        if pd.isna(state) or state is None:
            return "", True
        
        state_str = str(state).strip()
        state_lower = state_str.lower()
        
        # Check if already standardized
        if state_str in INDIA_STATES:
            return state_str, False
        
        # Try mapping
        if state_lower in STATE_NAME_MAPPING:
            return STATE_NAME_MAPPING[state_lower], True
        
        # Try title case
        state_title = state_str.title()
        if state_title in INDIA_STATES:
            return state_title, True
        
        # Return original with warning
        logger.warning(f"Unknown state name: '{state_str}'")
        return state_str, False
    
    def standardize_district_name(self, district: str) -> Tuple[str, bool]:
        """
        Standardize district name to Title Case.
        
        Args:
            district: Input district name
        
        Returns:
            Tuple of (standardized name, whether it was changed)
        """
        if pd.isna(district) or district is None:
            return "", True
        
        district_str = str(district).strip()
        
        # Handle special characters
        if district_str in ['?', '-', 'N/A', 'NA', 'null', 'NULL']:
            return "", True
        
        # Convert to Title Case
        district_title = district_str.title()
        
        return district_title, district_title != district_str
    
    def validate_pincode(self, pincode: Any) -> Tuple[str, bool]:
        """
        Validate and clean pincode.
        
        Args:
            pincode: Input pincode value
        
        Returns:
            Tuple of (cleaned pincode string, whether it's valid)
        """
        if pd.isna(pincode) or pincode is None:
            return "", False
        
        # Convert to string and clean
        pincode_str = str(pincode).strip()
        
        # Remove any non-numeric characters
        pincode_clean = re.sub(r'[^0-9]', '', pincode_str)
        
        # Validate Indian pincode (6 digits, starts with 1-9)
        if len(pincode_clean) == 6 and pincode_clean[0] != '0':
            return pincode_clean, True
        
        # Try to pad with zeros if short
        if len(pincode_clean) < 6 and pincode_clean:
            pincode_padded = pincode_clean.zfill(6)
            if pincode_padded[0] != '0':
                return pincode_padded, True
        
        return pincode_str, False
    
    def clean_value(self, value: Any) -> Any:
        """
        Clean individual cell values.
        
        Args:
            value: Input value
        
        Returns:
            Cleaned value
        """
        if pd.isna(value) or value is None:
            return ""
        
        value_str = str(value).strip()
        
        # Check for null-like values
        if value_str.lower() in ['null', 'none', 'nan', 'n/a', 'na', '']:
            return ""
        
        # Remove invalid characters
        for char in INVALID_CHARS:
            value_str = value_str.replace(char, '')
        
        # Remove excessive whitespace
        value_str = ' '.join(value_str.split())
        
        return value_str
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean a DataFrame according to validation rules.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        original_len = len(df)
        
        # Clean all string values
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(self.clean_value)
        
        # Standardize state names
        if 'state' in df.columns:
            state_results = df['state'].apply(self.standardize_state_name)
            df['state'] = [r[0] for r in state_results]
            self.stats['state_names_fixed'] += sum(1 for r in state_results if r[1])
        
        # Standardize district names
        if 'district' in df.columns:
            district_results = df['district'].apply(self.standardize_district_name)
            df['district'] = [r[0] for r in district_results]
            self.stats['district_names_fixed'] += sum(1 for r in district_results if r[1])
        
        # Validate pincodes
        if 'pincode' in df.columns:
            pincode_results = df['pincode'].apply(self.validate_pincode)
            df['pincode'] = [r[0] for r in pincode_results]
            self.stats['pincode_issues'] += sum(1 for r in pincode_results if not r[1])
        
        # Remove rows with critical null values
        critical_cols = [col for col in NON_NULL_FIELDS if col in df.columns]
        null_mask = df[critical_cols].isin(['', None]).any(axis=1)
        self.stats['null_values_fixed'] += null_mask.sum()
        df = df[~null_mask]
        
        # Remove duplicates
        dup_count = df.duplicated().sum()
        self.stats['duplicates_removed'] += dup_count
        df = df.drop_duplicates()
        
        self.stats['valid_records'] += len(df)
        self.stats['invalid_records'] += original_len - len(df)
        
        return df
    
    def process_file(self, input_file: Path) -> Tuple[bool, str]:
        """
        Process a single CSV file.
        
        Args:
            input_file: Path to input file
        
        Returns:
            Tuple of (success, message)
        """
        output_file = self.output_dir / f"cleaned_{input_file.name}"
        
        try:
            # Detect encoding
            encoding = self.detect_encoding(input_file)
            if encoding.lower() != 'utf-8':
                self.stats['encoding_issues'] += 1
            
            # Read CSV
            df = pd.read_csv(
                input_file,
                encoding=encoding,
                dtype=str,
                na_values=['', 'null', 'NULL', 'None', 'none', 'NaN', 'nan'],
                keep_default_na=True,
                low_memory=False
            )
            
            self.stats['total_records'] += len(df)
            
            # Clean the dataframe
            df = self.clean_dataframe(df)
            
            # Save cleaned file
            df.to_csv(output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
            
            self.stats['files_processed'] += 1
            
            return True, f"Processed {input_file.name}: {len(df)} valid records"
            
        except Exception as e:
            logger.error(f"Error processing {input_file.name}: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def process_all_files(self) -> Dict[str, Any]:
        """
        Process all CSV files in the input directory using multiprocessing.
        
        Returns:
            Statistics dictionary
        """
        csv_files = list(self.input_dir.glob("*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.input_dir}")
            return self.stats
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Process files in parallel
        with ProgressLogger(logger, f"Cleaning {self.dataset_type} files", len(csv_files)) as progress:
            with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
                futures = {executor.submit(self.process_file, f): f for f in csv_files}
                
                for future in tqdm(as_completed(futures), total=len(csv_files), desc="Processing"):
                    file_path = futures[future]
                    try:
                        success, message = future.result()
                        progress.update(1, error=not success)
                        if success:
                            logger.debug(message)
                        else:
                            logger.error(message)
                    except Exception as e:
                        progress.update(1, error=True)
                        logger.error(f"Failed to process {file_path.name}: {str(e)}")
        
        return self.stats


def merge_and_clean_dataset(
    input_dir: Path,
    output_dir: Path,
    dataset_type: str,
    output_filename: str
) -> Dict[str, Any]:
    """
    Merge multiple CSV files into one and clean the dataset.
    
    Args:
        input_dir: Directory containing input CSV files
        output_dir: Directory to write output
        dataset_type: Type of dataset
        output_filename: Name for the merged output file
    
    Returns:
        Statistics dictionary
    """
    logger.info(f"Processing {dataset_type} dataset from {input_dir}")
    
    cleaner = CSVCleaner(input_dir, output_dir, dataset_type)
    
    # First, clean all individual files
    stats = cleaner.process_all_files()
    
    # Then merge all cleaned files
    cleaned_files = list(output_dir.glob("cleaned_*.csv"))
    
    if cleaned_files:
        logger.info(f"Merging {len(cleaned_files)} cleaned files...")
        
        all_dfs = []
        for f in tqdm(cleaned_files, desc="Merging"):
            df = pd.read_csv(f, dtype=str)
            all_dfs.append(df)
        
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Final deduplication
        initial_len = len(merged_df)
        merged_df = merged_df.drop_duplicates()
        stats['duplicates_removed'] += initial_len - len(merged_df)
        
        # Save merged file
        merged_path = output_dir / output_filename
        merged_df.to_csv(merged_path, index=False, encoding='utf-8')
        
        logger.info(f"Saved merged dataset: {merged_path} ({len(merged_df):,} records)")
        
        # Clean up intermediate files
        for f in cleaned_files:
            os.remove(f)
    
    return stats


def generate_validation_report(all_stats: Dict[str, Dict[str, Any]], output_dir: Path):
    """
    Generate a comprehensive validation report.
    
    Args:
        all_stats: Statistics from all dataset types
        output_dir: Directory to save the report
    """
    report_path = output_dir / "cleaning_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# CSV Cleaning and Validation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for dataset_type, stats in all_stats.items():
            f.write(f"## {dataset_type.title()} Dataset\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            for key, value in stats.items():
                f.write(f"| {key.replace('_', ' ').title()} | {value:,} |\n")
            f.write("\n")
        
        f.write("## Summary\n\n")
        total_records = sum(s.get('total_records', 0) for s in all_stats.values())
        valid_records = sum(s.get('valid_records', 0) for s in all_stats.values())
        f.write(f"- **Total Records Processed**: {total_records:,}\n")
        f.write(f"- **Valid Records**: {valid_records:,}\n")
        f.write(f"- **Validity Rate**: {valid_records/total_records*100:.2f}%\n" if total_records > 0 else "")
    
    logger.info(f"Validation report saved to: {report_path}")


def main():
    """Main entry point for the CSV cleaning script."""
    parser = argparse.ArgumentParser(
        description="Clean and validate Aadhaar CSV datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python csv_cleaner.py --all
    python csv_cleaner.py --type biometric
    python csv_cleaner.py --input-dir ./data/raw --output-dir ./data/cleaned --type demographic
        """
    )
    
    parser.add_argument(
        '--all', action='store_true',
        help='Process all dataset types (biometric, demographic, enrollment)'
    )
    parser.add_argument(
        '--type', choices=['biometric', 'demographic', 'enrollment'],
        help='Specific dataset type to process'
    )
    parser.add_argument(
        '--input-dir', type=Path,
        help='Input directory containing CSV files'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=CLEANED_DATASET_PATH,
        help='Output directory for cleaned files'
    )
    parser.add_argument(
        '--workers', type=int, default=CPU_COUNT,
        help=f'Number of worker processes (default: {CPU_COUNT})'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting CSV Cleaning Pipeline")
    logger.info(f"Workers: {args.workers}, Output: {args.output_dir}")
    logger.info("=" * 60)
    
    all_stats = {}
    
    # Define dataset configurations
    datasets = {
        'biometric': {
            'input_dir': BIOMETRIC_PATH,
            'output_dir': args.output_dir / 'biometric',
            'output_filename': 'final_cleaned_biometric.csv'
        },
        'demographic': {
            'input_dir': DEMOGRAPHIC_PATH,
            'output_dir': args.output_dir / 'demographic',
            'output_filename': 'final_cleaned_demographic.csv'
        },
        'enrollment': {
            'input_dir': ENROLLMENT_PATH,
            'output_dir': args.output_dir / 'enrollment',
            'output_filename': 'final_cleaned_enrollment.csv'
        }
    }
    
    # Determine which datasets to process
    if args.all:
        to_process = list(datasets.keys())
    elif args.type:
        to_process = [args.type]
    else:
        logger.error("Please specify --all or --type")
        sys.exit(1)
    
    # Override input directory if specified
    if args.input_dir:
        for dtype in to_process:
            datasets[dtype]['input_dir'] = args.input_dir
    
    # Process each dataset
    for dtype in to_process:
        config = datasets[dtype]
        logger.info(f"\nProcessing {dtype} dataset...")
        
        stats = merge_and_clean_dataset(
            input_dir=config['input_dir'],
            output_dir=config['output_dir'],
            dataset_type=dtype,
            output_filename=config['output_filename']
        )
        all_stats[dtype] = stats
        
        logger.info(f"\n{dtype.title()} Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:,}")
    
    # Generate report
    generate_validation_report(all_stats, args.output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("CSV Cleaning Pipeline Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
