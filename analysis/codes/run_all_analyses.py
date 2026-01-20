#!/usr/bin/env python3
"""
Master Analysis Runner
======================
Author: Shuvam Banerji Seal's Team
Date: January 2026

Master script to run all analyses in parallel and generate comprehensive results.
Outputs CSV and JSON files for web consumption.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_time_series_wrapper(nrows: Optional[int] = None) -> Dict[str, Any]:
    """Wrapper for time series analysis."""
    try:
        from time_series.analyzer import run_time_series_analysis
        return {'time_series': run_time_series_analysis(nrows=nrows)}
    except Exception as e:
        logger.error(f"Time series analysis failed: {e}")
        return {'time_series': {'error': str(e)}}


def run_geographic_wrapper(nrows: Optional[int] = None) -> Dict[str, Any]:
    """Wrapper for geographic analysis."""
    try:
        from geographic.analyzer import run_geographic_analysis
        return {'geographic': run_geographic_analysis(nrows=nrows)}
    except Exception as e:
        logger.error(f"Geographic analysis failed: {e}")
        return {'geographic': {'error': str(e)}}


def run_demographic_wrapper(nrows: Optional[int] = None) -> Dict[str, Any]:
    """Wrapper for demographic analysis."""
    try:
        from demographic.analyzer import run_demographic_analysis
        return {'demographic': run_demographic_analysis(nrows=nrows)}
    except Exception as e:
        logger.error(f"Demographic analysis failed: {e}")
        return {'demographic': {'error': str(e)}}


def run_ml_wrapper(nrows: Optional[int] = None) -> Dict[str, Any]:
    """Wrapper for ML model training."""
    try:
        from ml_models.training import ModelTrainer, TimeSeriesForecaster, AnomalyDetector
        from utils.io_utils import load_dataset
        
        results = {}
        
        # Load data
        for dataset_name in ['biometric', 'demographic', 'enrollment']:
            try:
                df = load_dataset(dataset_name, nrows=nrows)
                
                # Skip if not enough data
                if len(df) < 100:
                    continue
                
                # Classification - predict state from features
                if 'state' in df.columns:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 2:
                        trainer = ModelTrainer(
                            df=df.dropna(subset=['state'] + numeric_cols[:5]),
                            target_column='state',
                            feature_columns=numeric_cols[:5],
                            task_type='classification'
                        )
                        trainer.train_all_models()
                        model_results = trainer.get_results()
                        results[f'{dataset_name}_classification'] = {
                            'best_model': model_results.get('best_model', 'unknown'),
                            'accuracy': model_results.get('best_accuracy', 0),
                            'models_trained': len(model_results.get('all_results', {})),
                            'feature_importance': model_results.get('feature_importance', {})
                        }
                
                # Anomaly detection
                if 'total_enrollment' in df.columns:
                    detector = AnomalyDetector(df)
                    anomalies = detector.detect_anomalies('total_enrollment')
                    results[f'{dataset_name}_anomaly_detection'] = {
                        'total_anomalies': int(anomalies.sum()),
                        'anomaly_rate': float(anomalies.mean() * 100)
                    }
                
            except Exception as e:
                logger.warning(f"ML training failed for {dataset_name}: {e}")
                results[f'{dataset_name}_ml_error'] = str(e)
        
        return {'ml_results': results}
    except Exception as e:
        logger.error(f"ML training failed: {e}")
        return {'ml_results': {'error': str(e)}}


def run_visualization_wrapper(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate visualizations from analysis results."""
    try:
        from visualization.visualizer import generate_all_visualizations
        
        output_paths = generate_all_visualizations(all_results)
        
        # Copy PDFs to web directory for download
        project_root = get_project_root()
        web_pdf_dir = project_root / 'web' / 'frontend' / 'public' / 'pdfs'
        web_pdf_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        pdf_files = {}
        for viz_type, path in output_paths.items():
            if path and path.exists():
                dest = web_pdf_dir / path.name
                shutil.copy2(path, dest)
                pdf_files[viz_type] = f'/pdfs/{path.name}'
        
        return {'visualization_paths': output_paths, 'web_pdf_files': pdf_files}
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        return {'visualization_error': str(e)}


def run_statistical_wrapper(nrows: Optional[int] = None) -> Dict[str, Any]:
    """Wrapper for statistical analysis."""
    try:
        from statistical.analyzer import run_statistical_analysis
        return {'statistical': run_statistical_analysis(nrows=nrows)}
    except Exception as e:
        logger.error(f"Statistical analysis failed: {e}")
        return {'statistical': {'error': str(e)}}


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    project_root = get_project_root()
    
    dirs = [
        'results/analysis/time_series',
        'results/analysis/geographic',
        'results/analysis/demographic',
        'results/analysis/statistical',
        'results/analysis/climate',
        'results/analysis/socioeconomic',
        'results/analysis/ml_predictions',
        'results/exports/json',
        'results/exports/csv',
        'results/data/processed',
        'results/data/aggregated',
        'web/frontend/public/data'
    ]
    
    for d in dirs:
        (project_root / d).mkdir(parents=True, exist_ok=True)
    
    logger.info("Output directories created")


def generate_summary(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of all analyses."""
    summary = {
        'generated_at': datetime.now().isoformat(),
        'author': "Shuvam Banerji Seal's Team",
        'project': 'UIDAI Hackathon Analysis',
        'analyses_completed': [],
        'analyses_failed': [],
        'total_analyses': 0
    }
    
    for analysis_type, results in all_results.items():
        if isinstance(results, dict):
            if 'error' in results:
                summary['analyses_failed'].append({
                    'type': analysis_type,
                    'error': results['error']
                })
            else:
                summary['analyses_completed'].append({
                    'type': analysis_type,
                    'result_count': len(results)
                })
    
    summary['total_analyses'] = len(summary['analyses_completed']) + len(summary['analyses_failed'])
    summary['success_rate'] = round(
        len(summary['analyses_completed']) / summary['total_analyses'] * 100, 2
    ) if summary['total_analyses'] > 0 else 0
    
    return summary


def clean_for_json(obj, seen=None):
    """Recursively clean data for JSON serialization (handle NaN, Infinity, circular refs)."""
    import numpy as np
    import pandas as pd
    
    if seen is None:
        seen = set()
    
    # Check for circular references
    obj_id = id(obj)
    if obj_id in seen:
        return None
    
    if obj is None:
        return None
    
    if isinstance(obj, (bool, str)):
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
        return [clean_for_json(v, seen) for v in obj.tolist()]
    
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    
    if isinstance(obj, (pd.Series,)):
        return clean_for_json(obj.to_dict(), seen)
    
    if isinstance(obj, pd.DataFrame):
        return clean_for_json(obj.to_dict('records'), seen)
    
    if isinstance(obj, dict):
        seen.add(obj_id)
        result = {}
        for k, v in obj.items():
            cleaned = clean_for_json(v, seen)
            if cleaned is not None or isinstance(cleaned, (bool, int, float, str, list, dict)):
                result[str(k)] = cleaned
        seen.discard(obj_id)
        return result
    
    if isinstance(obj, (list, tuple)):
        seen.add(obj_id)
        result = [clean_for_json(v, seen) for v in obj]
        seen.discard(obj_id)
        return result
    
    # Try to check for NaN
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass
    
    # Default: convert to string
    return str(obj)


def export_for_web(all_results: Dict[str, Any]):
    """Export results in web-friendly format."""
    import numpy as np
    
    project_root = get_project_root()
    web_data_dir = project_root / 'web' / 'frontend' / 'public' / 'data'
    web_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Export master summary
    summary = generate_summary(all_results)
    with open(web_data_dir / 'analysis_summary.json', 'w') as f:
        json.dump(clean_for_json(summary), f, indent=2, default=str)
    
    # Export each analysis type
    for analysis_type, results in all_results.items():
        if isinstance(results, dict) and 'error' not in results:
            # Flatten nested dictionaries for easier web consumption
            output_file = web_data_dir / f'{analysis_type}.json'
            cleaned_results = clean_for_json(results)
            with open(output_file, 'w') as f:
                json.dump(cleaned_results, f, indent=2, default=str)
            logger.info(f"Exported {analysis_type} to {output_file}")
    
    # Create index file for web
    index = {
        'generated_at': datetime.now().isoformat(),
        'available_analyses': [
            {
                'name': analysis_type,
                'file': f'{analysis_type}.json',
                'has_error': isinstance(results, dict) and 'error' in results
            }
            for analysis_type, results in all_results.items()
        ]
    }
    
    with open(web_data_dir / 'index.json', 'w') as f:
        json.dump(index, f, indent=2)
    
    logger.info(f"Web export complete. Files saved to {web_data_dir}")


def run_all_analyses(nrows: Optional[int] = None, parallel: bool = True, 
                    analyses: Optional[list] = None, run_ml: bool = True,
                    generate_viz: bool = True) -> Dict[str, Any]:
    """
    Run all analyses.
    
    Args:
        nrows: Number of rows to sample (None for all)
        parallel: Whether to run analyses in parallel
        analyses: List of specific analyses to run (None for all)
        run_ml: Whether to run ML model training
        generate_viz: Whether to generate PDF visualizations
        
    Returns:
        Dictionary with all results
    """
    logger.info("=" * 70)
    logger.info("UIDAI COMPREHENSIVE ANALYSIS SUITE")
    logger.info("Author: Shuvam Banerji Seal's Team")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    # Ensure output directories exist
    ensure_output_dirs()
    
    # Define available analyses
    analysis_functions = {
        'time_series': run_time_series_wrapper,
        'geographic': run_geographic_wrapper,
        'demographic': run_demographic_wrapper,
        'statistical': run_statistical_wrapper,
    }
    
    # Filter analyses if specified
    if analyses:
        analysis_functions = {k: v for k, v in analysis_functions.items() if k in analyses}
    
    all_results = {}
    
    if parallel and len(analysis_functions) > 1:
        # Run analyses in parallel
        logger.info(f"Running {len(analysis_functions)} analyses in parallel...")
        
        # Note: We use spawn method to avoid issues with fork
        with ProcessPoolExecutor(max_workers=min(4, len(analysis_functions))) as executor:
            futures = {
                executor.submit(func, nrows): name 
                for name, func in analysis_functions.items()
            }
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    all_results.update(result)
                    logger.info(f"✓ {name} analysis completed")
                except Exception as e:
                    logger.error(f"✗ {name} analysis failed: {e}")
                    all_results[name] = {'error': str(e)}
    else:
        # Run analyses sequentially
        logger.info(f"Running {len(analysis_functions)} analyses sequentially...")
        
        for name, func in analysis_functions.items():
            try:
                result = func(nrows)
                all_results.update(result)
                logger.info(f"✓ {name} analysis completed")
            except Exception as e:
                logger.error(f"✗ {name} analysis failed: {e}")
                all_results[name] = {'error': str(e)}
    
    # Export for web
    export_for_web(all_results)
    
    # Run ML training if requested
    if run_ml:
        logger.info("Running ML model training...")
        try:
            ml_results = run_ml_wrapper(nrows)
            all_results.update(ml_results)
            
            # Export ML results
            project_root = get_project_root()
            web_data_dir = project_root / 'web' / 'frontend' / 'public' / 'data'
            ml_file = web_data_dir / 'ml_results.json'
            with open(ml_file, 'w') as f:
                json.dump(clean_for_json(ml_results.get('ml_results', {})), f, indent=2)
            logger.info(f"ML results saved to {ml_file}")
        except Exception as e:
            logger.error(f"ML training failed: {e}")
    
    # Generate visualizations
    if generate_viz:
        logger.info("Generating visualizations...")
        try:
            viz_results = run_visualization_wrapper(all_results)
            all_results['visualization'] = viz_results
            
            # Export PDF index for web
            project_root = get_project_root()
            web_data_dir = project_root / 'web' / 'frontend' / 'public' / 'data'
            pdf_index_file = web_data_dir / 'pdf_index.json'
            with open(pdf_index_file, 'w') as f:
                json.dump(viz_results.get('web_pdf_files', {}), f, indent=2)
            logger.info(f"PDF index saved to {pdf_index_file}")
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
    
    # Generate and save summary
    summary = generate_summary(all_results)
    
    project_root = get_project_root()
    summary_file = project_root / 'results' / 'analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Success rate: {summary['success_rate']}%")
    logger.info(f"Results saved to: {project_root / 'results'}")
    logger.info(f"Web data saved to: {project_root / 'web' / 'frontend' / 'public' / 'data'}")
    logger.info("=" * 70)
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="UIDAI Comprehensive Analysis Suite - Author: Shuvam Banerji Seal's Team"
    )
    parser.add_argument(
        '--sample', '-s', 
        type=int, 
        default=None,
        help="Number of rows to sample (default: all)"
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help="Run analyses sequentially instead of parallel"
    )
    parser.add_argument(
        '--analyses', '-a',
        nargs='+',
        choices=['time_series', 'geographic', 'demographic', 'statistical'],
        help="Specific analyses to run (default: all)"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        '--no-ml',
        action='store_true',
        help="Skip ML model training"
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help="Skip visualization generation"
    )
    
    args = parser.parse_args()
    
    # Run analyses
    results = run_all_analyses(
        nrows=args.sample,
        parallel=not args.sequential,
        analyses=args.analyses,
        run_ml=not args.no_ml,
        generate_viz=not args.no_viz
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    
    summary = generate_summary(results)
    print(f"Total analyses: {summary['total_analyses']}")
    print(f"Completed: {len(summary['analyses_completed'])}")
    print(f"Failed: {len(summary['analyses_failed'])}")
    print(f"Success rate: {summary['success_rate']}%")
    
    if summary['analyses_failed']:
        print("\nFailed analyses:")
        for failed in summary['analyses_failed']:
            print(f"  - {failed['type']}: {failed['error']}")
    
    print("\nResults exported to web/frontend/public/data/")


if __name__ == "__main__":
    # Use spawn method for multiprocessing to avoid fork issues
    mp.set_start_method('spawn', force=True)
    main()
