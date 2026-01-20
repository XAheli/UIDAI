#!/usr/bin/env python3
"""
Generate PDF Visualizations from Analysis Results
==================================================
Author: Shuvam Banerji Seal's Team
Date: January 2026

Standalone script to generate high-quality PDF visualizations from
the analysis results JSON files.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

# Set up matplotlib for headless rendering BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker

# Try importing seaborn for enhanced plots
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="husl")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("Seaborn not available, using basic matplotlib styling")

# Default figure settings
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_analysis_results() -> Dict[str, Any]:
    """Load all analysis results from JSON files."""
    project_root = get_project_root()
    json_dir = project_root / "results" / "exports" / "json"
    
    results = {}
    
    # Load each analysis type
    analysis_files = {
        'time_series': 'time_series_analysis.json',
        'geographic': 'geographic_analysis.json',
        'statistical': 'statistical_analysis.json',
        'demographic': 'demographic_analysis.json'
    }
    
    for key, filename in analysis_files.items():
        filepath = json_dir / filename
        if filepath.exists():
            logger.info(f"Loading {filename}")
            with open(filepath, 'r') as f:
                results[key] = json.load(f)
        else:
            logger.warning(f"File not found: {filepath}")
    
    return results


def generate_time_series_pdf(data: Dict[str, Any], output_dir: Path) -> Path:
    """Generate time series visualization PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "time_series_analysis.pdf"
    
    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(12, 8))
        fig.text(0.5, 0.6, 'UIDAI Time Series Analysis', 
                ha='center', va='center', fontsize=28, fontweight='bold')
        fig.text(0.5, 0.45, "Shuvam Banerji Seal's Team", 
                ha='center', va='center', fontsize=16)
        fig.text(0.5, 0.35, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', va='center', fontsize=12)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        datasets = ['biometric', 'demographic', 'enrollment']
        
        for dataset in datasets:
            # Daily trends
            daily_key = f"{dataset}_daily_trends"
            if daily_key in data:
                try:
                    trend_data = data[daily_key]
                    daily_data = trend_data.get('daily_data', [])
                    
                    if daily_data:
                        df = pd.DataFrame(daily_data)
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                            
                            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
                            
                            # Enrollment trend
                            if 'total_enrollment' in df.columns:
                                axes[0].plot(df['date'], df['total_enrollment'], 
                                           linewidth=2, marker='o', markersize=4)
                                axes[0].set_title(f'{dataset.title()} - Daily Enrollment Trend', 
                                                fontsize=14, fontweight='bold')
                                axes[0].set_xlabel('Date')
                                axes[0].set_ylabel('Total Enrollment')
                                axes[0].grid(True, alpha=0.3)
                                
                                # Add moving average
                                if len(df) > 7:
                                    ma = df['total_enrollment'].rolling(window=7).mean()
                                    axes[0].plot(df['date'], ma, linewidth=2, 
                                               linestyle='--', label='7-day MA')
                                    axes[0].legend()
                            
                            # Summary statistics
                            summary = trend_data.get('trend_analysis', {})
                            summary_text = f"""
                            Trend Analysis Summary:
                            - Total Records: {summary.get('total_records', 'N/A')}
                            - Slope: {summary.get('slope', 'N/A')}
                            - R-squared: {summary.get('r_squared', 'N/A')}
                            """
                            axes[1].text(0.1, 0.5, summary_text, transform=axes[1].transAxes,
                                       fontsize=12, verticalalignment='center')
                            axes[1].axis('off')
                            axes[1].set_title(f'{dataset.title()} - Trend Statistics')
                            
                            plt.tight_layout()
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                            logger.info(f"Added {dataset} daily trends plot")
                except Exception as e:
                    logger.warning(f"Error creating {dataset} daily trends: {e}")
            
            # Seasonality
            seasonality_key = f"{dataset}_seasonality"
            if seasonality_key in data:
                try:
                    seasonal_data = data[seasonality_key]
                    
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    fig.suptitle(f'{dataset.title()} - Seasonality Analysis', 
                               fontsize=16, fontweight='bold')
                    
                    # Monthly pattern
                    monthly = seasonal_data.get('monthly_pattern', [])
                    if monthly:
                        months = [m['month'] for m in monthly]
                        values = [m['avg_enrollment'] for m in monthly]
                        axes[0, 0].bar(months, values, color='steelblue', alpha=0.7)
                        axes[0, 0].set_title('Monthly Enrollment Pattern')
                        axes[0, 0].set_xlabel('Month')
                        axes[0, 0].set_ylabel('Average Enrollment')
                        axes[0, 0].tick_params(axis='x', rotation=45)
                    
                    # Day of week pattern
                    dow = seasonal_data.get('day_of_week_pattern', [])
                    if dow:
                        days = [d['day_of_week'] for d in dow]
                        values = [d['avg_enrollment'] for d in dow]
                        axes[0, 1].bar(days, values, color='coral', alpha=0.7)
                        axes[0, 1].set_title('Day of Week Pattern')
                        axes[0, 1].set_xlabel('Day')
                        axes[0, 1].set_ylabel('Average Enrollment')
                        axes[0, 1].tick_params(axis='x', rotation=45)
                    
                    # Quarterly pattern
                    quarterly = seasonal_data.get('quarterly_pattern', [])
                    if quarterly:
                        quarters = [q['quarter'] for q in quarterly]
                        values = [q['avg_enrollment'] for q in quarterly]
                        axes[1, 0].bar([f'Q{q}' for q in quarters], values, 
                                      color='seagreen', alpha=0.7)
                        axes[1, 0].set_title('Quarterly Pattern')
                        axes[1, 0].set_xlabel('Quarter')
                        axes[1, 0].set_ylabel('Average Enrollment')
                    
                    # Summary
                    summary_text = f"""
                    Seasonality Summary:
                    - Peak month: {seasonal_data.get('peak_month', 'N/A')}
                    - Lowest month: {seasonal_data.get('lowest_month', 'N/A')}
                    - Peak day: {seasonal_data.get('peak_day', 'N/A')}
                    - Seasonality strength: {seasonal_data.get('strength', 'N/A')}
                    """
                    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                                  fontsize=12, verticalalignment='center')
                    axes[1, 1].axis('off')
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Added {dataset} seasonality plot")
                except Exception as e:
                    logger.warning(f"Error creating {dataset} seasonality: {e}")
            
            # Anomalies
            anomaly_key = f"{dataset}_anomalies"
            if anomaly_key in data:
                try:
                    anomaly_data = data[anomaly_key]
                    anomalies = anomaly_data.get('anomalies', [])
                    
                    if anomalies:
                        fig, ax = plt.subplots(figsize=(14, 8))
                        
                        # Create DataFrame
                        df = pd.DataFrame(anomalies)
                        if 'date' in df.columns and 'value' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                            
                            ax.scatter(df['date'], df['value'], c='red', 
                                      alpha=0.7, s=50, label='Anomalies')
                            ax.set_title(f'{dataset.title()} - Detected Anomalies', 
                                       fontsize=14, fontweight='bold')
                            ax.set_xlabel('Date')
                            ax.set_ylabel('Value')
                            ax.grid(True, alpha=0.3)
                            ax.legend()
                            
                            plt.tight_layout()
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                            logger.info(f"Added {dataset} anomalies plot")
                except Exception as e:
                    logger.warning(f"Error creating {dataset} anomalies: {e}")
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'UIDAI Time Series Analysis'
        d['Author'] = "Shuvam Banerji Seal's Team"
        d['CreationDate'] = datetime.now()
    
    logger.info(f"Saved time series PDF to {output_path}")
    return output_path


def generate_geographic_pdf(data: Dict[str, Any], output_dir: Path) -> Path:
    """Generate geographic visualization PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "geographic_analysis.pdf"
    
    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(12, 8))
        fig.text(0.5, 0.6, 'UIDAI Geographic Analysis', 
                ha='center', va='center', fontsize=28, fontweight='bold')
        fig.text(0.5, 0.45, "Shuvam Banerji Seal's Team", 
                ha='center', va='center', fontsize=16)
        fig.text(0.5, 0.35, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', va='center', fontsize=12)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        datasets = ['biometric', 'demographic', 'enrollment']
        
        for dataset in datasets:
            # State analysis
            state_key = f"{dataset}_state"
            if state_key in data:
                try:
                    state_data = data[state_key]
                    state_stats = state_data.get('state_stats', [])
                    
                    if state_stats:
                        df = pd.DataFrame(state_stats)
                        if 'state' in df.columns and 'enrollment_count' in df.columns:
                            # Sort by enrollment count
                            df = df.sort_values('enrollment_count', ascending=True).tail(20)
                            
                            fig, ax = plt.subplots(figsize=(14, 10))
                            
                            y_pos = np.arange(len(df))
                            ax.barh(y_pos, df['enrollment_count'], color='steelblue', alpha=0.7)
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(df['state'])
                            ax.set_xlabel('Enrollment Count')
                            ax.set_title(f'{dataset.title()} - Top 20 States by Enrollment',
                                       fontsize=14, fontweight='bold')
                            ax.grid(True, alpha=0.3, axis='x')
                            
                            plt.tight_layout()
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                            logger.info(f"Added {dataset} state analysis plot")
                except Exception as e:
                    logger.warning(f"Error creating {dataset} state analysis: {e}")
            
            # Regional analysis
            regional_key = f"{dataset}_regional"
            if regional_key in data:
                try:
                    regional_data = data[regional_key]
                    regional_stats = regional_data.get('regional_stats', [])
                    
                    if regional_stats:
                        df = pd.DataFrame(regional_stats)
                        
                        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                        fig.suptitle(f'{dataset.title()} - Regional Analysis', 
                                   fontsize=16, fontweight='bold')
                        
                        if 'region' in df.columns and 'enrollment_count' in df.columns:
                            # Pie chart
                            axes[0].pie(df['enrollment_count'], labels=df['region'], 
                                       autopct='%1.1f%%', startangle=90)
                            axes[0].set_title('Regional Distribution')
                        
                        if 'region' in df.columns and 'avg_enrollment' in df.columns:
                            # Bar chart
                            axes[1].bar(df['region'], df['avg_enrollment'], 
                                       color='coral', alpha=0.7)
                            axes[1].set_title('Average Enrollment by Region')
                            axes[1].set_xlabel('Region')
                            axes[1].set_ylabel('Average Enrollment')
                            axes[1].tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        logger.info(f"Added {dataset} regional analysis plot")
                except Exception as e:
                    logger.warning(f"Error creating {dataset} regional analysis: {e}")
            
            # District analysis
            district_key = f"{dataset}_district"
            if district_key in data:
                try:
                    district_data = data[district_key]
                    top_districts = district_data.get('top_districts', [])
                    
                    if top_districts:
                        df = pd.DataFrame(top_districts[:20])  # Top 20
                        
                        if 'district' in df.columns and 'enrollment_count' in df.columns:
                            fig, ax = plt.subplots(figsize=(14, 10))
                            
                            y_pos = np.arange(len(df))
                            ax.barh(y_pos, df['enrollment_count'], color='seagreen', alpha=0.7)
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(df['district'])
                            ax.set_xlabel('Enrollment Count')
                            ax.set_title(f'{dataset.title()} - Top 20 Districts by Enrollment',
                                       fontsize=14, fontweight='bold')
                            ax.grid(True, alpha=0.3, axis='x')
                            
                            plt.tight_layout()
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                            logger.info(f"Added {dataset} district analysis plot")
                except Exception as e:
                    logger.warning(f"Error creating {dataset} district analysis: {e}")
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'UIDAI Geographic Analysis'
        d['Author'] = "Shuvam Banerji Seal's Team"
        d['CreationDate'] = datetime.now()
    
    logger.info(f"Saved geographic PDF to {output_path}")
    return output_path


def generate_statistical_pdf(data: Dict[str, Any], output_dir: Path) -> Path:
    """Generate statistical visualization PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "statistical_analysis.pdf"
    
    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(12, 8))
        fig.text(0.5, 0.6, 'UIDAI Statistical Analysis', 
                ha='center', va='center', fontsize=28, fontweight='bold')
        fig.text(0.5, 0.45, "Shuvam Banerji Seal's Team", 
                ha='center', va='center', fontsize=16)
        fig.text(0.5, 0.35, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', va='center', fontsize=12)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        datasets = ['biometric', 'demographic', 'enrollment']
        
        for dataset in datasets:
            # Descriptive statistics
            desc_key = f"{dataset}_descriptive"
            if desc_key in data:
                try:
                    desc_data = data[desc_key]
                    
                    fig, ax = plt.subplots(figsize=(14, 10))
                    ax.axis('off')
                    
                    # Create table
                    table_data = []
                    headers = ['Metric', 'Value']
                    
                    for key, value in desc_data.items():
                        if isinstance(value, (int, float)):
                            if isinstance(value, float):
                                table_data.append([key, f'{value:.4f}'])
                            else:
                                table_data.append([key, f'{value:,}'])
                    
                    if table_data:
                        table = ax.table(cellText=table_data, colLabels=headers,
                                       loc='center', cellLoc='left')
                        table.auto_set_font_size(False)
                        table.set_fontsize(10)
                        table.scale(1.2, 1.5)
                        ax.set_title(f'{dataset.title()} - Descriptive Statistics',
                                   fontsize=14, fontweight='bold', pad=20)
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Added {dataset} descriptive statistics")
                except Exception as e:
                    logger.warning(f"Error creating {dataset} descriptive stats: {e}")
            
            # Distribution analysis
            dist_key = f"{dataset}_distribution"
            if dist_key in data:
                try:
                    dist_data = data[dist_key]
                    distributions = dist_data.get('distributions', {})
                    
                    if distributions:
                        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                        fig.suptitle(f'{dataset.title()} - Distribution Analysis', 
                                   fontsize=16, fontweight='bold')
                        axes = axes.flatten()
                        
                        for idx, (var_name, var_dist) in enumerate(list(distributions.items())[:4]):
                            if isinstance(var_dist, dict) and 'histogram' in var_dist:
                                hist = var_dist['histogram']
                                if 'bins' in hist and 'counts' in hist:
                                    axes[idx].bar(range(len(hist['counts'])), 
                                                 hist['counts'], alpha=0.7)
                                    axes[idx].set_title(f'{var_name} Distribution')
                                    axes[idx].set_xlabel('Bin')
                                    axes[idx].set_ylabel('Count')
                        
                        plt.tight_layout()
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        logger.info(f"Added {dataset} distribution analysis")
                except Exception as e:
                    logger.warning(f"Error creating {dataset} distribution: {e}")
            
            # Correlation analysis
            corr_key = f"{dataset}_correlation"
            if corr_key in data:
                try:
                    corr_data = data[corr_key]
                    corr_matrix = corr_data.get('correlation_matrix', {})
                    
                    if corr_matrix:
                        # Convert to DataFrame
                        df = pd.DataFrame(corr_matrix)
                        
                        fig, ax = plt.subplots(figsize=(12, 10))
                        
                        if SEABORN_AVAILABLE:
                            mask = np.triu(np.ones_like(df, dtype=bool))
                            sns.heatmap(df, mask=mask, annot=True, cmap='coolwarm',
                                       center=0, fmt='.2f', ax=ax)
                        else:
                            im = ax.imshow(df.values, cmap='coolwarm', aspect='auto')
                            ax.set_xticks(range(len(df.columns)))
                            ax.set_yticks(range(len(df.index)))
                            ax.set_xticklabels(df.columns, rotation=45, ha='right')
                            ax.set_yticklabels(df.index)
                            plt.colorbar(im, ax=ax)
                        
                        ax.set_title(f'{dataset.title()} - Correlation Matrix',
                                   fontsize=14, fontweight='bold')
                        
                        plt.tight_layout()
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        logger.info(f"Added {dataset} correlation matrix")
                except Exception as e:
                    logger.warning(f"Error creating {dataset} correlation: {e}")
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'UIDAI Statistical Analysis'
        d['Author'] = "Shuvam Banerji Seal's Team"
        d['CreationDate'] = datetime.now()
    
    logger.info(f"Saved statistical PDF to {output_path}")
    return output_path


def generate_demographic_pdf(data: Dict[str, Any], output_dir: Path) -> Path:
    """Generate demographic visualization PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "demographic_analysis.pdf"
    
    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(12, 8))
        fig.text(0.5, 0.6, 'UIDAI Demographic Analysis', 
                ha='center', va='center', fontsize=28, fontweight='bold')
        fig.text(0.5, 0.45, "Shuvam Banerji Seal's Team", 
                ha='center', va='center', fontsize=16)
        fig.text(0.5, 0.35, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', va='center', fontsize=12)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        datasets = ['biometric', 'demographic', 'enrollment']
        
        for dataset in datasets:
            # Age groups analysis
            age_key = f"{dataset}_age_groups"
            if age_key in data:
                try:
                    age_data = data[age_key]
                    age_stats = age_data.get('age_group_stats', [])
                    
                    if age_stats:
                        df = pd.DataFrame(age_stats)
                        
                        if 'age_group' in df.columns and 'enrollment_count' in df.columns:
                            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                            fig.suptitle(f'{dataset.title()} - Age Group Analysis',
                                       fontsize=16, fontweight='bold')
                            
                            # Bar chart
                            axes[0].bar(df['age_group'], df['enrollment_count'],
                                       color='steelblue', alpha=0.7)
                            axes[0].set_title('Enrollment by Age Group')
                            axes[0].set_xlabel('Age Group')
                            axes[0].set_ylabel('Enrollment Count')
                            axes[0].tick_params(axis='x', rotation=45)
                            
                            # Pie chart
                            axes[1].pie(df['enrollment_count'], labels=df['age_group'],
                                       autopct='%1.1f%%', startangle=90)
                            axes[1].set_title('Age Group Distribution')
                            
                            plt.tight_layout()
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                            logger.info(f"Added {dataset} age groups plot")
                except Exception as e:
                    logger.warning(f"Error creating {dataset} age groups: {e}")
            
            # Population correlation
            pop_key = f"{dataset}_population"
            if pop_key in data:
                try:
                    pop_data = data[pop_key]
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    scatter_data = pop_data.get('scatter_data', [])
                    if scatter_data:
                        df = pd.DataFrame(scatter_data)
                        if 'population' in df.columns and 'enrollment' in df.columns:
                            ax.scatter(df['population'], df['enrollment'], 
                                      alpha=0.5, c='steelblue')
                            
                            # Add trend line if enough points
                            if len(df) > 2:
                                z = np.polyfit(df['population'], df['enrollment'], 1)
                                p = np.poly1d(z)
                                ax.plot(df['population'].sort_values(), 
                                       p(df['population'].sort_values()),
                                       "r--", alpha=0.8, label='Trend')
                            
                            ax.set_xlabel('Population')
                            ax.set_ylabel('Enrollment')
                            ax.set_title(f'{dataset.title()} - Population vs Enrollment',
                                       fontsize=14, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            ax.legend()
                    else:
                        # Show summary statistics
                        summary_text = f"""
                        Population Correlation Analysis:
                        - Correlation coefficient: {pop_data.get('correlation', 'N/A')}
                        - P-value: {pop_data.get('p_value', 'N/A')}
                        - Total observations: {pop_data.get('n_observations', 'N/A')}
                        """
                        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                               fontsize=12, ha='center', va='center')
                        ax.axis('off')
                        ax.set_title(f'{dataset.title()} - Population Correlation')
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Added {dataset} population correlation plot")
                except Exception as e:
                    logger.warning(f"Error creating {dataset} population: {e}")
            
            # Literacy analysis
            lit_key = f"{dataset}_literacy"
            if lit_key in data:
                try:
                    lit_data = data[lit_key]
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    scatter_data = lit_data.get('scatter_data', [])
                    if scatter_data:
                        df = pd.DataFrame(scatter_data)
                        if 'literacy_rate' in df.columns and 'enrollment' in df.columns:
                            ax.scatter(df['literacy_rate'], df['enrollment'],
                                      alpha=0.5, c='seagreen')
                            ax.set_xlabel('Literacy Rate (%)')
                            ax.set_ylabel('Enrollment')
                            ax.set_title(f'{dataset.title()} - Literacy vs Enrollment',
                                       fontsize=14, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                    else:
                        summary_text = f"""
                        Literacy-Enrollment Analysis:
                        - Correlation: {lit_data.get('correlation', 'N/A')}
                        - Highest literacy state: {lit_data.get('highest_literacy_state', 'N/A')}
                        - Lowest literacy state: {lit_data.get('lowest_literacy_state', 'N/A')}
                        """
                        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                               fontsize=12, ha='center', va='center')
                        ax.axis('off')
                        ax.set_title(f'{dataset.title()} - Literacy Analysis')
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Added {dataset} literacy analysis plot")
                except Exception as e:
                    logger.warning(f"Error creating {dataset} literacy: {e}")
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'UIDAI Demographic Analysis'
        d['Author'] = "Shuvam Banerji Seal's Team"
        d['CreationDate'] = datetime.now()
    
    logger.info(f"Saved demographic PDF to {output_path}")
    return output_path


def generate_summary_pdf(results: Dict[str, Any], output_dir: Path) -> Path:
    """Generate a comprehensive summary PDF of all analyses."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "analysis_summary.pdf"
    
    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(12, 8))
        fig.text(0.5, 0.65, 'UIDAI Comprehensive Analysis', 
                ha='center', va='center', fontsize=32, fontweight='bold')
        fig.text(0.5, 0.50, 'Summary Report', 
                ha='center', va='center', fontsize=24)
        fig.text(0.5, 0.38, "Shuvam Banerji Seal's Team", 
                ha='center', va='center', fontsize=16)
        fig.text(0.5, 0.28, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', va='center', fontsize=12)
        fig.text(0.5, 0.15, 'Analysis includes: Time Series, Geographic,\nStatistical, and Demographic Analysis',
                ha='center', va='center', fontsize=11)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Table of contents
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        toc_text = """
        TABLE OF CONTENTS
        ═══════════════════════════════════════════════════════
        
        1. Time Series Analysis
           - Daily Enrollment Trends
           - Seasonality Patterns
           - Anomaly Detection
        
        2. Geographic Analysis
           - State-wise Distribution
           - Regional Analysis
           - District-level Insights
        
        3. Statistical Analysis
           - Descriptive Statistics
           - Distribution Analysis
           - Correlation Analysis
        
        4. Demographic Analysis
           - Age Group Distribution
           - Population Correlation
           - Literacy Impact Analysis
        
        ═══════════════════════════════════════════════════════
        """
        ax.text(0.1, 0.9, toc_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', family='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Summary statistics page
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        summary_text = """
        EXECUTIVE SUMMARY
        ═══════════════════════════════════════════════════════
        
        This comprehensive analysis covers the UIDAI (Unique Identification 
        Authority of India) enrollment data across three primary datasets:
        
        • Biometric Data: Fingerprint and iris scan information
        • Demographic Data: Personal and address information
        • Enrollment Data: Registration and processing records
        
        Key Findings:
        ─────────────────────────────────────────────────────────
        """
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', family='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'UIDAI Comprehensive Analysis Summary'
        d['Author'] = "Shuvam Banerji Seal's Team"
        d['CreationDate'] = datetime.now()
    
    logger.info(f"Saved summary PDF to {output_path}")
    return output_path


def copy_pdfs_to_web(output_dir: Path, web_dir: Path):
    """Copy generated PDFs to web frontend public folder."""
    web_pdf_dir = web_dir / "pdfs"
    web_pdf_dir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    
    for pdf_file in output_dir.rglob("*.pdf"):
        dest = web_pdf_dir / pdf_file.name
        shutil.copy2(pdf_file, dest)
        logger.info(f"Copied {pdf_file.name} to {dest}")
    
    # Create a manifest file
    manifest = {
        "pdfs": [f.name for f in web_pdf_dir.glob("*.pdf")],
        "generated": datetime.now().isoformat()
    }
    
    manifest_path = web_pdf_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Created PDF manifest at {manifest_path}")


def main():
    """Main function to generate all visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate UIDAI Analysis Visualizations")
    parser.add_argument("--output", type=str, help="Output directory for PDFs")
    parser.add_argument("--copy-to-web", action="store_true", 
                       help="Copy PDFs to web frontend")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("UIDAI Visualization Generator")
    logger.info("=" * 60)
    
    # Load analysis results
    results = load_analysis_results()
    
    if not results:
        logger.error("No analysis results found. Run analyses first.")
        sys.exit(1)
    
    # Set up output directory
    project_root = get_project_root()
    output_dir = Path(args.output) if args.output else project_root / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    generated_pdfs = []
    
    # Generate individual analysis PDFs
    if 'time_series' in results:
        pdf_path = generate_time_series_pdf(results['time_series'], output_dir)
        generated_pdfs.append(pdf_path)
    
    if 'geographic' in results:
        pdf_path = generate_geographic_pdf(results['geographic'], output_dir)
        generated_pdfs.append(pdf_path)
    
    if 'statistical' in results:
        pdf_path = generate_statistical_pdf(results['statistical'], output_dir)
        generated_pdfs.append(pdf_path)
    
    if 'demographic' in results:
        pdf_path = generate_demographic_pdf(results['demographic'], output_dir)
        generated_pdfs.append(pdf_path)
    
    # Generate summary PDF
    summary_pdf = generate_summary_pdf(results, output_dir)
    generated_pdfs.append(summary_pdf)
    
    # Copy to web frontend if requested
    if args.copy_to_web:
        web_dir = project_root / "web" / "frontend" / "public"
        copy_pdfs_to_web(output_dir, web_dir)
    
    logger.info("=" * 60)
    logger.info("VISUALIZATION GENERATION COMPLETE")
    logger.info(f"Generated {len(generated_pdfs)} PDFs")
    for pdf in generated_pdfs:
        logger.info(f"  - {pdf}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
