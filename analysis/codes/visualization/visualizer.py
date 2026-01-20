"""
Visualization Module for UIDAI Analysis
========================================
Author: Shuvam Banerji Seal's Team
Date: January 2026

Comprehensive visualization module that generates high-quality PDF plots
for all analysis types. Supports:
- Time series plots
- Geographic distribution maps
- Demographic analysis charts
- Statistical visualizations
- ML model results
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import json

import pandas as pd
import numpy as np

# Set up matplotlib for headless rendering
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

logger = logging.getLogger(__name__)

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
    current = Path(__file__).resolve()
    return current.parent.parent.parent.parent


def ensure_output_dir(subdir: str = "") -> Path:
    """Ensure output directory exists."""
    project_root = get_project_root()
    output_dir = project_root / "results" / "visualizations"
    if subdir:
        output_dir = output_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class TimeSeriesVisualizer:
    """Generate time series visualizations."""
    
    def __init__(self, data: Dict[str, Any], output_dir: Optional[Path] = None):
        self.data = data
        self.output_dir = output_dir or ensure_output_dir("time_series")
        self.figures = []
    
    def plot_enrollment_trend(self, dataset_name: str = "biometric") -> plt.Figure:
        """Plot enrollment trend over time."""
        key = f"{dataset_name}_daily_trends"
        if key not in self.data:
            logger.warning(f"No data found for {key}")
            return None
        
        trend_data = self.data[key]
        daily_data = trend_data.get('daily_data', [])
        
        if not daily_data:
            return None
        
        df = pd.DataFrame(daily_data)
        df['date'] = pd.to_datetime(df['date'])
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Main trend plot
        ax1 = axes[0]
        ax1.plot(df['date'], df['total_enrollment'], 'b-', linewidth=2, label='Daily Enrollment')
        
        if 'ma_7' in df.columns:
            ax1.plot(df['date'], df['ma_7'], 'r--', linewidth=1.5, label='7-day MA')
        if 'ma_30' in df.columns:
            ax1.plot(df['date'], df['ma_30'], 'g-.', linewidth=1.5, label='30-day MA')
        
        ax1.set_title(f'{dataset_name.capitalize()} Enrollment Trend', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Enrollment')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Distribution plot
        ax2 = axes[1]
        ax2.hist(df['total_enrollment'], bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(df['total_enrollment'].mean(), color='red', linestyle='--', label=f'Mean: {df["total_enrollment"].mean():,.0f}')
        ax2.axvline(df['total_enrollment'].median(), color='green', linestyle=':', label=f'Median: {df["total_enrollment"].median():,.0f}')
        ax2.set_title('Enrollment Distribution')
        ax2.set_xlabel('Daily Enrollment')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        self.figures.append(('enrollment_trend', fig))
        return fig
    
    def plot_seasonality(self, dataset_name: str = "biometric") -> plt.Figure:
        """Plot seasonality patterns."""
        key = f"{dataset_name}_seasonality"
        if key not in self.data:
            return None
        
        seasonality_data = self.data[key]
        dow_data = seasonality_data.get('day_of_week', {}).get('stats', [])
        
        if not dow_data:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        days = [d.get('day_name', '')[:3] for d in dow_data]
        means = [d.get('mean', 0) for d in dow_data]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(days)))
        bars = ax.bar(days, means, color=colors, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.01,
                   f'{val:,.0f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'{dataset_name.capitalize()} - Day of Week Pattern', fontsize=14, fontweight='bold')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Average Enrollment')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self.figures.append(('seasonality', fig))
        return fig
    
    def plot_anomalies(self, dataset_name: str = "biometric") -> plt.Figure:
        """Plot anomaly detection results."""
        trends_key = f"{dataset_name}_daily_trends"
        anomaly_key = f"{dataset_name}_anomalies"
        
        if trends_key not in self.data:
            return None
        
        trend_data = self.data[trends_key]
        daily_data = trend_data.get('daily_data', [])
        
        if not daily_data:
            return None
        
        df = pd.DataFrame(daily_data)
        df['date'] = pd.to_datetime(df['date'])
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(df['date'], df['total_enrollment'], 'b-', linewidth=1.5, label='Enrollment')
        
        # Highlight anomalies if present
        if 'is_anomaly' in df.columns:
            anomalies = df[df['is_anomaly'] == True]
            ax.scatter(anomalies['date'], anomalies['total_enrollment'], 
                      color='red', s=100, zorder=5, label='Anomalies')
        
        # Add confidence bands
        mean = df['total_enrollment'].mean()
        std = df['total_enrollment'].std()
        ax.axhline(mean, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean:,.0f}')
        ax.fill_between(df['date'], mean - 2*std, mean + 2*std, alpha=0.2, color='yellow', label='±2σ Band')
        
        ax.set_title(f'{dataset_name.capitalize()} - Anomaly Detection', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Enrollment')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures.append(('anomalies', fig))
        return fig
    
    def save_all_plots(self, filename: str = "time_series_analysis.pdf") -> Path:
        """Save all generated plots to a single PDF."""
        output_path = self.output_dir / filename
        
        with PdfPages(output_path) as pdf:
            for name, fig in self.figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'UIDAI Time Series Analysis'
            d['Author'] = "Shuvam Banerji Seal's Team"
            d['CreationDate'] = datetime.now()
        
        logger.info(f"Saved time series plots to {output_path}")
        return output_path


class GeographicVisualizer:
    """Generate geographic visualizations."""
    
    def __init__(self, data: Dict[str, Any], output_dir: Optional[Path] = None):
        self.data = data
        self.output_dir = output_dir or ensure_output_dir("geographic")
        self.figures = []
    
    def plot_state_distribution(self, dataset_name: str = "biometric") -> plt.Figure:
        """Plot state-wise distribution."""
        key = f"{dataset_name}_state"
        if key not in self.data:
            return None
        
        state_data = self.data[key]
        top_states = state_data.get('top_10_states', [])
        
        if not top_states:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Bar chart
        ax1 = axes[0]
        states = [s.get('state', '')[:15] for s in top_states]
        enrollments = [s.get('total_enrollment', 0) for s in top_states]
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(states)))[::-1]
        bars = ax1.barh(states, enrollments, color=colors, edgecolor='black')
        
        ax1.set_title(f'Top 10 States - {dataset_name.capitalize()}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Total Enrollment')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, enrollments):
            ax1.text(val + max(enrollments)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:,.0f}', va='center', fontsize=9)
        
        # Pie chart for market share
        ax2 = axes[1]
        shares = [s.get('market_share_pct', 0) for s in top_states]
        
        # Add "Others" category
        total_share = sum(shares)
        if total_share < 100:
            states_pie = states + ['Others']
            shares_pie = shares + [100 - total_share]
        else:
            states_pie = states
            shares_pie = shares
        
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(states_pie)))
        wedges, texts, autotexts = ax2.pie(shares_pie, labels=states_pie, autopct='%1.1f%%',
                                           colors=colors_pie, startangle=90)
        ax2.set_title('Market Share Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self.figures.append(('state_distribution', fig))
        return fig
    
    def plot_regional_analysis(self, dataset_name: str = "biometric") -> plt.Figure:
        """Plot regional analysis."""
        key = f"{dataset_name}_regional"
        if key not in self.data:
            return None
        
        regional_data = self.data[key]
        region_stats = regional_data.get('regional_stats', [])
        
        if not region_stats:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        regions = [r.get('region', '') for r in region_stats]
        enrollments = [r.get('total_enrollment', 0) for r in region_stats]
        shares = [r.get('share_pct', 0) for r in region_stats]
        
        # Bar chart
        ax1 = axes[0]
        colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(regions)))
        bars = ax1.bar(regions, enrollments, color=colors, edgecolor='black')
        
        ax1.set_title(f'Regional Distribution - {dataset_name.capitalize()}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Total Enrollment')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Donut chart
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(shares, labels=regions, autopct='%1.1f%%',
                                           colors=colors, startangle=90,
                                           wedgeprops=dict(width=0.6))
        ax2.set_title('Regional Share', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self.figures.append(('regional_analysis', fig))
        return fig
    
    def plot_concentration_analysis(self, dataset_name: str = "biometric") -> plt.Figure:
        """Plot Lorenz curve and Gini coefficient."""
        key = f"{dataset_name}_state"
        if key not in self.data:
            return None
        
        state_data = self.data[key]
        state_details = state_data.get('state_details', [])
        summary = state_data.get('summary', {})
        
        if not state_details:
            return None
        
        enrollments = sorted([s.get('total_enrollment', 0) for s in state_details])
        cumulative = np.cumsum(enrollments) / sum(enrollments)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        n = len(enrollments)
        x = np.arange(1, n + 1) / n
        
        # Lorenz curve
        ax.plot(np.insert(x, 0, 0), np.insert(cumulative, 0, 0), 'b-', linewidth=2, label='Lorenz Curve')
        
        # Line of equality
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Line of Equality')
        
        # Fill area
        ax.fill_between(np.insert(x, 0, 0), np.insert(cumulative, 0, 0), 
                       np.insert(x, 0, 0), alpha=0.3, color='blue')
        
        gini = summary.get('gini_coefficient', 0)
        ax.set_title(f'Concentration Analysis - {dataset_name.capitalize()}\nGini Coefficient: {gini:.3f}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Cumulative Share of States')
        ax.set_ylabel('Cumulative Share of Enrollment')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        self.figures.append(('concentration', fig))
        return fig
    
    def save_all_plots(self, filename: str = "geographic_analysis.pdf") -> Path:
        """Save all generated plots to a single PDF."""
        output_path = self.output_dir / filename
        
        with PdfPages(output_path) as pdf:
            for name, fig in self.figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            d = pdf.infodict()
            d['Title'] = 'UIDAI Geographic Analysis'
            d['Author'] = "Shuvam Banerji Seal's Team"
            d['CreationDate'] = datetime.now()
        
        logger.info(f"Saved geographic plots to {output_path}")
        return output_path


class StatisticalVisualizer:
    """Generate statistical visualizations."""
    
    def __init__(self, data: Dict[str, Any], output_dir: Optional[Path] = None):
        self.data = data
        self.output_dir = output_dir or ensure_output_dir("statistical")
        self.figures = []
    
    def plot_descriptive_stats(self, dataset_name: str = "biometric") -> plt.Figure:
        """Plot descriptive statistics summary."""
        key = f"{dataset_name}_descriptive"
        if key not in self.data:
            return None
        
        desc_data = self.data[key]
        statistics = desc_data.get('statistics', {})
        
        if not statistics:
            return None
        
        # Select numeric columns for visualization
        numeric_cols = {k: v for k, v in statistics.items() 
                       if isinstance(v, dict) and 'mean' in v}
        
        if not numeric_cols:
            return None
        
        # Take first 10 columns
        cols = list(numeric_cols.keys())[:10]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Mean values
        ax1 = axes[0, 0]
        means = [numeric_cols[c].get('mean', 0) for c in cols]
        ax1.barh([c[:20] for c in cols], means, color='steelblue', edgecolor='black')
        ax1.set_title('Mean Values', fontweight='bold')
        ax1.set_xlabel('Value')
        
        # Standard deviation
        ax2 = axes[0, 1]
        stds = [numeric_cols[c].get('std', 0) for c in cols]
        ax2.barh([c[:20] for c in cols], stds, color='coral', edgecolor='black')
        ax2.set_title('Standard Deviation', fontweight='bold')
        ax2.set_xlabel('Value')
        
        # Skewness
        ax3 = axes[1, 0]
        skews = [numeric_cols[c].get('skewness', 0) for c in cols]
        colors = ['green' if s >= 0 else 'red' for s in skews]
        ax3.barh([c[:20] for c in cols], skews, color=colors, edgecolor='black')
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Skewness', fontweight='bold')
        ax3.set_xlabel('Value')
        
        # Kurtosis
        ax4 = axes[1, 1]
        kurts = [numeric_cols[c].get('kurtosis', 0) for c in cols]
        ax4.barh([c[:20] for c in cols], kurts, color='mediumpurple', edgecolor='black')
        ax4.axvline(x=3, color='red', linestyle='--', linewidth=1, label='Normal (3)')
        ax4.set_title('Kurtosis', fontweight='bold')
        ax4.set_xlabel('Value')
        ax4.legend()
        
        plt.suptitle(f'Descriptive Statistics - {dataset_name.capitalize()}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.figures.append(('descriptive_stats', fig))
        return fig
    
    def plot_correlation_matrix(self, dataset_name: str = "biometric") -> plt.Figure:
        """Plot correlation matrix heatmap."""
        key = f"{dataset_name}_correlation"
        if key not in self.data:
            return None
        
        corr_data = self.data[key]
        matrix = corr_data.get('correlation_matrix', {})
        
        if not matrix:
            return None
        
        # Convert to DataFrame
        df_corr = pd.DataFrame(matrix)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if SEABORN_AVAILABLE:
            mask = np.triu(np.ones_like(df_corr, dtype=bool))
            sns.heatmap(df_corr, mask=mask, annot=True, fmt='.2f', 
                       cmap='RdBu_r', center=0, ax=ax,
                       square=True, linewidths=0.5,
                       cbar_kws={'shrink': 0.8})
        else:
            im = ax.imshow(df_corr.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(range(len(df_corr.columns)))
            ax.set_yticks(range(len(df_corr.index)))
            ax.set_xticklabels(df_corr.columns, rotation=90)
            ax.set_yticklabels(df_corr.index)
            plt.colorbar(im, ax=ax)
        
        ax.set_title(f'Correlation Matrix - {dataset_name.capitalize()}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self.figures.append(('correlation_matrix', fig))
        return fig
    
    def plot_distribution_analysis(self, dataset_name: str = "biometric") -> plt.Figure:
        """Plot distribution analysis."""
        key = f"{dataset_name}_distribution"
        if key not in self.data:
            return None
        
        dist_data = self.data[key]
        distributions = dist_data.get('distributions', {})
        
        if not distributions:
            return None
        
        # Select first 6 columns
        cols = list(distributions.keys())[:6]
        
        n_cols = min(len(cols), 6)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for i, col in enumerate(cols[:6]):
            ax = axes[i]
            dist_info = distributions[col]
            
            # Create histogram from quantiles if available
            if 'histogram' in dist_info:
                hist = dist_info['histogram']
                ax.bar(range(len(hist['counts'])), hist['counts'], 
                      color='steelblue', edgecolor='black', alpha=0.7)
            
            ax.set_title(col[:25], fontsize=10, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused axes
        for i in range(len(cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Distribution Analysis - {dataset_name.capitalize()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.figures.append(('distribution', fig))
        return fig
    
    def plot_outlier_analysis(self, dataset_name: str = "biometric") -> plt.Figure:
        """Plot outlier analysis results."""
        key = f"{dataset_name}_outliers"
        if key not in self.data:
            return None
        
        outlier_data = self.data[key]
        summary = outlier_data.get('summary', {})
        
        if not summary:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Outlier counts by method
        ax1 = axes[0]
        methods = ['iqr_outliers', 'zscore_outliers']
        counts = [summary.get(m, 0) for m in methods]
        
        bars = ax1.bar(['IQR Method', 'Z-Score Method'], counts, 
                      color=['steelblue', 'coral'], edgecolor='black')
        
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax1.set_title('Outliers Detected by Method', fontweight='bold')
        ax1.set_ylabel('Number of Outliers')
        ax1.grid(axis='y', alpha=0.3)
        
        # Outlier percentage
        ax2 = axes[1]
        total = summary.get('total_records', 1)
        iqr_pct = (summary.get('iqr_outliers', 0) / total) * 100
        zscore_pct = (summary.get('zscore_outliers', 0) / total) * 100
        
        sizes = [iqr_pct, zscore_pct, 100 - max(iqr_pct, zscore_pct)]
        labels = [f'IQR Outliers ({iqr_pct:.1f}%)', 
                 f'Z-Score Outliers ({zscore_pct:.1f}%)',
                 'Normal']
        colors = ['steelblue', 'coral', 'lightgray']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
        ax2.set_title('Outlier Distribution', fontweight='bold')
        
        plt.suptitle(f'Outlier Analysis - {dataset_name.capitalize()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.figures.append(('outliers', fig))
        return fig
    
    def save_all_plots(self, filename: str = "statistical_analysis.pdf") -> Path:
        """Save all generated plots to a single PDF."""
        output_path = self.output_dir / filename
        
        with PdfPages(output_path) as pdf:
            for name, fig in self.figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            d = pdf.infodict()
            d['Title'] = 'UIDAI Statistical Analysis'
            d['Author'] = "Shuvam Banerji Seal's Team"
            d['CreationDate'] = datetime.now()
        
        logger.info(f"Saved statistical plots to {output_path}")
        return output_path


class DemographicVisualizer:
    """Generate demographic visualizations."""
    
    def __init__(self, data: Dict[str, Any], output_dir: Optional[Path] = None):
        self.data = data
        self.output_dir = output_dir or ensure_output_dir("demographic")
        self.figures = []
    
    def plot_population_correlation(self, dataset_name: str = "biometric") -> plt.Figure:
        """Plot population vs enrollment correlation."""
        key = f"{dataset_name}_population"
        if key not in self.data:
            return None
        
        pop_data = self.data[key]
        state_data = pop_data.get('state_level', [])
        
        if not state_data:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        populations = [s.get('population', 0) for s in state_data]
        enrollments = [s.get('total_enrollment', 0) for s in state_data]
        states = [s.get('state', '')[:10] for s in state_data]
        
        scatter = ax.scatter(populations, enrollments, s=100, c=range(len(states)),
                            cmap='viridis', edgecolors='black', alpha=0.7)
        
        # Add labels
        for i, state in enumerate(states):
            ax.annotate(state, (populations[i], enrollments[i]), 
                       fontsize=8, alpha=0.8)
        
        # Trend line
        if len(populations) > 2:
            z = np.polyfit(populations, enrollments, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(populations), max(populations), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label='Trend')
        
        ax.set_title(f'Population vs Enrollment - {dataset_name.capitalize()}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Population')
        ax.set_ylabel('Total Enrollment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures.append(('population_correlation', fig))
        return fig
    
    def plot_literacy_analysis(self, dataset_name: str = "biometric") -> plt.Figure:
        """Plot literacy rate analysis."""
        key = f"{dataset_name}_literacy"
        if key not in self.data:
            return None
        
        lit_data = self.data[key]
        state_data = lit_data.get('state_level', [])
        
        if not state_data:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        literacy_rates = [s.get('literacy_rate', 0) for s in state_data]
        enrollments = [s.get('total_enrollment', 0) for s in state_data]
        states = [s.get('state', '')[:12] for s in state_data]
        
        # Scatter plot
        ax1 = axes[0]
        scatter = ax1.scatter(literacy_rates, enrollments, s=80, c='steelblue',
                             edgecolors='black', alpha=0.7)
        
        ax1.set_title('Literacy Rate vs Enrollment', fontweight='bold')
        ax1.set_xlabel('Literacy Rate (%)')
        ax1.set_ylabel('Total Enrollment')
        ax1.grid(True, alpha=0.3)
        
        # Bar chart sorted by literacy
        ax2 = axes[1]
        sorted_idx = np.argsort(literacy_rates)[::-1][:15]  # Top 15
        
        sorted_states = [states[i] for i in sorted_idx]
        sorted_lit = [literacy_rates[i] for i in sorted_idx]
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_states)))[::-1]
        ax2.barh(sorted_states, sorted_lit, color=colors, edgecolor='black')
        ax2.set_title('Top States by Literacy Rate', fontweight='bold')
        ax2.set_xlabel('Literacy Rate (%)')
        ax2.invert_yaxis()
        
        plt.suptitle(f'Literacy Analysis - {dataset_name.capitalize()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.figures.append(('literacy_analysis', fig))
        return fig
    
    def save_all_plots(self, filename: str = "demographic_analysis.pdf") -> Path:
        """Save all generated plots to a single PDF."""
        output_path = self.output_dir / filename
        
        with PdfPages(output_path) as pdf:
            for name, fig in self.figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            d = pdf.infodict()
            d['Title'] = 'UIDAI Demographic Analysis'
            d['Author'] = "Shuvam Banerji Seal's Team"
            d['CreationDate'] = datetime.now()
        
        logger.info(f"Saved demographic plots to {output_path}")
        return output_path


class MLVisualizer:
    """Generate ML model visualizations."""
    
    def __init__(self, data: Dict[str, Any], output_dir: Optional[Path] = None):
        self.data = data
        self.output_dir = output_dir or ensure_output_dir("ml_models")
        self.figures = []
    
    def plot_model_comparison(self) -> plt.Figure:
        """Plot model comparison results."""
        comparison = self.data.get('model_comparison', {})
        
        if not comparison:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        models = list(comparison.keys())[:10]
        
        # Accuracy/R2 scores
        ax1 = axes[0]
        scores = [comparison[m].get('accuracy', comparison[m].get('r2', 0)) for m in models]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        
        bars = ax1.barh(models, scores, color=colors, edgecolor='black')
        ax1.set_title('Model Performance Scores', fontweight='bold')
        ax1.set_xlabel('Score')
        ax1.set_xlim(0, 1)
        
        # Training time
        ax2 = axes[1]
        times = [comparison[m].get('training_time', 0) for m in models]
        ax2.barh(models, times, color='coral', edgecolor='black')
        ax2.set_title('Training Time', fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        
        plt.suptitle('ML Model Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.figures.append(('model_comparison', fig))
        return fig
    
    def plot_feature_importance(self) -> plt.Figure:
        """Plot feature importance."""
        importance = self.data.get('feature_importance', {})
        
        if not importance:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features = list(importance.keys())[:20]
        scores = [importance[f] for f in features]
        
        # Sort by importance
        sorted_idx = np.argsort(scores)
        features = [features[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(features)))
        ax.barh(features, scores, color=colors, edgecolor='black')
        
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        self.figures.append(('feature_importance', fig))
        return fig
    
    def plot_predictions(self) -> plt.Figure:
        """Plot actual vs predicted values."""
        predictions = self.data.get('predictions', {})
        
        if not predictions:
            return None
        
        actual = predictions.get('actual', [])
        predicted = predictions.get('predicted', [])
        
        if not actual or not predicted:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        ax1 = axes[0]
        ax1.scatter(actual, predicted, alpha=0.5, s=20, c='steelblue')
        
        # Perfect prediction line
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax1.set_title('Actual vs Predicted', fontweight='bold')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        ax2 = axes[1]
        residuals = np.array(actual) - np.array(predicted)
        ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('Residual Distribution', fontweight='bold')
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Frequency')
        
        plt.suptitle('Prediction Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.figures.append(('predictions', fig))
        return fig
    
    def save_all_plots(self, filename: str = "ml_analysis.pdf") -> Path:
        """Save all generated plots to a single PDF."""
        output_path = self.output_dir / filename
        
        with PdfPages(output_path) as pdf:
            for name, fig in self.figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            d = pdf.infodict()
            d['Title'] = 'UIDAI ML Analysis'
            d['Author'] = "Shuvam Banerji Seal's Team"
            d['CreationDate'] = datetime.now()
        
        logger.info(f"Saved ML plots to {output_path}")
        return output_path


def generate_all_visualizations(
    results: Dict[str, Any],
    output_base_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Generate all visualizations from analysis results.
    
    Args:
        results: Dictionary containing all analysis results
        output_base_dir: Base directory for output files
        
    Returns:
        Dictionary mapping visualization types to output paths
    """
    if output_base_dir is None:
        output_base_dir = ensure_output_dir()
    
    output_paths = {}
    
    # Time Series Visualizations
    if 'time_series' in results:
        logger.info("Generating time series visualizations...")
        ts_viz = TimeSeriesVisualizer(results['time_series'], output_base_dir / "time_series")
        
        for dataset in ['biometric', 'demographic', 'enrollment']:
            ts_viz.plot_enrollment_trend(dataset)
            ts_viz.plot_seasonality(dataset)
            ts_viz.plot_anomalies(dataset)
        
        output_paths['time_series'] = ts_viz.save_all_plots()
    
    # Geographic Visualizations
    if 'geographic' in results:
        logger.info("Generating geographic visualizations...")
        geo_viz = GeographicVisualizer(results['geographic'], output_base_dir / "geographic")
        
        for dataset in ['biometric', 'demographic', 'enrollment']:
            geo_viz.plot_state_distribution(dataset)
            geo_viz.plot_regional_analysis(dataset)
            geo_viz.plot_concentration_analysis(dataset)
        
        output_paths['geographic'] = geo_viz.save_all_plots()
    
    # Statistical Visualizations
    if 'statistical' in results:
        logger.info("Generating statistical visualizations...")
        stat_viz = StatisticalVisualizer(results['statistical'], output_base_dir / "statistical")
        
        for dataset in ['biometric', 'demographic', 'enrollment']:
            stat_viz.plot_descriptive_stats(dataset)
            stat_viz.plot_correlation_matrix(dataset)
            stat_viz.plot_distribution_analysis(dataset)
            stat_viz.plot_outlier_analysis(dataset)
        
        output_paths['statistical'] = stat_viz.save_all_plots()
    
    # Demographic Visualizations
    if 'demographic' in results:
        logger.info("Generating demographic visualizations...")
        demo_viz = DemographicVisualizer(results['demographic'], output_base_dir / "demographic")
        
        for dataset in ['biometric', 'demographic', 'enrollment']:
            demo_viz.plot_population_correlation(dataset)
            demo_viz.plot_literacy_analysis(dataset)
        
        output_paths['demographic'] = demo_viz.save_all_plots()
    
    # ML Visualizations
    if 'ml_results' in results:
        logger.info("Generating ML visualizations...")
        ml_viz = MLVisualizer(results['ml_results'], output_base_dir / "ml_models")
        
        ml_viz.plot_model_comparison()
        ml_viz.plot_feature_importance()
        ml_viz.plot_predictions()
        
        output_paths['ml_models'] = ml_viz.save_all_plots()
    
    logger.info(f"All visualizations generated. Output paths: {output_paths}")
    return output_paths


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate UIDAI Analysis Visualizations")
    parser.add_argument("--input", type=str, help="Path to analysis results JSON")
    parser.add_argument("--output", type=str, help="Output directory for PDFs")
    
    args = parser.parse_args()
    
    if args.input:
        with open(args.input, 'r') as f:
            results = json.load(f)
        
        output_dir = Path(args.output) if args.output else None
        generate_all_visualizations(results, output_dir)
    else:
        print("Please provide input JSON file with --input")
