#!/usr/bin/env python3
"""
Generate Publication-Quality Figures with Large Fonts
For UIDAI Aadhaar Data Analysis Research Paper

Author: Shuvam Banerji Seal, Alok Mishra, Aheli Poddar
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from pathlib import Path
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# LARGE FONT CONFIGURATION FOR PUBLICATION
# =====================================================================

# Set publication-quality fonts - LARGE SIZE
plt.rcParams.update({
    # Figure size
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    
    # Font sizes - INCREASED FOR PUBLICATION VISIBILITY
    'font.size': 18,                    # Base font size (was 16)
    'axes.titlesize': 24,               # Title font size (was 20)
    'axes.labelsize': 20,               # Axis label size (was 18)
    'xtick.labelsize': 16,              # X-tick labels (was 14)
    'ytick.labelsize': 16,              # Y-tick labels (was 14)
    'legend.fontsize': 16,              # Legend font size (was 14)
    'legend.title_fontsize': 18,        # Legend title (was 16)
    
    # Font family
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    
    # Figure appearance
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    
    # Lines
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
    
    # Ticks
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    
    # Legend
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    
    # Saving
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Custom color palette
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',     # Orange
    'success': '#2ca02c',       # Green
    'danger': '#d62728',        # Red
    'warning': '#bcbd22',       # Yellow
    'info': '#17becf',          # Cyan
    'purple': '#9467bd',        # Purple
    'brown': '#8c564b',         # Brown
    'pink': '#e377c2',          # Pink
    'gray': '#7f7f7f',          # Gray
}

REGION_COLORS = {
    'North': '#1f77b4',
    'South': '#2ca02c',
    'East': '#ff7f0e',
    'West': '#d62728',
    'Central': '#9467bd',
    'Northeast': '#8c564b',
}

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class PublicationFigureGenerator:
    """Generate publication-quality figures with large fonts."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_info = []
    
    def save_figure(self, fig, name: str, title: str):
        """Save figure with metadata."""
        filepath = self.output_dir / f"{name}.pdf"
        fig.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        self.figures_info.append({
            'filename': f"{name}.pdf",
            'title': title,
            'path': str(filepath)
        })
        print(f"  Saved: {name}.pdf")
    
    def plot_regional_distribution(self, df: pd.DataFrame, auth_col: str, region_col: str,
                                   dataset_name: str):
        """Plot regional distribution of authentication."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Aggregate by region
        regional_data = df.groupby(region_col)[auth_col].sum().sort_values(ascending=True)
        
        # Create horizontal bar chart
        colors = [REGION_COLORS.get(r, COLORS['primary']) for r in regional_data.index]
        bars = ax.barh(regional_data.index, regional_data.values, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bar, value in zip(bars, regional_data.values):
            ax.text(value + regional_data.max() * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value/1e6:.1f}M' if value > 1e6 else f'{value/1e3:.1f}K',
                   va='center', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Total Authentication Count', fontsize=18, fontweight='bold')
        ax.set_ylabel('Region', fontsize=18, fontweight='bold')
        ax.set_title(f'Regional Distribution of Aadhaar Authentications\n({dataset_name})',
                    fontsize=20, fontweight='bold', pad=20)
        
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(axis='x', alpha=0.3, linewidth=0.8)
        
        # Add statistical annotation
        total = regional_data.sum()
        ax.text(0.98, 0.02, f'Total: {total/1e6:.1f}M authentications',
               transform=ax.transAxes, fontsize=14, ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        self.save_figure(fig, f'regional_distribution_{dataset_name.lower()}',
                        f'Regional Distribution - {dataset_name}')
    
    def plot_top_states(self, df: pd.DataFrame, auth_col: str, state_col: str,
                       dataset_name: str, top_n: int = 10):
        """Plot top states by authentication count."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get top states
        state_data = df.groupby(state_col)[auth_col].sum().nlargest(top_n).sort_values()
        
        # Create horizontal bar chart with gradient colors
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(state_data)))
        bars = ax.barh(state_data.index, state_data.values, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bar, value in zip(bars, state_data.values):
            ax.text(value + state_data.max() * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value/1e6:.2f}M' if value > 1e6 else f'{value/1e3:.1f}K',
                   va='center', fontsize=13, fontweight='bold')
        
        ax.set_xlabel('Total Authentication Count', fontsize=18, fontweight='bold')
        ax.set_ylabel('State', fontsize=18, fontweight='bold')
        ax.set_title(f'Top {top_n} States by Aadhaar Authentications\n({dataset_name})',
                    fontsize=20, fontweight='bold', pad=20)
        
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(axis='x', alpha=0.3, linewidth=0.8)
        
        # Percentage annotation
        total = df[auth_col].sum()
        top_pct = state_data.sum() / total * 100 if total > 0 else 0
        ax.text(0.98, 0.02, f'Top {top_n} states: {top_pct:.1f}% of total',
               transform=ax.transAxes, fontsize=14, ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        self.save_figure(fig, f'top_states_{dataset_name.lower()}',
                        f'Top States - {dataset_name}')
    
    def plot_day_of_week_pattern(self, df: pd.DataFrame, auth_col: str, dow_col: str,
                                 dataset_name: str):
        """Plot day of week pattern."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Aggregate by day of week
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        if df[dow_col].dtype == 'object':
            dow_data = df.groupby(dow_col)[auth_col].mean().reindex(dow_order)
        else:
            dow_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                       4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            df['dow_name'] = df[dow_col].map(dow_map)
            dow_data = df.groupby('dow_name')[auth_col].mean().reindex(dow_order)
        
        dow_data = dow_data.dropna()
        
        # Create bar chart
        colors = [COLORS['primary'] if d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                  else COLORS['secondary'] for d in dow_data.index]
        
        bars = ax.bar(dow_data.index, dow_data.values, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add horizontal mean line
        mean_val = dow_data.mean()
        ax.axhline(mean_val, color=COLORS['danger'], linestyle='--', linewidth=2.5,
                  label=f'Mean: {mean_val:.0f}')
        
        ax.set_xlabel('Day of Week', fontsize=18, fontweight='bold')
        ax.set_ylabel('Average Authentication Count', fontsize=18, fontweight='bold')
        ax.set_title(f'Aadhaar Authentication Pattern by Day of Week\n({dataset_name})',
                    fontsize=20, fontweight='bold', pad=20)
        
        ax.tick_params(axis='both', labelsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=14, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linewidth=0.8)
        
        # Weekend vs Weekday annotation
        weekday_mean = dow_data[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean()
        weekend_mean = dow_data[['Saturday', 'Sunday']].mean()
        pct_diff = (weekend_mean - weekday_mean) / weekday_mean * 100 if weekday_mean > 0 else 0
        
        ax.text(0.02, 0.98, f'Weekend vs Weekday: {"+" if pct_diff > 0 else ""}{pct_diff:.1f}%',
               transform=ax.transAxes, fontsize=14, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        self.save_figure(fig, f'day_of_week_{dataset_name.lower()}',
                        f'Day of Week Pattern - {dataset_name}')
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, columns: list, dataset_name: str):
        """Plot correlation heatmap."""
        fig, ax = plt.subplots(figsize=(14, 11))
        
        # Get available numeric columns
        available_cols = [c for c in columns if c in df.columns and df[c].dtype in ['int64', 'float64']]
        
        if len(available_cols) < 3:
            print(f"  Insufficient columns for correlation heatmap ({dataset_name})")
            return
        
        # Calculate correlation matrix
        corr_matrix = df[available_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, square=True,
                   linewidths=0.5, linecolor='white',
                   annot_kws={'size': 12, 'weight': 'bold'},
                   cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                   ax=ax)
        
        ax.set_title(f'Correlation Matrix of Key Variables\n({dataset_name})',
                    fontsize=20, fontweight='bold', pad=20)
        ax.tick_params(axis='both', labelsize=12)
        
        # Shorten labels if needed
        labels = [l.replace('_', '\n').replace('population', 'pop') for l in available_cols]
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(labels, rotation=0, fontsize=12)
        
        plt.tight_layout()
        self.save_figure(fig, f'correlation_heatmap_{dataset_name.lower()}',
                        f'Correlation Heatmap - {dataset_name}')
    
    def plot_hdi_analysis(self, df: pd.DataFrame, auth_col: str, hdi_col: str,
                         dataset_name: str):
        """Plot HDI vs Authentication analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Scatter plot
        ax1 = axes[0]
        clean_data = df[[auth_col, hdi_col]].dropna()
        
        if len(clean_data) > 100:
            sample = clean_data.sample(n=min(5000, len(clean_data)))
        else:
            sample = clean_data
        
        scatter = ax1.scatter(sample[hdi_col], sample[auth_col],
                             alpha=0.5, s=80, c=COLORS['primary'], edgecolors='white', linewidth=0.5)
        
        # Add regression line
        if len(clean_data) > 10:
            z = np.polyfit(clean_data[hdi_col], clean_data[auth_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(clean_data[hdi_col].min(), clean_data[hdi_col].max(), 100)
            ax1.plot(x_line, p(x_line), color=COLORS['danger'], linewidth=3,
                    linestyle='--', label='Trend Line')
            
            # Calculate R-squared
            r, _ = stats.pearsonr(clean_data[hdi_col], clean_data[auth_col])
            ax1.text(0.05, 0.95, f'r = {r:.3f}\nR² = {r**2:.3f}',
                    transform=ax1.transAxes, fontsize=14, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax1.set_xlabel('Human Development Index (HDI)', fontsize=18, fontweight='bold')
        ax1.set_ylabel('Authentication Count', fontsize=18, fontweight='bold')
        ax1.set_title('HDI vs Authentication\n(Scatter)', fontsize=18, fontweight='bold', pad=15)
        ax1.tick_params(axis='both', labelsize=14)
        ax1.legend(fontsize=14)
        ax1.grid(alpha=0.3)
        
        # HDI category distribution
        ax2 = axes[1]
        
        # Categorize HDI
        hdi_bins = [0, 0.55, 0.7, 0.8, 1.0]
        hdi_labels = ['Low\n(<0.55)', 'Medium\n(0.55-0.7)', 'High\n(0.7-0.8)', 'Very High\n(>0.8)']
        clean_data['hdi_category'] = pd.cut(clean_data[hdi_col], bins=hdi_bins, labels=hdi_labels)
        
        category_data = clean_data.groupby('hdi_category')[auth_col].mean()
        
        colors = [COLORS['danger'], COLORS['warning'], COLORS['info'], COLORS['success']]
        bars = ax2.bar(category_data.index, category_data.values, color=colors,
                      edgecolor='black', linewidth=1.5)
        
        ax2.set_xlabel('HDI Category', fontsize=18, fontweight='bold')
        ax2.set_ylabel('Average Authentication Count', fontsize=18, fontweight='bold')
        ax2.set_title('Authentication by HDI Category\n(Bar)', fontsize=18, fontweight='bold', pad=15)
        ax2.tick_params(axis='both', labelsize=14)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, category_data.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + category_data.max()*0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        fig.suptitle(f'Development Level Analysis ({dataset_name})', fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.save_figure(fig, f'hdi_analysis_{dataset_name.lower()}',
                        f'HDI Analysis - {dataset_name}')
    
    def plot_climate_analysis(self, df: pd.DataFrame, auth_col: str, climate_col: str,
                             dataset_name: str):
        """Plot climate zone analysis."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        if climate_col not in df.columns:
            print(f"  Climate column not found ({dataset_name})")
            return
        
        # Aggregate by climate zone
        climate_data = df.groupby(climate_col)[auth_col].agg(['sum', 'mean', 'count']).reset_index()
        climate_data = climate_data.sort_values('sum', ascending=True)
        
        # Create horizontal bar chart
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(climate_data)))
        
        bars = ax.barh(climate_data[climate_col], climate_data['sum'], color=colors,
                      edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Total Authentication Count', fontsize=18, fontweight='bold')
        ax.set_ylabel('Climate Zone', fontsize=18, fontweight='bold')
        ax.set_title(f'Aadhaar Authentication Distribution by Climate Zone\n({dataset_name})',
                    fontsize=20, fontweight='bold', pad=20)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(axis='x', alpha=0.3, linewidth=0.8)
        
        # Add value labels
        for bar, value in zip(bars, climate_data['sum'].values):
            ax.text(value + climate_data['sum'].max() * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value/1e6:.1f}M', va='center', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, f'climate_analysis_{dataset_name.lower()}',
                        f'Climate Zone Analysis - {dataset_name}')
    
    def plot_air_quality_analysis(self, df: pd.DataFrame, auth_col: str,
                                  aqi_col: str, dataset_name: str):
        """Plot air quality vs authentication analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        if aqi_col not in df.columns:
            print(f"  AQI column not found ({dataset_name})")
            return
        
        clean_data = df[[auth_col, aqi_col]].dropna()
        
        # Scatter plot
        ax1 = axes[0]
        if len(clean_data) > 100:
            sample = clean_data.sample(n=min(3000, len(clean_data)))
        else:
            sample = clean_data
        
        # Color by AQI category
        colors = []
        for aqi in sample[aqi_col]:
            if aqi <= 50:
                colors.append(COLORS['success'])
            elif aqi <= 100:
                colors.append(COLORS['warning'])
            elif aqi <= 150:
                colors.append(COLORS['secondary'])
            else:
                colors.append(COLORS['danger'])
        
        scatter = ax1.scatter(sample[aqi_col], sample[auth_col], c=colors,
                             alpha=0.6, s=80, edgecolors='white', linewidth=0.5)
        
        ax1.set_xlabel('Air Quality Index (AQI)', fontsize=18, fontweight='bold')
        ax1.set_ylabel('Authentication Count', fontsize=18, fontweight='bold')
        ax1.set_title('AQI vs Authentication\n(Color-coded by Category)', fontsize=18, fontweight='bold', pad=15)
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['success'], label='Good (≤50)'),
            Patch(facecolor=COLORS['warning'], label='Moderate (51-100)'),
            Patch(facecolor=COLORS['secondary'], label='Unhealthy for Sensitive (101-150)'),
            Patch(facecolor=COLORS['danger'], label='Unhealthy (>150)'),
        ]
        ax1.legend(handles=legend_elements, fontsize=12, loc='upper right')
        
        # AQI category box plot
        ax2 = axes[1]
        
        aqi_bins = [0, 50, 100, 150, 500]
        aqi_labels = ['Good', 'Moderate', 'Unhealthy\n(Sensitive)', 'Unhealthy']
        clean_data['aqi_category'] = pd.cut(clean_data[aqi_col], bins=aqi_bins, labels=aqi_labels)
        
        box_colors = [COLORS['success'], COLORS['warning'], COLORS['secondary'], COLORS['danger']]
        
        bp = ax2.boxplot([clean_data[clean_data['aqi_category'] == cat][auth_col].dropna().values
                         for cat in aqi_labels if cat in clean_data['aqi_category'].values],
                        labels=[cat for cat in aqi_labels if cat in clean_data['aqi_category'].values],
                        patch_artist=True)
        
        for patch, color in zip(bp['boxes'], box_colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('AQI Category', fontsize=18, fontweight='bold')
        ax2.set_ylabel('Authentication Count', fontsize=18, fontweight='bold')
        ax2.set_title('Authentication Distribution\nby AQI Category', fontsize=18, fontweight='bold', pad=15)
        ax2.tick_params(axis='both', labelsize=14)
        ax2.grid(axis='y', alpha=0.3)
        
        fig.suptitle(f'Air Quality Impact Analysis ({dataset_name})', fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.save_figure(fig, f'air_quality_analysis_{dataset_name.lower()}',
                        f'Air Quality Analysis - {dataset_name}')
    
    def plot_infrastructure_analysis(self, df: pd.DataFrame, auth_col: str, dataset_name: str):
        """Plot infrastructure correlation analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        infra_cols = ['hospitals_per_100k', 'schools_per_100k', 'banks_per_100k', 'mobile_penetration']
        titles = ['Healthcare Infrastructure', 'Education Infrastructure',
                 'Banking Infrastructure', 'Mobile Connectivity']
        colors_list = [COLORS['danger'], COLORS['primary'], COLORS['success'], COLORS['purple']]
        
        for ax, col, title, color in zip(axes.flatten(), infra_cols, titles, colors_list):
            if col not in df.columns:
                ax.text(0.5, 0.5, f'{title}\nData Not Available',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_title(title, fontsize=16, fontweight='bold')
                continue
            
            clean_data = df[[auth_col, col]].dropna()
            
            if len(clean_data) > 100:
                sample = clean_data.sample(n=min(2000, len(clean_data)))
            else:
                sample = clean_data
            
            ax.scatter(sample[col], sample[auth_col], alpha=0.5, s=60, c=color,
                      edgecolors='white', linewidth=0.3)
            
            # Trend line
            if len(clean_data) > 10:
                z = np.polyfit(clean_data[col], clean_data[auth_col], 1)
                p = np.poly1d(z)
                x_line = np.linspace(clean_data[col].min(), clean_data[col].max(), 100)
                ax.plot(x_line, p(x_line), color='black', linewidth=2.5, linestyle='--')
                
                r, _ = stats.pearsonr(clean_data[col], clean_data[auth_col])
                ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
                       fontsize=14, va='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            ax.set_xlabel(col.replace('_', ' ').title(), fontsize=16, fontweight='bold')
            ax.set_ylabel('Authentication Count', fontsize=16, fontweight='bold')
            ax.set_title(title, fontsize=18, fontweight='bold', pad=10)
            ax.tick_params(axis='both', labelsize=13)
            ax.grid(alpha=0.3)
        
        fig.suptitle(f'Infrastructure Correlation Analysis ({dataset_name})',
                    fontsize=22, fontweight='bold', y=0.98)
        plt.tight_layout()
        self.save_figure(fig, f'infrastructure_analysis_{dataset_name.lower()}',
                        f'Infrastructure Analysis - {dataset_name}')
    
    def plot_socioeconomic_analysis(self, df: pd.DataFrame, auth_col: str, dataset_name: str):
        """Plot socioeconomic indicators analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        socio_cols = ['literacy_rate', 'sdg_score', 'financial_inclusion_index', 'health_index']
        titles = ['Literacy Rate', 'SDG Score', 'Financial Inclusion', 'Health Index']
        cmaps = ['Blues', 'Greens', 'Oranges', 'Reds']
        
        for ax, col, title, cmap in zip(axes.flatten(), socio_cols, titles, cmaps):
            if col not in df.columns:
                ax.text(0.5, 0.5, f'{title}\nData Not Available',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_title(title, fontsize=16, fontweight='bold')
                continue
            
            clean_data = df[[auth_col, col]].dropna()
            
            if len(clean_data) < 10:
                continue
            
            # Bin the socioeconomic indicator
            n_bins = 5
            clean_data['bin'] = pd.qcut(clean_data[col], q=n_bins, labels=False, duplicates='drop')
            
            bin_data = clean_data.groupby('bin')[auth_col].mean()
            bin_ranges = clean_data.groupby('bin')[col].agg(['min', 'max'])
            
            colors = plt.cm.get_cmap(cmap)(np.linspace(0.3, 0.9, len(bin_data)))
            
            bars = ax.bar(range(len(bin_data)), bin_data.values, color=colors,
                         edgecolor='black', linewidth=1.2)
            
            # Labels
            labels = [f'{bin_ranges.loc[i, "min"]:.1f}-{bin_ranges.loc[i, "max"]:.1f}'
                     for i in bin_data.index]
            ax.set_xticks(range(len(bin_data)))
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=12)
            
            ax.set_xlabel(f'{title} Range', fontsize=16, fontweight='bold')
            ax.set_ylabel('Mean Authentication Count', fontsize=16, fontweight='bold')
            ax.set_title(f'Authentication by {title}', fontsize=18, fontweight='bold', pad=10)
            ax.tick_params(axis='both', labelsize=13)
            ax.grid(axis='y', alpha=0.3)
        
        fig.suptitle(f'Socioeconomic Indicators Analysis ({dataset_name})',
                    fontsize=22, fontweight='bold', y=0.98)
        plt.tight_layout()
        self.save_figure(fig, f'socioeconomic_analysis_{dataset_name.lower()}',
                        f'Socioeconomic Analysis - {dataset_name}')
    
    def plot_ml_model_comparison(self, model_results: dict, dataset_name: str):
        """Plot ML model comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Regression models
        ax1 = axes[0]
        reg_models = {k: v['r2'] for k, v in model_results.items()
                     if 'r2' in v and v['r2'] is not None}
        
        if reg_models:
            models = list(reg_models.keys())
            scores = list(reg_models.values())
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
            
            bars = ax1.barh(models, scores, color=colors, edgecolor='black', linewidth=1.2)
            
            for bar, score in zip(bars, scores):
                ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', va='center', fontsize=14, fontweight='bold')
            
            ax1.set_xlabel('R² Score', fontsize=18, fontweight='bold')
            ax1.set_xlim(0, 1.0)
            ax1.set_title('Regression Model Performance', fontsize=18, fontweight='bold', pad=15)
            ax1.tick_params(axis='both', labelsize=14)
            ax1.grid(axis='x', alpha=0.3)
        
        # Classification models (if available)
        ax2 = axes[1]
        cls_models = {k: v.get('accuracy', v.get('f1_score'))
                     for k, v in model_results.items()
                     if 'accuracy' in v or 'f1_score' in v}
        
        if cls_models:
            models = list(cls_models.keys())
            scores = list(cls_models.values())
            colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(models)))
            
            bars = ax2.barh(models, scores, color=colors, edgecolor='black', linewidth=1.2)
            
            ax2.set_xlabel('Score', fontsize=18, fontweight='bold')
            ax2.set_xlim(0, 1.0)
            ax2.set_title('Classification Model Performance', fontsize=18, fontweight='bold', pad=15)
            ax2.tick_params(axis='both', labelsize=14)
            ax2.grid(axis='x', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Classification Results\nNot Available',
                    ha='center', va='center', fontsize=16, transform=ax2.transAxes)
            ax2.set_title('Classification Model Performance', fontsize=18, fontweight='bold')
        
        fig.suptitle(f'Machine Learning Model Comparison ({dataset_name})',
                    fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.save_figure(fig, f'ml_comparison_{dataset_name.lower()}',
                        f'ML Model Comparison - {dataset_name}')
    
    def plot_hypothesis_test_summary(self, test_results: dict, dataset_name: str):
        """Plot hypothesis test summary."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        tests = []
        p_values = []
        significant = []
        
        for key, result in test_results.items():
            if isinstance(result, dict) and 'p_value' in result:
                tests.append(key.replace('_', ' ').title()[:30])
                p_values.append(result['p_value'])
                significant.append(result.get('significant', result['p_value'] < 0.05))
        
        if not tests:
            ax.text(0.5, 0.5, 'No Hypothesis Test Results Available',
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            self.save_figure(fig, f'hypothesis_tests_{dataset_name.lower()}',
                           f'Hypothesis Tests - {dataset_name}')
            return
        
        # Create bar chart
        colors = [COLORS['success'] if s else COLORS['gray'] for s in significant]
        
        y_pos = np.arange(len(tests))
        log_p = [-np.log10(p) if p > 0 else 10 for p in p_values]
        
        bars = ax.barh(y_pos, log_p, color=colors, edgecolor='black', linewidth=1.2)
        
        # Significance threshold
        ax.axvline(-np.log10(0.05), color=COLORS['danger'], linestyle='--',
                  linewidth=2.5, label='α = 0.05')
        ax.axvline(-np.log10(0.01), color=COLORS['warning'], linestyle=':',
                  linewidth=2.5, label='α = 0.01')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tests, fontsize=13)
        ax.set_xlabel('-log₁₀(p-value)', fontsize=18, fontweight='bold')
        ax.set_title(f'Hypothesis Test Results ({dataset_name})',
                    fontsize=20, fontweight='bold', pad=20)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(fontsize=14, loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        # Add significance count
        n_sig = sum(significant)
        ax.text(0.02, 0.98, f'{n_sig}/{len(tests)} tests significant (p < 0.05)',
               transform=ax.transAxes, fontsize=14, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        self.save_figure(fig, f'hypothesis_tests_{dataset_name.lower()}',
                        f'Hypothesis Tests - {dataset_name}')
    
    def generate_all_figures(self, df: pd.DataFrame, dataset_name: str,
                            auth_col: str = 'total_authentication_count',
                            state_col: str = 'state',
                            region_col: str = 'region',
                            dow_col: str = 'day_of_week'):
        """Generate all figures for a dataset."""
        print(f"\nGenerating figures for {dataset_name}...")
        print("="*60)
        
        # Regional distribution
        if region_col in df.columns and auth_col in df.columns:
            self.plot_regional_distribution(df, auth_col, region_col, dataset_name)
        
        # Top states
        if state_col in df.columns and auth_col in df.columns:
            self.plot_top_states(df, auth_col, state_col, dataset_name)
        
        # Day of week
        if dow_col in df.columns and auth_col in df.columns:
            self.plot_day_of_week_pattern(df, auth_col, dow_col, dataset_name)
        
        # Correlation heatmap
        numeric_cols = ['total_authentication_count', 'accept_count', 'reject_count',
                       'male_population', 'female_population', 'total_population',
                       'population_density', 'literacy_rate', 'hdi',
                       'temperature_c', 'aqi', 'elevation_m']
        available_cols = [c for c in numeric_cols if c in df.columns]
        if len(available_cols) >= 3:
            self.plot_correlation_heatmap(df, available_cols, dataset_name)
        
        # HDI analysis
        if 'hdi' in df.columns and auth_col in df.columns:
            self.plot_hdi_analysis(df, auth_col, 'hdi', dataset_name)
        
        # Climate analysis
        if 'climate_zone' in df.columns and auth_col in df.columns:
            self.plot_climate_analysis(df, auth_col, 'climate_zone', dataset_name)
        
        # Air quality analysis
        if 'aqi' in df.columns and auth_col in df.columns:
            self.plot_air_quality_analysis(df, auth_col, 'aqi', dataset_name)
        
        # Infrastructure analysis
        self.plot_infrastructure_analysis(df, auth_col, dataset_name)
        
        # Socioeconomic analysis
        self.plot_socioeconomic_analysis(df, auth_col, dataset_name)
        
        print(f"\nGenerated {len(self.figures_info)} figures for {dataset_name}")
        return self.figures_info


def main():
    """Generate all publication figures."""
    
    print("="*80)
    print("GENERATING PUBLICATION-QUALITY FIGURES WITH LARGE FONTS")
    print("="*80)
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "paper" / "figures_large"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = PublicationFigureGenerator(output_dir)
    
    # Load data - USE CLEANED DATA (not corrected or augmented)
    # Priority: augmented (if available) > cleaned (correct) > corrected (fallback)
    data_paths = {
        'Biometric': project_root / "Dataset" / "api_augmented" / "api_augmented_biometric.csv",
        'Demographic': project_root / "Dataset" / "api_augmented" / "api_augmented_demographic.csv",
        'Enrollment': project_root / "Dataset" / "api_augmented" / "api_augmented_enrollment.csv",
    }
    
    # Cleaned paths (use these if augmented not available)
    cleaned_paths = {
        'Biometric': project_root / "Dataset" / "cleaned" / "biometric" / "biometric" / "final_cleaned_biometric.csv",
        'Demographic': project_root / "Dataset" / "cleaned" / "demographic" / "demographic" / "final_cleaned_demographic.csv",
        'Enrollment': project_root / "Dataset" / "cleaned" / "enrollment" / "enrollment" / "final_cleaned_enrollment.csv",
    }
    
    all_figures = []
    
    for name, path in data_paths.items():
        if not path.exists():
            path = cleaned_paths.get(name)
        
        if path and path.exists():
            print(f"\nLoading ALL {name} data from: {path}")
            df = pd.read_csv(path)  # NO nrows limit - process ALL data
            print(f"Loaded {len(df):,} records with {len(df.columns)} columns")
            
            # Determine auth column (depends on dataset type)
            auth_cols = ['total_authentication_count', 'bio_age_5_17', 'demo_age_5_17', 
                         'age_5_17', 'total_population', 'male_population']
            auth_col = next((c for c in auth_cols if c in df.columns), None)
            
            if not auth_col:
                # Sum numeric columns for total
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                age_cols = [c for c in numeric_cols if 'age' in c.lower()]
                if age_cols:
                    auth_col = age_cols[0]
            
            if auth_col:
                figures = generator.generate_all_figures(df, name, auth_col=auth_col)
                all_figures.extend(figures)
            else:
                print(f"  Warning: No suitable numeric column found for {name}")
            
            # Determine auth column
            auth_cols = ['total_authentication_count', 'total_population', 'male_population']
            auth_col = next((c for c in auth_cols if c in df.columns), None)
            
            if auth_col:
                figures = generator.generate_all_figures(df, name, auth_col=auth_col)
                all_figures.extend(figures)
    
    # Save figures metadata
    metadata_path = output_dir / "figures_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(all_figures, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"FIGURE GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total figures generated: {len(all_figures)}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
