#!/usr/bin/env python3
"""
Publication-Quality Visualization Generator for UIDAI Aadhaar Analysis
Authors: Shuvam Banerji Seal, Alok Mishra, Aheli Poddar
UIDAI Data Hackathon 2026

Generates high-quality plots for research paper inclusion.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Color palette (coffee green theme)
COLORS = {
    'primary': '#4a7c59',
    'secondary': '#87a96b',
    'accent': '#6b4423',
    'light': '#d4c5a9',
    'dark': '#2d4a3e',
    'regions': {
        'North': '#3498db',
        'South': '#e74c3c',
        'East': '#2ecc71',
        'West': '#f39c12',
        'Central': '#9b59b6',
        'Northeast': '#1abc9c',
        'Other': '#95a5a6'
    }
}

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / 'results' / 'comprehensive_analysis'
FIGURES_DIR = BASE_DIR / 'paper' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_analysis_results():
    """Load comprehensive analysis results"""
    results_file = RESULTS_DIR / 'comprehensive_analysis_results.json'
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def format_large_number(x, pos):
    """Format large numbers for axis labels"""
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    return f'{x:.0f}'


def plot_regional_distribution(results, dataset='biometric'):
    """Plot regional enrollment distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    regional = results['datasets'][dataset]['geographic']['regional_analysis']
    
    # Filter out 'Other' for cleaner visualization
    regions = [r for r in regional.keys() if r != 'Other']
    totals = [regional[r]['total_enrollment'] for r in regions]
    percentages = [regional[r]['percentage'] for r in regions]
    
    # Bar chart
    ax1 = axes[0]
    colors = [COLORS['regions'].get(r, COLORS['light']) for r in regions]
    bars = ax1.bar(regions, totals, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Region')
    ax1.set_ylabel('Total Enrollment')
    ax1.set_title(f'Regional Enrollment Distribution ({dataset.title()})')
    ax1.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars, totals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(totals)*0.02,
                f'{val/1e6:.2f}M', ha='center', va='bottom', fontsize=8)
    
    # Pie chart
    ax2 = axes[1]
    explode = [0.05] * len(regions)
    wedges, texts, autotexts = ax2.pie(percentages, labels=regions, autopct='%1.1f%%',
                                        colors=colors, explode=explode, startangle=90)
    ax2.set_title('Percentage Share by Region')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'regional_distribution_{dataset}.pdf')
    plt.savefig(FIGURES_DIR / f'regional_distribution_{dataset}.png')
    plt.close()
    print(f"✓ Saved regional distribution plot for {dataset}")


def plot_state_rankings(results, dataset='biometric'):
    """Plot top 10 state rankings"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_states = results['datasets'][dataset]['geographic']['state_analysis']['top_10_states']
    
    states = [s['state'] for s in top_states]
    enrollments = [s['total_enrollment'] for s in top_states]
    
    # Horizontal bar chart
    y_pos = np.arange(len(states))
    bars = ax.barh(y_pos, enrollments, color=COLORS['primary'], edgecolor='white')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(states)
    ax.invert_yaxis()
    ax.set_xlabel('Total Enrollment')
    ax.set_title(f'Top 10 States by Enrollment ({dataset.title()} Dataset)')
    ax.xaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    # Add value labels
    for bar, val in zip(bars, enrollments):
        ax.text(val + max(enrollments)*0.01, bar.get_y() + bar.get_height()/2,
               f'{val:,.0f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'top_states_{dataset}.pdf')
    plt.savefig(FIGURES_DIR / f'top_states_{dataset}.png')
    plt.close()
    print(f"✓ Saved state rankings plot for {dataset}")


def plot_day_of_week_pattern(results, dataset='biometric'):
    """Plot day of week enrollment pattern"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    dow_data = results['datasets'][dataset]['time_series']['day_of_week_pattern']
    
    # Order days correctly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days = [d for d in day_order if d in dow_data]
    totals = [dow_data[d]['total'] for d in days]
    means = [dow_data[d]['mean'] for d in days]
    
    x = np.arange(len(days))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, totals, width, label='Total Enrollment', 
                   color=COLORS['primary'], edgecolor='white')
    
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, means, width, label='Mean per Record',
                    color=COLORS['secondary'], edgecolor='white')
    
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Total Enrollment', color=COLORS['primary'])
    ax2.set_ylabel('Mean Enrollment per Record', color=COLORS['secondary'])
    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45)
    ax.set_title(f'Enrollment Pattern by Day of Week ({dataset.title()})')
    
    ax.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'day_of_week_{dataset}.pdf')
    plt.savefig(FIGURES_DIR / f'day_of_week_{dataset}.png')
    plt.close()
    print(f"✓ Saved day of week pattern plot for {dataset}")


def plot_weekend_vs_weekday(results):
    """Plot weekend vs weekday comparison across datasets"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = ['biometric', 'demographic', 'enrollment']
    x = np.arange(len(datasets))
    width = 0.35
    
    weekend_totals = []
    weekday_totals = []
    
    for ds in datasets:
        ww = results['datasets'][ds]['time_series']['weekend_vs_weekday']
        weekend_totals.append(ww['weekend']['total'])
        weekday_totals.append(ww['weekday']['total'])
    
    bars1 = ax.bar(x - width/2, weekend_totals, width, label='Weekend', color=COLORS['accent'])
    bars2 = ax.bar(x + width/2, weekday_totals, width, label='Weekday', color=COLORS['primary'])
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Total Enrollment')
    ax.set_title('Weekend vs Weekday Enrollment Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in datasets])
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max(weekday_totals)*0.02,
                   f'{height/1e6:.1f}M', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'weekend_weekday_comparison.pdf')
    plt.savefig(FIGURES_DIR / 'weekend_weekday_comparison.png')
    plt.close()
    print("✓ Saved weekend vs weekday comparison plot")


def plot_correlation_heatmap(results, dataset='biometric'):
    """Plot key correlations for a dataset"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_data = results['datasets'][dataset]['correlations'].get('enrollment_correlations', [])
    
    if not corr_data:
        print(f"⚠ No correlation data for {dataset}")
        return
    
    # Top 10 correlations
    top_corrs = corr_data[:10]
    variables = [c['variable'] for c in top_corrs]
    correlations = [c['correlation_with_enrollment'] for c in top_corrs]
    
    # Color based on positive/negative
    colors = [COLORS['primary'] if c > 0 else COLORS['accent'] for c in correlations]
    
    y_pos = np.arange(len(variables))
    bars = ax.barh(y_pos, correlations, color=colors, edgecolor='white')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel('Pearson Correlation with Total Enrollment')
    ax.set_title(f'Top Correlations with Enrollment ({dataset.title()})')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-1, 1)
    
    # Add correlation values
    for bar, corr in zip(bars, correlations):
        x_pos = corr + 0.02 if corr >= 0 else corr - 0.15
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{corr:.3f}',
               va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'correlations_{dataset}.pdf')
    plt.savefig(FIGURES_DIR / f'correlations_{dataset}.png')
    plt.close()
    print(f"✓ Saved correlation plot for {dataset}")


def plot_hypothesis_test_summary(results):
    """Plot summary of hypothesis tests across datasets"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_tests = []
    for ds in ['biometric', 'demographic', 'enrollment']:
        tests = results['datasets'][ds]['hypothesis_tests'].get('tests', [])
        for t in tests:
            all_tests.append({
                'dataset': ds,
                'test': t['test_name'].split(':')[1].strip() if ':' in t['test_name'] else t['test_name'],
                'p_value': t['p_value'],
                'significant': t['p_value'] < 0.05
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_tests)
    
    # Group by test
    test_names = df['test'].unique()
    x = np.arange(len(test_names))
    width = 0.25
    
    datasets = ['biometric', 'demographic', 'enrollment']
    colors_ds = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
    
    for i, (ds, color) in enumerate(zip(datasets, colors_ds)):
        ds_data = df[df['dataset'] == ds]
        p_values = []
        for test in test_names:
            test_data = ds_data[ds_data['test'] == test]
            if len(test_data) > 0:
                p_values.append(-np.log10(test_data.iloc[0]['p_value'] + 1e-100))
            else:
                p_values.append(0)
        
        ax.bar(x + i*width, p_values, width, label=ds.title(), color=color)
    
    # Add significance line
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
    
    ax.set_xlabel('Hypothesis Test')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Statistical Significance of Hypothesis Tests')
    ax.set_xticks(x + width)
    ax.set_xticklabels(test_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'hypothesis_tests_summary.pdf')
    plt.savefig(FIGURES_DIR / 'hypothesis_tests_summary.png')
    plt.close()
    print("✓ Saved hypothesis tests summary plot")


def plot_hdi_analysis(results, dataset='biometric'):
    """Plot HDI stratification analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    socio = results['datasets'][dataset]['socioeconomic']
    
    # HDI stratification
    if 'hdi_stratification' in socio:
        ax1 = axes[0]
        strat = socio['hdi_stratification']
        
        categories = ['High HDI\n(≥0.65)', 'Medium HDI\n(0.55-0.65)', 'Low HDI\n(<0.55)']
        counts = [strat['high_hdi_states']['count'], 
                 strat['medium_hdi_states']['count'],
                 strat['low_hdi_states']['count']]
        enrollments = [strat['high_hdi_states']['mean_enrollment'],
                      strat['medium_hdi_states']['mean_enrollment'],
                      strat['low_hdi_states']['mean_enrollment']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, counts, width, label='Number of States', color=COLORS['primary'])
        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x + width/2, enrollments, width, label='Mean Enrollment', color=COLORS['secondary'])
        
        ax1.set_xlabel('HDI Category')
        ax1.set_ylabel('Number of States', color=COLORS['primary'])
        ax1_twin.set_ylabel('Mean Enrollment', color=COLORS['secondary'])
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.set_title(f'HDI Stratification Analysis ({dataset.title()})')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax1_twin.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    # HDI correlation
    ax2 = axes[1]
    if 'hdi_analysis' in socio:
        hdi = socio['hdi_analysis']
        corr = hdi['correlation_with_enrollment']
        p_val = hdi['p_value']
        
        # Simple visualization
        ax2.bar(['HDI Correlation'], [corr], color=COLORS['primary'] if corr > 0 else COLORS['accent'])
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_ylabel('Pearson Correlation')
        ax2.set_title(f'HDI-Enrollment Correlation\n(r={corr:.3f}, p={p_val:.4f})')
        ax2.set_ylim(-1, 1)
        
        # Significance indicator
        if p_val < 0.05:
            ax2.text(0, corr + 0.05, '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*',
                    ha='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'hdi_analysis_{dataset}.pdf')
    plt.savefig(FIGURES_DIR / f'hdi_analysis_{dataset}.png')
    plt.close()
    print(f"✓ Saved HDI analysis plot for {dataset}")


def plot_climate_analysis(results, dataset='biometric'):
    """Plot climate zone analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    climate = results['datasets'][dataset]['climate']
    
    # Rainfall zone
    ax1 = axes[0]
    if 'rainfall_zone_analysis' in climate:
        rainfall = climate['rainfall_zone_analysis']
        zones = list(rainfall.keys())
        totals = [rainfall[z]['total_enrollment'] for z in zones]
        
        bars = ax1.bar(zones, totals, color=COLORS['primary'], edgecolor='white')
        ax1.set_xlabel('Rainfall Zone')
        ax1.set_ylabel('Total Enrollment')
        ax1.set_title(f'Enrollment by Rainfall Zone ({dataset.title()})')
        ax1.tick_params(axis='x', rotation=45)
        ax1.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    # Climate type
    ax2 = axes[1]
    if 'climate_type_analysis' in climate:
        clim = climate['climate_type_analysis']
        types = list(clim.keys())[:8]  # Top 8
        totals = [clim[t]['total_enrollment'] for t in types]
        
        bars = ax2.barh(types, totals, color=COLORS['secondary'], edgecolor='white')
        ax2.set_xlabel('Total Enrollment')
        ax2.set_title(f'Enrollment by Climate Type ({dataset.title()})')
        ax2.xaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'climate_analysis_{dataset}.pdf')
    plt.savefig(FIGURES_DIR / f'climate_analysis_{dataset}.png')
    plt.close()
    print(f"✓ Saved climate analysis plot for {dataset}")


def plot_ml_results():
    """Plot ML model comparison from training results"""
    ml_results_file = BASE_DIR / 'results' / 'models' / 'ml_training_results.json'
    
    if not ml_results_file.exists():
        print("⚠ ML results not found")
        return
    
    with open(ml_results_file) as f:
        ml_results = json.load(f)
    
    # Classification accuracy comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get biometric classification results
    bio_class = ml_results['datasets']['biometric']['classification']['models']
    
    models = list(bio_class.keys())
    accuracies = [bio_class[m]['accuracy'] for m in models]
    
    # Sort by accuracy
    sorted_idx = np.argsort(accuracies)[::-1]
    models = [models[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]
    
    ax1 = axes[0, 0]
    colors = [COLORS['primary'] if a > 0.9 else COLORS['secondary'] if a > 0.7 else COLORS['accent'] for a in accuracies]
    bars = ax1.barh(models, accuracies, color=colors, edgecolor='white')
    ax1.set_xlabel('Accuracy')
    ax1.set_title('Classification Model Accuracy (Biometric)')
    ax1.set_xlim(0, 1.1)
    ax1.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(acc + 0.01, bar.get_y() + bar.get_height()/2, f'{acc:.3f}',
                va='center', fontsize=7)
    
    # Regression R² comparison
    bio_reg = ml_results['datasets']['biometric']['regression']['models']
    
    models_reg = list(bio_reg.keys())
    r2_scores = [bio_reg[m]['r2'] for m in models_reg]
    
    # Filter out negative R²
    valid_idx = [i for i, r2 in enumerate(r2_scores) if r2 > 0]
    models_reg = [models_reg[i] for i in valid_idx]
    r2_scores = [r2_scores[i] for i in valid_idx]
    
    sorted_idx = np.argsort(r2_scores)[::-1]
    models_reg = [models_reg[i] for i in sorted_idx[:10]]
    r2_scores = [r2_scores[i] for i in sorted_idx[:10]]
    
    ax2 = axes[0, 1]
    bars = ax2.barh(models_reg, r2_scores, color=COLORS['secondary'], edgecolor='white')
    ax2.set_xlabel('R² Score')
    ax2.set_title('Top 10 Regression Models (Biometric)')
    ax2.set_xlim(0, 1.1)
    
    for bar, r2 in zip(bars, r2_scores):
        ax2.text(min(r2 + 0.01, 1.0), bar.get_y() + bar.get_height()/2, f'{r2:.4f}',
                va='center', fontsize=7)
    
    # Clustering silhouette scores
    bio_clust = ml_results['datasets']['biometric']['clustering']['models']
    
    models_clust = list(bio_clust.keys())
    silhouettes = [bio_clust[m]['silhouette_score'] for m in models_clust]
    
    ax3 = axes[1, 0]
    bars = ax3.bar(models_clust, silhouettes, color=COLORS['accent'], edgecolor='white')
    ax3.set_xlabel('Clustering Configuration')
    ax3.set_ylabel('Silhouette Score')
    ax3.set_title('Clustering Performance (Biometric)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Feature importance
    if 'random_forest' in bio_class:
        fi = bio_class['random_forest'].get('feature_importance', {})
        if fi:
            features = list(fi.keys())
            importance = list(fi.values())
            
            sorted_idx = np.argsort(importance)[::-1][:10]
            features = [features[i] for i in sorted_idx]
            importance = [importance[i] for i in sorted_idx]
            
            ax4 = axes[1, 1]
            bars = ax4.barh(features, importance, color=COLORS['primary'], edgecolor='white')
            ax4.set_xlabel('Feature Importance')
            ax4.set_title('Top 10 Features (Random Forest)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ml_model_comparison.pdf')
    plt.savefig(FIGURES_DIR / 'ml_model_comparison.png')
    plt.close()
    print("✓ Saved ML model comparison plot")


def generate_all_plots():
    """Generate all plots for the research paper"""
    print("="*60)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("="*60)
    
    results = load_analysis_results()
    
    if not results:
        print("⚠ Could not load analysis results")
        return
    
    # Generate plots for each dataset
    for dataset in ['biometric', 'demographic', 'enrollment']:
        print(f"\nGenerating plots for {dataset}...")
        plot_regional_distribution(results, dataset)
        plot_state_rankings(results, dataset)
        plot_day_of_week_pattern(results, dataset)
        plot_correlation_heatmap(results, dataset)
        plot_hdi_analysis(results, dataset)
        plot_climate_analysis(results, dataset)
    
    # Cross-dataset plots
    print("\nGenerating cross-dataset plots...")
    plot_weekend_vs_weekday(results)
    plot_hypothesis_test_summary(results)
    plot_ml_results()
    
    print(f"\n✓ All plots saved to {FIGURES_DIR}")
    print(f"  Total figures: {len(list(FIGURES_DIR.glob('*.pdf')))}")


if __name__ == '__main__':
    generate_all_plots()
