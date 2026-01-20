#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis and Inference Generation
============================================================
Analyzes ALL 4.3M cleaned records and generates detailed statistical
inferences for the research paper.

Author: Shuvam Banerji Seal, Alok Mishra, Aheli Poddar
Email: sbs22ms076@iiserkol.ac.in
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import (
    pearsonr, spearmanr, chi2_contingency, kruskal, 
    mannwhitneyu, f_oneway, ttest_ind, normaltest
)
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CLEANED_PATH = PROJECT_ROOT / "Dataset" / "cleaned"
OUTPUT_PATH = PROJECT_ROOT / "analysis" / "outputs" / "results"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Reference data for augmentation
REGION_MAPPING = {
    "Andhra Pradesh": "South", "Arunachal Pradesh": "Northeast", "Assam": "Northeast",
    "Bihar": "East", "Chhattisgarh": "Central", "Goa": "West", "Gujarat": "West",
    "Haryana": "North", "Himachal Pradesh": "North", "Jharkhand": "East",
    "Karnataka": "South", "Kerala": "South", "Madhya Pradesh": "Central",
    "Maharashtra": "West", "Manipur": "Northeast", "Meghalaya": "Northeast",
    "Mizoram": "Northeast", "Nagaland": "Northeast", "Odisha": "East",
    "Punjab": "North", "Rajasthan": "North", "Sikkim": "East",
    "Tamil Nadu": "South", "Telangana": "South", "Tripura": "Northeast",
    "Uttar Pradesh": "Central", "Uttarakhand": "North", "West Bengal": "East",
    "Delhi": "North", "Jammu And Kashmir": "North", "Ladakh": "North",
    "Puducherry": "South", "Chandigarh": "North", 
    "Andaman And Nicobar Islands": "Islands", "Lakshadweep": "South",
    "Dadra And Nagar Haveli And Daman And Diu": "West",
}

HDI_DATA = {
    "Kerala": 0.779, "Delhi": 0.746, "Goa": 0.761, "Himachal Pradesh": 0.725,
    "Punjab": 0.723, "Sikkim": 0.716, "Tamil Nadu": 0.708, "Haryana": 0.708,
    "Maharashtra": 0.695, "Uttarakhand": 0.684, "Karnataka": 0.682,
    "Gujarat": 0.672, "Telangana": 0.669, "Andhra Pradesh": 0.650,
    "West Bengal": 0.641, "Mizoram": 0.705, "Manipur": 0.696,
    "Tripura": 0.658, "Meghalaya": 0.655, "Nagaland": 0.679,
    "Arunachal Pradesh": 0.617, "Rajasthan": 0.621, "Madhya Pradesh": 0.606,
    "Uttar Pradesh": 0.596, "Jharkhand": 0.589, "Chhattisgarh": 0.613,
    "Odisha": 0.606, "Bihar": 0.576, "Assam": 0.614,
}

LITERACY_DATA = {
    "Kerala": 94.0, "Mizoram": 91.6, "Tripura": 87.8, "Goa": 87.4,
    "Himachal Pradesh": 83.8, "Maharashtra": 82.9, "Tamil Nadu": 80.3,
    "Uttarakhand": 79.6, "Gujarat": 79.3, "Delhi": 86.3,
    "West Bengal": 77.1, "Punjab": 76.7, "Haryana": 76.6,
    "Karnataka": 75.6, "Sikkim": 82.2, "Manipur": 79.9,
    "Nagaland": 80.1, "Meghalaya": 75.5, "Odisha": 73.5,
    "Assam": 73.2, "Chhattisgarh": 71.0, "Madhya Pradesh": 70.6,
    "Uttar Pradesh": 69.7, "Andhra Pradesh": 67.7, "Jharkhand": 67.6,
    "Rajasthan": 67.1, "Arunachal Pradesh": 66.9, "Bihar": 63.8,
    "Telangana": 66.5,
}

RAINFALL_ZONES = {
    "Kerala": "Very High", "Assam": "Very High", "Meghalaya": "Very High",
    "Arunachal Pradesh": "Very High", "West Bengal": "High", "Odisha": "High",
    "Karnataka": "Moderate", "Tamil Nadu": "Moderate", "Maharashtra": "Moderate",
    "Madhya Pradesh": "Moderate", "Uttar Pradesh": "Moderate", "Bihar": "Moderate",
    "Gujarat": "Low", "Rajasthan": "Very Low", "Punjab": "Low", "Haryana": "Low",
    "Delhi": "Low", "Himachal Pradesh": "High", "Uttarakhand": "High",
}


class ComprehensiveAnalyzer:
    """Performs comprehensive statistical analysis on UIDAI data."""
    
    def __init__(self):
        self.results = {}
        self.inferences = []
        
    def load_all_data(self):
        """Load ALL cleaned data (no sampling)."""
        print("Loading ALL cleaned data...")
        
        datasets = {}
        paths = {
            'biometric': CLEANED_PATH / "biometric" / "biometric" / "final_cleaned_biometric.csv",
            'demographic': CLEANED_PATH / "demographic" / "demographic" / "final_cleaned_demographic.csv",
            'enrollment': CLEANED_PATH / "enrollment" / "enrollment" / "final_cleaned_enrollment.csv",
        }
        
        for name, path in paths.items():
            if path.exists():
                df = pd.read_csv(path)
                print(f"  {name}: {len(df):,} records")
                datasets[name] = df
        
        return datasets
    
    def augment_with_reference_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add reference data columns."""
        df = df.copy()
        
        # Normalize state names
        df['state_normalized'] = df['state'].str.strip().str.title()
        
        # Add region
        df['region'] = df['state_normalized'].map(REGION_MAPPING).fillna('Other')
        
        # Add HDI
        df['hdi'] = df['state_normalized'].map(HDI_DATA)
        df['hdi_category'] = pd.cut(df['hdi'], bins=[0, 0.55, 0.65, 0.75, 1.0], 
                                     labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Add literacy
        df['literacy_rate'] = df['state_normalized'].map(LITERACY_DATA)
        
        # Add rainfall zone
        df['rainfall_zone'] = df['state_normalized'].map(RAINFALL_ZONES).fillna('Moderate')
        
        # Parse date and add temporal features
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_name'] = df['date'].dt.day_name()
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Pincode zone
        df['pincode_zone'] = (df['pincode'] // 100000).astype(int)
        
        return df
    
    def calculate_total_count(self, df: pd.DataFrame) -> pd.Series:
        """Calculate total count from age columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        age_cols = [c for c in numeric_cols if 'age' in c.lower() or 'bio_' in c.lower() or 'demo_' in c.lower()]
        if age_cols:
            return df[age_cols].sum(axis=1)
        return pd.Series([1] * len(df))
    
    def analyze_temporal_patterns(self, df: pd.DataFrame, name: str) -> dict:
        """Analyze temporal patterns with detailed inferences."""
        print(f"\n  Analyzing temporal patterns for {name}...")
        
        df['total_count'] = self.calculate_total_count(df)
        
        results = {'dataset': name}
        
        # Day of week analysis
        dow_stats = df.groupby('day_name')['total_count'].agg(['mean', 'std', 'sum', 'count'])
        dow_stats = dow_stats.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        results['day_of_week'] = dow_stats.to_dict()
        
        # Find peak and trough days
        peak_day = dow_stats['mean'].idxmax()
        trough_day = dow_stats['mean'].idxmin()
        peak_mean = dow_stats.loc[peak_day, 'mean']
        trough_mean = dow_stats.loc[trough_day, 'mean']
        
        # Statistical test: Kruskal-Wallis for day differences
        day_groups = [df[df['day_name'] == day]['total_count'].values for day in dow_stats.index if len(df[df['day_name'] == day]) > 0]
        if len(day_groups) >= 2:
            kw_stat, kw_p = kruskal(*day_groups)
            results['kruskal_wallis'] = {'statistic': kw_stat, 'p_value': kw_p}
        
        # Weekend vs weekday analysis
        weekend_data = df[df['is_weekend']]['total_count']
        weekday_data = df[~df['is_weekend']]['total_count']
        
        if len(weekend_data) > 0 and len(weekday_data) > 0:
            u_stat, u_p = mannwhitneyu(weekend_data, weekday_data, alternative='two-sided')
            t_stat, t_p = ttest_ind(weekend_data, weekday_data)
            
            results['weekend_vs_weekday'] = {
                'weekend_mean': weekend_data.mean(),
                'weekday_mean': weekday_data.mean(),
                'weekend_total': weekend_data.sum(),
                'weekday_total': weekday_data.sum(),
                'weekend_records': len(weekend_data),
                'weekday_records': len(weekday_data),
                'mann_whitney_u': u_stat,
                'mann_whitney_p': u_p,
                't_statistic': t_stat,
                't_p_value': t_p,
            }
        
        # Generate inference
        inference = f"""
### Temporal Analysis - {name.title()} Dataset

**Day of Week Patterns:**
- Peak enrollment day: **{peak_day}** (mean = {peak_mean:,.2f} per record)
- Lowest enrollment day: **{trough_day}** (mean = {trough_mean:,.2f} per record)
- Difference: {((peak_mean - trough_mean) / trough_mean * 100):.1f}% higher on {peak_day}

**Statistical Significance:**
- Kruskal-Wallis H-test: H = {kw_stat:.2f}, p = {kw_p:.2e}
- Interpretation: {"Significant differences exist between days (p < 0.05)" if kw_p < 0.05 else "No significant daily differences detected"}

**Weekend vs Weekday:**
- Weekend mean: {results['weekend_vs_weekday']['weekend_mean']:,.2f}
- Weekday mean: {results['weekend_vs_weekday']['weekday_mean']:,.2f}
- t-statistic: {results['weekend_vs_weekday']['t_statistic']:.2f}, p = {results['weekend_vs_weekday']['t_p_value']:.2e}
- Inference: {"Weekend enrollments are significantly higher, possibly due to working population availability" if results['weekend_vs_weekday']['weekend_mean'] > results['weekend_vs_weekday']['weekday_mean'] and results['weekend_vs_weekday']['t_p_value'] < 0.05 else "Weekday enrollments dominate, indicating office-hour-based enrollment centers"}
"""
        self.inferences.append(inference)
        return results
    
    def analyze_geographic_patterns(self, df: pd.DataFrame, name: str) -> dict:
        """Analyze geographic distribution with inferences."""
        print(f"  Analyzing geographic patterns for {name}...")
        
        df['total_count'] = self.calculate_total_count(df)
        
        results = {'dataset': name}
        
        # Regional analysis
        region_stats = df.groupby('region')['total_count'].agg(['mean', 'std', 'sum', 'count'])
        region_stats['percentage'] = region_stats['sum'] / region_stats['sum'].sum() * 100
        results['regional'] = region_stats.to_dict()
        
        # State analysis
        state_stats = df.groupby('state_normalized')['total_count'].agg(['mean', 'std', 'sum', 'count'])
        state_stats = state_stats.sort_values('sum', ascending=False)
        results['state_top10'] = state_stats.head(10).to_dict()
        
        # Calculate Gini coefficient for inequality
        values = state_stats['sum'].values
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
        results['gini_coefficient'] = gini
        
        # Top 5 and Top 10 concentration
        total_sum = state_stats['sum'].sum()
        top5_share = state_stats['sum'].head(5).sum() / total_sum * 100
        top10_share = state_stats['sum'].head(10).sum() / total_sum * 100
        results['concentration'] = {'top5_share': top5_share, 'top10_share': top10_share}
        
        # Regional Kruskal-Wallis test
        region_groups = [df[df['region'] == r]['total_count'].values for r in df['region'].unique() if len(df[df['region'] == r]) > 0]
        if len(region_groups) >= 2:
            kw_stat, kw_p = kruskal(*region_groups)
            results['regional_kruskal'] = {'statistic': kw_stat, 'p_value': kw_p}
        
        # Top states
        top_states = list(state_stats.head(5).index)
        
        inference = f"""
### Geographic Analysis - {name.title()} Dataset

**Regional Distribution:**
{chr(10).join([f"- {r}: {region_stats.loc[r, 'percentage']:.1f}% of total" for r in region_stats.sort_values('percentage', ascending=False).index[:6]])}

**Top 5 States:**
{chr(10).join([f"- {s}: {state_stats.loc[s, 'sum']:,.0f} ({state_stats.loc[s, 'sum']/total_sum*100:.1f}%)" for s in top_states])}

**Inequality Metrics:**
- Gini Coefficient: **{gini:.3f}** ({"High inequality" if gini > 0.6 else "Moderate inequality" if gini > 0.4 else "Low inequality"})
- Top 5 states account for **{top5_share:.1f}%** of all enrollments
- Top 10 states account for **{top10_share:.1f}%** of all enrollments

**Statistical Test:**
- Regional Kruskal-Wallis: H = {results['regional_kruskal']['statistic']:.2f}, p = {results['regional_kruskal']['p_value']:.2e}
- Interpretation: {"Significant regional differences exist" if results['regional_kruskal']['p_value'] < 0.05 else "No significant regional differences"}

**Key Insight:** The high Gini coefficient indicates concentrated enrollment in a few populous states, suggesting need for targeted outreach in smaller states.
"""
        self.inferences.append(inference)
        return results
    
    def analyze_socioeconomic_correlations(self, df: pd.DataFrame, name: str) -> dict:
        """Analyze correlations with HDI, literacy, etc."""
        print(f"  Analyzing socioeconomic correlations for {name}...")
        
        df['total_count'] = self.calculate_total_count(df)
        
        results = {'dataset': name}
        
        # Aggregate by state for correlation analysis
        state_agg = df.groupby('state_normalized').agg({
            'total_count': 'sum',
            'hdi': 'first',
            'literacy_rate': 'first'
        }).dropna()
        
        # HDI correlation
        if len(state_agg) > 5:
            hdi_corr, hdi_p = pearsonr(state_agg['total_count'], state_agg['hdi'])
            results['hdi_correlation'] = {'r': hdi_corr, 'p': hdi_p}
            
            literacy_corr, literacy_p = pearsonr(state_agg['total_count'], state_agg['literacy_rate'])
            results['literacy_correlation'] = {'r': literacy_corr, 'p': literacy_p}
        
        # HDI category analysis
        hdi_cat_stats = df.groupby('hdi_category')['total_count'].agg(['mean', 'sum', 'count'])
        results['hdi_category'] = hdi_cat_stats.to_dict()
        
        # ANOVA for HDI categories
        hdi_groups = [df[df['hdi_category'] == cat]['total_count'].values 
                      for cat in df['hdi_category'].dropna().unique() 
                      if len(df[df['hdi_category'] == cat]) > 0]
        if len(hdi_groups) >= 2:
            f_stat, f_p = f_oneway(*hdi_groups)
            results['hdi_anova'] = {'f_statistic': f_stat, 'p_value': f_p}
        
        # Extract values safely for f-string formatting
        hdi_r = results.get('hdi_correlation', {}).get('r', None)
        hdi_p = results.get('hdi_correlation', {}).get('p', None)
        lit_r = results.get('literacy_correlation', {}).get('r', None)
        hdi_f = results.get('hdi_anova', {}).get('f_statistic', None)
        hdi_anova_p = results.get('hdi_anova', {}).get('p_value', None)
        
        hdi_r_str = f"{hdi_r:.3f}" if isinstance(hdi_r, float) else 'N/A'
        hdi_p_str = f"{hdi_p:.2e}" if isinstance(hdi_p, float) else 'N/A'
        lit_r_str = f"{lit_r:.3f}" if isinstance(lit_r, float) else 'N/A'
        hdi_f_str = f"{hdi_f:.2f}" if isinstance(hdi_f, float) else 'N/A'
        hdi_anova_p_str = f"{hdi_anova_p:.2e}" if isinstance(hdi_anova_p, float) else 'N/A'
        
        hdi_interp = "Negative correlation - higher enrollment in lower HDI states indicates successful penetration in underdeveloped regions" if (hdi_r or 0) < 0 else "Positive correlation - enrollment follows development patterns"
        policy_impl = "The inverse relationship between HDI and enrollment volume suggests that Aadhaar enrollment drives have successfully targeted less-developed states, contributing to financial inclusion goals." if (hdi_r or 0) < 0 else "Enrollment patterns follow development, suggesting need for targeted interventions in low-HDI regions."
        
        inference = f"""
### Socioeconomic Analysis - {name.title()} Dataset

**HDI Correlation:**
- Pearson r = {hdi_r_str}
- p-value = {hdi_p_str}
- Interpretation: {hdi_interp}

**Literacy Rate Correlation:**
- Pearson r = {lit_r_str}

**HDI Category ANOVA:**
- F-statistic = {hdi_f_str}
- p-value = {hdi_anova_p_str}

**Policy Implication:** {policy_impl}
"""
        self.inferences.append(inference)
        return results
    
    def analyze_climate_patterns(self, df: pd.DataFrame, name: str) -> dict:
        """Analyze climate zone patterns."""
        print(f"  Analyzing climate patterns for {name}...")
        
        df['total_count'] = self.calculate_total_count(df)
        
        results = {'dataset': name}
        
        # Rainfall zone analysis
        rainfall_stats = df.groupby('rainfall_zone')['total_count'].agg(['mean', 'sum', 'count'])
        rainfall_stats['percentage'] = rainfall_stats['sum'] / rainfall_stats['sum'].sum() * 100
        results['rainfall_zone'] = rainfall_stats.to_dict()
        
        # ANOVA for rainfall zones
        rainfall_groups = [df[df['rainfall_zone'] == zone]['total_count'].values 
                          for zone in df['rainfall_zone'].unique() 
                          if len(df[df['rainfall_zone'] == zone]) > 0]
        if len(rainfall_groups) >= 2:
            f_stat, f_p = f_oneway(*rainfall_groups)
            results['rainfall_anova'] = {'f_statistic': f_stat, 'p_value': f_p}
        
        # Extract values safely for f-string formatting
        rf_f = results.get('rainfall_anova', {}).get('f_statistic', None)
        rf_p = results.get('rainfall_anova', {}).get('p_value', None)
        
        rf_f_str = f"{rf_f:.2f}" if isinstance(rf_f, float) else 'N/A'
        rf_p_str = f"{rf_p:.2e}" if isinstance(rf_p, float) else 'N/A'
        rf_interp = "Significant differences exist across climate zones" if (rf_p or 1) < 0.05 else "No significant climate-based differences"
        
        inference = f"""
### Climate Analysis - {name.title()} Dataset

**Rainfall Zone Distribution:**
{chr(10).join([f"- {zone}: {rainfall_stats.loc[zone, 'percentage']:.1f}% (mean: {rainfall_stats.loc[zone, 'mean']:.2f})" for zone in rainfall_stats.index if zone in rainfall_stats.index])}

**ANOVA Results:**
- F-statistic = {rf_f_str}
- p-value = {rf_p_str}

**Interpretation:** {rf_interp}
"""
        self.inferences.append(inference)
        return results
    
    def generate_summary_statistics(self, datasets: dict) -> dict:
        """Generate overall summary statistics."""
        print("\nGenerating summary statistics...")
        
        summary = {}
        
        for name, df in datasets.items():
            df['total_count'] = self.calculate_total_count(df)
            
            summary[name] = {
                'total_records': len(df),
                'unique_states': df['state'].nunique(),
                'unique_districts': df['district'].nunique() if 'district' in df.columns else 0,
                'unique_pincodes': df['pincode'].nunique() if 'pincode' in df.columns else 0,
                'date_range': {
                    'start': df['date'].min().strftime('%Y-%m-%d') if pd.notna(df['date'].min()) else 'N/A',
                    'end': df['date'].max().strftime('%Y-%m-%d') if pd.notna(df['date'].max()) else 'N/A',
                    'unique_days': df['date'].nunique()
                },
                'total_count_stats': {
                    'sum': df['total_count'].sum(),
                    'mean': df['total_count'].mean(),
                    'std': df['total_count'].std(),
                    'median': df['total_count'].median(),
                    'min': df['total_count'].min(),
                    'max': df['total_count'].max(),
                }
            }
        
        return summary
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("=" * 80)
        print("COMPREHENSIVE STATISTICAL ANALYSIS - UIDAI AADHAAR DATA")
        print("=" * 80)
        print(f"Started at: {datetime.now()}")
        
        # Load data
        datasets = self.load_all_data()
        
        # Augment with reference data
        print("\nAugmenting with reference data...")
        for name, df in datasets.items():
            datasets[name] = self.augment_with_reference_data(df)
        
        # Generate summary
        self.results['summary'] = self.generate_summary_statistics(datasets)
        
        # Run analyses for each dataset
        self.results['temporal'] = {}
        self.results['geographic'] = {}
        self.results['socioeconomic'] = {}
        self.results['climate'] = {}
        
        for name, df in datasets.items():
            print(f"\n{'='*60}")
            print(f"Analyzing {name.upper()} dataset ({len(df):,} records)")
            print('='*60)
            
            self.results['temporal'][name] = self.analyze_temporal_patterns(df, name)
            self.results['geographic'][name] = self.analyze_geographic_patterns(df, name)
            self.results['socioeconomic'][name] = self.analyze_socioeconomic_correlations(df, name)
            self.results['climate'][name] = self.analyze_climate_patterns(df, name)
        
        # Save results
        results_file = OUTPUT_PATH / "comprehensive_analysis_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        with open(results_file, 'w') as f:
            json.dump(convert_to_serializable(self.results), f, indent=2, default=str)
        
        # Save inferences
        inferences_file = OUTPUT_PATH / "statistical_inferences.md"
        with open(inferences_file, 'w') as f:
            f.write("# Statistical Inferences from UIDAI Aadhaar Data Analysis\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            f.write(f"Total Records Analyzed: {sum(self.results['summary'][k]['total_records'] for k in self.results['summary']):,}\n\n")
            f.write("---\n\n")
            for inference in self.inferences:
                f.write(inference)
                f.write("\n---\n")
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {results_file}")
        print(f"Inferences saved to: {inferences_file}")
        
        return self.results, self.inferences


if __name__ == "__main__":
    analyzer = ComprehensiveAnalyzer()
    results, inferences = analyzer.run_full_analysis()
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    
    for name, summary in results['summary'].items():
        print(f"\n{name.upper()}:")
        print(f"  Records: {summary['total_records']:,}")
        print(f"  States: {summary['unique_states']}")
        print(f"  Districts: {summary['unique_districts']}")
        print(f"  Total count: {summary['total_count_stats']['sum']:,.0f}")
