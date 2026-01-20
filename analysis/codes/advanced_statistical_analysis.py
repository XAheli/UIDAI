#!/usr/bin/env python3
"""
Advanced Statistical Analysis with Detailed Inferences
Includes: Time Series Analysis, Causal Inference, Spatial Analysis, Regression Models

Author: Shuvam Banerji Seal, Alok Mishra, Aheli Poddar
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
    pearsonr, spearmanr, kendalltau, chi2_contingency,
    kruskal, mannwhitneyu, f_oneway, normaltest,
    shapiro, levene, ttest_ind, wilcoxon
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DetailedStatisticalInference:
    """Generate detailed statistical inferences with interpretations."""
    
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
        self.inferences = []
    
    def interpret_correlation(self, r: float, p: float, var1: str, var2: str) -> str:
        """Generate detailed interpretation of correlation coefficient."""
        # Strength classification
        abs_r = abs(r)
        if abs_r < 0.1:
            strength = "negligible"
        elif abs_r < 0.3:
            strength = "weak"
        elif abs_r < 0.5:
            strength = "moderate"
        elif abs_r < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if r > 0 else "negative"
        significance = "statistically significant" if p < self.alpha else "not statistically significant"
        
        interpretation = f"""
**Correlation Analysis: {var1} vs {var2}**
- Pearson's r = {r:.4f} (p = {p:.2e})
- **Strength**: {strength.capitalize()} {direction} correlation
- **Statistical Significance**: {significance} at α = {self.alpha}
- **Interpretation**: """
        
        if p < self.alpha:
            if r > 0:
                interpretation += f"There is a {strength} positive association between {var1} and {var2}. As {var1} increases, {var2} tends to increase as well. "
            else:
                interpretation += f"There is a {strength} negative association between {var1} and {var2}. As {var1} increases, {var2} tends to decrease. "
            
            interpretation += f"This relationship is unlikely due to random chance (p = {p:.2e} < {self.alpha}). "
            interpretation += f"The correlation explains approximately {(r**2)*100:.1f}% of the variance in the data."
        else:
            interpretation += f"No statistically significant linear relationship was found between {var1} and {var2}. "
            interpretation += "Any observed association may be due to random sampling variation."
        
        self.inferences.append(interpretation)
        return interpretation
    
    def interpret_hypothesis_test(self, test_name: str, statistic: float, p_value: float,
                                  groups: list, variable: str) -> str:
        """Generate detailed interpretation of hypothesis test."""
        significance = "statistically significant" if p_value < self.alpha else "not statistically significant"
        
        interpretation = f"""
**{test_name}: {variable} across groups**
- Test Statistic = {statistic:.4f}
- p-value = {p_value:.2e}
- Groups compared: {', '.join(groups[:5])}{'...' if len(groups) > 5 else ''}
- **Result**: {significance} at α = {self.alpha}
- **Interpretation**: """
        
        if p_value < self.alpha:
            interpretation += f"There are significant differences in {variable} across the groups. "
            interpretation += f"The null hypothesis (that all groups have equal {variable}) is rejected (p = {p_value:.2e}). "
            interpretation += "Post-hoc analysis should be conducted to identify which specific groups differ."
        else:
            interpretation += f"No significant differences in {variable} were found across groups. "
            interpretation += "The observed variations are consistent with random sampling variability."
        
        self.inferences.append(interpretation)
        return interpretation
    
    def interpret_regression(self, r2: float, coefficients: dict, variable: str, predictors: list) -> str:
        """Generate detailed interpretation of regression analysis."""
        interpretation = f"""
**Regression Analysis: Predicting {variable}**
- R² = {r2:.4f} (Model explains {r2*100:.1f}% of variance)
- Predictors: {', '.join(predictors[:5])}{'...' if len(predictors) > 5 else ''}
- **Key Coefficients**:
"""
        for pred, coef in list(coefficients.items())[:5]:
            interpretation += f"  - {pred}: β = {coef:.4f}\n"
        
        interpretation += f"""
- **Model Quality**: """
        
        if r2 > 0.7:
            interpretation += f"Excellent predictive power (R² > 0.7). The model captures most of the variance in {variable}."
        elif r2 > 0.5:
            interpretation += f"Good predictive power (R² > 0.5). The model captures substantial variance in {variable}."
        elif r2 > 0.3:
            interpretation += f"Moderate predictive power. The model explains some variance in {variable}, but other factors may be important."
        else:
            interpretation += f"Limited predictive power. {variable} may be influenced by factors not captured in this model."
        
        self.inferences.append(interpretation)
        return interpretation
    
    def interpret_time_series(self, trend: str, seasonality: bool, stationarity: bool) -> str:
        """Generate detailed interpretation of time series analysis."""
        interpretation = f"""
**Time Series Analysis**
- **Trend**: {trend.capitalize()}
- **Seasonality Detected**: {'Yes' if seasonality else 'No'}
- **Stationarity**: {'Stationary' if stationarity else 'Non-stationary'}

**Interpretation**:
"""
        if trend == "increasing":
            interpretation += "The data shows an upward trend over time, indicating systematic growth or increase. "
        elif trend == "decreasing":
            interpretation += "The data shows a downward trend over time, indicating systematic decline. "
        else:
            interpretation += "The data shows no clear long-term trend. "
        
        if seasonality:
            interpretation += "Clear seasonal patterns are present, suggesting cyclical factors (e.g., day-of-week effects, monthly patterns). "
        
        if not stationarity:
            interpretation += "The series is non-stationary, meaning its statistical properties change over time. Differencing or transformation may be needed for forecasting."
        
        self.inferences.append(interpretation)
        return interpretation


class AdvancedStatisticalAnalysis:
    """
    Comprehensive statistical analysis with detailed inferences.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inference = DetailedStatisticalInference()
        self.results = {}
    
    def run_correlation_analysis(self, df: pd.DataFrame, columns: list) -> dict:
        """Run comprehensive correlation analysis with interpretations."""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        correlations = {}
        interpretations = []
        
        # Get numeric columns
        numeric_cols = [c for c in columns if c in df.columns and df[c].dtype in ['int64', 'float64']]
        
        if len(numeric_cols) < 2:
            print("  Insufficient numeric columns for correlation analysis")
            return correlations
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Upper triangle only
                    r = corr_matrix.loc[col1, col2]
                    if not np.isnan(r):
                        # Calculate p-value
                        valid_data = df[[col1, col2]].dropna()
                        if len(valid_data) > 10:
                            _, p = pearsonr(valid_data[col1], valid_data[col2])
                            if abs(r) > 0.3:  # Report moderate+ correlations
                                strong_correlations.append({
                                    'var1': col1, 'var2': col2, 'r': r, 'p': p
                                })
                                interpretation = self.inference.interpret_correlation(r, p, col1, col2)
                                interpretations.append(interpretation)
                                print(f"  {col1} vs {col2}: r = {r:.3f}, p = {p:.2e}")
        
        # Key correlation findings
        correlations['matrix'] = corr_matrix.to_dict()
        correlations['strong_correlations'] = sorted(strong_correlations, key=lambda x: abs(x['r']), reverse=True)
        correlations['interpretations'] = interpretations
        
        self.results['correlations'] = correlations
        return correlations
    
    def run_hypothesis_tests(self, df: pd.DataFrame, group_column: str, value_columns: list) -> dict:
        """Run comprehensive hypothesis testing with interpretations."""
        print("\n" + "="*60)
        print("HYPOTHESIS TESTING")
        print("="*60)
        
        tests = {}
        interpretations = []
        
        if group_column not in df.columns:
            print(f"  Group column {group_column} not found")
            return tests
        
        groups = df[group_column].dropna().unique()
        
        for value_col in value_columns:
            if value_col not in df.columns:
                continue
            
            print(f"\n  Testing: {value_col} by {group_column}")
            
            # Prepare data
            group_data = []
            valid_groups = []
            for g in groups:
                data = df[df[group_column] == g][value_col].dropna()
                if len(data) >= 10:
                    group_data.append(data.values)
                    valid_groups.append(str(g))
            
            if len(group_data) < 2:
                continue
            
            # Kruskal-Wallis H-test (non-parametric ANOVA)
            try:
                h_stat, h_p = kruskal(*group_data)
                interpretation = self.inference.interpret_hypothesis_test(
                    "Kruskal-Wallis H-test", h_stat, h_p, valid_groups, value_col
                )
                interpretations.append(interpretation)
                
                tests[f'{value_col}_kruskal'] = {
                    'statistic': h_stat,
                    'p_value': h_p,
                    'significant': h_p < 0.05,
                    'groups': valid_groups,
                    'interpretation': interpretation
                }
                
                print(f"    Kruskal-Wallis: H = {h_stat:.2f}, p = {h_p:.2e}")
            except Exception as e:
                print(f"    Kruskal-Wallis failed: {e}")
            
            # ANOVA (parametric)
            try:
                f_stat, f_p = f_oneway(*group_data)
                
                tests[f'{value_col}_anova'] = {
                    'statistic': f_stat,
                    'p_value': f_p,
                    'significant': f_p < 0.05
                }
                print(f"    ANOVA: F = {f_stat:.2f}, p = {f_p:.2e}")
            except Exception as e:
                print(f"    ANOVA failed: {e}")
            
            # Effect size (eta-squared)
            try:
                all_data = np.concatenate(group_data)
                grand_mean = np.mean(all_data)
                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_data)
                ss_total = sum((x - grand_mean)**2 for x in all_data)
                eta_sq = ss_between / ss_total if ss_total > 0 else 0
                
                tests[f'{value_col}_effect_size'] = {
                    'eta_squared': eta_sq,
                    'interpretation': 'small' if eta_sq < 0.06 else 'medium' if eta_sq < 0.14 else 'large'
                }
                print(f"    Effect size (η²): {eta_sq:.4f} ({tests[f'{value_col}_effect_size']['interpretation']})")
            except Exception as e:
                print(f"    Effect size calculation failed: {e}")
        
        tests['interpretations'] = interpretations
        self.results['hypothesis_tests'] = tests
        return tests
    
    def run_regression_analysis(self, df: pd.DataFrame, target: str, predictors: list) -> dict:
        """Run comprehensive regression analysis with interpretations."""
        print("\n" + "="*60)
        print("REGRESSION ANALYSIS")
        print("="*60)
        
        regression = {}
        
        # Prepare data
        available_predictors = [p for p in predictors if p in df.columns]
        if target not in df.columns or len(available_predictors) < 1:
            print("  Insufficient data for regression")
            return regression
        
        # Clean data
        analysis_cols = [target] + available_predictors
        clean_df = df[analysis_cols].dropna()
        
        if len(clean_df) < 50:
            print("  Insufficient clean data for regression")
            return regression
        
        X = clean_df[available_predictors].values
        y = clean_df[target].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Linear Regression
        print("\n  Linear Regression:")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        
        lr_r2 = r2_score(y_test, y_pred)
        lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        coefficients = dict(zip(available_predictors, lr.coef_))
        interpretation = self.inference.interpret_regression(lr_r2, coefficients, target, available_predictors)
        
        regression['linear'] = {
            'r2': lr_r2,
            'rmse': lr_rmse,
            'coefficients': coefficients,
            'intercept': lr.intercept_,
            'interpretation': interpretation
        }
        print(f"    R² = {lr_r2:.4f}, RMSE = {lr_rmse:.4f}")
        
        # Ridge Regression
        print("\n  Ridge Regression:")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        
        ridge_r2 = r2_score(y_test, y_pred_ridge)
        regression['ridge'] = {
            'r2': ridge_r2,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
            'coefficients': dict(zip(available_predictors, ridge.coef_))
        }
        print(f"    R² = {ridge_r2:.4f}")
        
        # Random Forest
        print("\n  Random Forest:")
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        rf_r2 = r2_score(y_test, y_pred_rf)
        feature_importance = dict(zip(available_predictors, rf.feature_importances_))
        
        regression['random_forest'] = {
            'r2': rf_r2,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'feature_importance': feature_importance
        }
        print(f"    R² = {rf_r2:.4f}")
        print("    Top features by importance:")
        for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {feat}: {imp:.4f}")
        
        # Gradient Boosting
        print("\n  Gradient Boosting:")
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)
        
        gb_r2 = r2_score(y_test, y_pred_gb)
        regression['gradient_boosting'] = {
            'r2': gb_r2,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
            'feature_importance': dict(zip(available_predictors, gb.feature_importances_))
        }
        print(f"    R² = {gb_r2:.4f}")
        
        # Best model summary
        models = {
            'linear': lr_r2,
            'ridge': ridge_r2,
            'random_forest': rf_r2,
            'gradient_boosting': gb_r2
        }
        best_model = max(models, key=models.get)
        regression['best_model'] = {
            'name': best_model,
            'r2': models[best_model]
        }
        
        print(f"\n  Best Model: {best_model} (R² = {models[best_model]:.4f})")
        
        self.results['regression'] = regression
        return regression
    
    def run_time_series_analysis(self, df: pd.DataFrame, date_column: str, value_column: str) -> dict:
        """Run time series analysis with trend and seasonality detection."""
        print("\n" + "="*60)
        print("TIME SERIES ANALYSIS")
        print("="*60)
        
        ts_results = {}
        
        if date_column not in df.columns or value_column not in df.columns:
            print("  Required columns not found")
            return ts_results
        
        # Prepare time series
        ts_df = df[[date_column, value_column]].dropna().copy()
        ts_df[date_column] = pd.to_datetime(ts_df[date_column], errors='coerce')
        ts_df = ts_df.dropna()
        
        if len(ts_df) < 30:
            print("  Insufficient data for time series analysis")
            return ts_results
        
        # Aggregate by date
        daily = ts_df.groupby(date_column)[value_column].sum().reset_index()
        daily = daily.sort_values(date_column)
        
        print(f"  Analyzing {len(daily)} time points")
        
        # Trend analysis
        if len(daily) >= 7:
            # Simple linear trend
            x = np.arange(len(daily))
            y = daily[value_column].values
            slope, intercept = np.polyfit(x, y, 1)
            
            trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            
            # Day of week pattern (seasonality proxy)
            ts_df['dayofweek'] = ts_df[date_column].dt.dayofweek
            dow_pattern = ts_df.groupby('dayofweek')[value_column].mean()
            
            # Check for weekly seasonality
            dow_cv = dow_pattern.std() / dow_pattern.mean() if dow_pattern.mean() > 0 else 0
            has_seasonality = dow_cv > 0.1
            
            # Stationarity (simplified using rolling statistics)
            window = min(7, len(daily) // 3)
            rolling_mean = daily[value_column].rolling(window=window).mean()
            rolling_std = daily[value_column].rolling(window=window).std()
            
            # Check if rolling statistics are stable
            mean_stability = rolling_mean.dropna().std() / rolling_mean.dropna().mean() if rolling_mean.dropna().mean() > 0 else 1
            is_stationary = mean_stability < 0.2
            
            interpretation = self.inference.interpret_time_series(trend_direction, has_seasonality, is_stationary)
            
            ts_results = {
                'trend': {
                    'direction': trend_direction,
                    'slope': slope,
                    'interpretation': f"Average change of {slope:.2f} per day"
                },
                'seasonality': {
                    'detected': has_seasonality,
                    'day_of_week_pattern': dow_pattern.to_dict(),
                    'coefficient_of_variation': dow_cv
                },
                'stationarity': {
                    'is_stationary': is_stationary,
                    'mean_stability': mean_stability
                },
                'summary': {
                    'n_observations': len(daily),
                    'date_range': f"{daily[date_column].min()} to {daily[date_column].max()}",
                    'mean_value': daily[value_column].mean(),
                    'std_value': daily[value_column].std()
                },
                'interpretation': interpretation
            }
            
            print(f"\n  Trend: {trend_direction} (slope = {slope:.4f})")
            print(f"  Seasonality detected: {has_seasonality}")
            print(f"  Is stationary: {is_stationary}")
            
            # Day of week effects
            print("\n  Day of Week Pattern:")
            dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            for i, val in dow_pattern.items():
                print(f"    {dow_names[i]}: {val:.2f}")
        
        self.results['time_series'] = ts_results
        return ts_results
    
    def run_spatial_analysis(self, df: pd.DataFrame, lat_col: str, lon_col: str, value_col: str) -> dict:
        """Run spatial analysis for geographic patterns."""
        print("\n" + "="*60)
        print("SPATIAL ANALYSIS")
        print("="*60)
        
        spatial = {}
        
        # Check required columns
        required = [lat_col, lon_col, value_col]
        if not all(c in df.columns for c in required):
            print("  Required columns not found")
            return spatial
        
        # Clean data
        spatial_df = df[required].dropna()
        
        if len(spatial_df) < 50:
            print("  Insufficient data for spatial analysis")
            return spatial
        
        print(f"  Analyzing {len(spatial_df)} spatial points")
        
        # Basic spatial statistics
        lat = spatial_df[lat_col].values
        lon = spatial_df[lon_col].values
        values = spatial_df[value_col].values
        
        # Centroid
        centroid = (np.mean(lat), np.mean(lon))
        
        # Spatial extent
        extent = {
            'lat_range': (np.min(lat), np.max(lat)),
            'lon_range': (np.min(lon), np.max(lon))
        }
        
        # Spatial clustering using K-Means
        coords = np.column_stack([lat, lon])
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # Find optimal clusters (simplified)
        inertias = []
        k_range = range(2, min(11, len(spatial_df) // 10))
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(coords_scaled)
            inertias.append(kmeans.inertia_)
        
        # Use elbow method (simplified)
        optimal_k = 4  # Default
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            diff2 = np.diff(diffs)
            if len(diff2) > 0:
                optimal_k = list(k_range)[np.argmax(diff2) + 1]
        
        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coords_scaled)
        
        # Cluster statistics
        cluster_stats = {}
        for c in range(optimal_k):
            mask = clusters == c
            cluster_stats[f'cluster_{c}'] = {
                'count': np.sum(mask),
                'mean_value': np.mean(values[mask]),
                'centroid_lat': np.mean(lat[mask]),
                'centroid_lon': np.mean(lon[mask])
            }
        
        # Regional distribution interpretation
        interpretation = f"""
**Spatial Analysis Results**

The data spans from latitude {extent['lat_range'][0]:.2f}° to {extent['lat_range'][1]:.2f}° N
and longitude {extent['lon_range'][0]:.2f}° to {extent['lon_range'][1]:.2f}° E.

**Spatial Clustering**: {optimal_k} distinct geographic clusters were identified.
"""
        
        for c in range(optimal_k):
            stats = cluster_stats[f'cluster_{c}']
            interpretation += f"""
- Cluster {c+1}: {stats['count']} records, centered at ({stats['centroid_lat']:.2f}°, {stats['centroid_lon']:.2f}°), mean value = {stats['mean_value']:.2f}
"""
        
        interpretation += """
**Interpretation**: Geographic clustering reveals distinct regional patterns in the data.
Different clusters may correspond to different administrative regions or economic zones.
"""
        
        self.inference.inferences.append(interpretation)
        
        spatial = {
            'centroid': {'lat': centroid[0], 'lon': centroid[1]},
            'extent': extent,
            'clustering': {
                'n_clusters': optimal_k,
                'cluster_stats': cluster_stats
            },
            'n_points': len(spatial_df),
            'interpretation': interpretation
        }
        
        print(f"\n  Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")
        print(f"  Optimal clusters: {optimal_k}")
        
        for c in range(optimal_k):
            stats = cluster_stats[f'cluster_{c}']
            print(f"  Cluster {c}: n={stats['count']}, mean={stats['mean_value']:.2f}")
        
        self.results['spatial'] = spatial
        return spatial
    
    def run_causal_analysis(self, df: pd.DataFrame, treatment_col: str, outcome_col: str,
                           confounders: list) -> dict:
        """Run causal inference analysis (propensity score approach)."""
        print("\n" + "="*60)
        print("CAUSAL INFERENCE ANALYSIS")
        print("="*60)
        
        causal = {}
        
        # Check required columns
        available_confounders = [c for c in confounders if c in df.columns]
        if treatment_col not in df.columns or outcome_col not in df.columns:
            print("  Required columns not found")
            return causal
        
        if len(available_confounders) < 1:
            print("  No confounders available")
            return causal
        
        # Prepare data
        analysis_cols = [treatment_col, outcome_col] + available_confounders
        clean_df = df[analysis_cols].dropna()
        
        if len(clean_df) < 100:
            print("  Insufficient data for causal analysis")
            return causal
        
        # Binarize treatment if needed
        treatment = clean_df[treatment_col].values
        if len(np.unique(treatment)) > 2:
            median_treatment = np.median(treatment)
            treatment_binary = (treatment > median_treatment).astype(int)
            print(f"  Treatment binarized at median ({median_treatment:.2f})")
        else:
            treatment_binary = treatment
        
        outcome = clean_df[outcome_col].values
        confounders_data = clean_df[available_confounders].values
        
        # Propensity score estimation (using logistic-like approach via linear model)
        # Simplified version
        from sklearn.linear_model import LogisticRegression
        
        scaler = StandardScaler()
        confounders_scaled = scaler.fit_transform(confounders_data)
        
        # Estimate propensity scores
        ps_model = LogisticRegression(random_state=42, max_iter=1000)
        ps_model.fit(confounders_scaled, treatment_binary)
        propensity_scores = ps_model.predict_proba(confounders_scaled)[:, 1]
        
        # Naive estimate (unadjusted)
        treated_mask = treatment_binary == 1
        naive_ate = np.mean(outcome[treated_mask]) - np.mean(outcome[~treated_mask])
        
        # Inverse probability weighting (IPW) estimate
        weights_treated = 1 / propensity_scores
        weights_control = 1 / (1 - propensity_scores)
        
        # Stabilized weights
        weights_treated = weights_treated * np.mean(treatment_binary)
        weights_control = weights_control * (1 - np.mean(treatment_binary))
        
        # IPW-adjusted ATE
        ipw_treated = np.sum(outcome * treatment_binary * weights_treated) / np.sum(treatment_binary * weights_treated)
        ipw_control = np.sum(outcome * (1 - treatment_binary) * weights_control) / np.sum((1 - treatment_binary) * weights_control)
        ipw_ate = ipw_treated - ipw_control
        
        # Stratification approach
        strata_ates = []
        n_strata = 5
        strata_bounds = np.percentile(propensity_scores, np.linspace(0, 100, n_strata + 1))
        
        for i in range(n_strata):
            mask = (propensity_scores >= strata_bounds[i]) & (propensity_scores < strata_bounds[i + 1])
            if np.sum(mask) > 10:
                strata_treated = outcome[mask & treated_mask]
                strata_control = outcome[mask & ~treated_mask]
                if len(strata_treated) > 0 and len(strata_control) > 0:
                    strata_ate = np.mean(strata_treated) - np.mean(strata_control)
                    strata_ates.append(strata_ate)
        
        stratified_ate = np.mean(strata_ates) if strata_ates else naive_ate
        
        interpretation = f"""
**Causal Inference Analysis: Effect of {treatment_col} on {outcome_col}**

**Methods Used**:
1. Naive (unadjusted) comparison
2. Inverse Probability Weighting (IPW)
3. Stratification on Propensity Scores

**Results**:
- Naive ATE: {naive_ate:.4f}
- IPW-adjusted ATE: {ipw_ate:.4f}
- Stratified ATE: {stratified_ate:.4f}

**Confounders controlled**: {', '.join(available_confounders)}

**Interpretation**:
"""
        
        if abs(ipw_ate - naive_ate) > abs(naive_ate) * 0.2:
            interpretation += """
The IPW-adjusted estimate differs substantially from the naive estimate, 
suggesting confounding bias in the unadjusted comparison. The adjusted 
estimate provides a more reliable causal effect estimate.
"""
        else:
            interpretation += """
The IPW-adjusted estimate is similar to the naive estimate, suggesting 
that confounding may be limited, or the included confounders do not 
strongly predict treatment assignment.
"""
        
        if ipw_ate > 0:
            interpretation += f"""
**Conclusion**: Higher {treatment_col} appears to causally increase {outcome_col} 
by approximately {ipw_ate:.4f} units, after controlling for confounders.
"""
        else:
            interpretation += f"""
**Conclusion**: Higher {treatment_col} appears to causally decrease {outcome_col} 
by approximately {abs(ipw_ate):.4f} units, after controlling for confounders.
"""
        
        self.inference.inferences.append(interpretation)
        
        causal = {
            'naive_ate': naive_ate,
            'ipw_ate': ipw_ate,
            'stratified_ate': stratified_ate,
            'propensity_score_stats': {
                'mean': np.mean(propensity_scores),
                'std': np.std(propensity_scores),
                'min': np.min(propensity_scores),
                'max': np.max(propensity_scores)
            },
            'confounders': available_confounders,
            'n_treated': np.sum(treated_mask),
            'n_control': np.sum(~treated_mask),
            'interpretation': interpretation
        }
        
        print(f"\n  Naive ATE: {naive_ate:.4f}")
        print(f"  IPW-adjusted ATE: {ipw_ate:.4f}")
        print(f"  Stratified ATE: {stratified_ate:.4f}")
        print(f"  Propensity score range: [{np.min(propensity_scores):.3f}, {np.max(propensity_scores):.3f}]")
        
        self.results['causal'] = causal
        return causal
    
    def run_pca_analysis(self, df: pd.DataFrame, columns: list) -> dict:
        """Run PCA for dimensionality reduction and interpretation."""
        print("\n" + "="*60)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("="*60)
        
        pca_results = {}
        
        available_cols = [c for c in columns if c in df.columns and df[c].dtype in ['int64', 'float64']]
        
        if len(available_cols) < 3:
            print("  Insufficient numeric columns for PCA")
            return pca_results
        
        # Prepare data
        data = df[available_cols].dropna()
        
        if len(data) < 50:
            print("  Insufficient clean data for PCA")
            return pca_results
        
        print(f"  Analyzing {len(available_cols)} variables with {len(data)} observations")
        
        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # PCA
        n_components = min(len(available_cols), 5)
        pca = PCA(n_components=n_components)
        pca.fit(data_scaled)
        
        # Variance explained
        var_explained = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(var_explained)
        
        # Loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=available_cols
        )
        
        interpretation = f"""
**Principal Component Analysis (PCA)**

**Variance Explained**:
"""
        for i, (var, cum) in enumerate(zip(var_explained, cumulative_var)):
            interpretation += f"- PC{i+1}: {var*100:.1f}% (cumulative: {cum*100:.1f}%)\n"
        
        interpretation += f"""
**Top Loadings for PC1** (most variance):
"""
        pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False)
        for var in pc1_loadings.head(5).index:
            interpretation += f"- {var}: {loadings.loc[var, 'PC1']:.3f}\n"
        
        interpretation += """
**Interpretation**: 
"""
        if cumulative_var[1] > 0.6:
            interpretation += f"The first two components capture {cumulative_var[1]*100:.1f}% of the variance, suggesting a relatively low-dimensional structure in the data. "
        else:
            interpretation += f"The data shows higher dimensionality, with the first two components capturing only {cumulative_var[1]*100:.1f}% of variance. "
        
        # Identify dominant features
        top_pc1_features = pc1_loadings.head(3).index.tolist()
        interpretation += f"The primary factors driving variation are: {', '.join(top_pc1_features)}."
        
        self.inference.inferences.append(interpretation)
        
        pca_results = {
            'n_components': n_components,
            'variance_explained': var_explained.tolist(),
            'cumulative_variance': cumulative_var.tolist(),
            'loadings': loadings.to_dict(),
            'interpretation': interpretation
        }
        
        print("\n  Variance explained:")
        for i, (var, cum) in enumerate(zip(var_explained, cumulative_var)):
            print(f"    PC{i+1}: {var*100:.1f}% (cumulative: {cum*100:.1f}%)")
        
        print("\n  Top loadings for PC1:")
        for var in pc1_loadings.head(5).index:
            print(f"    {var}: {loadings.loc[var, 'PC1']:.3f}")
        
        self.results['pca'] = pca_results
        return pca_results
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report with all inferences."""
        report = """
================================================================================
COMPREHENSIVE STATISTICAL ANALYSIS REPORT
UIDAI Aadhaar Authentication Data
================================================================================

This report presents a detailed statistical analysis of the UIDAI Aadhaar 
authentication data, incorporating multiple analytical approaches and providing
interpretations of the findings.

"""
        
        # Add all inferences
        for i, inference in enumerate(self.inference.inferences, 1):
            report += f"\n{'='*60}\n"
            report += f"FINDING {i}\n"
            report += f"{'='*60}\n"
            report += inference + "\n"
        
        # Summary statistics
        report += f"""

================================================================================
METHODOLOGY NOTES
================================================================================

**Statistical Tests Used**:
- Pearson and Spearman correlation for linear and monotonic relationships
- Kruskal-Wallis H-test for non-parametric group comparisons
- ANOVA F-test for parametric group comparisons
- Linear regression, Ridge regression, and Random Forest for prediction
- Principal Component Analysis for dimensionality reduction
- Propensity score methods for causal inference

**Significance Level**: α = 0.05 (95% confidence)

**Effect Size Interpretations**:
- Correlation: |r| < 0.3 (weak), 0.3-0.5 (moderate), 0.5-0.7 (strong), >0.7 (very strong)
- Eta-squared: η² < 0.06 (small), 0.06-0.14 (medium), >0.14 (large)
- R-squared: R² < 0.3 (low), 0.3-0.5 (moderate), 0.5-0.7 (good), >0.7 (excellent)

**Causal Inference Assumptions**:
- Positivity: Treatment probabilities bounded away from 0 and 1
- Unconfoundedness: No unmeasured confounders
- SUTVA: No interference between units

"""
        
        return report
    
    def save_results(self):
        """Save all results to files."""
        # Save JSON results
        results_path = self.output_dir / "advanced_analysis_results.json"
        
        # Convert numpy types for JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        with open(results_path, 'w') as f:
            json.dump(convert_numpy(self.results), f, indent=2)
        
        # Save report
        report = self.generate_summary_report()
        report_path = self.output_dir / "statistical_inference_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n\nResults saved to:")
        print(f"  - {results_path}")
        print(f"  - {report_path}")


def main():
    """Run complete advanced statistical analysis."""
    
    print("="*80)
    print("ADVANCED STATISTICAL ANALYSIS WITH DETAILED INFERENCES")
    print("="*80)
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "analysis" / "advanced_results"
    
    # Initialize analyzer
    analyzer = AdvancedStatisticalAnalysis(output_dir)
    
    # Load augmented data (try API-augmented first, then corrected)
    data_paths = [
        project_root / "Dataset" / "api_augmented" / "api_augmented_biometric.csv",
        project_root / "Dataset" / "corrected_dataset" / "biometric" / "final_main_corrected_biometric.csv",
        project_root / "Dataset" / "corrected_dataset" / "demographic" / "corrected_api_data_aadhar_demographic_0_500000.csv",
    ]
    
    df = None
    for path in data_paths:
        if path.exists():
            print(f"\nLoading data from: {path}")
            df = pd.read_csv(path, nrows=300000)
            print(f"Loaded {len(df)} records with {len(df.columns)} columns")
            break
    
    if df is None:
        print("No data files found!")
        return
    
    # Define analysis columns
    numeric_cols = [
        'total_authentication_count', 'accept_count', 'reject_count',
        'male_population', 'female_population', 'total_population',
        'population_density', 'urban_population_percentage',
        'literacy_rate', 'hdi',
        'temperature_c', 'humidity_pct', 'aqi', 'pm2_5',
        'elevation_m', 'hospitals_per_100k', 'schools_per_100k',
        'mobile_penetration', 'sdg_score', 'health_index'
    ]
    
    # Run analyses
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE STATISTICAL ANALYSES")
    print("="*80)
    
    # 1. Correlation Analysis
    analyzer.run_correlation_analysis(df, numeric_cols)
    
    # 2. Hypothesis Tests
    analyzer.run_hypothesis_tests(
        df, 
        group_column='region' if 'region' in df.columns else 'state',
        value_columns=['total_authentication_count', 'accept_count', 'reject_count'] if 'total_authentication_count' in df.columns else ['male_population', 'female_population']
    )
    
    # 3. Regression Analysis
    if 'total_authentication_count' in df.columns:
        target = 'total_authentication_count'
        predictors = ['male_population', 'female_population', 'population_density', 
                     'literacy_rate', 'hdi', 'urban_population_percentage']
    else:
        target = 'total_population' if 'total_population' in df.columns else numeric_cols[0]
        predictors = [c for c in numeric_cols if c != target][:6]
    
    available_predictors = [p for p in predictors if p in df.columns]
    if target in df.columns and len(available_predictors) >= 2:
        analyzer.run_regression_analysis(df, target, available_predictors)
    
    # 4. Time Series Analysis
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'day' in c.lower()]
    if date_cols:
        value_col = 'total_authentication_count' if 'total_authentication_count' in df.columns else numeric_cols[0]
        analyzer.run_time_series_analysis(df, date_cols[0], value_col)
    
    # 5. Spatial Analysis
    if 'latitude' in df.columns and 'longitude' in df.columns:
        value_col = 'total_authentication_count' if 'total_authentication_count' in df.columns else numeric_cols[0]
        analyzer.run_spatial_analysis(df, 'latitude', 'longitude', value_col)
    
    # 6. PCA Analysis
    available_numeric = [c for c in numeric_cols if c in df.columns]
    if len(available_numeric) >= 3:
        analyzer.run_pca_analysis(df, available_numeric)
    
    # 7. Causal Inference
    if 'literacy_rate' in df.columns:
        outcome = 'total_authentication_count' if 'total_authentication_count' in df.columns else 'total_population'
        confounders = ['population_density', 'urban_population_percentage', 'hdi']
        available_confounders = [c for c in confounders if c in df.columns]
        if outcome in df.columns and len(available_confounders) >= 1:
            analyzer.run_causal_analysis(df, 'literacy_rate', outcome, available_confounders)
    
    # Save results
    analyzer.save_results()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
