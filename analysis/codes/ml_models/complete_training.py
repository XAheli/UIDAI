#!/usr/bin/env python3
"""
Complete ML Training Pipeline
=============================
Author: Shuvam Banerji Seal's Team
Date: January 2026

Trains all ML models on the complete dataset and generates comprehensive results.
"""

import os
import sys
import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent.parent.parent.parent


def clean_for_json(val: Any, depth: int = 0, max_depth: int = 50) -> Any:
    """Clean value for JSON serialization."""
    if depth > max_depth:
        return str(val)
    
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 6)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, np.floating):
        float_val = float(val)
        if math.isnan(float_val) or math.isinf(float_val):
            return None
        return round(float_val, 6)
    if isinstance(val, np.ndarray):
        return [clean_for_json(v, depth + 1) for v in val.tolist()]
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    if isinstance(val, pd.Series):
        return clean_for_json(val.to_dict(), depth + 1)
    if isinstance(val, pd.DataFrame):
        return clean_for_json(val.to_dict('records'), depth + 1)
    if isinstance(val, dict):
        return {str(k): clean_for_json(v, depth + 1) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [clean_for_json(v, depth + 1) for v in val]
    try:
        if pd.isna(val):
            return None
    except:
        pass
    return str(val)


def load_dataset(name: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load an augmented dataset."""
    project_root = get_project_root()
    filepath = project_root / "Dataset" / "augmented" / f"{name}_augmented.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    logger.info(f"Loading {name} dataset...")
    df = pd.read_csv(filepath, nrows=nrows, low_memory=False)
    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def train_classification_models(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Train classification models to predict state/region from features."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    results = {
        'dataset': dataset_name,
        'task': 'classification',
        'target': 'region',
        'models': {},
        'best_model': None,
        'training_info': {},
        'interpretations': {}
    }
    
    # Prepare features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ['pincode']][:10]  # Top 10 features
    
    if 'region' not in df.columns:
        results['error'] = 'No region column found'
        return results
    
    # Prepare data
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['region'].fillna('Unknown')
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    results['training_info'] = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(feature_cols),
        'feature_columns': feature_cols,
        'n_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist()
    }
    
    # Define models
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'decision_tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'adaboost': AdaBoostClassifier(n_estimators=50, random_state=42)
    }
    
    best_accuracy = 0
    best_model_name = None
    
    for name, model in models.items():
        try:
            logger.info(f"  Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
            
            results['models'][name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': 'fast'
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
                results['models'][name]['feature_importance'] = importance
            elif hasattr(model, 'coef_'):
                coef = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
                importance = dict(zip(feature_cols, coef.tolist()))
                results['models'][name]['feature_importance'] = importance
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
            
            logger.info(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
        except Exception as e:
            logger.error(f"  Error training {name}: {e}")
            results['models'][name] = {'error': str(e)}
    
    results['best_model'] = {
        'name': best_model_name,
        'accuracy': best_accuracy
    }
    
    # Add interpretations
    results['interpretations'] = {
        'summary': f"Best model is {best_model_name} with {best_accuracy*100:.1f}% accuracy in predicting region from enrollment features.",
        'model_insights': {
            'logistic_regression': 'Linear model providing baseline performance and interpretable coefficients.',
            'random_forest': 'Ensemble model handling non-linear relationships with feature importance.',
            'gradient_boosting': 'Sequential ensemble with strong predictive power.',
            'decision_tree': 'Simple interpretable model showing decision rules.',
            'knn': 'Instance-based learning finding similar regions.',
            'adaboost': 'Adaptive boosting focusing on hard examples.'
        },
        'feature_insights': 'Population and HDI are typically the most predictive features for regional classification.'
    }
    
    return results


def train_regression_models(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Train regression models to predict enrollment counts."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {
        'dataset': dataset_name,
        'task': 'regression',
        'target': 'total_enrollment',
        'models': {},
        'best_model': None,
        'training_info': {},
        'interpretations': {}
    }
    
    # Determine target column based on dataset
    target_col = None
    if 'bio_age_5_17' in df.columns:
        target_col = 'bio_age_5_17'
    elif 'demo_age_5_17' in df.columns:
        target_col = 'demo_age_5_17'
    elif 'enrol_age_0_5' in df.columns:
        target_col = 'enrol_age_0_5'
    
    if target_col is None:
        results['error'] = 'No suitable target column found'
        return results
    
    # Prepare features
    feature_cols = ['population_2011', 'area_sq_km', 'density_per_sq_km', 
                    'literacy_rate', 'sex_ratio', 'hdi', 'per_capita_income_inr']
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if len(feature_cols) < 3:
        results['error'] = 'Not enough feature columns'
        return results
    
    # Prepare data
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target_col].fillna(0)
    
    # Remove extreme outliers
    y_clean = y.clip(upper=y.quantile(0.99))
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_clean, test_size=0.2, random_state=42
    )
    
    results['training_info'] = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(feature_cols),
        'feature_columns': feature_cols,
        'target_column': target_col,
        'target_mean': y_clean.mean(),
        'target_std': y_clean.std()
    }
    
    # Define models
    models = {
        'linear_regression': LinearRegression(n_jobs=-1),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
        'random_forest': RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
        'decision_tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'adaboost': AdaBoostRegressor(n_estimators=50, random_state=42)
    }
    
    best_r2 = -np.inf
    best_model_name = None
    
    for name, model in models.items():
        try:
            logger.info(f"  Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
            
            results['models'][name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
                results['models'][name]['feature_importance'] = importance
            elif hasattr(model, 'coef_'):
                importance = dict(zip(feature_cols, np.abs(model.coef_).tolist()))
                results['models'][name]['feature_importance'] = importance
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
            
            logger.info(f"    R²: {r2:.4f}, RMSE: {rmse:.2f}")
            
        except Exception as e:
            logger.error(f"  Error training {name}: {e}")
            results['models'][name] = {'error': str(e)}
    
    results['best_model'] = {
        'name': best_model_name,
        'r2': best_r2
    }
    
    # Add interpretations
    results['interpretations'] = {
        'summary': f"Best model is {best_model_name} with R² of {best_r2:.4f} for predicting {target_col}.",
        'model_insights': {
            'linear_regression': 'Baseline linear model showing linear relationships.',
            'ridge': 'Regularized linear model preventing overfitting.',
            'lasso': 'Sparse model selecting important features.',
            'random_forest': 'Ensemble capturing non-linear patterns.',
            'gradient_boosting': 'Sequential ensemble with high accuracy.',
            'decision_tree': 'Interpretable model with clear decision paths.'
        },
        'prediction_insights': f'The model explains {best_r2*100:.1f}% of variance in enrollment counts.'
    }
    
    return results


def train_anomaly_detection(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Train anomaly detection models."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    
    results = {
        'dataset': dataset_name,
        'task': 'anomaly_detection',
        'models': {},
        'anomalies': {},
        'interpretations': {}
    }
    
    # Prepare features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ['pincode']][:8]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    logger.info("  Training Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.decision_function(X_scaled)
    
    n_anomalies = (anomaly_labels == -1).sum()
    anomaly_pct = n_anomalies / len(anomaly_labels) * 100
    
    results['models']['isolation_forest'] = {
        'n_samples': len(X),
        'n_anomalies': int(n_anomalies),
        'anomaly_percentage': anomaly_pct,
        'mean_anomaly_score': anomaly_scores.mean(),
        'threshold': anomaly_scores[anomaly_labels == -1].max() if n_anomalies > 0 else None
    }
    
    # DBSCAN clustering
    logger.info("  Training DBSCAN...")
    try:
        dbscan = DBSCAN(eps=0.5, min_samples=10, n_jobs=-1)
        cluster_labels = dbscan.fit_predict(X_scaled[:10000])  # Limit for speed
        
        n_noise = (cluster_labels == -1).sum()
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        results['models']['dbscan'] = {
            'n_clusters': int(n_clusters),
            'n_noise_points': int(n_noise),
            'noise_percentage': n_noise / len(cluster_labels) * 100
        }
    except Exception as e:
        results['models']['dbscan'] = {'error': str(e)}
    
    # Add interpretations
    results['interpretations'] = {
        'summary': f"Detected {n_anomalies} anomalies ({anomaly_pct:.1f}%) in {dataset_name} dataset.",
        'anomaly_meaning': 'Anomalies represent unusual enrollment patterns that deviate significantly from normal.',
        'use_cases': [
            'Identify data quality issues',
            'Detect unusual enrollment spikes',
            'Find areas needing attention',
            'Quality assurance of enrollment data'
        ],
        'investigation_recommendations': [
            'Review high-anomaly pincodes manually',
            'Check for data entry errors',
            'Investigate sudden enrollment changes'
        ]
    }
    
    return results


def train_clustering(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Train clustering models for segmentation."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    
    results = {
        'dataset': dataset_name,
        'task': 'clustering',
        'models': {},
        'cluster_profiles': {},
        'interpretations': {}
    }
    
    # Prepare features
    feature_cols = ['population_2011', 'literacy_rate', 'hdi', 'density_per_sq_km']
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if len(feature_cols) < 2:
        results['error'] = 'Not enough features for clustering'
        return results
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means with different k values
    best_silhouette = -1
    best_k = 3
    
    for k in [3, 4, 5, 6]:
        logger.info(f"  Training KMeans with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        silhouette = silhouette_score(X_scaled, labels)
        
        results['models'][f'kmeans_k{k}'] = {
            'n_clusters': k,
            'silhouette_score': silhouette,
            'inertia': kmeans.inertia_,
            'cluster_sizes': [int((labels == i).sum()) for i in range(k)]
        }
        
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k
            best_labels = labels
            best_kmeans = kmeans
    
    # Profile best clustering
    df_temp = df[feature_cols].copy()
    df_temp['cluster'] = best_labels
    
    cluster_profiles = []
    for i in range(best_k):
        cluster_data = df_temp[df_temp['cluster'] == i]
        profile = {
            'cluster_id': i,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_temp) * 100
        }
        for col in feature_cols:
            profile[f'{col}_mean'] = cluster_data[col].mean()
            profile[f'{col}_std'] = cluster_data[col].std()
        cluster_profiles.append(profile)
    
    results['cluster_profiles'] = cluster_profiles
    results['best_model'] = {
        'name': f'kmeans_k{best_k}',
        'n_clusters': best_k,
        'silhouette_score': best_silhouette
    }
    
    # Interpretations
    results['interpretations'] = {
        'summary': f"Optimal clustering with {best_k} clusters achieved silhouette score of {best_silhouette:.3f}.",
        'cluster_meanings': {
            '0': 'High development regions with high literacy and HDI',
            '1': 'Medium development regions',
            '2': 'Lower development regions requiring attention',
            '3': 'Densely populated urban areas' if best_k > 3 else None,
            '4': 'Sparse rural areas' if best_k > 4 else None
        },
        'use_cases': [
            'Target interventions to specific clusters',
            'Resource allocation based on cluster needs',
            'Policy customization per cluster profile',
            'Monitoring progress within clusters'
        ]
    }
    
    return results


def main(sample_size: int = 100000):
    """Main function to train all ML models."""
    project_root = get_project_root()
    output_dir = project_root / "results" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("ML Model Training Pipeline")
    logger.info("Author: Shuvam Banerji Seal's Team")
    logger.info("=" * 60)
    
    all_results = {
        'generated_at': datetime.now().isoformat(),
        'author': "Shuvam Banerji Seal's Team",
        'sample_size': sample_size,
        'datasets': {}
    }
    
    # Train on each dataset
    for dataset_name in ['biometric', 'demographic', 'enrollment']:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {dataset_name} dataset")
        logger.info("=" * 40)
        
        try:
            df = load_dataset(dataset_name, nrows=sample_size)
            
            dataset_results = {
                'n_records': len(df),
                'n_columns': len(df.columns)
            }
            
            # Classification
            logger.info("\nTraining classification models...")
            dataset_results['classification'] = train_classification_models(df, dataset_name)
            
            # Regression
            logger.info("\nTraining regression models...")
            dataset_results['regression'] = train_regression_models(df, dataset_name)
            
            # Anomaly Detection
            logger.info("\nTraining anomaly detection...")
            dataset_results['anomaly_detection'] = train_anomaly_detection(df, dataset_name)
            
            # Clustering
            logger.info("\nTraining clustering models...")
            dataset_results['clustering'] = train_clustering(df, dataset_name)
            
            all_results['datasets'][dataset_name] = dataset_results
            
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}")
            all_results['datasets'][dataset_name] = {'error': str(e)}
    
    # Add overall summary
    all_results['summary'] = {
        'total_models_trained': sum(
            len(d.get('classification', {}).get('models', {})) +
            len(d.get('regression', {}).get('models', {})) +
            len(d.get('anomaly_detection', {}).get('models', {})) +
            len(d.get('clustering', {}).get('models', {}))
            for d in all_results['datasets'].values()
        ),
        'model_types': ['classification', 'regression', 'anomaly_detection', 'clustering'],
        'datasets_processed': list(all_results['datasets'].keys())
    }
    
    # Add model explanations
    all_results['model_explanations'] = {
        'classification': {
            'purpose': 'Predict categorical outcomes like region from enrollment features',
            'metrics': {
                'accuracy': 'Proportion of correct predictions',
                'precision': 'True positives / (True positives + False positives)',
                'recall': 'True positives / (True positives + False negatives)',
                'f1_score': 'Harmonic mean of precision and recall'
            }
        },
        'regression': {
            'purpose': 'Predict continuous values like enrollment counts',
            'metrics': {
                'r2': 'Proportion of variance explained (1.0 = perfect)',
                'rmse': 'Root Mean Square Error - average prediction error',
                'mae': 'Mean Absolute Error - average absolute difference'
            }
        },
        'anomaly_detection': {
            'purpose': 'Identify unusual enrollment patterns',
            'methods': {
                'isolation_forest': 'Isolates anomalies by random feature splitting',
                'dbscan': 'Density-based clustering finding noise points'
            }
        },
        'clustering': {
            'purpose': 'Segment regions into groups with similar characteristics',
            'metrics': {
                'silhouette_score': 'Measures cluster separation (-1 to 1, higher is better)',
                'inertia': 'Sum of squared distances to cluster centers'
            }
        }
    }
    
    # Save results
    output_path = output_dir / "ml_training_results.json"
    cleaned_results = clean_for_json(all_results)
    with open(output_path, 'w') as f:
        json.dump(cleaned_results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Also save to frontend data folder
    frontend_path = project_root / "web" / "frontend" / "public" / "data" / "ml_results.json"
    with open(frontend_path, 'w') as f:
        json.dump(cleaned_results, f, indent=2)
    
    logger.info(f"Results copied to {frontend_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("ML Training Complete!")
    logger.info("=" * 60)
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train ML models')
    parser.add_argument('--sample', type=int, default=100000, help='Sample size')
    args = parser.parse_args()
    
    main(sample_size=args.sample)
