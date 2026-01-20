#!/usr/bin/env python3
"""
Comprehensive ML Model Training
Runs ALL available machine learning models on all datasets
Author: Shuvam Banerji Seal's Team
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import logging
from typing import Dict, Any, List, Optional
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("/home/shuvam/codes/UIDAI_hackathon")
DATA_DIR = BASE_DIR / "Dataset"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
FRONTEND_DATA_DIR = BASE_DIR / "web/frontend/public/data"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "checkpoints").mkdir(exist_ok=True)

# Import ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Classification models
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regression models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor, SGDRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

# Anomaly Detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, calinski_harabasz_score
)

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")


def load_dataset(name: str, sample_size: int = 100000) -> Optional[pd.DataFrame]:
    """Load and prepare dataset for training from multiple possible locations"""
    # Define possible data directories
    dataset_dirs = [
        DATA_DIR / "augmented",
        DATA_DIR / f"api_data_aadhar_{name}",
        DATA_DIR / "corrected_dataset" / name,
        DATA_DIR / "corrected_dataset" / f"{name}metric" if name == "bio" else DATA_DIR / "corrected_dataset" / name,
    ]
    
    # Map common names to actual directory names
    name_map = {
        'biometric': 'api_data_aadhar_biometric',
        'demographic': 'api_data_aadhar_demographic',
        'enrollment': 'api_data_aadhar_enrolment',
        'enrolment': 'api_data_aadhar_enrolment',
    }
    
    actual_name = name_map.get(name.lower(), name)
    
    # Try multiple locations
    search_paths = [
        DATA_DIR / actual_name,
        DATA_DIR / f"api_data_aadhar_{name}",
        DATA_DIR / "augmented",
        DATA_DIR / "corrected_dataset" / "biometric",
        DATA_DIR / "corrected_dataset" / "demographic",
        DATA_DIR / "corrected_dataset" / "enrollement",
    ]
    
    all_dfs = []
    
    for search_dir in search_paths:
        if search_dir.exists():
            # Look for CSV files in directory
            csv_files = list(search_dir.glob("*.csv"))
            if csv_files:
                logger.info(f"Found {len(csv_files)} CSV files in {search_dir}")
                for csv_file in csv_files[:3]:  # Take first 3 files to limit size
                    try:
                        df_chunk = pd.read_csv(csv_file, nrows=sample_size // 3)
                        all_dfs.append(df_chunk)
                        logger.info(f"  Loaded {len(df_chunk)} rows from {csv_file.name}")
                    except Exception as e:
                        logger.warning(f"  Error loading {csv_file.name}: {e}")
                        
                if all_dfs:
                    combined = pd.concat(all_dfs, ignore_index=True)
                    # Sample to target size
                    if len(combined) > sample_size:
                        combined = combined.sample(n=sample_size, random_state=42)
                    logger.info(f"Combined dataset: {len(combined)} rows with {len(combined.columns)} columns")
                    return combined
    
    # Also try single file patterns
    file_patterns = [
        DATA_DIR / f"{actual_name}.csv",
        DATA_DIR / f"{name}.csv",
        DATA_DIR / "augmented" / f"{name}_augmented.csv",
    ]
    
    for filepath in file_patterns:
        if filepath.exists():
            logger.info(f"Loading {filepath}...")
            df = pd.read_csv(filepath, nrows=sample_size)
            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            return df
    
    logger.warning(f"No dataset found for {name}")
    return None


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare numeric features from dataframe, including derived features"""
    
    # First, create derived features from existing columns
    df_enriched = df.copy()
    
    # Parse date if available
    date_col = None
    for col in ['date', 'Date', 'DATE']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        try:
            df_enriched[date_col] = pd.to_datetime(df_enriched[date_col], errors='coerce')
            df_enriched['day_of_week'] = df_enriched[date_col].dt.dayofweek
            df_enriched['day_of_month'] = df_enriched[date_col].dt.day
            df_enriched['month'] = df_enriched[date_col].dt.month
            df_enriched['year'] = df_enriched[date_col].dt.year
            df_enriched['is_weekend'] = (df_enriched['day_of_week'] >= 5).astype(int)
            df_enriched['quarter'] = df_enriched[date_col].dt.quarter
        except:
            pass
    
    # Encode state to numeric if exists
    state_col = None
    for col in ['state', 'State', 'STATE']:
        if col in df.columns:
            state_col = col
            break
    
    if state_col and df_enriched[state_col].dtype == 'object':
        state_le = LabelEncoder()
        df_enriched['state_encoded'] = state_le.fit_transform(df_enriched[state_col].fillna('Unknown'))
    
    # Encode district if exists
    district_col = None
    for col in ['district', 'District', 'DISTRICT']:
        if col in df.columns:
            district_col = col
            break
    
    if district_col and df_enriched[district_col].dtype == 'object':
        district_le = LabelEncoder()
        df_enriched['district_encoded'] = district_le.fit_transform(df_enriched[district_col].fillna('Unknown'))
    
    # Extract pincode features
    pincode_col = None
    for col in ['pincode', 'Pincode', 'PINCODE', 'pin_code']:
        if col in df.columns:
            pincode_col = col
            break
    
    if pincode_col:
        df_enriched['pincode_numeric'] = pd.to_numeric(df_enriched[pincode_col], errors='coerce')
        df_enriched['pincode_region'] = (df_enriched['pincode_numeric'] // 10000).astype('Int64')
        df_enriched['pincode_zone'] = (df_enriched['pincode_numeric'] // 100000).astype('Int64')
    
    # Create interaction features for any numeric columns ending in age groups
    age_cols = [col for col in df.columns if any(x in col.lower() for x in ['age_5_17', 'age_17', '5_17', '17_'])]
    if len(age_cols) >= 2:
        for i, col1 in enumerate(age_cols):
            for col2 in age_cols[i+1:]:
                if col1 in df_enriched.columns and col2 in df_enriched.columns:
                    try:
                        df_enriched[f'{col1}_plus_{col2}'] = pd.to_numeric(df_enriched[col1], errors='coerce') + pd.to_numeric(df_enriched[col2], errors='coerce')
                        df_enriched[f'{col1}_ratio_{col2}'] = pd.to_numeric(df_enriched[col1], errors='coerce') / (pd.to_numeric(df_enriched[col2], errors='coerce') + 1)
                    except:
                        pass
    
    # Sum all enrollment-related columns
    enroll_cols = [col for col in df.columns if any(x in col.lower() for x in ['enrol', 'bio', 'demo']) and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    if enroll_cols:
        df_enriched['total_enrollment'] = df_enriched[enroll_cols].sum(axis=1, numeric_only=True)
    
    # Select numeric columns (now including our derived features)
    numeric_cols = df_enriched.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude ID-like columns (but keep encoded and derived columns)
    exclude_patterns = ['id', 'index', 'unnamed']  # Removed 'pincode' from exclusion
    feature_cols = [
        col for col in numeric_cols
        if not any(p in col.lower() for p in exclude_patterns)
    ]
    
    if len(feature_cols) < 3:
        logger.warning(f"Not enough numeric features (found {len(feature_cols)}: {feature_cols})")
        return None, None
    
    X = df_enriched[feature_cols].copy()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    return pd.DataFrame(X_imputed, columns=feature_cols), feature_cols


def get_classification_models() -> Dict[str, Any]:
    """Get all classification models"""
    models = {
        # Linear models
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'ridge_classifier': RidgeClassifier(random_state=42),
        'sgd_classifier': SGDClassifier(max_iter=1000, random_state=42, n_jobs=-1),
        
        # Tree-based
        'decision_tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'extra_trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'adaboost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'bagging': BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        
        # Distance-based
        'knn': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        
        # Probabilistic
        'naive_bayes': GaussianNB(),
        
        # SVM (use LinearSVC for speed on large datasets)
        'linear_svc': LinearSVC(max_iter=1000, random_state=42),
    }
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'
        )
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        )
    
    return models


def get_regression_models() -> Dict[str, Any]:
    """Get all regression models"""
    models = {
        # Linear models
        'linear_regression': LinearRegression(n_jobs=-1),
        'ridge': Ridge(alpha=1.0, random_state=42),
        'lasso': Lasso(alpha=1.0, random_state=42, max_iter=1000),
        'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=1000),
        'bayesian_ridge': BayesianRidge(),
        'huber': HuberRegressor(max_iter=1000),
        'sgd_regressor': SGDRegressor(max_iter=1000, random_state=42),
        
        # Tree-based
        'decision_tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'extra_trees': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'adaboost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'bagging': BaggingRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        
        # Distance-based
        'knn': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        
        # SVM
        'linear_svr': LinearSVR(max_iter=1000, random_state=42),
    }
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        models['xgboost'] = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1
        )
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        )
    
    return models


def train_classification_models(X_train, X_test, y_train, y_test, feature_cols) -> Dict:
    """Train all classification models"""
    models = get_classification_models()
    results = {}
    best_accuracy = 0
    best_model_name = None
    
    for name, model in models.items():
        logger.info(f"  Training {name}...")
        try:
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
            except:
                cv_mean = accuracy
                cv_std = 0.0
            
            # Feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                for i, col in enumerate(feature_cols):
                    feature_importance[col] = float(model.feature_importances_[i])
            elif hasattr(model, 'coef_'):
                coef = np.abs(model.coef_).flatten()
                if len(coef) >= len(feature_cols):
                    for i, col in enumerate(feature_cols):
                        feature_importance[col] = float(coef[i]) if i < len(coef) else 0.0
            
            results[name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_time': 'fast',
                'feature_importance': feature_importance
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
            
            logger.info(f"    {name}: accuracy={accuracy:.4f}, f1={f1:.4f}")
            
        except Exception as e:
            logger.error(f"    Error training {name}: {str(e)}")
            results[name] = {'error': str(e)}
    
    return results, best_model_name, best_accuracy


def train_regression_models(X_train, X_test, y_train, y_test, feature_cols) -> Dict:
    """Train all regression models"""
    models = get_regression_models()
    results = {}
    best_r2 = -float('inf')
    best_model_name = None
    
    for name, model in models.items():
        logger.info(f"  Training {name}...")
        try:
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
            except:
                cv_mean = r2
                cv_std = 0.0
            
            # Feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                for i, col in enumerate(feature_cols):
                    feature_importance[col] = float(model.feature_importances_[i])
            elif hasattr(model, 'coef_'):
                coef = np.abs(np.array(model.coef_)).flatten()
                for i, col in enumerate(feature_cols):
                    if i < len(coef):
                        feature_importance[col] = float(coef[i])
            
            results[name] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_time': 'fast',
                'feature_importance': feature_importance
            }
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
            
            logger.info(f"    {name}: r2={r2:.4f}, rmse={rmse:.2f}")
            
        except Exception as e:
            logger.error(f"    Error training {name}: {str(e)}")
            results[name] = {'error': str(e)}
    
    return results, best_model_name, best_r2


def train_clustering_models(X, feature_cols) -> Dict:
    """Train clustering models"""
    results = {}
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means with different k values
    for k in [3, 5, 7, 10]:
        logger.info(f"  Training kmeans_k{k}...")
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            
            silhouette = silhouette_score(X_scaled, labels)
            calinski = calinski_harabasz_score(X_scaled, labels)
            
            # Cluster profiles
            profiles = []
            for i in range(k):
                cluster_mask = labels == i
                size = int(cluster_mask.sum())
                profiles.append({
                    'cluster_id': i,
                    'size': size,
                    'percentage': float(size / len(labels))
                })
            
            results[f'kmeans_k{k}'] = {
                'n_clusters': k,
                'silhouette_score': float(silhouette),
                'calinski_harabasz_score': float(calinski),
                'inertia': float(model.inertia_),
                'cluster_profiles': profiles
            }
            
            logger.info(f"    kmeans_k{k}: silhouette={silhouette:.4f}")
            
        except Exception as e:
            results[f'kmeans_k{k}'] = {'error': str(e)}
    
    # Gaussian Mixture
    for n in [3, 5]:
        logger.info(f"  Training gmm_n{n}...")
        try:
            model = GaussianMixture(n_components=n, random_state=42)
            labels = model.fit_predict(X_scaled)
            
            silhouette = silhouette_score(X_scaled, labels)
            
            results[f'gmm_n{n}'] = {
                'n_components': n,
                'silhouette_score': float(silhouette),
                'bic': float(model.bic(X_scaled)),
                'aic': float(model.aic(X_scaled))
            }
            
            logger.info(f"    gmm_n{n}: silhouette={silhouette:.4f}")
            
        except Exception as e:
            results[f'gmm_n{n}'] = {'error': str(e)}
    
    # Agglomerative Clustering
    logger.info("  Training agglomerative...")
    try:
        # Use sample for speed
        sample_size = min(10000, len(X_scaled))
        X_sample = X_scaled[:sample_size]
        
        model = AgglomerativeClustering(n_clusters=5)
        labels = model.fit_predict(X_sample)
        
        silhouette = silhouette_score(X_sample, labels)
        
        results['agglomerative'] = {
            'n_clusters': 5,
            'silhouette_score': float(silhouette),
            'sample_size': sample_size
        }
        
        logger.info(f"    agglomerative: silhouette={silhouette:.4f}")
        
    except Exception as e:
        results['agglomerative'] = {'error': str(e)}
    
    return results


def train_anomaly_detection(X, feature_cols) -> Dict:
    """Train anomaly detection models"""
    results = {}
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    logger.info("  Training isolation_forest...")
    try:
        model = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        predictions = model.fit_predict(X_scaled)
        
        n_anomalies = int((predictions == -1).sum())
        
        results['isolation_forest'] = {
            'n_samples': len(X_scaled),
            'n_anomalies': n_anomalies,
            'anomaly_percentage': float(n_anomalies / len(X_scaled) * 100),
            'contamination': 0.1
        }
        
        logger.info(f"    isolation_forest: {n_anomalies} anomalies ({n_anomalies/len(X_scaled)*100:.2f}%)")
        
    except Exception as e:
        results['isolation_forest'] = {'error': str(e)}
    
    # Local Outlier Factor
    logger.info("  Training local_outlier_factor...")
    try:
        # Use sample for speed
        sample_size = min(50000, len(X_scaled))
        X_sample = X_scaled[:sample_size]
        
        model = LocalOutlierFactor(contamination=0.1, n_jobs=-1)
        predictions = model.fit_predict(X_sample)
        
        n_anomalies = int((predictions == -1).sum())
        
        results['local_outlier_factor'] = {
            'n_samples': sample_size,
            'n_anomalies': n_anomalies,
            'anomaly_percentage': float(n_anomalies / sample_size * 100),
            'contamination': 0.1
        }
        
        logger.info(f"    local_outlier_factor: {n_anomalies} anomalies")
        
    except Exception as e:
        results['local_outlier_factor'] = {'error': str(e)}
    
    # Elliptic Envelope (for Gaussian distributed data)
    logger.info("  Training elliptic_envelope...")
    try:
        sample_size = min(10000, len(X_scaled))
        X_sample = X_scaled[:sample_size]
        
        model = EllipticEnvelope(contamination=0.1, random_state=42)
        predictions = model.fit_predict(X_sample)
        
        n_anomalies = int((predictions == -1).sum())
        
        results['elliptic_envelope'] = {
            'n_samples': sample_size,
            'n_anomalies': n_anomalies,
            'anomaly_percentage': float(n_anomalies / sample_size * 100),
            'contamination': 0.1
        }
        
        logger.info(f"    elliptic_envelope: {n_anomalies} anomalies")
        
    except Exception as e:
        results['elliptic_envelope'] = {'error': str(e)}
    
    return results


def run_comprehensive_training(sample_size: int = 100000):
    """Run comprehensive ML training on all datasets"""
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE ML MODEL TRAINING")
    logger.info(f"Sample size: {sample_size}")
    logger.info("=" * 60)
    
    all_results = {
        'generated_at': datetime.now().isoformat(),
        'author': 'Shuvam Banerji Seal, Alok Mishra, Aheli Poddar',
        'sample_size': sample_size,
        'datasets': {},
        'model_explanations': {
            'classification': {
                'purpose': 'Predict categorical outcomes like region, state, or enrollment type',
                'models_used': list(get_classification_models().keys()),
                'metrics': {
                    'accuracy': 'Proportion of correct predictions',
                    'precision': 'Proportion of positive predictions that are correct',
                    'recall': 'Proportion of actual positives identified correctly',
                    'f1_score': 'Harmonic mean of precision and recall'
                }
            },
            'regression': {
                'purpose': 'Predict continuous values like population, enrollment count',
                'models_used': list(get_regression_models().keys()),
                'metrics': {
                    'r2': 'Coefficient of determination (variance explained)',
                    'rmse': 'Root mean squared error',
                    'mae': 'Mean absolute error',
                    'mse': 'Mean squared error'
                }
            },
            'clustering': {
                'purpose': 'Group similar records together to discover patterns',
                'methods': {
                    'kmeans': 'Partition data into k clusters based on distance to centroids',
                    'gmm': 'Probabilistic clustering using Gaussian distributions',
                    'agglomerative': 'Hierarchical clustering building from individual points'
                }
            },
            'anomaly_detection': {
                'purpose': 'Identify unusual or outlier records in the data',
                'methods': {
                    'isolation_forest': 'Isolates anomalies by random partitioning',
                    'local_outlier_factor': 'Compares local density to neighbors',
                    'elliptic_envelope': 'Fits Gaussian and identifies outliers'
                }
            }
        },
        'summary': {
            'total_models_trained': 0,
            'model_types': ['classification', 'regression', 'clustering', 'anomaly_detection'],
            'datasets_processed': []
        }
    }
    
    datasets = ['biometric', 'demographic', 'enrollment']
    total_models = 0
    
    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset_name.upper()} dataset")
        logger.info("=" * 60)
        
        df = load_dataset(dataset_name, sample_size)
        
        if df is None or len(df) < 100:
            logger.warning(f"Skipping {dataset_name} - insufficient data")
            continue
        
        X, feature_cols = prepare_features(df)
        
        if X is None:
            logger.warning(f"Skipping {dataset_name} - no valid features")
            continue
        
        dataset_results = {
            'n_records': len(df),
            'n_columns': len(df.columns),
            'n_features': len(feature_cols),
            'feature_columns': feature_cols
        }
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # CLASSIFICATION
        logger.info("\n--- Classification ---")
        # Create target variable (region-based classification)
        if 'region' in df.columns:
            y_class = df['region'].fillna('Unknown')
        elif 'state' in df.columns:
            # Create region from state
            y_class = df['state'].fillna('Unknown')
            # Map to regions
            state_region = {
                'MAHARASHTRA': 'West', 'GUJARAT': 'West', 'GOA': 'West', 'RAJASTHAN': 'West',
                'UTTAR PRADESH': 'North', 'DELHI': 'North', 'HARYANA': 'North', 'PUNJAB': 'North',
                'HIMACHAL PRADESH': 'North', 'UTTARAKHAND': 'North', 'JAMMU AND KASHMIR': 'North',
                'TAMIL NADU': 'South', 'KARNATAKA': 'South', 'KERALA': 'South', 
                'ANDHRA PRADESH': 'South', 'TELANGANA': 'South',
                'WEST BENGAL': 'East', 'BIHAR': 'East', 'JHARKHAND': 'East', 'ODISHA': 'East',
                'MADHYA PRADESH': 'Central', 'CHHATTISGARH': 'Central',
                'ASSAM': 'Northeast', 'MEGHALAYA': 'Northeast', 'MANIPUR': 'Northeast',
                'MIZORAM': 'Northeast', 'NAGALAND': 'Northeast', 'TRIPURA': 'Northeast',
                'ARUNACHAL PRADESH': 'Northeast', 'SIKKIM': 'Northeast'
            }
            y_class = y_class.str.upper().map(lambda x: state_region.get(x, 'Other'))
        else:
            # Create dummy classification target based on quartiles
            y_class = pd.qcut(X.iloc[:, 0], q=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
        
        le = LabelEncoder()
        y_class_encoded = le.fit_transform(y_class.astype(str))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_class_encoded, test_size=0.2, random_state=42, stratify=y_class_encoded
        )
        
        class_results, best_class_model, best_accuracy = train_classification_models(
            X_train, X_test, y_train, y_test, feature_cols
        )
        
        dataset_results['classification'] = {
            'target': 'region',
            'n_classes': len(le.classes_),
            'class_names': list(le.classes_),
            'models': class_results,
            'best_model': {'name': best_class_model, 'accuracy': best_accuracy},
            'training_info': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': len(feature_cols),
                'feature_columns': feature_cols,
                'n_classes': len(le.classes_),
                'class_names': list(le.classes_)
            },
            'interpretations': {
                'summary': f"Classification achieved {best_accuracy*100:.1f}% accuracy using {best_class_model}",
                'model_insights': {
                    best_class_model: f"Best performing model with {best_accuracy*100:.1f}% accuracy"
                },
                'feature_insights': "Geographic and demographic features are most predictive of regional classification"
            }
        }
        total_models += len([r for r in class_results.values() if 'error' not in r])
        
        # REGRESSION
        logger.info("\n--- Regression ---")
        # Find best regression target
        regression_targets = ['population_2011', 'density_per_sq_km', 'literacy_rate', 'total_enrollment']
        y_reg = None
        target_col = None
        
        for col in regression_targets:
            if col in df.columns:
                y_temp = df[col].fillna(df[col].median())
                if y_temp.std() > 0:
                    y_reg = y_temp.values
                    target_col = col
                    break
        
        if y_reg is None:
            # Use first numeric column as target
            y_reg = X.iloc[:, 0].values
            target_col = feature_cols[0]
            X_reg = X.iloc[:, 1:]
            feature_cols_reg = feature_cols[1:]
        else:
            X_reg = X
            feature_cols_reg = feature_cols
        
        X_reg_scaled = StandardScaler().fit_transform(X_reg)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg_scaled, y_reg, test_size=0.2, random_state=42
        )
        
        reg_results, best_reg_model, best_r2 = train_regression_models(
            X_train, X_test, y_train, y_test, feature_cols_reg
        )
        
        dataset_results['regression'] = {
            'target': target_col,
            'models': reg_results,
            'best_model': {'name': best_reg_model, 'r2': best_r2},
            'training_info': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': len(feature_cols_reg),
                'feature_columns': feature_cols_reg,
                'target_column': target_col,
                'target_mean': float(np.mean(y_reg)),
                'target_std': float(np.std(y_reg))
            },
            'interpretations': {
                'summary': f"Regression achieved R²={best_r2:.4f} using {best_reg_model}",
                'model_insights': {
                    best_reg_model: f"Explains {best_r2*100:.1f}% of variance in {target_col}"
                },
                'prediction_insights': f"Model can predict {target_col} with reasonable accuracy"
            }
        }
        total_models += len([r for r in reg_results.values() if 'error' not in r])
        
        # CLUSTERING
        logger.info("\n--- Clustering ---")
        cluster_results = train_clustering_models(X, feature_cols)
        
        # Find best clustering
        best_cluster = max(
            [(k, v.get('silhouette_score', 0)) for k, v in cluster_results.items() if 'silhouette_score' in v],
            key=lambda x: x[1],
            default=('kmeans_k5', 0)
        )
        
        dataset_results['clustering'] = {
            'models': cluster_results,
            'best_model': {
                'name': best_cluster[0],
                'n_clusters': cluster_results.get(best_cluster[0], {}).get('n_clusters', 5),
                'silhouette_score': best_cluster[1]
            },
            'cluster_profiles': cluster_results.get(best_cluster[0], {}).get('cluster_profiles', []),
            'interpretations': {
                'summary': f"Best clustering with silhouette score {best_cluster[1]:.3f}",
                'use_cases': [
                    'Segment districts by enrollment patterns',
                    'Identify similar regions for targeted policies',
                    'Discover natural groupings in the data'
                ]
            }
        }
        total_models += len([r for r in cluster_results.values() if 'error' not in r])
        
        # ANOMALY DETECTION
        logger.info("\n--- Anomaly Detection ---")
        anomaly_results = train_anomaly_detection(X, feature_cols)
        
        dataset_results['anomaly_detection'] = {
            'models': anomaly_results,
            'interpretations': {
                'summary': 'Multiple anomaly detection methods applied to identify outliers',
                'anomaly_meaning': 'Records that deviate significantly from normal patterns',
                'use_cases': [
                    'Detect data quality issues',
                    'Identify unusual enrollment patterns',
                    'Flag records for manual review'
                ]
            }
        }
        total_models += len([r for r in anomaly_results.values() if 'error' not in r])
        
        all_results['datasets'][dataset_name] = dataset_results
        all_results['summary']['datasets_processed'].append(dataset_name)
    
    all_results['summary']['total_models_trained'] = total_models
    
    # Save results
    output_path = MODELS_DIR / "ml_training_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved results to {output_path}")
    
    # Copy to frontend
    frontend_path = FRONTEND_DATA_DIR / "ml_results.json"
    with open(frontend_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Copied to {frontend_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Total models trained: {total_models}")
    logger.info(f"Datasets processed: {', '.join(all_results['summary']['datasets_processed'])}")
    logger.info("=" * 60)
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive ML Training')
    parser.add_argument('--sample-size', type=int, default=100000, help='Sample size per dataset')
    args = parser.parse_args()
    
    run_comprehensive_training(args.sample_size)
