"""
ML Model Training Module
Author: Shuvam Banerji Seal's Team
UIDAI Hackathon - Aadhaar Data Analysis

Provides model training capabilities for:
- Time series forecasting
- Anomaly detection
- Classification tasks
- Regression analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
import warnings
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        IsolationForest, AdaBoostClassifier, AdaBoostRegressor
    )
    from sklearn.linear_model import (
        LinearRegression, LogisticRegression, Ridge, Lasso,
        ElasticNet, SGDClassifier, SGDRegressor
    )
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix, silhouette_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")

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

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class ModelTrainer:
    """
    Generic model trainer for classification and regression tasks
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        task_type: str = 'classification',
        n_jobs: int = -1
    ):
        """
        Initialize the model trainer
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            feature_columns: List of feature column names (None = use all numeric)
            task_type: 'classification' or 'regression'
            n_jobs: Number of parallel jobs (-1 = use all cores)
        """
        self.df = df.copy()
        self.target_column = target_column
        self.task_type = task_type
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        
        # Determine feature columns
        if feature_columns is None:
            self.feature_columns = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col != target_column
            ]
        else:
            self.feature_columns = feature_columns
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for training"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for model training")
        
        # Select features and target
        X = self.df[self.feature_columns].copy()
        y = self.df[self.target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Encode categorical target if classification
        if self.task_type == 'classification' and y.dtype == 'object':
            self.encoders['target'] = LabelEncoder()
            y = self.encoders['target'].fit_transform(y)
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_scaled = self.scalers['features'].fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if self.task_type == 'classification' else None
        )
        
        logger.info(f"Data prepared: {len(self.X_train)} train, {len(self.X_test)} test samples")
    
    def _get_models(self) -> Dict[str, Any]:
        """Get models based on task type"""
        if self.task_type == 'classification':
            models = {
                'logistic_regression': LogisticRegression(max_iter=1000, n_jobs=self.n_jobs),
                'random_forest': RandomForestClassifier(n_estimators=100, n_jobs=self.n_jobs, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'svm': SVC(kernel='rbf', probability=True, random_state=42),
                'knn': KNeighborsClassifier(n_neighbors=5, n_jobs=self.n_jobs),
                'decision_tree': DecisionTreeClassifier(random_state=42),
                'adaboost': AdaBoostClassifier(n_estimators=100, random_state=42),
            }
            
            if XGBOOST_AVAILABLE:
                models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=100, n_jobs=self.n_jobs, random_state=42,
                    use_label_encoder=False, eval_metric='logloss'
                )
            
            if LIGHTGBM_AVAILABLE:
                models['lightgbm'] = lgb.LGBMClassifier(
                    n_estimators=100, n_jobs=self.n_jobs, random_state=42, verbose=-1
                )
        else:
            models = {
                'linear_regression': LinearRegression(n_jobs=self.n_jobs),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=1.0),
                'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
                'random_forest': RandomForestRegressor(n_estimators=100, n_jobs=self.n_jobs, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'svr': SVR(kernel='rbf'),
                'knn': KNeighborsRegressor(n_neighbors=5, n_jobs=self.n_jobs),
                'decision_tree': DecisionTreeRegressor(random_state=42),
                'adaboost': AdaBoostRegressor(n_estimators=100, random_state=42),
            }
            
            if XGBOOST_AVAILABLE:
                models['xgboost'] = xgb.XGBRegressor(
                    n_estimators=100, n_jobs=self.n_jobs, random_state=42
                )
            
            if LIGHTGBM_AVAILABLE:
                models['lightgbm'] = lgb.LGBMRegressor(
                    n_estimators=100, n_jobs=self.n_jobs, random_state=42, verbose=-1
                )
        
        return models
    
    def train_all_models(self) -> Dict[str, Dict]:
        """Train all available models and compare performance"""
        models = self._get_models()
        results = {}
        
        logger.info(f"Training {len(models)} models...")
        
        for name, model in models.items():
            try:
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Predict
                y_pred = model.predict(self.X_test)
                
                # Evaluate
                if self.task_type == 'classification':
                    metrics = {
                        'accuracy': accuracy_score(self.y_test, y_pred),
                        'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                        'f1': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
                    }
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy', n_jobs=self.n_jobs)
                    metrics['cv_mean'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()
                else:
                    metrics = {
                        'mse': mean_squared_error(self.y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                        'mae': mean_absolute_error(self.y_test, y_pred),
                        'r2': r2_score(self.y_test, y_pred),
                    }
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2', n_jobs=self.n_jobs)
                    metrics['cv_mean'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()
                
                results[name] = metrics
                self.models[name] = model
                
                logger.info(f"  {name}: {metrics}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        # Find best model
        if self.task_type == 'classification':
            valid_results = {k: v for k, v in results.items() if 'accuracy' in v}
            if valid_results:
                self.best_model_name = max(valid_results, key=lambda x: valid_results[x]['accuracy'])
        else:
            valid_results = {k: v for k, v in results.items() if 'r2' in v}
            if valid_results:
                self.best_model_name = max(valid_results, key=lambda x: valid_results[x]['r2'])
        
        if self.best_model_name:
            self.best_model = self.models[self.best_model_name]
            logger.info(f"Best model: {self.best_model_name}")
        
        self.results = results
        return results
    
    def hyperparameter_tuning(
        self,
        model_name: str,
        param_grid: Dict[str, List],
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning with GridSearchCV
        
        Args:
            model_name: Name of the model to tune
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            
        Returns:
            Dict with best parameters and score
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self._get_models()[model_name]
        scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=self.n_jobs, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
            }
        }
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        model_name = model_name or self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
            if len(importance) != len(self.feature_columns):
                importance = np.abs(model.coef_[0])
        else:
            raise ValueError(f"Model {model_name} doesn't support feature importance")
        
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, model_name: str, path: Union[str, Path]):
        """Save a trained model to disk"""
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib required for model saving")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.models[model_name],
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'task_type': self.task_type,
            'metrics': self.results.get(model_name, {}),
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def export_results(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Export training results to JSON"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export metrics
        metrics_path = output_dir / 'model_comparison.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'task_type': self.task_type,
                'target_column': self.target_column,
                'feature_columns': self.feature_columns,
                'num_train_samples': len(self.X_train),
                'num_test_samples': len(self.X_test),
                'results': self.results,
                'best_model': self.best_model_name,
                'timestamp': datetime.now().isoformat(),
                'author': "Shuvam Banerji Seal's Team"
            }, f, indent=2)
        
        # Export feature importance for best model
        if self.best_model_name:
            try:
                importance_df = self.get_feature_importance()
                importance_path = output_dir / 'feature_importance.json'
                importance_df.to_json(importance_path, orient='records', indent=2)
            except ValueError:
                pass
        
        return {'metrics': str(metrics_path)}


class TimeSeriesForecaster:
    """
    Time series forecasting with multiple algorithms
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        freq: str = 'D'
    ):
        """
        Initialize the time series forecaster
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            value_column: Name of the value column to forecast
            freq: Time series frequency ('D', 'W', 'M', etc.)
        """
        self.df = df.copy()
        self.date_column = date_column
        self.value_column = value_column
        self.freq = freq
        
        self.models = {}
        self.forecasts = {}
        self.results = {}
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare time series data"""
        # Ensure date column is datetime
        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
        
        # Sort by date
        self.df = self.df.sort_values(self.date_column)
        
        # Create time series
        self.ts = self.df.set_index(self.date_column)[self.value_column]
        
        # Resample if needed
        if self.freq:
            self.ts = self.ts.resample(self.freq).sum()
        
        # Handle missing values
        self.ts = self.ts.fillna(method='ffill').fillna(method='bfill')
        
        # Train/test split (80/20)
        split_idx = int(len(self.ts) * 0.8)
        self.ts_train = self.ts[:split_idx]
        self.ts_test = self.ts[split_idx:]
        
        logger.info(f"Time series prepared: {len(self.ts_train)} train, {len(self.ts_test)} test points")
    
    def fit_arima(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """Fit ARIMA/SARIMAX model"""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ARIMA")
        
        try:
            if seasonal_order:
                model = SARIMAX(
                    self.ts_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_name = f'SARIMAX{order}x{seasonal_order}'
            else:
                model = ARIMA(self.ts_train, order=order)
                model_name = f'ARIMA{order}'
            
            fitted = model.fit(disp=False)
            
            # Forecast
            forecast = fitted.forecast(steps=len(self.ts_test))
            
            # Evaluate
            mse = mean_squared_error(self.ts_test, forecast)
            mae = mean_absolute_error(self.ts_test, forecast)
            
            self.models[model_name] = fitted
            self.forecasts[model_name] = forecast
            
            result = {
                'model': model_name,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'aic': fitted.aic,
                'bic': fitted.bic
            }
            
            self.results[model_name] = result
            logger.info(f"ARIMA fitted: RMSE={np.sqrt(mse):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"ARIMA fitting error: {str(e)}")
            return {'error': str(e)}
    
    def fit_exponential_smoothing(
        self,
        seasonal_periods: Optional[int] = None,
        trend: str = 'add',
        seasonal: str = 'add'
    ) -> Dict[str, Any]:
        """Fit Exponential Smoothing (Holt-Winters) model"""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for Exponential Smoothing")
        
        try:
            model = ExponentialSmoothing(
                self.ts_train,
                trend=trend,
                seasonal=seasonal if seasonal_periods else None,
                seasonal_periods=seasonal_periods
            )
            
            fitted = model.fit()
            
            # Forecast
            forecast = fitted.forecast(steps=len(self.ts_test))
            
            # Evaluate
            mse = mean_squared_error(self.ts_test, forecast)
            mae = mean_absolute_error(self.ts_test, forecast)
            
            model_name = 'ExponentialSmoothing'
            self.models[model_name] = fitted
            self.forecasts[model_name] = forecast
            
            result = {
                'model': model_name,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'aic': fitted.aic,
                'bic': fitted.bic
            }
            
            self.results[model_name] = result
            logger.info(f"Exponential Smoothing fitted: RMSE={np.sqrt(mse):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Exponential Smoothing error: {str(e)}")
            return {'error': str(e)}
    
    def fit_prophet(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False
    ) -> Dict[str, Any]:
        """Fit Prophet model"""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet required")
        
        try:
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': self.ts_train.index,
                'y': self.ts_train.values
            })
            
            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality
            )
            
            model.fit(prophet_df)
            
            # Forecast
            future = pd.DataFrame({'ds': self.ts_test.index})
            forecast = model.predict(future)
            
            # Evaluate
            y_pred = forecast['yhat'].values
            mse = mean_squared_error(self.ts_test, y_pred)
            mae = mean_absolute_error(self.ts_test, y_pred)
            
            model_name = 'Prophet'
            self.models[model_name] = model
            self.forecasts[model_name] = pd.Series(y_pred, index=self.ts_test.index)
            
            result = {
                'model': model_name,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae
            }
            
            self.results[model_name] = result
            logger.info(f"Prophet fitted: RMSE={np.sqrt(mse):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prophet error: {str(e)}")
            return {'error': str(e)}
    
    def fit_all_models(self) -> Dict[str, Dict]:
        """Fit all available time series models"""
        results = {}
        
        # ARIMA variants
        for order in [(1, 1, 1), (2, 1, 2), (1, 0, 1)]:
            result = self.fit_arima(order=order)
            if 'error' not in result:
                results[result['model']] = result
        
        # Exponential Smoothing
        result = self.fit_exponential_smoothing()
        if 'error' not in result:
            results[result['model']] = result
        
        # Prophet
        if PROPHET_AVAILABLE:
            result = self.fit_prophet()
            if 'error' not in result:
                results[result['model']] = result
        
        return results
    
    def forecast_future(
        self,
        model_name: str,
        periods: int
    ) -> pd.Series:
        """Generate future forecasts"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not fitted")
        
        model = self.models[model_name]
        
        if 'Prophet' in model_name:
            future_dates = pd.date_range(
                start=self.ts.index[-1] + pd.Timedelta(days=1),
                periods=periods,
                freq=self.freq
            )
            future = pd.DataFrame({'ds': future_dates})
            forecast = model.predict(future)
            return pd.Series(forecast['yhat'].values, index=future_dates)
        else:
            forecast = model.forecast(steps=periods)
            return forecast
    
    def export_results(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Export forecasting results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export model comparison
        comparison_path = output_dir / 'forecast_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump({
                'value_column': self.value_column,
                'frequency': self.freq,
                'train_size': len(self.ts_train),
                'test_size': len(self.ts_test),
                'results': self.results,
                'best_model': min(self.results, key=lambda x: self.results[x].get('rmse', float('inf'))) if self.results else None,
                'timestamp': datetime.now().isoformat(),
                'author': "Shuvam Banerji Seal's Team"
            }, f, indent=2)
        
        # Export forecasts
        for name, forecast in self.forecasts.items():
            forecast_path = output_dir / f'forecast_{name.lower().replace(" ", "_")}.json'
            forecast_df = pd.DataFrame({
                'date': forecast.index.astype(str).tolist(),
                'actual': self.ts_test.values.tolist() if len(forecast) == len(self.ts_test) else [],
                'forecast': forecast.values.tolist()
            })
            forecast_df.to_json(forecast_path, orient='records', indent=2)
        
        return {'comparison': str(comparison_path)}


class AnomalyDetector:
    """
    Anomaly detection for Aadhaar data
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        n_jobs: int = -1
    ):
        """
        Initialize anomaly detector
        
        Args:
            df: Input DataFrame
            feature_columns: Columns to use for anomaly detection
            n_jobs: Number of parallel jobs
        """
        self.df = df.copy()
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        
        if feature_columns is None:
            self.feature_columns = list(df.select_dtypes(include=[np.number]).columns)
        else:
            self.feature_columns = feature_columns
        
        self.models = {}
        self.results = {}
        self.anomalies = {}
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for anomaly detection"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for anomaly detection")
        
        X = self.df[self.feature_columns].copy()
        X = X.fillna(X.median())
        
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Data prepared: {len(self.X_scaled)} samples, {len(self.feature_columns)} features")
    
    def detect_isolation_forest(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100
    ) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest"""
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            n_jobs=self.n_jobs,
            random_state=42
        )
        
        predictions = model.fit_predict(self.X_scaled)
        anomaly_scores = model.decision_function(self.X_scaled)
        
        anomaly_mask = predictions == -1
        n_anomalies = anomaly_mask.sum()
        
        self.models['isolation_forest'] = model
        self.anomalies['isolation_forest'] = {
            'mask': anomaly_mask,
            'scores': anomaly_scores,
            'indices': np.where(anomaly_mask)[0].tolist()
        }
        
        result = {
            'method': 'Isolation Forest',
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(n_anomalies / len(predictions)),
            'contamination': contamination,
            'mean_score': float(np.mean(anomaly_scores)),
            'min_score': float(np.min(anomaly_scores)),
            'max_score': float(np.max(anomaly_scores))
        }
        
        self.results['isolation_forest'] = result
        logger.info(f"Isolation Forest: {n_anomalies} anomalies detected ({result['anomaly_rate']:.2%})")
        
        return result
    
    def detect_dbscan(
        self,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> Dict[str, Any]:
        """Detect anomalies using DBSCAN clustering"""
        # Use PCA for dimensionality reduction if many features
        if len(self.feature_columns) > 10:
            pca = PCA(n_components=min(10, len(self.feature_columns)))
            X_reduced = pca.fit_transform(self.X_scaled)
        else:
            X_reduced = self.X_scaled
        
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=self.n_jobs)
        clusters = model.fit_predict(X_reduced)
        
        # Anomalies are points labeled as -1 (noise)
        anomaly_mask = clusters == -1
        n_anomalies = anomaly_mask.sum()
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        self.models['dbscan'] = model
        self.anomalies['dbscan'] = {
            'mask': anomaly_mask,
            'clusters': clusters,
            'indices': np.where(anomaly_mask)[0].tolist()
        }
        
        result = {
            'method': 'DBSCAN',
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(n_anomalies / len(clusters)),
            'n_clusters': n_clusters,
            'eps': eps,
            'min_samples': min_samples
        }
        
        self.results['dbscan'] = result
        logger.info(f"DBSCAN: {n_anomalies} anomalies detected, {n_clusters} clusters")
        
        return result
    
    def detect_statistical(
        self,
        threshold: float = 3.0
    ) -> Dict[str, Any]:
        """Detect anomalies using statistical methods (Z-score)"""
        # Calculate Z-scores
        z_scores = np.abs(self.X_scaled)
        
        # Anomaly if any feature has Z-score > threshold
        anomaly_mask = np.any(z_scores > threshold, axis=1)
        n_anomalies = anomaly_mask.sum()
        
        # Get max Z-score for each sample
        max_z_scores = np.max(z_scores, axis=1)
        
        self.anomalies['statistical'] = {
            'mask': anomaly_mask,
            'max_z_scores': max_z_scores,
            'indices': np.where(anomaly_mask)[0].tolist()
        }
        
        result = {
            'method': 'Statistical (Z-score)',
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(n_anomalies / len(anomaly_mask)),
            'threshold': threshold,
            'mean_max_z': float(np.mean(max_z_scores)),
            'max_z': float(np.max(max_z_scores))
        }
        
        self.results['statistical'] = result
        logger.info(f"Statistical: {n_anomalies} anomalies detected (Z > {threshold})")
        
        return result
    
    def detect_all_methods(self) -> Dict[str, Dict]:
        """Run all anomaly detection methods"""
        results = {}
        
        results['isolation_forest'] = self.detect_isolation_forest()
        results['dbscan'] = self.detect_dbscan()
        results['statistical'] = self.detect_statistical()
        
        return results
    
    def get_anomaly_records(self, method: str = 'isolation_forest') -> pd.DataFrame:
        """Get DataFrame of anomaly records"""
        if method not in self.anomalies:
            raise ValueError(f"Method {method} not run yet")
        
        mask = self.anomalies[method]['mask']
        return self.df[mask].copy()
    
    def export_results(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Export anomaly detection results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export summary
        summary_path = output_dir / 'anomaly_detection_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'feature_columns': self.feature_columns,
                'total_samples': len(self.df),
                'results': self.results,
                'timestamp': datetime.now().isoformat(),
                'author': "Shuvam Banerji Seal's Team"
            }, f, indent=2)
        
        # Export anomaly indices for each method
        for method, data in self.anomalies.items():
            indices_path = output_dir / f'anomaly_indices_{method}.json'
            with open(indices_path, 'w') as f:
                json.dump({
                    'method': method,
                    'indices': data['indices'][:1000],  # Limit to first 1000
                    'total_anomalies': len(data['indices'])
                }, f, indent=2)
        
        return {'summary': str(summary_path)}


# Export all classes
__all__ = ['ModelTrainer', 'TimeSeriesForecaster', 'AnomalyDetector']
