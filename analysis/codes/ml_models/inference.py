"""
ML Model Inference Module
Author: Shuvam Banerji Seal's Team
UIDAI Hackathon - Aadhaar Data Analysis

Provides model inference capabilities for:
- Loading saved models
- Batch predictions
- Real-time inference
- Model serving
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
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

# Try to import joblib for model loading
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available")

# Try to import sklearn
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ModelInference:
    """
    Load and run inference on saved ML models
    """
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize model inference
        
        Args:
            model_path: Path to saved model file
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib required for model loading")
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self._load_model()
    
    def _load_model(self):
        """Load model from disk"""
        model_data = joblib.load(self.model_path)
        
        self.model = model_data['model']
        self.scalers = model_data.get('scalers', {})
        self.encoders = model_data.get('encoders', {})
        self.feature_columns = model_data.get('feature_columns', [])
        self.target_column = model_data.get('target_column', '')
        self.task_type = model_data.get('task_type', 'classification')
        self.metrics = model_data.get('metrics', {})
        self.timestamp = model_data.get('timestamp', '')
        
        logger.info(f"Model loaded from {self.model_path}")
        logger.info(f"Task type: {self.task_type}")
        logger.info(f"Features: {len(self.feature_columns)} columns")
    
    def predict(
        self,
        df: pd.DataFrame,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            df: Input DataFrame
            return_proba: Whether to return probabilities (classification only)
            
        Returns:
            Array of predictions
        """
        # Ensure required columns exist
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Select and order columns
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        if 'features' in self.scalers:
            X_scaled = self.scalers['features'].transform(X)
        else:
            X_scaled = X.values
        
        # Make predictions
        if return_proba and hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(X_scaled)
        else:
            predictions = self.model.predict(X_scaled)
        
        # Decode labels if necessary
        if 'target' in self.encoders and not return_proba:
            predictions = self.encoders['target'].inverse_transform(predictions.astype(int))
        
        return predictions
    
    def predict_single(
        self,
        features: Dict[str, Any],
        return_proba: bool = False
    ) -> Any:
        """
        Make prediction for a single sample
        
        Args:
            features: Dictionary of feature values
            return_proba: Whether to return probabilities
            
        Returns:
            Single prediction
        """
        df = pd.DataFrame([features])
        predictions = self.predict(df, return_proba=return_proba)
        return predictions[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_path': str(self.model_path),
            'task_type': self.task_type,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'timestamp': self.timestamp,
            'model_type': type(self.model).__name__
        }


class BatchPredictor:
    """
    Batch prediction processing with parallel execution
    """
    
    def __init__(
        self,
        model: ModelInference,
        batch_size: int = 10000,
        n_workers: int = -1
    ):
        """
        Initialize batch predictor
        
        Args:
            model: ModelInference instance
            batch_size: Size of each batch
            n_workers: Number of parallel workers (-1 = auto)
        """
        self.model = model
        self.batch_size = batch_size
        self.n_workers = n_workers if n_workers > 0 else os.cpu_count()
    
    def predict_dataframe(
        self,
        df: pd.DataFrame,
        output_column: str = 'prediction',
        return_proba: bool = False
    ) -> pd.DataFrame:
        """
        Make predictions on a DataFrame with progress tracking
        
        Args:
            df: Input DataFrame
            output_column: Name for prediction column
            return_proba: Whether to return probabilities
            
        Returns:
            DataFrame with predictions
        """
        result_df = df.copy()
        predictions = []
        
        n_batches = (len(df) + self.batch_size - 1) // self.batch_size
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(df))
            
            batch = df.iloc[start_idx:end_idx]
            batch_pred = self.model.predict(batch, return_proba=return_proba)
            predictions.extend(batch_pred)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{n_batches} batches")
        
        if return_proba and len(predictions[0].shape) > 0:
            # Multiple probability columns
            for j in range(len(predictions[0])):
                result_df[f'{output_column}_prob_{j}'] = [p[j] for p in predictions]
        else:
            result_df[output_column] = predictions
        
        return result_df
    
    def predict_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        output_column: str = 'prediction',
        return_proba: bool = False,
        chunksize: int = 100000
    ) -> Dict[str, Any]:
        """
        Make predictions on a file with chunked processing
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output CSV file
            output_column: Name for prediction column
            return_proba: Whether to return probabilities
            chunksize: Size of chunks for reading
            
        Returns:
            Dictionary with processing stats
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_rows = 0
        chunk_num = 0
        
        # Process in chunks
        for chunk in pd.read_csv(input_path, chunksize=chunksize):
            # Make predictions
            predictions = self.model.predict(chunk, return_proba=return_proba)
            
            if return_proba and len(predictions[0].shape) > 0:
                for j in range(len(predictions[0])):
                    chunk[f'{output_column}_prob_{j}'] = [p[j] for p in predictions]
            else:
                chunk[output_column] = predictions
            
            # Write to output
            mode = 'w' if chunk_num == 0 else 'a'
            header = chunk_num == 0
            chunk.to_csv(output_path, mode=mode, header=header, index=False)
            
            total_rows += len(chunk)
            chunk_num += 1
            
            logger.info(f"Processed chunk {chunk_num}: {total_rows} total rows")
        
        return {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'total_rows': total_rows,
            'chunks_processed': chunk_num
        }


class ModelRegistry:
    """
    Registry for managing multiple models
    """
    
    def __init__(self, models_dir: Union[str, Path]):
        """
        Initialize model registry
        
        Args:
            models_dir: Directory containing saved models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.metadata = {}
        
        self._scan_models()
    
    def _scan_models(self):
        """Scan directory for saved models"""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        for model_file in self.models_dir.glob('*.joblib'):
            model_name = model_file.stem
            try:
                model_data = joblib.load(model_file)
                self.metadata[model_name] = {
                    'path': str(model_file),
                    'task_type': model_data.get('task_type', 'unknown'),
                    'target_column': model_data.get('target_column', ''),
                    'metrics': model_data.get('metrics', {}),
                    'timestamp': model_data.get('timestamp', ''),
                    'model_type': type(model_data.get('model')).__name__
                }
            except Exception as e:
                logger.error(f"Error scanning {model_file}: {str(e)}")
        
        logger.info(f"Found {len(self.metadata)} models in registry")
    
    def load_model(self, model_name: str) -> ModelInference:
        """Load a model by name"""
        if model_name not in self.metadata:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if model_name not in self.models:
            self.models[model_name] = ModelInference(self.metadata[model_name]['path'])
        
        return self.models[model_name]
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        return [
            {'name': name, **meta}
            for name, meta in self.metadata.items()
        ]
    
    def get_best_model(
        self,
        task_type: str,
        metric: str = 'accuracy'
    ) -> Optional[str]:
        """Get the best model for a task type based on metric"""
        candidates = {
            name: meta for name, meta in self.metadata.items()
            if meta['task_type'] == task_type
        }
        
        if not candidates:
            return None
        
        best_model = max(
            candidates,
            key=lambda x: candidates[x]['metrics'].get(metric, 0)
        )
        
        return best_model


class ForecastService:
    """
    Service for time series forecasting inference
    """
    
    def __init__(self, models_dir: Union[str, Path]):
        """
        Initialize forecast service
        
        Args:
            models_dir: Directory containing forecast models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
    
    def load_prophet_model(self, model_path: Union[str, Path]) -> Any:
        """Load a saved Prophet model"""
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib required")
        
        return joblib.load(model_path)
    
    def forecast(
        self,
        model_name: str,
        periods: int,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """
        Generate forecast using a loaded model
        
        Args:
            model_name: Name of the model to use
            periods: Number of periods to forecast
            freq: Frequency of forecast
            
        Returns:
            DataFrame with forecasts
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        
        # Generate future dates
        future_dates = pd.date_range(
            start=datetime.now(),
            periods=periods,
            freq=freq
        )
        
        # This would depend on the model type
        # For Prophet models:
        if hasattr(model, 'predict'):
            future = pd.DataFrame({'ds': future_dates})
            forecast = model.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        raise ValueError(f"Unknown model type for {model_name}")


# Export classes
__all__ = ['ModelInference', 'BatchPredictor', 'ModelRegistry', 'ForecastService']
