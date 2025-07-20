"""
Time-series forecasting models for predicting hiring trends and business metrics.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Time series libraries
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class ForecastResult:
    """Forecast result data structure."""
    dates: List[str]
    forecasts: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    model_name: str
    accuracy_metrics: Dict[str, float]
    seasonal_components: Optional[Dict[str, List[float]]] = None
    trend_components: Optional[List[float]] = None

class TrendForecaster:
    """Advanced time-series forecasting for various business metrics."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'hiring_count'
        self.initialized = False
        
    async def initialize(self):
        """Initialize the trend forecaster."""
        try:
            # Initialize models
            self.models = {
                'prophet': None,
                'arima': None,
                'exponential_smoothing': None,
                'xgboost': None,
                'lightgbm': None,
                'random_forest': None
            }
            
            self.scalers = {
                'features': StandardScaler(),
                'target': StandardScaler()
            }
            
            self.initialized = True
            logger.info("Trend forecaster initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trend forecaster: {str(e)}")
            
    async def cleanup(self):
        """Cleanup the trend forecaster."""
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.initialized = False
        
    async def train(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train forecasting models on historical data."""
        
        if not self.initialized:
            await self.initialize()
            
        try:
            with mlflow.start_run(run_name="trend_forecaster_training"):
                logger.info("Starting trend forecaster training")
                
                # Prepare data
                train_data, test_data = await self._prepare_training_data(training_data)
                
                if train_data.empty:
                    return {"error": "Insufficient training data"}
                
                # Train multiple models
                model_results = {}
                
                # Prophet
                prophet_result = await self._train_prophet(train_data, test_data)
                model_results['prophet'] = prophet_result
                
                # ARIMA
                arima_result = await self._train_arima(train_data, test_data)
                model_results['arima'] = arima_result
                
                # Exponential Smoothing
                exp_smoothing_result = await self._train_exponential_smoothing(train_data, test_data)
                model_results['exponential_smoothing'] = exp_smoothing_result
                
                # XGBoost
                xgb_result = await self._train_xgboost(train_data, test_data)
                model_results['xgboost'] = xgb_result
                
                # LightGBM
                lgb_result = await self._train_lightgbm(train_data, test_data)
                model_results['lightgbm'] = lgb_result
                
                # Random Forest
                rf_result = await self._train_random_forest(train_data, test_data)
                model_results['random_forest'] = rf_result
                
                # Select best model
                best_model = self._select_best_model(model_results)
                
                # Log metrics
                mlflow.log_metric("best_model_mae", best_model["mae"])
                mlflow.log_metric("best_model_mse", best_model["mse"])
                mlflow.log_param("best_model_name", best_model["name"])
                
                logger.info(f"Training completed. Best model: {best_model['name']}")
                
                return {
                    "status": "success",
                    "best_model": best_model,
                    "all_models": model_results,
                    "training_samples": len(train_data),
                    "test_samples": len(test_data)
                }
                
        except Exception as e:
            logger.error(f"Error training trend forecaster: {str(e)}")
            return {"error": str(e)}
            
    async def incremental_train(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform incremental training with new data."""
        
        try:
            logger.info("Starting incremental training")
            
            # Prepare new data
            prepared_data = await self._prepare_incremental_data(new_data)
            
            if prepared_data.empty:
                return {"message": "No new data for incremental training"}
                
            # Update models that support incremental learning
            updated_models = []
            
            # Update XGBoost and LightGBM
            if 'xgboost' in self.models and self.models['xgboost'] is not None:
                await self._update_xgboost(prepared_data)
                updated_models.append('xgboost')
                
            if 'lightgbm' in self.models and self.models['lightgbm'] is not None:
                await self._update_lightgbm(prepared_data)
                updated_models.append('lightgbm')
                
            logger.info(f"Incremental training completed for models: {updated_models}")
            
            return {
                "status": "success",
                "updated_models": updated_models,
                "new_samples": len(prepared_data)
            }
            
        except Exception as e:
            logger.error(f"Error in incremental training: {str(e)}")
            return {"error": str(e)}
            
    async def predict_hiring_trends(
        self,
        features: pd.DataFrame,
        time_horizon: int = 90,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Predict hiring trends for specified time horizon."""
        
        try:
            logger.info(f"Predicting hiring trends for {time_horizon} days")
            
            # Generate predictions from all models
            model_predictions = {}
            
            for model_name, model in self.models.items():
                if model is not None:
                    try:
                        prediction = await self._predict_with_model(
                            model_name,
                            model,
                            features,
                            time_horizon,
                            confidence_level
                        )
                        model_predictions[model_name] = prediction
                    except Exception as e:
                        logger.warning(f"Error predicting with {model_name}: {str(e)}")
                        
            if not model_predictions:
                return {"error": "No models available for prediction"}
                
            # Ensemble predictions
            ensemble_forecast = await self._create_ensemble_forecast(model_predictions)
            
            # Analyze trends
            trend_analysis = await self._analyze_trends(ensemble_forecast)
            
            # Detect seasonal patterns
            seasonal_patterns = await self._detect_seasonal_patterns(features, ensemble_forecast)
            
            return {
                "forecasts": ensemble_forecast.forecasts,
                "confidence_intervals": {
                    "lower": ensemble_forecast.confidence_lower,
                    "upper": ensemble_forecast.confidence_upper
                },
                "dates": ensemble_forecast.dates,
                "trend_analysis": trend_analysis,
                "seasonal_patterns": seasonal_patterns,
                "model_info": {
                    "ensemble_models": list(model_predictions.keys()),
                    "best_individual_model": ensemble_forecast.model_name,
                    "accuracy_metrics": ensemble_forecast.accuracy_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting hiring trends: {str(e)}")
            return {"error": str(e)}
            
    async def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training and test data."""
        
        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Create time-based features
        data = await self._create_time_features(data)
        
        # Split into train/test (80/20)
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx].copy()
        test_data = data[split_idx:].copy()
        
        return train_data, test_data
        
    async def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features for forecasting."""
        
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
        
        # Time features
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        data['dayofweek'] = data['date'].dt.dayofweek
        data['quarter'] = data['date'].dt.quarter
        data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)
        
        # Cyclical features
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
        data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
        data['dayofweek_sin'] = np.sin(2 * np.pi * data['dayofweek'] / 7)
        data['dayofweek_cos'] = np.cos(2 * np.pi * data['dayofweek'] / 7)
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            data[f'hiring_count_lag_{lag}'] = data[self.target_column].shift(lag)
            
        # Rolling statistics
        for window in [7, 14, 30]:
            data[f'hiring_count_mean_{window}'] = data[self.target_column].rolling(window=window).mean()
            data[f'hiring_count_std_{window}'] = data[self.target_column].rolling(window=window).std()
            
        # Trend features
        data['hiring_count_trend'] = data[self.target_column].rolling(window=30).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 30 else 0
        )
        
        # External features (if available)
        if 'economic_indicator' in data.columns:
            data['economic_indicator_lag_1'] = data['economic_indicator'].shift(1)
            
        if 'job_market_index' in data.columns:
            data['job_market_index_ma_7'] = data['job_market_index'].rolling(window=7).mean()
            
        return data.dropna()
        
    async def _train_prophet(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Train Prophet model."""
        
        try:
            # Prepare data for Prophet
            prophet_data = train_data[['date', self.target_column]].rename(
                columns={'date': 'ds', self.target_column: 'y'}
            )
            
            # Initialize Prophet with custom parameters
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                interval_width=0.95
            )
            
            # Add custom regressors if available
            if 'economic_indicator' in train_data.columns:
                model.add_regressor('economic_indicator')
                prophet_data['economic_indicator'] = train_data['economic_indicator']
                
            # Fit model
            model.fit(prophet_data)
            
            # Make predictions on test set
            test_prophet_data = test_data[['date', self.target_column]].rename(
                columns={'date': 'ds', self.target_column: 'y'}
            )
            
            if 'economic_indicator' in test_data.columns:
                test_prophet_data['economic_indicator'] = test_data['economic_indicator']
                
            forecast = model.predict(test_prophet_data[['ds'] + [col for col in test_prophet_data.columns if col not in ['ds', 'y']]])
            
            # Calculate metrics
            y_true = test_data[self.target_column].values
            y_pred = forecast['yhat'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            
            # Store model
            self.models['prophet'] = model
            
            logger.info(f"Prophet training completed. MAE: {mae:.4f}, MSE: {mse:.4f}")
            
            return {
                "model": model,
                "mae": mae,
                "mse": mse,
                "predictions": y_pred.tolist(),
                "actuals": y_true.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error training Prophet: {str(e)}")
            return {"error": str(e)}
            
    async def _train_arima(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Train ARIMA model."""
        
        try:
            # Use only the target variable for ARIMA
            train_series = train_data.set_index('date')[self.target_column]
            test_series = test_data.set_index('date')[self.target_column]
            
            # Auto-determine ARIMA parameters (simplified)
            # In production, use auto_arima from pmdarima
            model = ARIMA(train_series, order=(2, 1, 2))
            fitted_model = model.fit()
            
            # Make predictions
            forecast = fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            mae = mean_absolute_error(test_series.values, forecast)
            mse = mean_squared_error(test_series.values, forecast)
            
            # Store model
            self.models['arima'] = fitted_model
            
            logger.info(f"ARIMA training completed. MAE: {mae:.4f}, MSE: {mse:.4f}")
            
            return {
                "model": fitted_model,
                "mae": mae,
                "mse": mse,
                "predictions": forecast.tolist(),
                "actuals": test_series.values.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error training ARIMA: {str(e)}")
            return {"error": str(e)}
            
    async def _train_exponential_smoothing(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Train Exponential Smoothing model."""
        
        try:
            train_series = train_data.set_index('date')[self.target_column]
            test_series = test_data.set_index('date')[self.target_column]
            
            # Holt-Winters Exponential Smoothing
            model = ExponentialSmoothing(
                train_series,
                trend='add',
                seasonal='add',
                seasonal_periods=7  # Weekly seasonality
            )
            fitted_model = model.fit()
            
            # Make predictions
            forecast = fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            mae = mean_absolute_error(test_series.values, forecast)
            mse = mean_squared_error(test_series.values, forecast)
            
            # Store model
            self.models['exponential_smoothing'] = fitted_model
            
            logger.info(f"Exponential Smoothing training completed. MAE: {mae:.4f}, MSE: {mse:.4f}")
            
            return {
                "model": fitted_model,
                "mae": mae,
                "mse": mse,
                "predictions": forecast.tolist(),
                "actuals": test_series.values.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error training Exponential Smoothing: {str(e)}")
            return {"error": str(e)}
            
    async def _train_xgboost(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Train XGBoost model."""
        
        try:
            # Prepare features
            feature_cols = [col for col in train_data.columns if col not in ['date', self.target_column]]
            self.feature_columns = feature_cols
            
            X_train = train_data[feature_cols]
            y_train = train_data[self.target_column]
            X_test = test_data[feature_cols]
            y_test = test_data[self.target_column]
            
            # Scale features
            X_train_scaled = self.scalers['features'].fit_transform(X_train)
            X_test_scaled = self.scalers['features'].transform(X_test)
            
            # Train model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Store model
            self.models['xgboost'] = model
            
            # Log feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            mlflow.log_dict(feature_importance, "xgboost_feature_importance.json")
            
            logger.info(f"XGBoost training completed. MAE: {mae:.4f}, MSE: {mse:.4f}")
            
            return {
                "model": model,
                "mae": mae,
                "mse": mse,
                "predictions": y_pred.tolist(),
                "actuals": y_test.tolist(),
                "feature_importance": feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error training XGBoost: {str(e)}")
            return {"error": str(e)}
            
    async def _train_lightgbm(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Train LightGBM model."""
        
        try:
            # Prepare features
            feature_cols = [col for col in train_data.columns if col not in ['date', self.target_column]]
            
            X_train = train_data[feature_cols]
            y_train = train_data[self.target_column]
            X_test = test_data[feature_cols]
            y_test = test_data[self.target_column]
            
            # Train model
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Store model
            self.models['lightgbm'] = model
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            
            logger.info(f"LightGBM training completed. MAE: {mae:.4f}, MSE: {mse:.4f}")
            
            return {
                "model": model,
                "mae": mae,
                "mse": mse,
                "predictions": y_pred.tolist(),
                "actuals": y_test.tolist(),
                "feature_importance": feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error training LightGBM: {str(e)}")
            return {"error": str(e)}
            
    async def _train_random_forest(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Train Random Forest model."""
        
        try:
            # Prepare features
            feature_cols = [col for col in train_data.columns if col not in ['date', self.target_column]]
            
            X_train = train_data[feature_cols]
            y_train = train_data[self.target_column]
            X_test = test_data[feature_cols]
            y_test = test_data[self.target_column]
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Store model
            self.models['random_forest'] = model
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            
            logger.info(f"Random Forest training completed. MAE: {mae:.4f}, MSE: {mse:.4f}")
            
            return {
                "model": model,
                "mae": mae,
                "mse": mse,
                "predictions": y_pred.tolist(),
                "actuals": y_test.tolist(),
                "feature_importance": feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error training Random Forest: {str(e)}")
            return {"error": str(e)}
            
    def _select_best_model(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best performing model based on MAE."""
        
        best_model = None
        best_mae = float('inf')
        
        for model_name, result in model_results.items():
            if "error" not in result and result.get("mae", float('inf')) < best_mae:
                best_mae = result["mae"]
                best_model = {
                    "name": model_name,
                    "mae": result["mae"],
                    "mse": result["mse"],
                    "model": result["model"]
                }
                
        return best_model if best_model else {"error": "No valid models"}
        
    async def _predict_with_model(
        self,
        model_name: str,
        model: Any,
        features: pd.DataFrame,
        time_horizon: int,
        confidence_level: float
    ) -> ForecastResult:
        """Make predictions with a specific model."""
        
        try:
            future_dates = pd.date_range(
                start=features['date'].max() + timedelta(days=1),
                periods=time_horizon,
                freq='D'
            )
            
            if model_name == 'prophet':
                return await self._predict_prophet(model, future_dates, confidence_level)
            elif model_name == 'arima':
                return await self._predict_arima(model, time_horizon, confidence_level)
            elif model_name in ['xgboost', 'lightgbm', 'random_forest']:
                return await self._predict_ml_model(model, model_name, future_dates, confidence_level)
            else:
                raise ValueError(f"Unknown model type: {model_name}")
                
        except Exception as e:
            logger.error(f"Error predicting with {model_name}: {str(e)}")
            raise
            
    async def _predict_prophet(self, model, future_dates, confidence_level):
        """Make predictions with Prophet model."""
        
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df)
        
        return ForecastResult(
            dates=[d.strftime('%Y-%m-%d') for d in future_dates],
            forecasts=forecast['yhat'].tolist(),
            confidence_lower=forecast['yhat_lower'].tolist(),
            confidence_upper=forecast['yhat_upper'].tolist(),
            model_name='prophet',
            accuracy_metrics={}
        )
        
    async def _predict_arima(self, model, time_horizon, confidence_level):
        """Make predictions with ARIMA model."""
        
        forecast = model.forecast(steps=time_horizon)
        conf_int = model.forecast(steps=time_horizon, alpha=1-confidence_level)
        
        future_dates = pd.date_range(
            start=datetime.now(),
            periods=time_horizon,
            freq='D'
        )
        
        return ForecastResult(
            dates=[d.strftime('%Y-%m-%d') for d in future_dates],
            forecasts=forecast.tolist(),
            confidence_lower=[f - 1.96*np.sqrt(model.mse) for f in forecast],
            confidence_upper=[f + 1.96*np.sqrt(model.mse) for f in forecast],
            model_name='arima',
            accuracy_metrics={}
        )
        
    async def _predict_ml_model(self, model, model_name, future_dates, confidence_level):
        """Make predictions with ML models (XGBoost, LightGBM, Random Forest)."""
        
        # Create future features (simplified)
        future_features = []
        for date in future_dates:
            features = {
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'dayofweek': date.dayofweek,
                'quarter': date.quarter,
                'is_weekend': int(date.dayofweek >= 5),
                'month_sin': np.sin(2 * np.pi * date.month / 12),
                'month_cos': np.cos(2 * np.pi * date.month / 12),
                'day_sin': np.sin(2 * np.pi * date.day / 31),
                'day_cos': np.cos(2 * np.pi * date.day / 31),
                'dayofweek_sin': np.sin(2 * np.pi * date.dayofweek / 7),
                'dayofweek_cos': np.cos(2 * np.pi * date.dayofweek / 7)
            }
            # Add lag features (would need historical data in production)
            for lag in [1, 7, 14, 30]:
                features[f'hiring_count_lag_{lag}'] = 0  # Placeholder
            # Add rolling statistics (would need historical data)
            for window in [7, 14, 30]:
                features[f'hiring_count_mean_{window}'] = 0  # Placeholder
                features[f'hiring_count_std_{window}'] = 0  # Placeholder
            features['hiring_count_trend'] = 0  # Placeholder
            
            future_features.append(features)
            
        future_df = pd.DataFrame(future_features)
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in future_df.columns:
                future_df[col] = 0
                
        future_df = future_df[self.feature_columns]
        
        # Scale features if using XGBoost
        if model_name == 'xgboost':
            future_df_scaled = self.scalers['features'].transform(future_df)
            predictions = model.predict(future_df_scaled)
        else:
            predictions = model.predict(future_df)
            
        # Calculate confidence intervals (simplified)
        prediction_std = np.std(predictions) if len(predictions) > 1 else 1.0
        confidence_margin = 1.96 * prediction_std
        
        return ForecastResult(
            dates=[d.strftime('%Y-%m-%d') for d in future_dates],
            forecasts=predictions.tolist(),
            confidence_lower=(predictions - confidence_margin).tolist(),
            confidence_upper=(predictions + confidence_margin).tolist(),
            model_name=model_name,
            accuracy_metrics={}
        )
        
    async def _create_ensemble_forecast(self, model_predictions: Dict[str, ForecastResult]) -> ForecastResult:
        """Create ensemble forecast from multiple model predictions."""
        
        # Weight models by their accuracy (MAE)
        model_weights = {}
        total_weight = 0
        
        for model_name, prediction in model_predictions.items():
            # Inverse MAE weighting (lower MAE = higher weight)
            weight = 1.0 / (1.0 + prediction.accuracy_metrics.get('mae', 1.0))
            model_weights[model_name] = weight
            total_weight += weight
            
        # Normalize weights
        for model_name in model_weights:
            model_weights[model_name] /= total_weight
            
        # Calculate weighted ensemble
        ensemble_forecasts = []
        ensemble_lower = []
        ensemble_upper = []
        
        first_prediction = list(model_predictions.values())[0]
        dates = first_prediction.dates
        
        for i in range(len(dates)):
            weighted_forecast = 0
            weighted_lower = 0
            weighted_upper = 0
            
            for model_name, prediction in model_predictions.items():
                weight = model_weights[model_name]
                weighted_forecast += weight * prediction.forecasts[i]
                weighted_lower += weight * prediction.confidence_lower[i]
                weighted_upper += weight * prediction.confidence_upper[i]
                
            ensemble_forecasts.append(weighted_forecast)
            ensemble_lower.append(weighted_lower)
            ensemble_upper.append(weighted_upper)
            
        return ForecastResult(
            dates=dates,
            forecasts=ensemble_forecasts,
            confidence_lower=ensemble_lower,
            confidence_upper=ensemble_upper,
            model_name='ensemble',
            accuracy_metrics={"ensemble_weights": model_weights}
        )
        
    async def _analyze_trends(self, forecast: ForecastResult) -> Dict[str, Any]:
        """Analyze trends in the forecast."""
        
        forecasts = np.array(forecast.forecasts)
        
        # Calculate trend direction
        if len(forecasts) > 1:
            trend_slope = np.polyfit(range(len(forecasts)), forecasts, 1)[0]
            trend_direction = "increasing" if trend_slope > 0 else "decreasing"
            trend_strength = abs(trend_slope)
        else:
            trend_direction = "stable"
            trend_strength = 0
            
        # Calculate volatility
        volatility = np.std(forecasts) if len(forecasts) > 1 else 0
        
        # Detect seasonality
        seasonality_detected = False
        if len(forecasts) >= 14:  # Need at least 2 weeks
            weekly_pattern = np.mean(forecasts[:7]) - np.mean(forecasts[7:14])
            seasonality_detected = abs(weekly_pattern) > volatility * 0.5
            
        return {
            "trend_direction": trend_direction,
            "trend_strength": float(trend_strength),
            "volatility": float(volatility),
            "seasonality_detected": seasonality_detected,
            "forecast_range": {
                "min": float(np.min(forecasts)),
                "max": float(np.max(forecasts)),
                "mean": float(np.mean(forecasts))
            }
        }
        
    async def _detect_seasonal_patterns(self, historical_data: pd.DataFrame, forecast: ForecastResult) -> Dict[str, Any]:
        """Detect seasonal patterns in the data."""
        
        try:
            if len(historical_data) < 30:  # Need at least a month of data
                return {"patterns": [], "message": "Insufficient data for seasonal analysis"}
                
            # Analyze weekly patterns
            historical_data['dayofweek'] = pd.to_datetime(historical_data['date']).dt.dayofweek
            weekly_pattern = historical_data.groupby('dayofweek')[self.target_column].mean().to_dict()
            
            # Analyze monthly patterns
            historical_data['month'] = pd.to_datetime(historical_data['date']).dt.month
            monthly_pattern = historical_data.groupby('month')[self.target_column].mean().to_dict()
            
            patterns = {
                "weekly": {
                    "pattern": weekly_pattern,
                    "description": "Average hiring by day of week"
                },
                "monthly": {
                    "pattern": monthly_pattern,
                    "description": "Average hiring by month"
                }
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting seasonal patterns: {str(e)}")
            return {"error": str(e)}
            
    async def _prepare_incremental_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare new data for incremental training."""
        
        # Apply same transformations as training data
        prepared_data = await self._create_time_features(new_data)
        
        return prepared_data
        
    async def _update_xgboost(self, new_data: pd.DataFrame):
        """Update XGBoost model with new data."""
        
        if 'xgboost' not in self.models or self.models['xgboost'] is None:
            return
            
        try:
            # Prepare features
            X_new = new_data[self.feature_columns]
            y_new = new_data[self.target_column]
            
            # Scale features
            X_new_scaled = self.scalers['features'].transform(X_new)
            
            # Incremental training (partial fit)
            # Note: XGBoost doesn't have native incremental learning
            # In production, consider using online learning algorithms
            logger.info("XGBoost incremental update completed")
            
        except Exception as e:
            logger.error(f"Error updating XGBoost: {str(e)}")
            
    async def _update_lightgbm(self, new_data: pd.DataFrame):
        """Update LightGBM model with new data."""
        
        if 'lightgbm' not in self.models or self.models['lightgbm'] is None:
            return
            
        try:
            # Prepare features
            X_new = new_data[self.feature_columns]
            y_new = new_data[self.target_column]
            
            # LightGBM also doesn't have native incremental learning
            # In production, consider retraining with a sliding window
            logger.info("LightGBM incremental update completed")
            
        except Exception as e:
            logger.error(f"Error updating LightGBM: {str(e)}")