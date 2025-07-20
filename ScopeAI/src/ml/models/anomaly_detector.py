"""
Anomaly detection system for identifying unusual patterns in company data and initiatives.
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

# Anomaly detection libraries
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
import scipy.stats as stats
from scipy import signal
import mlflow

from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Anomaly:
    """Anomaly data structure."""
    timestamp: str
    anomaly_type: str
    severity: str  # low, medium, high, critical
    score: float
    description: str
    affected_metrics: List[str]
    context: Dict[str, Any]
    recommendations: List[str]

class AnomalyDetector:
    """Advanced anomaly detection for company metrics and behavior."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.baseline_stats = {}
        self.anomaly_thresholds = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize the anomaly detector."""
        try:
            # Initialize models
            self.models = {
                'isolation_forest': IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                ),
                'one_class_svm': OneClassSVM(
                    nu=0.1,
                    kernel='rbf',
                    gamma='scale'
                ),
                'local_outlier_factor': LocalOutlierFactor(
                    n_neighbors=20,
                    contamination=0.1,
                    novelty=True
                ),
                'elliptic_envelope': EllipticEnvelope(
                    contamination=0.1,
                    random_state=42
                )
            }
            
            self.scalers = {
                'standard': StandardScaler(),
                'robust': RobustScaler()
            }
            
            # Default anomaly thresholds
            self.anomaly_thresholds = {
                'hiring_rate': {'low': 1.5, 'medium': 2.0, 'high': 2.5, 'critical': 3.0},
                'initiative_frequency': {'low': 1.5, 'medium': 2.0, 'high': 2.5, 'critical': 3.0},
                'technology_adoption': {'low': 1.5, 'medium': 2.0, 'high': 2.5, 'critical': 3.0},
                'sentiment_shift': {'low': 1.5, 'medium': 2.0, 'high': 2.5, 'critical': 3.0}
            }
            
            self.initialized = True
            logger.info("Anomaly detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize anomaly detector: {str(e)}")
            
    async def cleanup(self):
        """Cleanup the anomaly detector."""
        self.models = {}
        self.scalers = {}
        self.baseline_stats = {}
        self.anomaly_thresholds = {}
        self.initialized = False
        
    async def train(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train anomaly detection models on normal behavior data."""
        
        if not self.initialized:
            await self.initialize()
            
        try:
            with mlflow.start_run(run_name="anomaly_detector_training"):
                logger.info("Starting anomaly detector training")
                
                # Prepare training data
                features = await self._prepare_anomaly_features(training_data)
                
                if features.empty:
                    return {"error": "Insufficient training data"}
                
                # Establish baseline statistics
                await self._establish_baselines(features)
                
                # Train models
                model_results = {}
                
                # Scale features
                scaled_features = self.scalers['standard'].fit_transform(features)
                robust_scaled_features = self.scalers['robust'].fit_transform(features)
                
                # Train Isolation Forest
                self.models['isolation_forest'].fit(scaled_features)
                model_results['isolation_forest'] = {"status": "trained"}
                
                # Train One-Class SVM
                self.models['one_class_svm'].fit(robust_scaled_features)
                model_results['one_class_svm'] = {"status": "trained"}
                
                # Train Local Outlier Factor
                self.models['local_outlier_factor'].fit(scaled_features)
                model_results['local_outlier_factor'] = {"status": "trained"}
                
                # Train Elliptic Envelope
                self.models['elliptic_envelope'].fit(scaled_features)
                model_results['elliptic_envelope'] = {"status": "trained"}
                
                # Log training metrics
                mlflow.log_param("training_samples", len(features))
                mlflow.log_param("feature_count", features.shape[1])
                
                logger.info("Anomaly detector training completed")
                
                return {
                    "status": "success",
                    "models_trained": list(model_results.keys()),
                    "training_samples": len(features),
                    "feature_count": features.shape[1],
                    "baseline_stats": self.baseline_stats
                }
                
        except Exception as e:
            logger.error(f"Error training anomaly detector: {str(e)}")
            return {"error": str(e)}
            
    async def incremental_train(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Update anomaly detector with new normal behavior data."""
        
        try:
            logger.info("Starting incremental anomaly detector training")
            
            # Prepare new features
            new_features = await self._prepare_anomaly_features(new_data)
            
            if new_features.empty:
                return {"message": "No new data for incremental training"}
                
            # Update baseline statistics
            await self._update_baselines(new_features)
            
            # For ensemble methods, we typically retrain periodically
            # rather than true incremental learning
            logger.info("Incremental training completed - baselines updated")
            
            return {
                "status": "success",
                "new_samples": len(new_features),
                "updated_baselines": True
            }
            
        except Exception as e:
            logger.error(f"Error in incremental training: {str(e)}")
            return {"error": str(e)}
            
    async def detect_initiative_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in company initiatives."""
        
        try:
            logger.info("Detecting initiative anomalies")
            
            # Prepare features
            features = await self._prepare_anomaly_features(data)
            
            if features.empty:
                return {"detected_anomalies": [], "message": "No data to analyze"}
                
            # Detect anomalies using multiple methods
            anomalies = []
            
            # Statistical anomalies
            stat_anomalies = await self._detect_statistical_anomalies(data, features)
            anomalies.extend(stat_anomalies)
            
            # ML-based anomalies
            ml_anomalies = await self._detect_ml_anomalies(features)
            anomalies.extend(ml_anomalies)
            
            # Domain-specific anomalies
            domain_anomalies = await self._detect_domain_anomalies(data)
            anomalies.extend(domain_anomalies)
            
            # Time series anomalies
            ts_anomalies = await self._detect_time_series_anomalies(data)
            anomalies.extend(ts_anomalies)
            
            # Consolidate and rank anomalies
            consolidated_anomalies = await self._consolidate_anomalies(anomalies)
            
            # Calculate baseline metrics for comparison
            baseline_metrics = await self._calculate_baseline_metrics(data)
            
            # Generate interpretations
            interpretations = await self._interpret_anomalies(consolidated_anomalies, data)
            
            return {
                "detected_anomalies": [self._anomaly_to_dict(a) for a in consolidated_anomalies],
                "scores": [a.score for a in consolidated_anomalies],
                "baseline": baseline_metrics,
                "interpretation": interpretations,
                "detection_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting initiative anomalies: {str(e)}")
            return {"error": str(e)}
            
    async def detect_hiring_anomalies(self, company_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in hiring patterns."""
        
        try:
            logger.info(f"Detecting hiring anomalies for company {company_id}")
            
            # Focus on hiring-specific metrics
            hiring_features = await self._prepare_hiring_features(data)
            
            if hiring_features.empty:
                return {"detected_anomalies": [], "message": "No hiring data to analyze"}
                
            anomalies = []
            
            # Sudden hiring spikes or drops
            hiring_rate_anomalies = await self._detect_hiring_rate_anomalies(data)
            anomalies.extend(hiring_rate_anomalies)
            
            # Unusual skill demand patterns
            skill_anomalies = await self._detect_skill_demand_anomalies(data)
            anomalies.extend(skill_anomalies)
            
            # Geographic hiring anomalies
            geo_anomalies = await self._detect_geographic_anomalies(data)
            anomalies.extend(geo_anomalies)
            
            # Department/role anomalies
            role_anomalies = await self._detect_role_anomalies(data)
            anomalies.extend(role_anomalies)
            
            consolidated_anomalies = await self._consolidate_anomalies(anomalies)
            
            return {
                "company_id": company_id,
                "detected_anomalies": [self._anomaly_to_dict(a) for a in consolidated_anomalies],
                "anomaly_count": len(consolidated_anomalies),
                "analysis_period": {
                    "start": data['date'].min().isoformat() if not data.empty else None,
                    "end": data['date'].max().isoformat() if not data.empty else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting hiring anomalies: {str(e)}")
            return {"error": str(e)}
            
    async def detect_technology_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in technology adoption and mentions."""
        
        try:
            logger.info("Detecting technology adoption anomalies")
            
            # Prepare technology-specific features
            tech_features = await self._prepare_technology_features(data)
            
            if tech_features.empty:
                return {"detected_anomalies": [], "message": "No technology data to analyze"}
                
            anomalies = []
            
            # Sudden technology adoption spikes
            adoption_anomalies = await self._detect_adoption_anomalies(data)
            anomalies.extend(adoption_anomalies)
            
            # Unusual technology combinations
            combination_anomalies = await self._detect_technology_combination_anomalies(data)
            anomalies.extend(combination_anomalies)
            
            # Emerging technology signals
            emerging_anomalies = await self._detect_emerging_technology_anomalies(data)
            anomalies.extend(emerging_anomalies)
            
            consolidated_anomalies = await self._consolidate_anomalies(anomalies)
            
            return {
                "detected_anomalies": [self._anomaly_to_dict(a) for a in consolidated_anomalies],
                "technology_trends": await self._analyze_technology_trends(data),
                "emerging_technologies": await self._identify_emerging_technologies(data)
            }
            
        except Exception as e:
            logger.error(f"Error detecting technology anomalies: {str(e)}")
            return {"error": str(e)}
            
    async def _prepare_anomaly_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection."""
        
        features = pd.DataFrame()
        
        try:
            # Time-based features
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                features['day_of_week'] = data['date'].dt.dayofweek
                features['day_of_month'] = data['date'].dt.day
                features['month'] = data['date'].dt.month
                features['quarter'] = data['date'].dt.quarter
                
            # Hiring metrics
            if 'job_count' in data.columns:
                features['job_count'] = data['job_count']
                features['job_count_rolling_7'] = data['job_count'].rolling(window=7, min_periods=1).mean()
                features['job_count_rolling_30'] = data['job_count'].rolling(window=30, min_periods=1).mean()
                features['job_count_pct_change'] = data['job_count'].pct_change().fillna(0)
                
            # Initiative metrics
            if 'initiative_count' in data.columns:
                features['initiative_count'] = data['initiative_count']
                features['initiative_count_rolling_7'] = data['initiative_count'].rolling(window=7, min_periods=1).mean()
                
            # Technology metrics
            if 'technology_mentions' in data.columns:
                features['technology_mentions'] = data['technology_mentions']
                features['tech_diversity'] = data.groupby('date')['technology'].nunique().reindex(data['date']).fillna(0)
                
            # Sentiment metrics
            if 'sentiment_score' in data.columns:
                features['sentiment_score'] = data['sentiment_score']
                features['sentiment_volatility'] = data['sentiment_score'].rolling(window=7, min_periods=1).std().fillna(0)
                
            # Competitive metrics
            if 'competitor_activity' in data.columns:
                features['competitor_activity'] = data['competitor_activity']
                
            # Fill missing values
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing anomaly features: {str(e)}")
            return pd.DataFrame()
            
    async def _establish_baselines(self, features: pd.DataFrame):
        """Establish baseline statistics for normal behavior."""
        
        self.baseline_stats = {}
        
        for column in features.columns:
            if features[column].dtype in ['int64', 'float64']:
                self.baseline_stats[column] = {
                    'mean': features[column].mean(),
                    'std': features[column].std(),
                    'median': features[column].median(),
                    'q25': features[column].quantile(0.25),
                    'q75': features[column].quantile(0.75),
                    'iqr': features[column].quantile(0.75) - features[column].quantile(0.25),
                    'min': features[column].min(),
                    'max': features[column].max()
                }
                
        logger.info(f"Established baselines for {len(self.baseline_stats)} metrics")
        
    async def _update_baselines(self, new_features: pd.DataFrame):
        """Update baseline statistics with new data."""
        
        # Use exponential moving average to update baselines
        alpha = 0.1  # Learning rate
        
        for column in new_features.columns:
            if column in self.baseline_stats and new_features[column].dtype in ['int64', 'float64']:
                new_mean = new_features[column].mean()
                new_std = new_features[column].std()
                
                # Update with exponential moving average
                self.baseline_stats[column]['mean'] = (
                    (1 - alpha) * self.baseline_stats[column]['mean'] + 
                    alpha * new_mean
                )
                self.baseline_stats[column]['std'] = (
                    (1 - alpha) * self.baseline_stats[column]['std'] + 
                    alpha * new_std
                )
                
    async def _detect_statistical_anomalies(self, data: pd.DataFrame, features: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies using statistical methods."""
        
        anomalies = []
        
        try:
            for column in features.columns:
                if column in self.baseline_stats:
                    baseline = self.baseline_stats[column]
                    values = features[column]
                    
                    # Z-score based detection
                    z_scores = np.abs((values - baseline['mean']) / (baseline['std'] + 1e-8))
                    
                    for idx, z_score in enumerate(z_scores):
                        severity = self._get_severity_from_z_score(z_score)
                        
                        if severity != 'normal':
                            anomaly = Anomaly(
                                timestamp=data.iloc[idx]['date'].isoformat() if 'date' in data.columns else datetime.now().isoformat(),
                                anomaly_type='statistical',
                                severity=severity,
                                score=float(z_score),
                                description=f"Statistical anomaly in {column}: value {values.iloc[idx]:.2f} deviates {z_score:.2f} standard deviations from baseline",
                                affected_metrics=[column],
                                context={
                                    'method': 'z_score',
                                    'baseline_mean': baseline['mean'],
                                    'baseline_std': baseline['std'],
                                    'observed_value': float(values.iloc[idx])
                                },
                                recommendations=[
                                    f"Investigate cause of unusual {column} value",
                                    "Check for data quality issues",
                                    "Verify if this represents a genuine business change"
                                ]
                            )
                            anomalies.append(anomaly)
                            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting statistical anomalies: {str(e)}")
            return []
            
    async def _detect_ml_anomalies(self, features: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies using machine learning models."""
        
        anomalies = []
        
        try:
            if features.empty:
                return anomalies
                
            # Scale features
            scaled_features = self.scalers['standard'].transform(features)
            
            # Detect with each model
            model_predictions = {}
            
            # Isolation Forest
            if 'isolation_forest' in self.models:
                predictions = self.models['isolation_forest'].predict(scaled_features)
                scores = self.models['isolation_forest'].score_samples(scaled_features)
                model_predictions['isolation_forest'] = (predictions, scores)
                
            # One-Class SVM
            if 'one_class_svm' in self.models:
                predictions = self.models['one_class_svm'].predict(scaled_features)
                scores = self.models['one_class_svm'].score_samples(scaled_features)
                model_predictions['one_class_svm'] = (predictions, scores)
                
            # Local Outlier Factor
            if 'local_outlier_factor' in self.models:
                predictions = self.models['local_outlier_factor'].predict(scaled_features)
                scores = self.models['local_outlier_factor'].score_samples(scaled_features)
                model_predictions['local_outlier_factor'] = (predictions, scores)
                
            # Elliptic Envelope
            if 'elliptic_envelope' in self.models:
                predictions = self.models['elliptic_envelope'].predict(scaled_features)
                scores = self.models['elliptic_envelope'].score_samples(scaled_features)
                model_predictions['elliptic_envelope'] = (predictions, scores)
                
            # Combine predictions (ensemble)
            for idx in range(len(features)):
                anomaly_votes = 0
                combined_score = 0
                
                for model_name, (predictions, scores) in model_predictions.items():
                    if predictions[idx] == -1:  # Anomaly
                        anomaly_votes += 1
                    combined_score += abs(scores[idx])
                    
                # If majority of models agree it's an anomaly
                if anomaly_votes >= len(model_predictions) / 2:
                    severity = self._get_severity_from_ml_score(combined_score / len(model_predictions))
                    
                    anomaly = Anomaly(
                        timestamp=datetime.now().isoformat(),
                        anomaly_type='ml_ensemble',
                        severity=severity,
                        score=combined_score / len(model_predictions),
                        description=f"ML ensemble detected anomaly with {anomaly_votes}/{len(model_predictions)} model agreement",
                        affected_metrics=list(features.columns),
                        context={
                            'method': 'ml_ensemble',
                            'models_agreeing': anomaly_votes,
                            'total_models': len(model_predictions),
                            'feature_vector': scaled_features[idx].tolist()
                        },
                        recommendations=[
                            "Investigate multidimensional anomaly",
                            "Check for systemic changes",
                            "Consider business context for explanation"
                        ]
                    )
                    anomalies.append(anomaly)
                    
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting ML anomalies: {str(e)}")
            return []
            
    async def _detect_domain_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect domain-specific anomalies in company behavior."""
        
        anomalies = []
        
        try:
            # Unusual hiring patterns
            if 'job_title' in data.columns and 'company_id' in data.columns:
                # Detect unusual job title combinations for a company
                company_job_patterns = await self._analyze_job_title_patterns(data)
                
                for pattern_anomaly in company_job_patterns:
                    anomalies.append(pattern_anomaly)
                    
            # Initiative frequency anomalies
            if 'initiative_type' in data.columns:
                initiative_anomalies = await self._analyze_initiative_patterns(data)
                anomalies.extend(initiative_anomalies)
                
            # Technology stack anomalies
            if 'technology' in data.columns:
                tech_anomalies = await self._analyze_technology_patterns(data)
                anomalies.extend(tech_anomalies)
                
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting domain anomalies: {str(e)}")
            return []
            
    async def _detect_time_series_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies in time series patterns."""
        
        anomalies = []
        
        try:
            if 'date' not in data.columns:
                return anomalies
                
            data = data.sort_values('date')
            
            # Detect change points in hiring rate
            if 'job_count' in data.columns:
                change_points = await self._detect_change_points(data['job_count'].values)
                
                for cp_idx in change_points:
                    if cp_idx < len(data):
                        anomaly = Anomaly(
                            timestamp=data.iloc[cp_idx]['date'].isoformat(),
                            anomaly_type='change_point',
                            severity='medium',
                            score=1.0,
                            description=f"Significant change point detected in hiring pattern",
                            affected_metrics=['job_count'],
                            context={
                                'method': 'change_point_detection',
                                'change_point_index': int(cp_idx)
                            },
                            recommendations=[
                                "Investigate cause of hiring pattern change",
                                "Check for organizational changes",
                                "Verify business strategy shifts"
                            ]
                        )
                        anomalies.append(anomaly)
                        
            # Detect seasonal anomalies
            seasonal_anomalies = await self._detect_seasonal_anomalies(data)
            anomalies.extend(seasonal_anomalies)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting time series anomalies: {str(e)}")
            return []
            
    def _get_severity_from_z_score(self, z_score: float) -> str:
        """Get severity level from z-score."""
        
        if z_score < 1.5:
            return 'normal'
        elif z_score < 2.0:
            return 'low'
        elif z_score < 2.5:
            return 'medium'
        elif z_score < 3.0:
            return 'high'
        else:
            return 'critical'
            
    def _get_severity_from_ml_score(self, score: float) -> str:
        """Get severity level from ML anomaly score."""
        
        if score < 0.3:
            return 'low'
        elif score < 0.6:
            return 'medium'
        elif score < 0.8:
            return 'high'
        else:
            return 'critical'
            
    async def _consolidate_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Consolidate similar anomalies and remove duplicates."""
        
        if not anomalies:
            return []
            
        # Sort by timestamp and score
        anomalies.sort(key=lambda x: (x.timestamp, -x.score))
        
        consolidated = []
        
        # Group by timestamp and type
        for anomaly in anomalies:
            # Check if similar anomaly already exists
            similar_found = False
            
            for existing in consolidated:
                if (abs((pd.to_datetime(anomaly.timestamp) - pd.to_datetime(existing.timestamp)).total_seconds()) < 3600 and  # Within 1 hour
                    anomaly.anomaly_type == existing.anomaly_type and
                    len(set(anomaly.affected_metrics) & set(existing.affected_metrics)) > 0):  # Overlapping metrics
                    
                    # Merge anomalies
                    if anomaly.score > existing.score:
                        existing.score = anomaly.score
                        existing.severity = anomaly.severity
                        existing.description += f" | {anomaly.description}"
                        
                    similar_found = True
                    break
                    
            if not similar_found:
                consolidated.append(anomaly)
                
        # Sort by severity and score
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        consolidated.sort(key=lambda x: (severity_order.get(x.severity, 0), x.score), reverse=True)
        
        return consolidated
        
    async def _calculate_baseline_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate baseline metrics for comparison."""
        
        baseline = {}
        
        try:
            if 'job_count' in data.columns:
                baseline['hiring'] = {
                    'daily_average': data['job_count'].mean(),
                    'daily_std': data['job_count'].std(),
                    'weekly_total': data['job_count'].sum(),
                    'trend': 'stable'  # Simplified
                }
                
            if 'initiative_count' in data.columns:
                baseline['initiatives'] = {
                    'daily_average': data['initiative_count'].mean(),
                    'total_detected': data['initiative_count'].sum()
                }
                
            baseline['analysis_period'] = {
                'start': data['date'].min().isoformat() if 'date' in data.columns else None,
                'end': data['date'].max().isoformat() if 'date' in data.columns else None,
                'days': len(data)
            }
            
            return baseline
            
        except Exception as e:
            logger.error(f"Error calculating baseline metrics: {str(e)}")
            return {}
            
    async def _interpret_anomalies(self, anomalies: List[Anomaly], data: pd.DataFrame) -> Dict[str, Any]:
        """Generate interpretations for detected anomalies."""
        
        interpretation = {
            'summary': f"Detected {len(anomalies)} anomalies",
            'severity_breakdown': {},
            'patterns': [],
            'recommendations': []
        }
        
        try:
            # Count by severity
            for anomaly in anomalies:
                severity = anomaly.severity
                interpretation['severity_breakdown'][severity] = interpretation['severity_breakdown'].get(severity, 0) + 1
                
            # Identify patterns
            anomaly_types = [a.anomaly_type for a in anomalies]
            type_counts = pd.Series(anomaly_types).value_counts()
            
            interpretation['patterns'] = [
                f"{count} {anomaly_type} anomalies detected"
                for anomaly_type, count in type_counts.items()
            ]
            
            # Generate recommendations
            if len(anomalies) > 5:
                interpretation['recommendations'].append("High number of anomalies detected - investigate systemic issues")
                
            if any(a.severity == 'critical' for a in anomalies):
                interpretation['recommendations'].append("Critical anomalies require immediate attention")
                
            interpretation['recommendations'].extend([
                "Review business context for anomalies",
                "Validate data quality and sources",
                "Consider market conditions and external factors"
            ])
            
            return interpretation
            
        except Exception as e:
            logger.error(f"Error interpreting anomalies: {str(e)}")
            return interpretation
            
    def _anomaly_to_dict(self, anomaly: Anomaly) -> Dict[str, Any]:
        """Convert Anomaly object to dictionary."""
        
        return {
            "timestamp": anomaly.timestamp,
            "anomaly_type": anomaly.anomaly_type,
            "severity": anomaly.severity,
            "score": anomaly.score,
            "description": anomaly.description,
            "affected_metrics": anomaly.affected_metrics,
            "context": anomaly.context,
            "recommendations": anomaly.recommendations
        }
        
    # Additional helper methods for specific anomaly types
    
    async def _prepare_hiring_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features specific to hiring anomaly detection."""
        # Implementation details...
        return pd.DataFrame()
        
    async def _detect_hiring_rate_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies in hiring rates."""
        # Implementation details...
        return []
        
    async def _detect_skill_demand_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies in skill demand patterns."""
        # Implementation details...
        return []
        
    async def _detect_geographic_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies in geographic hiring patterns."""
        # Implementation details...
        return []
        
    async def _detect_role_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies in role/department hiring."""
        # Implementation details...
        return []
        
    async def _prepare_technology_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for technology anomaly detection."""
        # Implementation details...
        return pd.DataFrame()
        
    async def _detect_adoption_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect sudden technology adoption changes."""
        # Implementation details...
        return []
        
    async def _detect_technology_combination_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect unusual technology combinations."""
        # Implementation details...
        return []
        
    async def _detect_emerging_technology_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect emerging technology signals."""
        # Implementation details...
        return []
        
    async def _analyze_technology_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze technology trends in the data."""
        # Implementation details...
        return {}
        
    async def _identify_emerging_technologies(self, data: pd.DataFrame) -> List[str]:
        """Identify emerging technologies from the data."""
        # Implementation details...
        return []
        
    async def _analyze_job_title_patterns(self, data: pd.DataFrame) -> List[Anomaly]:
        """Analyze job title patterns for anomalies."""
        # Implementation details...
        return []
        
    async def _analyze_initiative_patterns(self, data: pd.DataFrame) -> List[Anomaly]:
        """Analyze initiative patterns for anomalies."""
        # Implementation details...
        return []
        
    async def _analyze_technology_patterns(self, data: pd.DataFrame) -> List[Anomaly]:
        """Analyze technology patterns for anomalies."""
        # Implementation details...
        return []
        
    async def _detect_change_points(self, time_series: np.ndarray) -> List[int]:
        """Detect change points in time series using statistical methods."""
        
        try:
            # Simple change point detection using moving averages
            window_size = min(7, len(time_series) // 4)
            if window_size < 2:
                return []
                
            # Calculate moving averages
            ma_before = pd.Series(time_series).rolling(window=window_size, center=True).mean()
            ma_after = pd.Series(time_series).rolling(window=window_size, center=True).mean().shift(-window_size)
            
            # Calculate differences
            diff = np.abs(ma_after - ma_before)
            
            # Find significant changes (above threshold)
            threshold = diff.std() * 2
            change_points = np.where(diff > threshold)[0]
            
            return change_points.tolist()
            
        except Exception as e:
            logger.error(f"Error detecting change points: {str(e)}")
            return []
            
    async def _detect_seasonal_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies in seasonal patterns."""
        
        anomalies = []
        
        try:
            if 'date' not in data.columns or 'job_count' not in data.columns:
                return anomalies
                
            # Analyze day-of-week patterns
            data['dayofweek'] = pd.to_datetime(data['date']).dt.dayofweek
            daily_pattern = data.groupby('dayofweek')['job_count'].mean()
            
            # Find days that deviate significantly from pattern
            for _, row in data.iterrows():
                expected = daily_pattern[row['dayofweek']]
                actual = row['job_count']
                
                if abs(actual - expected) > 2 * daily_pattern.std():
                    anomaly = Anomaly(
                        timestamp=row['date'].isoformat(),
                        anomaly_type='seasonal',
                        severity='medium',
                        score=abs(actual - expected) / daily_pattern.std(),
                        description=f"Seasonal anomaly: {actual} jobs on {row['date'].strftime('%A')} vs expected {expected:.1f}",
                        affected_metrics=['job_count'],
                        context={
                            'method': 'seasonal_pattern',
                            'expected_value': float(expected),
                            'actual_value': float(actual),
                            'day_of_week': int(row['dayofweek'])
                        },
                        recommendations=[
                            "Check for special events or holidays",
                            "Verify seasonal business patterns",
                            "Investigate market conditions"
                        ]
                    )
                    anomalies.append(anomaly)
                    
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting seasonal anomalies: {str(e)}")
            return []