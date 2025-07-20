"""
Initiative prediction system for forecasting future company strategic initiatives.
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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import mlflow

from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class InitiativePrediction:
    """Initiative prediction data structure."""
    initiative_type: str
    probability: float
    confidence_interval: Tuple[float, float]
    expected_timeline: str
    strategic_rationale: List[str]
    supporting_evidence: List[str]
    risk_factors: List[str]

class InitiativePredictor:
    """Advanced system for predicting future company initiatives."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.initiative_types = []
        self.initialized = False
        
    async def initialize(self):
        """Initialize the initiative predictor."""
        try:
            # Initialize models for different initiative types
            self.models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    eval_metric='logloss'
                ),
                'logistic_regression': LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )
            }
            
            self.scalers = {
                'features': StandardScaler(),
                'text': StandardScaler()
            }
            
            # Common initiative types
            self.initiative_types = [
                'product_development',
                'market_expansion', 
                'technology_adoption',
                'talent_acquisition',
                'strategic_partnership',
                'operational_efficiency',
                'digital_transformation',
                'innovation_lab',
                'acquisition',
                'funding_round'
            ]
            
            # Initialize label encoders
            for init_type in self.initiative_types:
                self.label_encoders[init_type] = LabelEncoder()
                
            self.initialized = True
            logger.info("Initiative predictor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize initiative predictor: {str(e)}")
            
    async def cleanup(self):
        """Cleanup the initiative predictor."""
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.initiative_types = []
        self.initialized = False
        
    async def train(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train initiative prediction models."""
        
        if not self.initialized:
            await self.initialize()
            
        try:
            with mlflow.start_run(run_name="initiative_predictor_training"):
                logger.info("Starting initiative predictor training")
                
                # Prepare training data
                X, y = await self._prepare_training_data(training_data)
                
                if X.empty or len(y) == 0:
                    return {"error": "Insufficient training data"}
                
                # Train models
                model_results = {}
                
                for model_name, model in self.models.items():
                    try:
                        # Multi-label classification for multiple initiative types
                        multi_output_model = MultiOutputClassifier(model)
                        multi_output_model.fit(X, y)
                        
                        # Store trained model
                        self.models[model_name] = multi_output_model
                        
                        # Evaluate model
                        predictions = multi_output_model.predict(X)
                        accuracy = accuracy_score(y, predictions)
                        
                        model_results[model_name] = {
                            "accuracy": accuracy,
                            "status": "trained"
                        }
                        
                        logger.info(f"Trained {model_name} with accuracy: {accuracy:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error training {model_name}: {str(e)}")
                        model_results[model_name] = {"error": str(e)}
                
                # Log training metrics
                mlflow.log_param("training_samples", len(X))
                mlflow.log_param("initiative_types", len(self.initiative_types))
                
                best_model = max(
                    [(name, result) for name, result in model_results.items() 
                     if "accuracy" in result],
                    key=lambda x: x[1]["accuracy"],
                    default=(None, {})
                )
                
                if best_model[0]:
                    mlflow.log_metric("best_accuracy", best_model[1]["accuracy"])
                    mlflow.log_param("best_model", best_model[0])
                
                logger.info("Initiative predictor training completed")
                
                return {
                    "status": "success",
                    "models_trained": list(model_results.keys()),
                    "best_model": best_model[0],
                    "model_results": model_results
                }
                
        except Exception as e:
            logger.error(f"Error training initiative predictor: {str(e)}")
            return {"error": str(e)}
            
    async def predict_initiatives(
        self,
        company_id: str,
        features: pd.DataFrame,
        initiative_types: Optional[List[str]] = None,
        time_horizon: int = 180
    ) -> Dict[str, Any]:
        """Predict future initiatives for a company."""
        
        try:
            logger.info(f"Predicting initiatives for company {company_id}")
            
            # Prepare prediction features
            X = await self._prepare_prediction_features(company_id, features)
            
            if X.empty:
                return {"error": "Insufficient data for prediction"}
                
            # Filter initiative types if specified
            target_types = initiative_types or self.initiative_types
            
            # Get predictions from ensemble of models
            predictions = await self._get_ensemble_predictions(X, target_types)
            
            # Generate timeline predictions
            timeline = await self._predict_timeline(company_id, predictions, time_horizon)
            
            # Calculate probability scores
            probabilities = await self._calculate_probabilities(predictions)
            
            # Generate strategic themes
            themes = await self._identify_strategic_themes(predictions, features)
            
            # Estimate investment areas
            investments = await self._estimate_investment_areas(predictions, features)
            
            return {
                "initiatives": [self._prediction_to_dict(pred) for pred in predictions],
                "timeline": timeline,
                "probabilities": probabilities,
                "themes": themes,
                "investments": investments,
                "prediction_metadata": {
                    "model_confidence": await self._calculate_model_confidence(predictions),
                    "data_quality_score": await self._assess_data_quality(features),
                    "prediction_horizon_days": time_horizon
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting initiatives: {str(e)}")
            return {"error": str(e)}
            
    async def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data for initiative prediction."""
        
        try:
            # Create features from historical data
            features = await self._create_predictive_features(data)
            
            # Create target labels (multi-label)
            targets = await self._create_initiative_labels(data)
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.scalers['features'].fit_transform(features),
                columns=features.columns,
                index=features.index
            )
            
            return X_scaled, targets
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return pd.DataFrame(), np.array([])
            
    async def _create_predictive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features that predict future initiatives."""
        
        features = pd.DataFrame()
        
        try:
            # Company metrics
            if 'hiring_rate' in data.columns:
                features['hiring_rate_mean'] = data.groupby('company_id')['hiring_rate'].mean()
                features['hiring_rate_trend'] = data.groupby('company_id')['hiring_rate'].apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                
            # Technology signals
            if 'technology_mentions' in data.columns:
                features['tech_diversity'] = data.groupby('company_id')['technology_mentions'].nunique()
                features['emerging_tech_adoption'] = data.groupby('company_id').apply(
                    lambda x: self._count_emerging_technologies(x)
                )
                
            # Financial indicators
            if 'funding_amount' in data.columns:
                features['recent_funding'] = data.groupby('company_id')['funding_amount'].sum()
                features['funding_growth'] = data.groupby('company_id')['funding_amount'].apply(
                    lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1) if len(x) > 1 else 0
                )
                
            # Market presence
            if 'job_posting_count' in data.columns:
                features['market_activity'] = data.groupby('company_id')['job_posting_count'].sum()
                features['expansion_signal'] = data.groupby('company_id')['job_posting_count'].apply(
                    lambda x: 1 if x.iloc[-1] > x.mean() * 1.5 else 0
                )
                
            # Innovation metrics
            if 'patent_count' in data.columns:
                features['innovation_activity'] = data.groupby('company_id')['patent_count'].sum()
                
            # Competitive metrics
            if 'competitor_mentions' in data.columns:
                features['competitive_pressure'] = data.groupby('company_id')['competitor_mentions'].mean()
                
            # Sentiment and communication
            if 'sentiment_score' in data.columns:
                features['sentiment_trend'] = data.groupby('company_id')['sentiment_score'].apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                
            # Time-based features
            if 'date' in data.columns:
                data['quarter'] = pd.to_datetime(data['date']).dt.quarter
                features['recent_quarter'] = data.groupby('company_id')['quarter'].last()
                
            # Fill missing values
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating predictive features: {str(e)}")
            return pd.DataFrame()
            
    async def _create_initiative_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Create multi-label targets for initiative types."""
        
        try:
            if 'initiative_type' not in data.columns:
                # Create synthetic labels based on patterns
                return await self._create_synthetic_labels(data)
                
            # Create binary matrix for multi-label classification
            company_ids = data['company_id'].unique()
            labels = np.zeros((len(company_ids), len(self.initiative_types)))
            
            for i, company_id in enumerate(company_ids):
                company_data = data[data['company_id'] == company_id]
                
                for initiative in company_data['initiative_type'].unique():
                    if initiative in self.initiative_types:
                        j = self.initiative_types.index(initiative)
                        labels[i, j] = 1
                        
            return labels
            
        except Exception as e:
            logger.error(f"Error creating initiative labels: {str(e)}")
            return np.array([])
            
    async def _create_synthetic_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Create synthetic labels based on data patterns."""
        
        try:
            company_ids = data['company_id'].unique()
            labels = np.zeros((len(company_ids), len(self.initiative_types)))
            
            for i, company_id in enumerate(company_ids):
                company_data = data[data['company_id'] == company_id]
                
                # Infer initiative types from patterns
                if company_data['hiring_rate'].mean() > 0.1:
                    # High hiring suggests talent acquisition initiative
                    j = self.initiative_types.index('talent_acquisition')
                    labels[i, j] = 1
                    
                if company_data.get('technology_mentions', pd.Series()).nunique() > 5:
                    # High tech diversity suggests technology adoption
                    j = self.initiative_types.index('technology_adoption')
                    labels[i, j] = 1
                    
                if company_data.get('funding_amount', pd.Series()).sum() > 1000000:
                    # Recent funding suggests product development
                    j = self.initiative_types.index('product_development')
                    labels[i, j] = 1
                    
            return labels
            
        except Exception as e:
            logger.error(f"Error creating synthetic labels: {str(e)}")
            return np.array([])
            
    def _count_emerging_technologies(self, company_data: pd.DataFrame) -> int:
        """Count mentions of emerging technologies."""
        
        emerging_techs = [
            'artificial intelligence', 'machine learning', 'blockchain',
            'quantum computing', 'edge computing', 'ar/vr', 'iot',
            'kubernetes', 'serverless', 'microservices'
        ]
        
        count = 0
        if 'technology' in company_data.columns:
            text_data = ' '.join(company_data['technology'].astype(str)).lower()
            for tech in emerging_techs:
                if tech in text_data:
                    count += 1
                    
        return count
        
    async def _prepare_prediction_features(self, company_id: str, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        
        try:
            # Filter for specific company
            company_features = features[features['company_id'] == company_id]
            
            if company_features.empty:
                return pd.DataFrame()
                
            # Create same features as training
            pred_features = await self._create_predictive_features(company_features)
            
            # Scale features
            if not pred_features.empty:
                pred_features_scaled = pd.DataFrame(
                    self.scalers['features'].transform(pred_features),
                    columns=pred_features.columns,
                    index=pred_features.index
                )
                return pred_features_scaled
                
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {str(e)}")
            return pd.DataFrame()
            
    async def _get_ensemble_predictions(self, X: pd.DataFrame, target_types: List[str]) -> List[InitiativePrediction]:
        """Get predictions from ensemble of models."""
        
        predictions = []
        
        try:
            # Get predictions from all models
            model_predictions = {}
            
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    model_predictions[model_name] = proba
                else:
                    pred = model.predict(X)
                    model_predictions[model_name] = pred
                    
            # Ensemble predictions
            for i, init_type in enumerate(self.initiative_types):
                if init_type in target_types:
                    # Average probabilities across models
                    avg_proba = 0
                    count = 0
                    
                    for model_name, pred in model_predictions.items():
                        if len(pred.shape) > 1 and pred.shape[1] > i:
                            avg_proba += pred[0, i]
                            count += 1
                            
                    if count > 0:
                        avg_proba /= count
                        
                        if avg_proba > 0.3:  # Threshold for considering prediction
                            prediction = InitiativePrediction(
                                initiative_type=init_type,
                                probability=float(avg_proba),
                                confidence_interval=(
                                    max(0, avg_proba - 0.1),
                                    min(1, avg_proba + 0.1)
                                ),
                                expected_timeline=await self._estimate_timeline(init_type, avg_proba),
                                strategic_rationale=await self._generate_rationale(init_type, X),
                                supporting_evidence=await self._find_supporting_evidence(init_type, X),
                                risk_factors=await self._identify_risk_factors(init_type, X)
                            )
                            predictions.append(prediction)
                            
            # Sort by probability
            predictions.sort(key=lambda x: x.probability, reverse=True)
            
            return predictions[:5]  # Top 5 predictions
            
        except Exception as e:
            logger.error(f"Error getting ensemble predictions: {str(e)}")
            return []
            
    async def _predict_timeline(
        self,
        company_id: str,
        predictions: List[InitiativePrediction],
        time_horizon: int
    ) -> Dict[str, Any]:
        """Predict timeline for initiatives."""
        
        timeline = {}
        
        try:
            for pred in predictions:
                # Estimate timeline based on initiative type and probability
                if pred.probability > 0.8:
                    timeline_months = np.random.randint(1, 6)  # 1-6 months
                elif pred.probability > 0.6:
                    timeline_months = np.random.randint(3, 12)  # 3-12 months
                else:
                    timeline_months = np.random.randint(6, 18)  # 6-18 months
                    
                timeline[pred.initiative_type] = {
                    "estimated_months": timeline_months,
                    "confidence": pred.probability,
                    "earliest_start": (datetime.now() + timedelta(days=timeline_months*30//2)).isoformat(),
                    "most_likely": (datetime.now() + timedelta(days=timeline_months*30)).isoformat(),
                    "latest_start": (datetime.now() + timedelta(days=timeline_months*30*2)).isoformat()
                }
                
            return timeline
            
        except Exception as e:
            logger.error(f"Error predicting timeline: {str(e)}")
            return {}
            
    async def _calculate_probabilities(self, predictions: List[InitiativePrediction]) -> Dict[str, float]:
        """Calculate probability scores for predictions."""
        
        probabilities = {}
        
        for pred in predictions:
            probabilities[pred.initiative_type] = pred.probability
            
        return probabilities
        
    async def _identify_strategic_themes(
        self,
        predictions: List[InitiativePrediction],
        features: pd.DataFrame
    ) -> List[str]:
        """Identify strategic themes from predictions."""
        
        themes = []
        
        # Analyze prediction patterns
        tech_initiatives = [p for p in predictions if 'technology' in p.initiative_type]
        growth_initiatives = [p for p in predictions if p.initiative_type in ['market_expansion', 'product_development']]
        talent_initiatives = [p for p in predictions if 'talent' in p.initiative_type]
        
        if len(tech_initiatives) >= 2:
            themes.append("Technology Transformation")
            
        if len(growth_initiatives) >= 2:
            themes.append("Aggressive Growth Strategy")
            
        if len(talent_initiatives) >= 1:
            themes.append("Talent-Centric Approach")
            
        if not themes:
            themes.append("Operational Excellence")
            
        return themes
        
    async def _estimate_investment_areas(
        self,
        predictions: List[InitiativePrediction],
        features: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Estimate investment areas and amounts."""
        
        investments = []
        
        investment_mapping = {
            'product_development': {'area': 'R&D', 'relative_amount': 'high'},
            'technology_adoption': {'area': 'Technology Infrastructure', 'relative_amount': 'medium'},
            'talent_acquisition': {'area': 'Human Resources', 'relative_amount': 'medium'},
            'market_expansion': {'area': 'Sales & Marketing', 'relative_amount': 'high'},
            'strategic_partnership': {'area': 'Business Development', 'relative_amount': 'low'}
        }
        
        for pred in predictions:
            if pred.initiative_type in investment_mapping:
                investment = investment_mapping[pred.initiative_type].copy()
                investment['initiative'] = pred.initiative_type
                investment['probability'] = pred.probability
                investments.append(investment)
                
        return investments
        
    def _prediction_to_dict(self, prediction: InitiativePrediction) -> Dict[str, Any]:
        """Convert InitiativePrediction to dictionary."""
        
        return {
            "initiative_type": prediction.initiative_type,
            "probability": prediction.probability,
            "confidence_interval": prediction.confidence_interval,
            "expected_timeline": prediction.expected_timeline,
            "strategic_rationale": prediction.strategic_rationale,
            "supporting_evidence": prediction.supporting_evidence,
            "risk_factors": prediction.risk_factors
        }
        
    # Additional helper methods
    
    async def _estimate_timeline(self, init_type: str, probability: float) -> str:
        """Estimate timeline for initiative type."""
        
        timeline_mapping = {
            'product_development': '6-12 months',
            'market_expansion': '3-9 months',
            'technology_adoption': '2-6 months',
            'talent_acquisition': '1-3 months',
            'strategic_partnership': '3-12 months'
        }
        
        return timeline_mapping.get(init_type, '6-12 months')
        
    async def _generate_rationale(self, init_type: str, features: pd.DataFrame) -> List[str]:
        """Generate strategic rationale for initiative."""
        
        rationale_templates = {
            'product_development': [
                "Market opportunity identified",
                "Competitive positioning advantage",
                "Customer demand signals"
            ],
            'technology_adoption': [
                "Technology maturity reached",
                "Competitive necessity",
                "Operational efficiency gains"
            ],
            'talent_acquisition': [
                "Skills gap identified",
                "Growth scaling requirements",
                "Competitive talent market"
            ]
        }
        
        return rationale_templates.get(init_type, ["Strategic necessity"])
        
    async def _find_supporting_evidence(self, init_type: str, features: pd.DataFrame) -> List[str]:
        """Find supporting evidence for initiative prediction."""
        
        evidence = []
        
        # Analyze features for evidence
        if not features.empty:
            feature_values = features.iloc[0]
            
            if feature_values.get('hiring_rate_trend', 0) > 0:
                evidence.append("Increasing hiring rate indicates growth")
                
            if feature_values.get('tech_diversity', 0) > 5:
                evidence.append("High technology diversity suggests innovation focus")
                
            if feature_values.get('recent_funding', 0) > 0:
                evidence.append("Recent funding provides capital for initiatives")
                
        return evidence[:3]
        
    async def _identify_risk_factors(self, init_type: str, features: pd.DataFrame) -> List[str]:
        """Identify risk factors for initiative."""
        
        risk_factors = [
            "Market competition intensity",
            "Execution capability requirements",
            "Resource allocation challenges"
        ]
        
        return risk_factors[:2]
        
    async def _calculate_model_confidence(self, predictions: List[InitiativePrediction]) -> float:
        """Calculate overall model confidence."""
        
        if not predictions:
            return 0.0
            
        avg_probability = sum(p.probability for p in predictions) / len(predictions)
        return min(1.0, avg_probability * 1.2)  # Boost confidence slightly
        
    async def _assess_data_quality(self, features: pd.DataFrame) -> float:
        """Assess data quality for predictions."""
        
        if features.empty:
            return 0.0
            
        # Simple data quality score
        completeness = 1 - features.isnull().sum().sum() / (features.shape[0] * features.shape[1])
        return float(completeness)