"""
Pydantic models for prediction-related API endpoints.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator

class PredictionType(str, Enum):
    """Types of predictions available."""
    HIRING_TRENDS = "hiring_trends"
    INITIATIVES = "initiatives"
    MARKET_FORECAST = "market_forecast"
    ANOMALY_DETECTION = "anomaly_detection"
    COMPETITIVE_ANALYSIS = "competitive_analysis"

class InitiativeType(str, Enum):
    """Types of strategic initiatives."""
    PRODUCT_DEVELOPMENT = "product_development"
    MARKET_EXPANSION = "market_expansion"
    TECHNOLOGY_ADOPTION = "technology_adoption"
    TALENT_ACQUISITION = "talent_acquisition"
    STRATEGIC_PARTNERSHIP = "strategic_partnership"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    DIGITAL_TRANSFORMATION = "digital_transformation"
    INNOVATION_LAB = "innovation_lab"
    ACQUISITION = "acquisition"
    FUNDING_ROUND = "funding_round"

class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""
    STATISTICAL = "statistical"
    SEASONAL = "seasonal"
    TREND = "trend"
    PATTERN = "pattern"
    OUTLIER = "outlier"

class SeverityLevel(str, Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Base request models
class PredictionRequest(BaseModel):
    """Base request model for predictions."""
    company_id: Optional[str] = None
    time_horizon: int = Field(90, ge=7, le=365, description="Prediction horizon in days")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Confidence level")
    features: Optional[Dict[str, Any]] = None
    lookback_days: Optional[int] = Field(365, ge=30, le=1095, description="Historical data lookback period")
    initiative_types: Optional[List[InitiativeType]] = None
    
    @validator('time_horizon')
    def validate_time_horizon(cls, v):
        if not 7 <= v <= 365:
            raise ValueError('Time horizon must be between 7 and 365 days')
        return v

class BatchPredictionOptions(BaseModel):
    """Options for batch prediction processing."""
    priority: str = Field("normal", regex="^(low|normal|high)$")
    notification_webhook: Optional[str] = None
    export_format: str = Field("json", regex="^(json|csv|excel)$")
    include_metadata: bool = True

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    prediction_requests: List[PredictionRequest] = Field(..., min_items=1, max_items=50)
    batch_options: Optional[BatchPredictionOptions] = None

# Prediction result models
class ConfidenceInterval(BaseModel):
    """Confidence interval for predictions."""
    lower: float
    upper: float
    confidence_level: float = Field(ge=0.5, le=0.99)

class TimeSeriesPoint(BaseModel):
    """Single point in a time series prediction."""
    date: datetime
    value: float
    confidence_interval: ConfidenceInterval

class TrendAnalysis(BaseModel):
    """Trend analysis results."""
    direction: str = Field(regex="^(increasing|decreasing|stable)$")
    strength: float = Field(ge=0, le=1, description="Trend strength (0-1)")
    volatility: float = Field(ge=0, description="Volatility measure")
    seasonality_detected: bool
    change_points: List[datetime] = Field(default_factory=list)

class HiringTrendPrediction(BaseModel):
    """Hiring trend prediction response."""
    company_id: Optional[str] = None
    prediction_type: PredictionType = PredictionType.HIRING_TRENDS
    forecast_horizon_days: int
    
    # Forecast data
    forecast: List[TimeSeriesPoint]
    overall_trend: TrendAnalysis
    seasonal_patterns: Dict[str, Any] = Field(default_factory=dict)
    
    # Model information
    model_used: str
    model_confidence: float = Field(ge=0, le=1)
    accuracy_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    data_quality_score: float = Field(ge=0, le=1)
    
    class Config:
        schema_extra = {
            "example": {
                "company_id": "comp_123456789",
                "prediction_type": "hiring_trends",
                "forecast_horizon_days": 90,
                "forecast": [
                    {
                        "date": "2024-02-01T00:00:00Z",
                        "value": 25.5,
                        "confidence_interval": {
                            "lower": 20.2,
                            "upper": 30.8,
                            "confidence_level": 0.95
                        }
                    }
                ],
                "overall_trend": {
                    "direction": "increasing",
                    "strength": 0.75,
                    "volatility": 0.25,
                    "seasonality_detected": True,
                    "change_points": []
                },
                "model_used": "ensemble",
                "model_confidence": 0.87,
                "generated_at": "2024-01-15T10:30:00Z",
                "data_quality_score": 0.92
            }
        }

class InitiativePredictionItem(BaseModel):
    """Single initiative prediction."""
    initiative_type: InitiativeType
    probability: float = Field(ge=0, le=1)
    confidence_interval: ConfidenceInterval
    expected_timeline: str
    strategic_rationale: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)

class InitiativePrediction(BaseModel):
    """Initiative prediction response."""
    company_id: str
    prediction_type: PredictionType = PredictionType.INITIATIVES
    
    # Predictions
    predicted_initiatives: List[InitiativePredictionItem]
    timeline_forecast: Dict[str, Any] = Field(default_factory=dict)
    strategic_themes: List[str] = Field(default_factory=list)
    investment_areas: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Model information
    model_confidence: float = Field(ge=0, le=1)
    prediction_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    time_horizon_days: int
    
    class Config:
        schema_extra = {
            "example": {
                "company_id": "comp_123456789",
                "prediction_type": "initiatives",
                "predicted_initiatives": [
                    {
                        "initiative_type": "product_development",
                        "probability": 0.85,
                        "confidence_interval": {
                            "lower": 0.75,
                            "upper": 0.95,
                            "confidence_level": 0.95
                        },
                        "expected_timeline": "6-12 months",
                        "strategic_rationale": [
                            "Market opportunity identified",
                            "Competitive positioning advantage"
                        ],
                        "supporting_evidence": [
                            "Increasing hiring rate indicates growth",
                            "Recent funding provides capital for initiatives"
                        ],
                        "risk_factors": [
                            "Market competition intensity",
                            "Execution capability requirements"
                        ]
                    }
                ],
                "strategic_themes": ["Technology Transformation"],
                "model_confidence": 0.82,
                "generated_at": "2024-01-15T10:30:00Z",
                "time_horizon_days": 180
            }
        }

class MarketForecast(BaseModel):
    """Market-level forecast response."""
    industry: str
    geography: Optional[str] = None
    prediction_type: PredictionType = PredictionType.MARKET_FORECAST
    
    # Forecast data
    market_metrics: Dict[str, List[TimeSeriesPoint]] = Field(default_factory=dict)
    growth_projections: Dict[str, float] = Field(default_factory=dict)
    trend_analysis: TrendAnalysis
    
    # Market insights
    key_drivers: List[str] = Field(default_factory=list)
    market_opportunities: List[str] = Field(default_factory=list)
    competitive_landscape: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    forecast_horizon_days: int
    data_sources: List[str] = Field(default_factory=list)

class AnomalyItem(BaseModel):
    """Single anomaly detection result."""
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: SeverityLevel
    score: float = Field(ge=0, description="Anomaly score")
    description: str
    affected_metrics: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)

class AnomalyDetectionResult(BaseModel):
    """Anomaly detection response."""
    company_id: Optional[str] = None
    prediction_type: PredictionType = PredictionType.ANOMALY_DETECTION
    detection_type: str
    
    # Anomalies
    detected_anomalies: List[AnomalyItem]
    anomaly_summary: Dict[str, int] = Field(default_factory=dict)
    baseline_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Analysis
    interpretation: Dict[str, Any] = Field(default_factory=dict)
    patterns: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    analysis_period: Dict[str, datetime] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "company_id": "comp_123456789",
                "prediction_type": "anomaly_detection",
                "detection_type": "comprehensive",
                "detected_anomalies": [
                    {
                        "timestamp": "2024-01-10T14:30:00Z",
                        "anomaly_type": "statistical",
                        "severity": "medium",
                        "score": 2.5,
                        "description": "Statistical anomaly in hiring_rate",
                        "affected_metrics": ["hiring_rate"],
                        "context": {
                            "method": "z_score",
                            "baseline_mean": 0.15,
                            "observed_value": 0.45
                        },
                        "recommendations": [
                            "Investigate cause of unusual hiring_rate value",
                            "Check for data quality issues"
                        ]
                    }
                ],
                "anomaly_summary": {
                    "total": 3,
                    "critical": 0,
                    "high": 1,
                    "medium": 2,
                    "low": 0
                },
                "generated_at": "2024-01-15T10:30:00Z"
            }
        }

class CompetitivePosition(BaseModel):
    """Competitive position data."""
    company_id: str
    company_name: str
    position_coordinates: Dict[str, float] = Field(default_factory=dict)
    cluster_id: Optional[int] = None
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)
    market_share_estimate: Optional[float] = Field(None, ge=0, le=1)

class CompetitiveAnalysis(BaseModel):
    """Competitive analysis response."""
    prediction_type: PredictionType = PredictionType.COMPETITIVE_ANALYSIS
    analysis_type: str
    
    # Companies analyzed
    companies: List[str]
    competitive_positions: List[CompetitivePosition]
    
    # Analysis results
    positioning_map: Dict[str, Any] = Field(default_factory=dict)
    clusters: Dict[str, Any] = Field(default_factory=dict)
    market_dynamics: Dict[str, Any] = Field(default_factory=dict)
    recommendations: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    analysis_dimensions: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "prediction_type": "competitive_analysis",
                "analysis_type": "positioning",
                "companies": ["comp_123", "comp_456", "comp_789"],
                "competitive_positions": [
                    {
                        "company_id": "comp_123",
                        "company_name": "TechCorp Inc",
                        "position_coordinates": {"x": 0.75, "y": 0.45},
                        "cluster_id": 1,
                        "strengths": ["Innovation Capability", "Talent Quality"],
                        "weaknesses": ["Market Presence"],
                        "opportunities": ["Invest in Market Presence"],
                        "threats": ["Increasing competition in core markets"],
                        "market_share_estimate": 0.25
                    }
                ],
                "market_dynamics": {
                    "market_concentration": {
                        "hhi_index": 1500.0,
                        "interpretation": "Highly competitive market"
                    },
                    "competitive_intensity": {
                        "number_of_competitors": 3,
                        "intensity_score": 7.5
                    }
                },
                "generated_at": "2024-01-15T10:30:00Z"
            }
        }

# Batch processing models
class BatchTaskStatus(str, Enum):
    """Status of batch processing tasks."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BatchPredictionStatus(BaseModel):
    """Status of a batch prediction task."""
    task_id: str
    status: BatchTaskStatus
    progress_percentage: float = Field(ge=0, le=100)
    
    # Request info
    total_requests: int
    completed_requests: int
    failed_requests: int
    
    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Results
    results: Optional[List[Dict[str, Any]]] = None
    error_summary: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": "batch_abc123",
                "status": "processing",
                "progress_percentage": 45.0,
                "total_requests": 10,
                "completed_requests": 4,
                "failed_requests": 1,
                "created_at": "2024-01-15T10:00:00Z",
                "started_at": "2024-01-15T10:01:00Z",
                "estimated_completion": "2024-01-15T10:15:00Z"
            }
        }

# Model performance and monitoring
class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics."""
    model_name: str
    accuracy: Optional[float] = Field(None, ge=0, le=1)
    precision: Optional[float] = Field(None, ge=0, le=1)
    recall: Optional[float] = Field(None, ge=0, le=1)
    f1_score: Optional[float] = Field(None, ge=0, le=1)
    mae: Optional[float] = Field(None, ge=0)
    mse: Optional[float] = Field(None, ge=0)
    last_evaluation: Optional[datetime] = None

class ModelStatus(BaseModel):
    """Status of an ML model."""
    model_name: str
    initialized: bool
    trained: bool
    last_training: Optional[datetime] = None
    performance_metrics: Optional[ModelPerformanceMetrics] = None
    health_status: str = Field(regex="^(healthy|degraded|unhealthy)$")

class PredictionHistory(BaseModel):
    """User's prediction history entry."""
    prediction_id: str
    prediction_type: PredictionType
    company_id: Optional[str] = None
    created_at: datetime
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: str
    result_summary: Optional[Dict[str, Any]] = None

class PredictionHistoryResponse(BaseModel):
    """Response for prediction history."""
    predictions: List[PredictionHistory]
    total_count: int
    page: int
    page_size: int
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction_id": "pred_abc123",
                        "prediction_type": "hiring_trends",
                        "company_id": "comp_123456789",
                        "created_at": "2024-01-15T10:30:00Z",
                        "parameters": {
                            "time_horizon": 90,
                            "confidence_level": 0.95
                        },
                        "status": "completed",
                        "result_summary": {
                            "trend_direction": "increasing",
                            "model_confidence": 0.87
                        }
                    }
                ],
                "total_count": 25,
                "page": 1,
                "page_size": 20
            }
        }

# Error models
class PredictionError(BaseModel):
    """Error model for prediction operations."""
    error_code: str
    message: str
    prediction_type: Optional[PredictionType] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)