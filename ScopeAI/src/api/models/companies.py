"""
Pydantic models for company-related API endpoints.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, HttpUrl
from pydantic.types import constr

class CompanySize(str, Enum):
    """Company size categories."""
    STARTUP = "startup"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"

class Industry(str, Enum):
    """Industry categories."""
    TECHNOLOGY = "technology"
    FINANCIAL_SERVICES = "financial_services"
    HEALTHCARE = "healthcare"
    ECOMMERCE = "ecommerce"
    MEDIA = "media"
    GAMING = "gaming"
    TRANSPORTATION = "transportation"
    ENERGY = "energy"
    EDUCATION = "education"
    MANUFACTURING = "manufacturing"

class CompanyStatus(str, Enum):
    """Company status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ACQUIRED = "acquired"
    CLOSED = "closed"

# Base models
class CompanyBase(BaseModel):
    """Base company model with common fields."""
    name: constr(min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    website: Optional[HttpUrl] = None
    industry: Optional[Industry] = None
    size: Optional[CompanySize] = None
    employee_count: Optional[int] = Field(None, ge=0)
    founded_year: Optional[int] = Field(None, ge=1800, le=2030)
    headquarters: Optional[str] = Field(None, max_length=200)
    
    @validator('employee_count')
    def validate_employee_count(cls, v):
        if v is not None and v < 0:
            raise ValueError('Employee count must be non-negative')
        return v

class CompanyCreate(CompanyBase):
    """Model for creating a new company."""
    name: constr(min_length=1, max_length=200)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "TechCorp Inc",
                "description": "Leading technology company specializing in AI and machine learning",
                "website": "https://techcorp.com",
                "industry": "technology",
                "size": "large",
                "employee_count": 5000,
                "founded_year": 2010,
                "headquarters": "San Francisco, CA"
            }
        }

class CompanyUpdate(BaseModel):
    """Model for updating company information."""
    name: Optional[constr(min_length=1, max_length=200)] = None
    description: Optional[str] = Field(None, max_length=2000)
    website: Optional[HttpUrl] = None
    industry: Optional[Industry] = None
    size: Optional[CompanySize] = None
    employee_count: Optional[int] = Field(None, ge=0)
    founded_year: Optional[int] = Field(None, ge=1800, le=2030)
    headquarters: Optional[str] = Field(None, max_length=200)
    status: Optional[CompanyStatus] = None

# Request models
class CompanyCreateRequest(CompanyCreate):
    """Request model for creating a company."""
    tags: Optional[List[str]] = Field(None, max_items=10)
    funding_info: Optional[Dict[str, Any]] = None
    social_links: Optional[Dict[str, str]] = None

class CompanyUpdateRequest(CompanyUpdate):
    """Request model for updating a company."""
    tags: Optional[List[str]] = Field(None, max_items=10)
    funding_info: Optional[Dict[str, Any]] = None
    social_links: Optional[Dict[str, str]] = None

class CompanyFilterParams(BaseModel):
    """Parameters for filtering companies."""
    industry: Optional[Industry] = None
    location: Optional[str] = None
    size_min: Optional[int] = Field(None, ge=1)
    size_max: Optional[int] = Field(None, ge=1)
    founded_after: Optional[datetime] = None
    founded_before: Optional[datetime] = None
    search: Optional[str] = Field(None, min_length=2, max_length=100)
    sort_by: Optional[str] = Field("name", regex="^(name|size|founded_date|created_at)$")
    sort_order: Optional[str] = Field("asc", regex="^(asc|desc)$")

# Nested models for responses
class JobPosting(BaseModel):
    """Job posting summary for company responses."""
    job_id: str
    title: str
    location: Optional[str] = None
    posted_date: datetime
    is_active: bool
    department: Optional[str] = None
    seniority_level: Optional[str] = None

class Initiative(BaseModel):
    """Strategic initiative for company responses."""
    initiative_id: str
    type: str
    description: str
    confidence_score: float = Field(ge=0, le=1)
    detected_date: datetime
    evidence: List[str] = Field(default_factory=list)
    impact_assessment: Optional[str] = None

class CompanyMetrics(BaseModel):
    """Company metrics and analytics."""
    hiring_rate: Optional[float] = None
    growth_rate: Optional[float] = None
    technology_adoption_rate: Optional[float] = None
    innovation_score: Optional[float] = None
    market_presence_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    
    # Time-series data
    monthly_job_postings: Optional[Dict[str, int]] = None
    quarterly_growth: Optional[Dict[str, float]] = None
    technology_trends: Optional[Dict[str, List[str]]] = None

class CompanyAnalytics(BaseModel):
    """Comprehensive company analytics."""
    metrics: CompanyMetrics
    trends: Dict[str, Any] = Field(default_factory=dict)
    predictions: Dict[str, Any] = Field(default_factory=dict)
    benchmarks: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime

# Response models
class CompanyResponse(CompanyBase):
    """Response model for company data."""
    company_id: str
    status: CompanyStatus
    created_at: datetime
    updated_at: datetime
    
    # Optional enriched data
    recent_jobs: Optional[List[JobPosting]] = None
    detected_initiatives: Optional[List[Initiative]] = None
    analytics: Optional[CompanyAnalytics] = None
    
    # Metadata
    data_quality_score: Optional[float] = Field(None, ge=0, le=1)
    last_analyzed: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "company_id": "comp_123456789",
                "name": "TechCorp Inc",
                "description": "Leading technology company specializing in AI and machine learning",
                "website": "https://techcorp.com",
                "industry": "technology",
                "size": "large",
                "employee_count": 5000,
                "founded_year": 2010,
                "headquarters": "San Francisco, CA",
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "data_quality_score": 0.95,
                "tags": ["artificial-intelligence", "machine-learning", "enterprise"]
            }
        }

class CompanyListResponse(BaseModel):
    """Response model for paginated company lists."""
    companies: List[CompanyResponse]
    total_count: int
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total_pages: int
    has_next: bool
    has_previous: bool
    
    class Config:
        schema_extra = {
            "example": {
                "companies": [],
                "total_count": 150,
                "page": 1,
                "page_size": 20,
                "total_pages": 8,
                "has_next": True,
                "has_previous": False
            }
        }

class CompanyAnalyticsResponse(BaseModel):
    """Response model for company analytics."""
    company_id: str
    company_name: str
    analytics: CompanyAnalytics
    comparison_data: Optional[Dict[str, Any]] = None
    market_context: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "company_id": "comp_123456789",
                "company_name": "TechCorp Inc",
                "analytics": {
                    "metrics": {
                        "hiring_rate": 0.15,
                        "growth_rate": 0.25,
                        "innovation_score": 0.85
                    },
                    "trends": {
                        "hiring_trend": "increasing",
                        "technology_adoption": "high"
                    },
                    "last_updated": "2024-01-15T10:30:00Z"
                }
            }
        }

class CompanySearchResult(BaseModel):
    """Search result for company search."""
    company_id: str
    name: str
    description: Optional[str] = None
    industry: Optional[Industry] = None
    size: Optional[CompanySize] = None
    headquarters: Optional[str] = None
    relevance_score: float = Field(ge=0, le=1)
    
    class Config:
        schema_extra = {
            "example": {
                "company_id": "comp_123456789",
                "name": "TechCorp Inc",
                "description": "Leading technology company...",
                "industry": "technology",
                "size": "large",
                "headquarters": "San Francisco, CA",
                "relevance_score": 0.95
            }
        }

class CompanySearchResponse(BaseModel):
    """Response model for company search."""
    results: List[CompanySearchResult]
    total_count: int
    page: int
    page_size: int
    query: str
    search_time_ms: float
    
    class Config:
        schema_extra = {
            "example": {
                "results": [],
                "total_count": 25,
                "page": 1,
                "page_size": 20,
                "query": "artificial intelligence",
                "search_time_ms": 45.2
            }
        }

class TrendingCompany(BaseModel):
    """Trending company model."""
    company_id: str
    name: str
    industry: Optional[Industry] = None
    trending_metric: str
    metric_value: float
    change_percentage: float
    trend_direction: str  # "up", "down", "stable"
    
    class Config:
        schema_extra = {
            "example": {
                "company_id": "comp_123456789",
                "name": "TechCorp Inc",
                "industry": "technology",
                "trending_metric": "hiring_activity",
                "metric_value": 0.35,
                "change_percentage": 25.5,
                "trend_direction": "up"
            }
        }

class TrendingCompaniesResponse(BaseModel):
    """Response model for trending companies."""
    trending_companies: List[TrendingCompany]
    time_range: str
    metric: str
    generated_at: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "trending_companies": [],
                "time_range": "7d",
                "metric": "hiring_activity",
                "generated_at": "2024-01-15T10:30:00Z"
            }
        }

# Job-related models for company endpoints
class CompanyJobResponse(BaseModel):
    """Response model for company job listings."""
    company_id: str
    jobs: List[JobPosting]
    total_count: int
    page: int
    page_size: int
    filters_applied: Dict[str, Any] = Field(default_factory=dict)

class CompanyInitiativesResponse(BaseModel):
    """Response model for company initiatives."""
    company_id: str
    initiatives: List[Initiative]
    total_count: int
    time_range: str
    confidence_threshold: float
    analysis_date: datetime

# Error models
class CompanyError(BaseModel):
    """Error model for company operations."""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Validation helpers
def validate_company_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean company data."""
    # Remove empty strings and None values
    cleaned_data = {k: v for k, v in data.items() if v not in [None, "", []]}
    
    # Validate employee count matches size category
    if "employee_count" in cleaned_data and "size" in cleaned_data:
        employee_count = cleaned_data["employee_count"]
        size = cleaned_data["size"]
        
        size_ranges = {
            CompanySize.STARTUP: (1, 50),
            CompanySize.SMALL: (51, 200),
            CompanySize.MEDIUM: (201, 1000),
            CompanySize.LARGE: (1001, 5000),
            CompanySize.ENTERPRISE: (5001, float('inf'))
        }
        
        if size in size_ranges:
            min_size, max_size = size_ranges[size]
            if not (min_size <= employee_count <= max_size):
                # Auto-correct size based on employee count
                for size_cat, (min_emp, max_emp) in size_ranges.items():
                    if min_emp <= employee_count <= max_emp:
                        cleaned_data["size"] = size_cat
                        break
    
    return cleaned_data