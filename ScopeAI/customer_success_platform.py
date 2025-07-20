#!/usr/bin/env python3
"""
Customer Success Platform - User Success Optimization & Retention
Part of the ScopeAI Business Intelligence Ecosystem
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import random
from datetime import datetime, timedelta
import asyncio
import json
import os

app = FastAPI(
    title="Customer Success Platform",
    description="User success optimization, retention, and engagement analytics",
    version="1.0.0"
)

# Data Models
class CustomerHealth(BaseModel):
    customer_id: str
    company_name: str
    health_score: float
    health_status: str
    product_adoption: float
    user_engagement: float
    support_health: float
    contract_health: float
    growth_potential: float
    trend: str
    risk_factors: List[str]
    last_updated: datetime

class OnboardingWorkflow(BaseModel):
    workflow_id: str
    customer_id: str
    user_role: str
    current_step: int
    total_steps: int
    completion_percentage: float
    started_at: datetime
    expected_completion: datetime
    milestones: List[Dict]
    blockers: List[str]

class LearningPath(BaseModel):
    path_id: str
    title: str
    description: str
    difficulty_level: str
    modules: List[Dict]
    estimated_duration: str
    certification_available: bool
    prerequisites: List[str]

class UserProgress(BaseModel):
    user_id: str
    customer_id: str
    path_id: str
    progress_percentage: float
    modules_completed: int
    quiz_scores: List[float]
    time_spent: int
    last_activity: datetime
    certification_earned: bool

class InterventionCampaign(BaseModel):
    campaign_id: str
    name: str
    campaign_type: str
    target_criteria: Dict
    intervention_actions: List[str]
    success_metrics: List[str]
    active: bool
    customers_targeted: int
    success_rate: float

class EngagementMetrics(BaseModel):
    customer_id: str
    period: str
    active_users: int
    total_users: int
    session_duration_avg: float
    feature_usage: Dict[str, int]
    api_calls: int
    support_tickets: int
    nps_score: Optional[float]

# In-memory storage
customer_health_scores = []
onboarding_workflows = []
learning_paths = []
user_progress = []
intervention_campaigns = []
engagement_metrics = []

# Data generators
def generate_customer_health(customer_id: str = None, company_name: str = None):
    """Generate customer health score"""
    if not customer_id:
        customer_id = f"cust_{random.randint(1000, 9999)}"
    if not company_name:
        company_name = random.choice(["TechCorp", "InnovateCo", "StartupX", "ScaleUp", "EnterpriseY"])
    
    # Generate individual metrics
    product_adoption = round(random.uniform(40, 95), 1)
    user_engagement = round(random.uniform(30, 90), 1)
    support_health = round(random.uniform(60, 100), 1)
    contract_health = round(random.uniform(50, 100), 1)
    growth_potential = round(random.uniform(20, 85), 1)
    
    # Calculate overall health score (weighted average)
    health_score = round((
        product_adoption * 0.25 +
        user_engagement * 0.25 +
        support_health * 0.15 +
        contract_health * 0.20 +
        growth_potential * 0.15
    ), 1)
    
    # Determine health status
    if health_score >= 80:
        health_status = "Excellent"
    elif health_score >= 65:
        health_status = "Good"
    elif health_score >= 45:
        health_status = "At Risk"
    else:
        health_status = "Critical"
    
    # Generate risk factors based on low scores
    risk_factors = []
    if product_adoption < 60:
        risk_factors.append("Low product adoption")
    if user_engagement < 50:
        risk_factors.append("Poor user engagement")
    if support_health < 70:
        risk_factors.append("High support burden")
    if contract_health < 70:
        risk_factors.append("Contract issues")
    
    return CustomerHealth(
        customer_id=customer_id,
        company_name=company_name,
        health_score=health_score,
        health_status=health_status,
        product_adoption=product_adoption,
        user_engagement=user_engagement,
        support_health=support_health,
        contract_health=contract_health,
        growth_potential=growth_potential,
        trend=random.choice(["improving", "stable", "declining"]),
        risk_factors=risk_factors,
        last_updated=datetime.now()
    )

def generate_onboarding_workflow(customer_id: str, user_role: str):
    """Generate onboarding workflow based on user role"""
    role_workflows = {
        "admin": {
            "steps": [
                {"step": 1, "title": "Account Setup", "status": "completed"},
                {"step": 2, "title": "Team Invitation", "status": "completed"},
                {"step": 3, "title": "Initial Configuration", "status": "in_progress"},
                {"step": 4, "title": "Integration Setup", "status": "pending"},
                {"step": 5, "title": "First Report Creation", "status": "pending"}
            ],
            "total": 5
        },
        "user": {
            "steps": [
                {"step": 1, "title": "Profile Setup", "status": "completed"},
                {"step": 2, "title": "Feature Tour", "status": "completed"},
                {"step": 3, "title": "First Task", "status": "in_progress"},
                {"step": 4, "title": "Collaboration Setup", "status": "pending"}
            ],
            "total": 4
        },
        "trial": {
            "steps": [
                {"step": 1, "title": "Quick Start", "status": "completed"},
                {"step": 2, "title": "Demo Data", "status": "in_progress"},
                {"step": 3, "title": "Key Features", "status": "pending"}
            ],
            "total": 3
        }
    }
    
    workflow = role_workflows.get(user_role, role_workflows["user"])
    completed_steps = len([s for s in workflow["steps"] if s["status"] == "completed"])
    current_step = completed_steps + 1
    
    return OnboardingWorkflow(
        workflow_id=f"onboard_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        customer_id=customer_id,
        user_role=user_role,
        current_step=current_step,
        total_steps=workflow["total"],
        completion_percentage=round((completed_steps / workflow["total"]) * 100, 1),
        started_at=datetime.now() - timedelta(days=random.randint(1, 14)),
        expected_completion=datetime.now() + timedelta(days=random.randint(3, 21)),
        milestones=workflow["steps"],
        blockers=random.sample(["Missing integration", "Approval pending", "Technical issue"], k=random.randint(0, 2))
    )

def generate_learning_paths():
    """Generate available learning paths"""
    paths = [
        LearningPath(
            path_id="path_beginner",
            title="Platform Fundamentals",
            description="Essential skills for new users",
            difficulty_level="Beginner",
            modules=[
                {"module": 1, "title": "Getting Started", "duration": "30 min"},
                {"module": 2, "title": "Basic Navigation", "duration": "45 min"},
                {"module": 3, "title": "Core Features", "duration": "60 min"},
                {"module": 4, "title": "Best Practices", "duration": "45 min"}
            ],
            estimated_duration="3 hours",
            certification_available=True,
            prerequisites=[]
        ),
        LearningPath(
            path_id="path_advanced",
            title="Advanced Analytics",
            description="Deep dive into analytics and reporting",
            difficulty_level="Advanced",
            modules=[
                {"module": 1, "title": "Advanced Queries", "duration": "90 min"},
                {"module": 2, "title": "Custom Dashboards", "duration": "120 min"},
                {"module": 3, "title": "API Integration", "duration": "150 min"},
                {"module": 4, "title": "Automation Setup", "duration": "90 min"}
            ],
            estimated_duration="7.5 hours",
            certification_available=True,
            prerequisites=["path_beginner"]
        ),
        LearningPath(
            path_id="path_admin",
            title="Administration Mastery",
            description="Complete admin training program",
            difficulty_level="Expert",
            modules=[
                {"module": 1, "title": "User Management", "duration": "60 min"},
                {"module": 2, "title": "Security Configuration", "duration": "90 min"},
                {"module": 3, "title": "Advanced Integrations", "duration": "120 min"},
                {"module": 4, "title": "Performance Optimization", "duration": "90 min"},
                {"module": 5, "title": "Troubleshooting", "duration": "75 min"}
            ],
            estimated_duration="8.5 hours",
            certification_available=True,
            prerequisites=["path_beginner", "path_advanced"]
        )
    ]
    
    return paths

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Customer Success Platform - Your user success optimization engine"}

@app.get("/api/v1/health-scores")
async def get_customer_health_scores(status: Optional[str] = None):
    """Get customer health scores"""
    scores = customer_health_scores
    if status:
        scores = [s for s in scores if s["health_status"] == status]
    return scores

@app.get("/api/v1/health-scores/{customer_id}")
async def get_customer_health(customer_id: str):
    """Get specific customer health score"""
    for score in customer_health_scores:
        if score["customer_id"] == customer_id:
            return score
    raise HTTPException(status_code=404, detail="Customer not found")

@app.post("/api/v1/health-scores/calculate/{customer_id}")
async def calculate_health_score(customer_id: str, company_name: Optional[str] = None):
    """Calculate/update customer health score"""
    health = generate_customer_health(customer_id, company_name)
    
    # Update existing or add new
    for i, existing in enumerate(customer_health_scores):
        if existing["customer_id"] == customer_id:
            customer_health_scores[i] = health.dict()
            return health
    
    customer_health_scores.append(health.dict())
    return health

@app.get("/api/v1/onboarding/workflows")
async def get_onboarding_workflows(customer_id: Optional[str] = None):
    """Get onboarding workflows"""
    workflows = onboarding_workflows
    if customer_id:
        workflows = [w for w in workflows if w["customer_id"] == customer_id]
    return workflows

@app.post("/api/v1/onboarding/create")
async def create_onboarding_workflow(customer_id: str, user_role: str):
    """Create new onboarding workflow"""
    workflow = generate_onboarding_workflow(customer_id, user_role)
    onboarding_workflows.append(workflow.dict())
    return workflow

@app.put("/api/v1/onboarding/update-step/{workflow_id}")
async def update_onboarding_step(workflow_id: str, step_number: int, status: str):
    """Update onboarding workflow step"""
    for workflow in onboarding_workflows:
        if workflow["workflow_id"] == workflow_id:
            for milestone in workflow["milestones"]:
                if milestone["step"] == step_number:
                    milestone["status"] = status
                    
            # Recalculate completion percentage
            completed = len([m for m in workflow["milestones"] if m["status"] == "completed"])
            workflow["completion_percentage"] = round((completed / workflow["total_steps"]) * 100, 1)
            
            return {"status": "updated", "workflow": workflow}
    
    raise HTTPException(status_code=404, detail="Workflow not found")

@app.get("/api/v1/learning-paths")
async def get_learning_paths():
    """Get available learning paths"""
    return learning_paths

@app.get("/api/v1/learning-paths/{path_id}")
async def get_learning_path(path_id: str):
    """Get specific learning path"""
    for path in learning_paths:
        if path["path_id"] == path_id:
            return path
    raise HTTPException(status_code=404, detail="Learning path not found")

@app.get("/api/v1/user-progress/{user_id}")
async def get_user_progress(user_id: str):
    """Get user learning progress"""
    progress = [p for p in user_progress if p["user_id"] == user_id]
    return progress

@app.post("/api/v1/user-progress/update")
async def update_user_progress(
    user_id: str,
    customer_id: str,
    path_id: str,
    modules_completed: int,
    quiz_score: float
):
    """Update user learning progress"""
    # Find existing progress or create new
    for progress in user_progress:
        if progress["user_id"] == user_id and progress["path_id"] == path_id:
            progress["modules_completed"] = modules_completed
            progress["quiz_scores"].append(quiz_score)
            progress["last_activity"] = datetime.now().isoformat()
            # Calculate progress percentage
            total_modules = len([p for p in learning_paths if p["path_id"] == path_id][0]["modules"])
            progress["progress_percentage"] = round((modules_completed / total_modules) * 100, 1)
            return progress
    
    # Create new progress record
    total_modules = 4  # Default
    for path in learning_paths:
        if path["path_id"] == path_id:
            total_modules = len(path["modules"])
            break
    
    new_progress = UserProgress(
        user_id=user_id,
        customer_id=customer_id,
        path_id=path_id,
        progress_percentage=round((modules_completed / total_modules) * 100, 1),
        modules_completed=modules_completed,
        quiz_scores=[quiz_score],
        time_spent=random.randint(30, 180),
        last_activity=datetime.now(),
        certification_earned=modules_completed >= total_modules and quiz_score >= 80.0
    )
    
    user_progress.append(new_progress.dict())
    return new_progress

@app.get("/api/v1/interventions/campaigns")
async def get_intervention_campaigns():
    """Get active intervention campaigns"""
    return intervention_campaigns

@app.post("/api/v1/interventions/create")
async def create_intervention_campaign(
    name: str,
    campaign_type: str,
    target_health_threshold: float = 65.0
):
    """Create intervention campaign"""
    campaign = InterventionCampaign(
        campaign_id=f"campaign_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        name=name,
        campaign_type=campaign_type,
        target_criteria={
            "health_score_below": target_health_threshold,
            "last_login_days": 7,
            "support_tickets": 3
        },
        intervention_actions=[
            "Send personalized email",
            "Schedule CSM call",
            "Assign training resources",
            "Executive outreach"
        ],
        success_metrics=[
            "Health score improvement",
            "Engagement increase",
            "Support ticket reduction"
        ],
        active=True,
        customers_targeted=0,
        success_rate=0.0
    )
    
    intervention_campaigns.append(campaign.dict())
    return campaign

@app.get("/api/v1/engagement/metrics")
async def get_engagement_metrics(customer_id: Optional[str] = None, period: str = "week"):
    """Get customer engagement metrics"""
    metrics = engagement_metrics
    if customer_id:
        metrics = [m for m in metrics if m["customer_id"] == customer_id]
    if period:
        metrics = [m for m in metrics if m["period"] == period]
    return metrics

@app.get("/api/v1/retention/analysis")
async def get_retention_analysis():
    """Get retention and churn analysis"""
    analysis = []
    
    for health in customer_health_scores:
        churn_probability = max(0, (100 - health["health_score"]) / 100)
        
        analysis.append({
            "customer_id": health["customer_id"],
            "company_name": health["company_name"],
            "churn_probability": round(churn_probability, 2),
            "risk_level": "High" if churn_probability > 0.6 else "Medium" if churn_probability > 0.3 else "Low",
            "lifetime_value": random.randint(10000, 100000),
            "contract_end_date": (datetime.now() + timedelta(days=random.randint(30, 365))).date().isoformat(),
            "expansion_potential": health["growth_potential"],
            "recommended_actions": [
                "Schedule executive review" if churn_probability > 0.6 else "Increase engagement",
                "Provide training resources",
                "Review contract terms"
            ]
        })
    
    return analysis

@app.get("/api/v1/dashboard/overview")
async def get_dashboard_overview():
    """Get customer success dashboard overview"""
    total_customers = len(customer_health_scores)
    healthy_customers = len([c for c in customer_health_scores if c["health_score"] >= 70])
    at_risk_customers = len([c for c in customer_health_scores if c["health_score"] < 50])
    
    return {
        "summary": {
            "total_customers": total_customers,
            "healthy_customers": healthy_customers,
            "at_risk_customers": at_risk_customers,
            "health_score_average": round(sum(c["health_score"] for c in customer_health_scores) / max(total_customers, 1), 1)
        },
        "onboarding": {
            "active_workflows": len(onboarding_workflows),
            "completion_rate": round(sum(w["completion_percentage"] for w in onboarding_workflows) / max(len(onboarding_workflows), 1), 1),
            "average_time_to_complete": "12 days"
        },
        "learning": {
            "active_learners": len(user_progress),
            "completion_rate": round(sum(p["progress_percentage"] for p in user_progress) / max(len(user_progress), 1), 1),
            "certifications_earned": len([p for p in user_progress if p.get("certification_earned")])
        },
        "interventions": {
            "active_campaigns": len([c for c in intervention_campaigns if c["active"]]),
            "customers_reached": sum(c["customers_targeted"] for c in intervention_campaigns),
            "average_success_rate": round(sum(c["success_rate"] for c in intervention_campaigns) / max(len(intervention_campaigns), 1), 1)
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Customer Success Platform", "timestamp": datetime.now().isoformat()}

# Serve dashboard
@app.get("/customer_success_dashboard.html", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the Customer Success dashboard"""
    dashboard_path = "customer_success_dashboard.html"
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Customer Success Dashboard</h1><p>Dashboard file not found.</p>")

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    # Generate sample customers and health scores
    companies = ["TechCorp Inc", "InnovateCo", "StartupX", "ScaleUp Solutions", "EnterpriseY", "FutureScale", "DisruptX", "MegaTech", "CloudFirst", "DataDriven"]
    
    for i, company in enumerate(companies):
        customer_id = f"cust_{1000 + i}"
        health = generate_customer_health(customer_id, company)
        customer_health_scores.append(health.dict())
        
        # Create sample onboarding workflows
        for role in ["admin", "user"]:
            workflow = generate_onboarding_workflow(customer_id, role)
            onboarding_workflows.append(workflow.dict())
        
        # Create sample engagement metrics
        metrics = EngagementMetrics(
            customer_id=customer_id,
            period="week",
            active_users=random.randint(5, 50),
            total_users=random.randint(10, 100),
            session_duration_avg=round(random.uniform(15, 45), 1),
            feature_usage={
                "reports": random.randint(10, 100),
                "dashboards": random.randint(5, 50),
                "integrations": random.randint(1, 20),
                "api_calls": random.randint(100, 1000)
            },
            api_calls=random.randint(500, 5000),
            support_tickets=random.randint(0, 10),
            nps_score=round(random.uniform(6, 10), 1)
        )
        engagement_metrics.append(metrics.dict())
    
    # Initialize learning paths
    learning_paths.extend([path.dict() for path in generate_learning_paths()])
    
    # Create sample intervention campaigns
    campaigns = [
        ("Health Recovery Campaign", "email"),
        ("Onboarding Boost", "training"),
        ("Executive Outreach", "call")
    ]
    
    for name, camp_type in campaigns:
        campaign = await create_intervention_campaign(name, camp_type, random.uniform(50, 70))
        # Update with sample data
        for camp in intervention_campaigns:
            if camp["name"] == name:
                camp["customers_targeted"] = random.randint(5, 25)
                camp["success_rate"] = round(random.uniform(60, 85), 1)
    
    print("🎯 Customer Success Platform started successfully!")
    print("📍 API available at: http://localhost:8018")
    print("📊 Dashboard available at: http://localhost:8018/customer_success_dashboard.html")
    print("📚 API docs available at: http://localhost:8018/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8018)