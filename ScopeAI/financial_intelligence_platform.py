#!/usr/bin/env python3
"""
Financial Intelligence Platform - Investment Analytics & Financial Intelligence
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
import math

app = FastAPI(
    title="Financial Intelligence Platform",
    description="Investment analytics, company tracking, and financial intelligence",
    version="1.0.0"
)

# Data Models
class CompanyFinancials(BaseModel):
    company_id: str
    company_name: str
    ticker_symbol: Optional[str]
    industry: str
    market_cap: float
    revenue_annual: float
    revenue_growth: float
    profit_margin: float
    debt_to_equity: float
    cash_position: float
    employees: int
    founded_year: int
    last_funding_round: Optional[Dict]
    valuation_multiple: float
    financial_health_score: float

class InvestmentTracking(BaseModel):
    investment_id: str
    company_id: str
    investment_type: str
    amount_invested: float
    investment_date: datetime
    current_valuation: float
    roi_percentage: float
    risk_rating: str
    investment_stage: str
    exit_strategy: str
    projected_exit_date: Optional[datetime]

class ValuationModel(BaseModel):
    model_id: str
    company_id: str
    model_type: str
    valuation_amount: float
    confidence_level: float
    key_assumptions: Dict
    comparable_companies: List[str]
    calculated_date: datetime
    analyst_notes: str

class FinancialAlert(BaseModel):
    alert_id: str
    alert_type: str
    company_id: str
    severity: str
    message: str
    financial_indicator: str
    threshold_breached: float
    current_value: float
    impact_assessment: str
    created_at: datetime

class MarketAnalysis(BaseModel):
    analysis_id: str
    market_segment: str
    analysis_type: str
    key_metrics: Dict
    growth_projections: Dict
    risk_factors: List[str]
    opportunities: List[str]
    analyst_rating: str
    confidence_score: float

# In-memory storage
company_financials = []
investment_tracking = []
valuation_models = []
financial_alerts = []
market_analyses = []

# Data generators
def generate_company_financials():
    """Generate company financial data"""
    companies = [
        ("TechVenture", "TV"),
        ("InnovateCorp", "INC"),
        ("ScaleUp Solutions", "SUS"),
        ("DataDriven Inc", "DDI"),
        ("CloudFirst", "CF"),
        ("AITech Systems", "ATS"),
        ("NextGen Platform", "NGP"),
        ("DigitalTransform", "DT")
    ]
    
    company_name, ticker = random.choice(companies)
    revenue = random.uniform(10e6, 500e6)  # 10M to 500M
    
    # Calculate derived metrics
    growth_rate = random.uniform(-10, 50)
    profit_margin = random.uniform(-5, 25)
    market_cap = revenue * random.uniform(3, 15)
    
    # Funding round data
    funding_round = None
    if random.choice([True, False]):
        funding_round = {
            "round_type": random.choice(["Series A", "Series B", "Series C", "Series D"]),
            "amount": random.uniform(5e6, 100e6),
            "date": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
            "lead_investor": random.choice(["Sequoia", "A16Z", "Kleiner Perkins", "GV", "Insight Partners"])
        }
    
    # Calculate financial health score
    health_factors = []
    health_factors.append(min(growth_rate / 30 * 25, 25))  # Growth component (max 25)
    health_factors.append(max(profit_margin / 20 * 20, 0))  # Profitability (max 20)
    health_factors.append(25 if revenue > 50e6 else revenue / 50e6 * 25)  # Scale (max 25)
    health_factors.append(random.uniform(15, 30))  # Market position (15-30)
    
    health_score = sum(health_factors)
    
    return CompanyFinancials(
        company_id=f"comp_{random.randint(10000, 99999)}",
        company_name=company_name,
        ticker_symbol=ticker if random.choice([True, False]) else None,
        industry=random.choice(["SaaS", "FinTech", "HealthTech", "EdTech", "E-commerce", "AI/ML"]),
        market_cap=market_cap,
        revenue_annual=revenue,
        revenue_growth=growth_rate,
        profit_margin=profit_margin,
        debt_to_equity=random.uniform(0.1, 2.5),
        cash_position=revenue * random.uniform(0.2, 1.5),
        employees=int(revenue / random.uniform(100000, 300000)),
        founded_year=random.randint(2010, 2020),
        last_funding_round=funding_round,
        valuation_multiple=random.uniform(3, 20),
        financial_health_score=round(health_score, 1)
    )

def generate_investment_tracking(company_id: str):
    """Generate investment tracking record"""
    investment_amount = random.uniform(100000, 10000000)
    investment_date = datetime.now() - timedelta(days=random.randint(30, 1095))
    time_held = (datetime.now() - investment_date).days
    
    # Calculate current valuation with some volatility
    growth_factor = random.uniform(0.8, 3.5)
    current_val = investment_amount * growth_factor
    roi = ((current_val - investment_amount) / investment_amount) * 100
    
    return InvestmentTracking(
        investment_id=f"inv_{random.randint(10000, 99999)}",
        company_id=company_id,
        investment_type=random.choice(["Equity", "Convertible Note", "SAFE", "Direct Investment"]),
        amount_invested=investment_amount,
        investment_date=investment_date,
        current_valuation=current_val,
        roi_percentage=round(roi, 1),
        risk_rating=random.choice(["Low", "Medium", "High", "Very High"]),
        investment_stage=random.choice(["Seed", "Series A", "Series B", "Series C", "Growth"]),
        exit_strategy=random.choice(["IPO", "Acquisition", "Secondary Sale", "Hold"]),
        projected_exit_date=datetime.now() + timedelta(days=random.randint(365, 1825)) if random.choice([True, False]) else None
    )

def generate_valuation_model(company_id: str):
    """Generate valuation model"""
    model_types = {
        "DCF": {
            "assumptions": {
                "discount_rate": round(random.uniform(8, 15), 1),
                "terminal_growth_rate": round(random.uniform(2, 4), 1),
                "projection_years": 5
            }
        },
        "Comparable Companies": {
            "assumptions": {
                "revenue_multiple": round(random.uniform(3, 12), 1),
                "ebitda_multiple": round(random.uniform(10, 25), 1),
                "peer_group_size": random.randint(5, 15)
            }
        },
        "Risk-Adjusted NPV": {
            "assumptions": {
                "success_probability": round(random.uniform(60, 90), 1),
                "market_size": f"${random.randint(1, 50)}B",
                "market_penetration": round(random.uniform(1, 10), 1)
            }
        }
    }
    
    model_type = random.choice(list(model_types.keys()))
    base_valuation = random.uniform(50e6, 2e9)
    
    return ValuationModel(
        model_id=f"val_{random.randint(10000, 99999)}",
        company_id=company_id,
        model_type=model_type,
        valuation_amount=base_valuation,
        confidence_level=round(random.uniform(65, 90), 1),
        key_assumptions=model_types[model_type]["assumptions"],
        comparable_companies=random.sample([
            "Salesforce", "Zoom", "Slack", "Dropbox", "Atlassian", "ServiceNow"
        ], k=random.randint(3, 5)),
        calculated_date=datetime.now(),
        analyst_notes=f"Valuation based on {model_type} methodology with strong growth assumptions"
    )

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Financial Intelligence Platform - Your investment analytics engine"}

@app.get("/api/v1/companies/financials")
async def get_company_financials(industry: Optional[str] = None, min_revenue: Optional[float] = None):
    """Get company financial data"""
    financials = company_financials
    
    if industry:
        financials = [f for f in financials if f["industry"] == industry]
    if min_revenue:
        financials = [f for f in financials if f["revenue_annual"] >= min_revenue]
    
    return financials

@app.get("/api/v1/companies/{company_id}/financials")
async def get_company_financial_details(company_id: str):
    """Get detailed financial information for specific company"""
    for financial in company_financials:
        if financial["company_id"] == company_id:
            return financial
    raise HTTPException(status_code=404, detail="Company not found")

@app.post("/api/v1/companies/track")
async def add_company_tracking():
    """Add new company to financial tracking"""
    financial = generate_company_financials()
    company_financials.append(financial.dict())
    return financial

@app.get("/api/v1/investments/portfolio")
async def get_investment_portfolio():
    """Get investment portfolio overview"""
    total_invested = sum(inv["amount_invested"] for inv in investment_tracking)
    current_value = sum(inv["current_valuation"] for inv in investment_tracking)
    total_roi = ((current_value - total_invested) / total_invested * 100) if total_invested > 0 else 0
    
    portfolio = {
        "summary": {
            "total_invested": round(total_invested, 2),
            "current_value": round(current_value, 2),
            "total_roi": round(total_roi, 1),
            "number_of_investments": len(investment_tracking),
            "unrealized_gains": round(current_value - total_invested, 2)
        },
        "investments": investment_tracking,
        "performance_by_stage": {},
        "performance_by_industry": {}
    }
    
    # Calculate performance by stage
    stages = {}
    for inv in investment_tracking:
        stage = inv["investment_stage"]
        if stage not in stages:
            stages[stage] = {"count": 0, "invested": 0, "current_value": 0}
        stages[stage]["count"] += 1
        stages[stage]["invested"] += inv["amount_invested"]
        stages[stage]["current_value"] += inv["current_valuation"]
    
    for stage, data in stages.items():
        roi = ((data["current_value"] - data["invested"]) / data["invested"] * 100) if data["invested"] > 0 else 0
        portfolio["performance_by_stage"][stage] = {
            "count": data["count"],
            "total_invested": round(data["invested"], 2),
            "current_value": round(data["current_value"], 2),
            "roi": round(roi, 1)
        }
    
    return portfolio

@app.post("/api/v1/investments/add")
async def add_investment(company_id: str, amount: float, investment_type: str):
    """Add new investment tracking"""
    # Verify company exists
    company_exists = any(c["company_id"] == company_id for c in company_financials)
    if not company_exists:
        raise HTTPException(status_code=404, detail="Company not found")
    
    investment = InvestmentTracking(
        investment_id=f"inv_{random.randint(10000, 99999)}",
        company_id=company_id,
        investment_type=investment_type,
        amount_invested=amount,
        investment_date=datetime.now(),
        current_valuation=amount,  # Initial valuation equals investment
        roi_percentage=0.0,
        risk_rating="Medium",
        investment_stage="Series A",
        exit_strategy="TBD",
        projected_exit_date=None
    )
    
    investment_tracking.append(investment.dict())
    return investment

@app.get("/api/v1/valuations/models")
async def get_valuation_models(company_id: Optional[str] = None):
    """Get valuation models"""
    models = valuation_models
    if company_id:
        models = [m for m in models if m["company_id"] == company_id]
    return models

@app.post("/api/v1/valuations/calculate/{company_id}")
async def calculate_valuation(company_id: str, model_type: str):
    """Calculate company valuation using specified model"""
    # Verify company exists
    company_exists = any(c["company_id"] == company_id for c in company_financials)
    if not company_exists:
        raise HTTPException(status_code=404, detail="Company not found")
    
    model = generate_valuation_model(company_id)
    model.model_type = model_type
    valuation_models.append(model.dict())
    return model

@app.get("/api/v1/alerts/financial")
async def get_financial_alerts(severity: Optional[str] = None):
    """Get financial alerts"""
    alerts = financial_alerts
    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]
    return alerts

@app.get("/api/v1/analysis/market")
async def get_market_analysis(segment: Optional[str] = None):
    """Get market analysis reports"""
    analyses = market_analyses
    if segment:
        analyses = [a for a in analyses if a["market_segment"] == segment]
    return analyses

@app.get("/api/v1/metrics/dashboard")
async def get_financial_dashboard():
    """Get financial intelligence dashboard metrics"""
    # Calculate portfolio metrics
    total_companies = len(company_financials)
    total_investments = len(investment_tracking)
    total_invested = sum(inv["amount_invested"] for inv in investment_tracking)
    current_value = sum(inv["current_valuation"] for inv in investment_tracking)
    
    # Top performers
    top_performers = sorted(investment_tracking, key=lambda x: x["roi_percentage"], reverse=True)[:5]
    
    # Industry breakdown
    industry_breakdown = {}
    for company in company_financials:
        industry = company["industry"]
        if industry not in industry_breakdown:
            industry_breakdown[industry] = {"count": 0, "total_revenue": 0}
        industry_breakdown[industry]["count"] += 1
        industry_breakdown[industry]["total_revenue"] += company["revenue_annual"]
    
    # Financial health distribution
    health_distribution = {
        "Excellent (80+)": len([c for c in company_financials if c["financial_health_score"] >= 80]),
        "Good (60-79)": len([c for c in company_financials if 60 <= c["financial_health_score"] < 80]),
        "Fair (40-59)": len([c for c in company_financials if 40 <= c["financial_health_score"] < 60]),
        "Poor (<40)": len([c for c in company_financials if c["financial_health_score"] < 40])
    }
    
    return {
        "portfolio_overview": {
            "total_companies_tracked": total_companies,
            "total_investments": total_investments,
            "total_invested": round(total_invested, 2),
            "current_portfolio_value": round(current_value, 2),
            "overall_roi": round(((current_value - total_invested) / total_invested * 100), 1) if total_invested > 0 else 0
        },
        "top_performers": [
            {
                "company_id": perf["company_id"],
                "roi": perf["roi_percentage"],
                "current_value": perf["current_valuation"]
            } for perf in top_performers
        ],
        "industry_breakdown": industry_breakdown,
        "financial_health_distribution": health_distribution,
        "recent_alerts": len([a for a in financial_alerts if datetime.fromisoformat(a["created_at"]) > datetime.now() - timedelta(days=7)]),
        "avg_financial_health": round(sum(c["financial_health_score"] for c in company_financials) / max(total_companies, 1), 1)
    }

@app.get("/api/v1/reports/investment-performance")
async def get_investment_performance_report():
    """Get detailed investment performance report"""
    # Calculate various performance metrics
    investments_by_month = {}
    for inv in investment_tracking:
        month_key = inv["investment_date"][:7]  # YYYY-MM format
        if month_key not in investments_by_month:
            investments_by_month[month_key] = {"count": 0, "amount": 0, "current_value": 0}
        investments_by_month[month_key]["count"] += 1
        investments_by_month[month_key]["amount"] += inv["amount_invested"]
        investments_by_month[month_key]["current_value"] += inv["current_valuation"]
    
    # Risk analysis
    risk_analysis = {}
    for inv in investment_tracking:
        risk = inv["risk_rating"]
        if risk not in risk_analysis:
            risk_analysis[risk] = {"count": 0, "total_invested": 0, "avg_roi": 0}
        risk_analysis[risk]["count"] += 1
        risk_analysis[risk]["total_invested"] += inv["amount_invested"]
    
    # Calculate average ROI by risk level
    for risk, data in risk_analysis.items():
        risk_investments = [inv for inv in investment_tracking if inv["risk_rating"] == risk]
        avg_roi = sum(inv["roi_percentage"] for inv in risk_investments) / len(risk_investments)
        risk_analysis[risk]["avg_roi"] = round(avg_roi, 1)
    
    return {
        "report_id": f"performance_{datetime.now().strftime('%Y%m%d')}",
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_investments": len(investment_tracking),
            "best_performing": max(investment_tracking, key=lambda x: x["roi_percentage"])["roi_percentage"] if investment_tracking else 0,
            "worst_performing": min(investment_tracking, key=lambda x: x["roi_percentage"])["roi_percentage"] if investment_tracking else 0,
            "average_roi": round(sum(inv["roi_percentage"] for inv in investment_tracking) / max(len(investment_tracking), 1), 1)
        },
        "monthly_investments": investments_by_month,
        "risk_analysis": risk_analysis,
        "exit_pipeline": [
            {
                "company_id": inv["company_id"],
                "investment_amount": inv["amount_invested"],
                "current_value": inv["current_valuation"],
                "projected_exit": inv["projected_exit_date"],
                "exit_strategy": inv["exit_strategy"]
            } for inv in investment_tracking if inv.get("projected_exit_date")
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Financial Intelligence Platform", "timestamp": datetime.now().isoformat()}

# Serve dashboard
@app.get("/financial_intelligence_dashboard.html", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the Financial Intelligence dashboard"""
    dashboard_path = "financial_intelligence_dashboard.html"
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Financial Intelligence Dashboard</h1><p>Dashboard file not found.</p>")

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    # Generate sample company financials
    for _ in range(12):
        financial = generate_company_financials()
        company_financials.append(financial.dict())
        
        # Generate investment tracking for some companies
        if random.choice([True, False]):
            investment = generate_investment_tracking(financial.company_id)
            investment_tracking.append(investment.dict())
        
        # Generate valuation models for some companies
        if random.choice([True, False]):
            valuation = generate_valuation_model(financial.company_id)
            valuation_models.append(valuation.dict())
    
    # Generate sample financial alerts
    alert_types = ["Revenue Drop", "Cash Burn Rate", "Valuation Change", "Market Risk"]
    for i in range(6):
        alert = FinancialAlert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}",
            alert_type=random.choice(alert_types),
            company_id=random.choice(company_financials)["company_id"],
            severity=random.choice(["High", "Medium", "Low"]),
            message=f"Company showing {random.choice(['concerning', 'positive', 'unusual'])} financial indicators",
            financial_indicator=random.choice(["Revenue Growth", "Burn Rate", "Cash Position", "Valuation"]),
            threshold_breached=random.uniform(10, 50),
            current_value=random.uniform(5, 45),
            impact_assessment=random.choice(["Significant", "Moderate", "Minor"]),
            created_at=datetime.now() - timedelta(hours=random.randint(1, 48))
        )
        financial_alerts.append(alert.dict())
    
    # Generate market analyses
    segments = ["SaaS", "FinTech", "HealthTech", "E-commerce"]
    for segment in segments:
        analysis = MarketAnalysis(
            analysis_id=f"market_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            market_segment=segment,
            analysis_type="Quarterly Review",
            key_metrics={
                "market_size": f"${random.randint(10, 100)}B",
                "growth_rate": f"{random.randint(15, 45)}%",
                "competition_level": random.choice(["High", "Medium", "Low"])
            },
            growth_projections={
                "next_quarter": f"{random.randint(5, 25)}%",
                "next_year": f"{random.randint(20, 60)}%",
                "three_year": f"{random.randint(50, 150)}%"
            },
            risk_factors=[
                "Market saturation",
                "Regulatory changes",
                "Economic downturn"
            ],
            opportunities=[
                "AI integration",
                "International expansion",
                "New customer segments"
            ],
            analyst_rating=random.choice(["Buy", "Hold", "Sell"]),
            confidence_score=round(random.uniform(70, 95), 1)
        )
        market_analyses.append(analysis.dict())
    
    print("💰 Financial Intelligence Platform started successfully!")
    print("📍 API available at: http://localhost:8020")
    print("📊 Dashboard available at: http://localhost:8020/financial_intelligence_dashboard.html")
    print("📚 API docs available at: http://localhost:8020/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)