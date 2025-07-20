#!/usr/bin/env python3
"""
Real-time Alerts API - Market Monitoring & Intelligence Alerts
Part of the ScopeAI Business Intelligence Ecosystem
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Set
import random
from datetime import datetime, timedelta
import asyncio
import json
import os

app = FastAPI(
    title="Real-time Alerts API",
    description="Market monitoring, workforce intelligence, and real-time alerts",
    version="1.0.0"
)

# Data Models
class Alert(BaseModel):
    alert_id: str
    alert_type: str
    severity: str
    category: str
    title: str
    description: str
    impact_assessment: str
    affected_entities: List[str]
    data_source: str
    confidence_score: float
    created_at: datetime
    expires_at: Optional[datetime]
    action_required: bool
    recommended_actions: List[str]

class MarketEvent(BaseModel):
    event_id: str
    event_type: str
    market_sector: str
    description: str
    magnitude: float
    companies_affected: List[str]
    predicted_impact: str
    timeline: str

class WorkforceUpdate(BaseModel):
    update_id: str
    company: str
    update_type: str
    department: str
    size: int
    percentage_change: float
    reason: str
    source: str
    confidence: float

class RiskIndicator(BaseModel):
    indicator_id: str
    risk_type: str
    current_level: float
    threshold: float
    trend: str
    factors: List[str]
    mitigation_strategies: List[str]

# In-memory storage and WebSocket management
alerts = []
market_events = []
workforce_updates = []
risk_indicators = []
active_websockets: Set[WebSocket] = set()

# Alert generators
def generate_alert():
    """Generate real-time market alert"""
    alert_types = {
        "Workforce Change": {
            "categories": ["Layoffs", "Hiring Surge", "Executive Departure", "Restructuring"],
            "severities": ["High", "Medium", "Low"],
            "descriptions": [
                "Major tech company announces 15% workforce reduction",
                "Startup unicorn plans to double headcount by Q4",
                "Fortune 500 CEO announces unexpected departure"
            ]
        },
        "Market Disruption": {
            "categories": ["Regulatory", "Technology", "Competition", "Economic"],
            "severities": ["Critical", "High", "Medium"],
            "descriptions": [
                "New AI regulations impact 80% of tech companies",
                "Breakthrough technology threatens incumbent players",
                "Major merger creates new market leader"
            ]
        },
        "Investment Signal": {
            "categories": ["Funding Round", "IPO", "Acquisition", "Partnership"],
            "severities": ["High", "Medium", "Low"],
            "descriptions": [
                "Series C funding of $500M for AI startup",
                "Tech giant files for $10B IPO",
                "Strategic acquisition reshapes industry landscape"
            ]
        }
    }
    
    alert_type = random.choice(list(alert_types.keys()))
    config = alert_types[alert_type]
    
    return Alert(
        alert_id=f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        alert_type=alert_type,
        severity=random.choice(config["severities"]),
        category=random.choice(config["categories"]),
        title=f"{alert_type}: {random.choice(config['categories'])}",
        description=random.choice(config["descriptions"]),
        impact_assessment="Significant impact on market dynamics expected",
        affected_entities=random.sample(["TechCorp", "InnovateCo", "FutureScale", "DisruptX", "MegaTech"], k=random.randint(1, 3)),
        data_source=random.choice(["SEC Filings", "News Analysis", "Social Media", "Industry Reports"]),
        confidence_score=round(random.uniform(75, 95), 1),
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=random.randint(1, 48)),
        action_required=random.choice([True, False]),
        recommended_actions=[
            "Monitor situation closely",
            "Update risk assessments",
            "Consider portfolio adjustments"
        ]
    )

def generate_market_event():
    """Generate market event"""
    events = [
        MarketEvent(
            event_id=f"event_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            event_type="Technology Breakthrough",
            market_sector="AI/ML",
            description="New AI model achieves AGI-level performance in specific domains",
            magnitude=9.2,
            companies_affected=["OpenAI", "Google", "Microsoft", "Meta"],
            predicted_impact="Paradigm shift in AI applications",
            timeline="6-12 months"
        ),
        MarketEvent(
            event_id=f"event_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            event_type="Regulatory Change",
            market_sector="FinTech",
            description="Central bank announces digital currency framework",
            magnitude=8.5,
            companies_affected=["PayPal", "Square", "Coinbase", "Stripe"],
            predicted_impact="Major shift in payment infrastructure",
            timeline="12-18 months"
        )
    ]
    
    return random.choice(events)

def generate_workforce_update():
    """Generate workforce intelligence update"""
    companies = ["TechGiant Inc", "StartupUnicorn", "LegacyCorp", "InnovateTech"]
    
    return WorkforceUpdate(
        update_id=f"workforce_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        company=random.choice(companies),
        update_type=random.choice(["Layoffs", "Hiring", "Restructuring", "Expansion"]),
        department=random.choice(["Engineering", "Sales", "Marketing", "Operations", "R&D"]),
        size=random.randint(50, 5000),
        percentage_change=round(random.uniform(-30, 50), 1),
        reason=random.choice([
            "Cost optimization",
            "Strategic refocus",
            "Market expansion",
            "Automation impact",
            "Post-merger integration"
        ]),
        source=random.choice(["WARN Notice", "Company Filing", "News Report", "Internal Source"]),
        confidence=round(random.uniform(70, 95), 1)
    )

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Real-time Alerts API - Your market intelligence radar"}

@app.get("/api/v1/alerts/live")
async def get_live_alerts(
    severity: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 20
):
    """Get live market alerts"""
    filtered_alerts = alerts
    
    if severity:
        filtered_alerts = [a for a in filtered_alerts if a["severity"] == severity]
    if category:
        filtered_alerts = [a for a in filtered_alerts if a["category"] == category]
    
    return filtered_alerts[-limit:]

@app.post("/api/v1/alerts/create")
async def create_custom_alert(
    title: str,
    description: str,
    severity: str,
    category: str
):
    """Create custom alert"""
    alert = Alert(
        alert_id=f"custom_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        alert_type="Custom Alert",
        severity=severity,
        category=category,
        title=title,
        description=description,
        impact_assessment="User-defined impact",
        affected_entities=[],
        data_source="User Input",
        confidence_score=100.0,
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=24),
        action_required=True,
        recommended_actions=["Review and assess impact"]
    )
    
    alerts.append(alert.dict())
    
    # Broadcast to WebSocket clients
    await broadcast_alert(alert.dict())
    
    return alert

@app.get("/api/v1/market-events")
async def get_market_events(sector: Optional[str] = None):
    """Get significant market events"""
    if not market_events:
        for _ in range(5):
            event = generate_market_event()
            market_events.append(event.dict())
    
    events = market_events
    if sector:
        events = [e for e in events if e["market_sector"] == sector]
    
    return events

@app.get("/api/v1/workforce/updates")
async def get_workforce_updates(company: Optional[str] = None):
    """Get workforce intelligence updates"""
    if not workforce_updates:
        for _ in range(10):
            update = generate_workforce_update()
            workforce_updates.append(update.dict())
    
    updates = workforce_updates
    if company:
        updates = [u for u in updates if u["company"] == company]
    
    return updates

@app.get("/api/v1/workforce/predictions")
async def get_layoff_predictions():
    """Get AI-powered layoff predictions"""
    predictions = [
        {
            "company": "TechCorp",
            "probability": round(random.uniform(20, 80), 1),
            "estimated_size": f"{random.randint(100, 1000)} employees",
            "timeline": "3-6 months",
            "risk_factors": [
                "Declining revenue",
                "Market saturation",
                "Automation adoption"
            ],
            "confidence": round(random.uniform(70, 90), 1)
        },
        {
            "company": "StartupX",
            "probability": round(random.uniform(10, 60), 1),
            "estimated_size": f"{random.randint(50, 500)} employees",
            "timeline": "6-12 months",
            "risk_factors": [
                "Funding challenges",
                "Market competition",
                "Burn rate"
            ],
            "confidence": round(random.uniform(65, 85), 1)
        }
    ]
    
    return predictions

@app.get("/api/v1/risk-indicators")
async def get_risk_indicators():
    """Get current risk indicators"""
    indicators = [
        RiskIndicator(
            indicator_id=f"risk_{datetime.now().strftime('%Y%m%d%H%M%S')}_1",
            risk_type="Market Volatility",
            current_level=round(random.uniform(60, 85), 1),
            threshold=75.0,
            trend="increasing",
            factors=["Geopolitical tensions", "Interest rate uncertainty", "Tech sector rotation"],
            mitigation_strategies=["Diversify portfolio", "Increase cash reserves", "Hedge positions"]
        ),
        RiskIndicator(
            indicator_id=f"risk_{datetime.now().strftime('%Y%m%d%H%M%S')}_2",
            risk_type="Talent Shortage",
            current_level=round(random.uniform(70, 90), 1),
            threshold=80.0,
            trend="stable",
            factors=["AI engineer demand", "Remote work competition", "Skill gap"],
            mitigation_strategies=["Upskill programs", "Retention incentives", "Strategic partnerships"]
        )
    ]
    
    return indicators

@app.post("/api/v1/alerts/subscribe")
async def subscribe_to_alerts(
    alert_types: List[str],
    severity_threshold: str = "Medium"
):
    """Subscribe to specific alert types"""
    subscription = {
        "subscription_id": f"sub_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "alert_types": alert_types,
        "severity_threshold": severity_threshold,
        "status": "active",
        "created_at": datetime.now().isoformat()
    }
    
    return subscription

@app.get("/api/v1/alerts/statistics")
async def get_alert_statistics():
    """Get alert statistics and trends"""
    return {
        "total_alerts_24h": len([a for a in alerts if datetime.fromisoformat(a["created_at"]) > datetime.now() - timedelta(days=1)]),
        "alerts_by_severity": {
            "Critical": len([a for a in alerts if a.get("severity") == "Critical"]),
            "High": len([a for a in alerts if a.get("severity") == "High"]),
            "Medium": len([a for a in alerts if a.get("severity") == "Medium"]),
            "Low": len([a for a in alerts if a.get("severity") == "Low"])
        },
        "top_categories": ["Workforce Change", "Market Disruption", "Investment Signal"],
        "average_confidence": round(sum(a.get("confidence_score", 0) for a in alerts) / max(len(alerts), 1), 1),
        "action_required_percentage": round(len([a for a in alerts if a.get("action_required")]) / max(len(alerts), 1) * 100, 1)
    }

# WebSocket endpoint for real-time alerts
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alert streaming"""
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to Real-time Alerts",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and send periodic updates
        while True:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat()
            })
    
    except WebSocketDisconnect:
        active_websockets.remove(websocket)

async def broadcast_alert(alert_data: dict):
    """Broadcast alert to all connected WebSocket clients"""
    disconnected = set()
    
    for websocket in active_websockets:
        try:
            await websocket.send_json({
                "type": "alert",
                "data": alert_data,
                "timestamp": datetime.now().isoformat()
            })
        except:
            disconnected.add(websocket)
    
    # Remove disconnected clients
    active_websockets.difference_update(disconnected)

# Background task to generate alerts
async def generate_periodic_alerts():
    """Generate alerts periodically"""
    while True:
        # Generate random alert every 30-60 seconds
        await asyncio.sleep(random.randint(30, 60))
        
        alert = generate_alert()
        alerts.append(alert.dict())
        
        # Keep only last 100 alerts
        if len(alerts) > 100:
            alerts.pop(0)
        
        # Broadcast to WebSocket clients
        await broadcast_alert(alert.dict())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Real-time Alerts API",
        "timestamp": datetime.now().isoformat(),
        "active_websockets": len(active_websockets)
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    # Generate initial alerts
    for _ in range(10):
        alert = generate_alert()
        alerts.append(alert.dict())
    
    # Generate initial workforce updates
    for _ in range(5):
        update = generate_workforce_update()
        workforce_updates.append(update.dict())
    
    # Start background alert generation
    asyncio.create_task(generate_periodic_alerts())
    
    print("🚨 Real-time Alerts API started successfully!")
    print("📍 API available at: http://localhost:8013")
    print("🔌 WebSocket available at: ws://localhost:8013/ws/alerts")
    print("📚 API docs available at: http://localhost:8013/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)