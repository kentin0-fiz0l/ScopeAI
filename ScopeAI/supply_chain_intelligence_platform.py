#!/usr/bin/env python3
"""
Supply Chain Intelligence Platform - Operational Excellence & Supply Chain Optimization
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
    title="Supply Chain Intelligence Platform",
    description="Supply chain optimization, logistics tracking, and operational intelligence",
    version="1.0.0"
)

# Data Models
class SupplierProfile(BaseModel):
    supplier_id: str
    company_name: str
    location: str
    supplier_type: str
    products_services: List[str]
    performance_score: float
    reliability_rating: str
    cost_competitiveness: float
    quality_score: float
    delivery_performance: float
    risk_level: str
    contract_value: float
    relationship_duration: int

class ShipmentTracking(BaseModel):
    shipment_id: str
    supplier_id: str
    origin: str
    destination: str
    product_category: str
    quantity: int
    value: float
    shipping_method: str
    departure_date: datetime
    estimated_arrival: datetime
    actual_arrival: Optional[datetime]
    status: str
    delays: List[Dict]
    tracking_events: List[Dict]

class InventoryOptimization(BaseModel):
    item_id: str
    product_name: str
    category: str
    current_stock: int
    optimal_stock_level: int
    reorder_point: int
    lead_time_days: int
    demand_forecast: Dict
    carrying_cost: float
    stockout_risk: float
    recommendations: List[str]

class RiskAssessment(BaseModel):
    risk_id: str
    risk_type: str
    category: str
    description: str
    probability: float
    impact_severity: str
    affected_suppliers: List[str]
    mitigation_strategies: List[str]
    monitoring_indicators: List[str]
    risk_score: float
    last_assessment: datetime

class PerformanceMetrics(BaseModel):
    metric_id: str
    metric_name: str
    category: str
    current_value: float
    target_value: float
    trend: str
    benchmark: float
    improvement_potential: float
    factors: List[str]
    action_items: List[str]

# In-memory storage
supplier_profiles = []
shipment_tracking = []
inventory_optimization = []
risk_assessments = []
performance_metrics = []

# Data generators
def generate_supplier_profile():
    """Generate supplier profile data"""
    company_names = [
        "Global Manufacturing Co", "Premium Logistics Ltd", "TechComponents Inc",
        "EcoSupply Solutions", "FastTrack Delivery", "QualityFirst Materials",
        "InnovateSupply Corp", "ReliablePartners LLC", "GreenChain Industries",
        "PrecisionMakers Ltd"
    ]
    
    locations = [
        "Shenzhen, China", "Mumbai, India", "Ho Chi Minh City, Vietnam",
        "Tijuana, Mexico", "Istanbul, Turkey", "Bangkok, Thailand",
        "Guangzhou, China", "Bangalore, India", "Manila, Philippines",
        "São Paulo, Brazil"
    ]
    
    supplier_types = ["Manufacturer", "Distributor", "Logistics Provider", "Raw Materials", "Components"]
    
    # Generate performance scores
    reliability = round(random.uniform(70, 98), 1)
    quality = round(random.uniform(75, 99), 1)
    delivery = round(random.uniform(65, 95), 1)
    cost_comp = round(random.uniform(60, 90), 1)
    
    # Calculate overall performance score
    performance = round((reliability * 0.3 + quality * 0.25 + delivery * 0.25 + cost_comp * 0.2), 1)
    
    # Determine reliability rating
    if performance >= 90:
        reliability_rating = "Excellent"
        risk_level = "Low"
    elif performance >= 80:
        reliability_rating = "Good"
        risk_level = "Medium"
    elif performance >= 70:
        reliability_rating = "Fair"
        risk_level = "Medium"
    else:
        reliability_rating = "Poor"
        risk_level = "High"
    
    return SupplierProfile(
        supplier_id=f"sup_{random.randint(10000, 99999)}",
        company_name=random.choice(company_names),
        location=random.choice(locations),
        supplier_type=random.choice(supplier_types),
        products_services=random.sample([
            "Electronic Components", "Raw Materials", "Packaging", "Transportation",
            "Manufacturing", "Assembly", "Quality Control", "Warehousing"
        ], k=random.randint(2, 4)),
        performance_score=performance,
        reliability_rating=reliability_rating,
        cost_competitiveness=cost_comp,
        quality_score=quality,
        delivery_performance=delivery,
        risk_level=risk_level,
        contract_value=random.uniform(100000, 10000000),
        relationship_duration=random.randint(1, 15)
    )

def generate_shipment_tracking(supplier_id: str = None):
    """Generate shipment tracking data"""
    if not supplier_id:
        supplier_id = f"sup_{random.randint(10000, 99999)}"
    
    departure = datetime.now() - timedelta(days=random.randint(1, 14))
    transit_time = random.randint(3, 21)
    estimated_arrival = departure + timedelta(days=transit_time)
    
    # Generate tracking events
    events = []
    current_date = departure
    event_types = ["Picked up", "In transit", "Customs clearance", "Out for delivery"]
    
    for i, event_type in enumerate(event_types[:random.randint(2, 4)]):
        current_date += timedelta(days=random.randint(1, 3))
        events.append({
            "timestamp": current_date.isoformat(),
            "event": event_type,
            "location": random.choice(["Origin Hub", "Transit Hub", "Destination Hub", "Customs"]),
            "status": "completed" if current_date < datetime.now() else "pending"
        })
    
    # Generate delays if any
    delays = []
    if random.choice([True, False]):
        delays.append({
            "reason": random.choice(["Weather", "Customs", "Transport delay", "Documentation"]),
            "duration_hours": random.randint(12, 72),
            "impact": random.choice(["Minor", "Moderate", "Significant"])
        })
    
    # Determine status
    if datetime.now() > estimated_arrival:
        status = random.choice(["Delivered", "Delayed", "In Transit"])
        actual_arrival = estimated_arrival + timedelta(days=random.randint(0, 3)) if status == "Delivered" else None
    else:
        status = "In Transit"
        actual_arrival = None
    
    return ShipmentTracking(
        shipment_id=f"ship_{random.randint(100000, 999999)}",
        supplier_id=supplier_id,
        origin=random.choice(["Shanghai", "Mumbai", "Bangkok", "Tijuana", "Istanbul"]),
        destination=random.choice(["Los Angeles", "New York", "Chicago", "Houston", "Atlanta"]),
        product_category=random.choice(["Electronics", "Textiles", "Machinery", "Raw Materials", "Consumer Goods"]),
        quantity=random.randint(100, 10000),
        value=random.uniform(10000, 500000),
        shipping_method=random.choice(["Sea Freight", "Air Freight", "Land Transport", "Multimodal"]),
        departure_date=departure,
        estimated_arrival=estimated_arrival,
        actual_arrival=actual_arrival,
        status=status,
        delays=delays,
        tracking_events=events
    )

def generate_inventory_optimization():
    """Generate inventory optimization data"""
    products = [
        "Microprocessors", "Memory Modules", "Power Supplies", "Circuit Boards",
        "Cables & Connectors", "Sensors", "Displays", "Batteries",
        "Packaging Materials", "Assembly Components"
    ]
    
    product_name = random.choice(products)
    current_stock = random.randint(50, 1000)
    lead_time = random.randint(7, 30)
    daily_demand = random.randint(5, 50)
    
    # Calculate optimal levels
    safety_stock = daily_demand * random.randint(3, 7)
    reorder_point = (daily_demand * lead_time) + safety_stock
    optimal_stock = reorder_point + (daily_demand * random.randint(10, 20))
    
    # Generate demand forecast
    forecast = {}
    for i in range(1, 13):
        base_demand = daily_demand * 30
        seasonal_factor = random.uniform(0.8, 1.2)
        forecast[f"month_{i}"] = int(base_demand * seasonal_factor)
    
    # Determine stockout risk
    stock_coverage = current_stock / daily_demand
    if stock_coverage < lead_time:
        stockout_risk = round(random.uniform(60, 90), 1)
        recommendations = ["Immediate reorder", "Expedite delivery", "Find alternative suppliers"]
    elif stock_coverage < reorder_point / daily_demand:
        stockout_risk = round(random.uniform(30, 60), 1)
        recommendations = ["Schedule reorder", "Monitor demand closely"]
    else:
        stockout_risk = round(random.uniform(5, 30), 1)
        recommendations = ["Maintain current levels", "Optimize order frequency"]
    
    return InventoryOptimization(
        item_id=f"item_{random.randint(10000, 99999)}",
        product_name=product_name,
        category=random.choice(["Electronics", "Components", "Materials", "Packaging"]),
        current_stock=current_stock,
        optimal_stock_level=optimal_stock,
        reorder_point=reorder_point,
        lead_time_days=lead_time,
        demand_forecast=forecast,
        carrying_cost=round(random.uniform(0.15, 0.35), 2),
        stockout_risk=stockout_risk,
        recommendations=recommendations
    )

def generate_risk_assessment():
    """Generate supply chain risk assessment"""
    risk_types = {
        "Supplier Risk": [
            "Single source dependency",
            "Financial instability of key supplier",
            "Quality control issues"
        ],
        "Logistics Risk": [
            "Transportation disruptions",
            "Port congestion delays",
            "Customs and regulatory changes"
        ],
        "Demand Risk": [
            "Demand volatility",
            "Seasonal fluctuations",
            "Market demand shifts"
        ],
        "Operational Risk": [
            "Production capacity constraints",
            "Technology failures",
            "Natural disasters"
        ]
    }
    
    risk_category = random.choice(list(risk_types.keys()))
    risk_description = random.choice(risk_types[risk_category])
    
    probability = round(random.uniform(20, 80), 1)
    impact_severity = random.choice(["Low", "Medium", "High", "Critical"])
    
    # Calculate risk score
    impact_weights = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    risk_score = round((probability / 100) * impact_weights[impact_severity] * 25, 1)
    
    return RiskAssessment(
        risk_id=f"risk_{random.randint(10000, 99999)}",
        risk_type=risk_category,
        category=random.choice(["Strategic", "Operational", "Financial", "Compliance"]),
        description=risk_description,
        probability=probability,
        impact_severity=impact_severity,
        affected_suppliers=random.sample([s["supplier_id"] for s in supplier_profiles], k=random.randint(1, 3)) if supplier_profiles else [],
        mitigation_strategies=[
            "Diversify supplier base",
            "Increase safety stock",
            "Develop contingency plans",
            "Implement monitoring systems"
        ],
        monitoring_indicators=[
            "Supplier performance metrics",
            "Inventory levels",
            "Lead time variations",
            "Quality indicators"
        ],
        risk_score=risk_score,
        last_assessment=datetime.now()
    )

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Supply Chain Intelligence Platform - Your operational excellence engine"}

@app.get("/api/v1/suppliers")
async def get_suppliers(risk_level: Optional[str] = None, supplier_type: Optional[str] = None):
    """Get supplier profiles"""
    suppliers = supplier_profiles
    
    if risk_level:
        suppliers = [s for s in suppliers if s["risk_level"] == risk_level]
    if supplier_type:
        suppliers = [s for s in suppliers if s["supplier_type"] == supplier_type]
    
    return suppliers

@app.get("/api/v1/suppliers/{supplier_id}")
async def get_supplier_details(supplier_id: str):
    """Get detailed supplier information"""
    for supplier in supplier_profiles:
        if supplier["supplier_id"] == supplier_id:
            # Add related shipments
            related_shipments = [s for s in shipment_tracking if s["supplier_id"] == supplier_id]
            supplier["recent_shipments"] = related_shipments[-5:]
            return supplier
    raise HTTPException(status_code=404, detail="Supplier not found")

@app.post("/api/v1/suppliers/add")
async def add_supplier():
    """Add new supplier to tracking"""
    supplier = generate_supplier_profile()
    supplier_profiles.append(supplier.dict())
    return supplier

@app.get("/api/v1/shipments/tracking")
async def get_shipment_tracking(status: Optional[str] = None, supplier_id: Optional[str] = None):
    """Get shipment tracking information"""
    shipments = shipment_tracking
    
    if status:
        shipments = [s for s in shipments if s["status"] == status]
    if supplier_id:
        shipments = [s for s in shipments if s["supplier_id"] == supplier_id]
    
    return shipments

@app.get("/api/v1/shipments/{shipment_id}")
async def get_shipment_details(shipment_id: str):
    """Get detailed shipment information"""
    for shipment in shipment_tracking:
        if shipment["shipment_id"] == shipment_id:
            return shipment
    raise HTTPException(status_code=404, detail="Shipment not found")

@app.post("/api/v1/shipments/create")
async def create_shipment(supplier_id: str):
    """Create new shipment tracking"""
    # Verify supplier exists
    supplier_exists = any(s["supplier_id"] == supplier_id for s in supplier_profiles)
    if not supplier_exists:
        raise HTTPException(status_code=404, detail="Supplier not found")
    
    shipment = generate_shipment_tracking(supplier_id)
    shipment_tracking.append(shipment.dict())
    return shipment

@app.get("/api/v1/inventory/optimization")
async def get_inventory_optimization(category: Optional[str] = None):
    """Get inventory optimization recommendations"""
    inventory = inventory_optimization
    if category:
        inventory = [i for i in inventory if i["category"] == category]
    return inventory

@app.get("/api/v1/inventory/alerts")
async def get_inventory_alerts():
    """Get inventory alerts and notifications"""
    alerts = []
    
    for item in inventory_optimization:
        if item["stockout_risk"] > 70:
            alerts.append({
                "alert_id": f"inv_alert_{item['item_id']}",
                "alert_type": "High Stockout Risk",
                "item_id": item["item_id"],
                "product_name": item["product_name"],
                "current_stock": item["current_stock"],
                "risk_level": item["stockout_risk"],
                "recommended_action": "Immediate reorder required",
                "urgency": "High"
            })
        elif item["current_stock"] < item["reorder_point"]:
            alerts.append({
                "alert_id": f"inv_alert_{item['item_id']}",
                "alert_type": "Reorder Point Reached",
                "item_id": item["item_id"],
                "product_name": item["product_name"],
                "current_stock": item["current_stock"],
                "reorder_point": item["reorder_point"],
                "recommended_action": "Schedule reorder",
                "urgency": "Medium"
            })
    
    return alerts

@app.get("/api/v1/risks/assessments")
async def get_risk_assessments(category: Optional[str] = None, risk_level: Optional[str] = None):
    """Get supply chain risk assessments"""
    risks = risk_assessments
    
    if category:
        risks = [r for r in risks if r["category"] == category]
    if risk_level:
        # Convert risk_level to score range for filtering
        score_ranges = {"Low": (0, 25), "Medium": (25, 50), "High": (50, 75), "Critical": (75, 100)}
        if risk_level in score_ranges:
            min_score, max_score = score_ranges[risk_level]
            risks = [r for r in risks if min_score <= r["risk_score"] < max_score]
    
    return risks

@app.post("/api/v1/risks/assess")
async def create_risk_assessment():
    """Create new risk assessment"""
    risk = generate_risk_assessment()
    risk_assessments.append(risk.dict())
    return risk

@app.get("/api/v1/performance/metrics")
async def get_performance_metrics(category: Optional[str] = None):
    """Get supply chain performance metrics"""
    metrics = performance_metrics
    if category:
        metrics = [m for m in metrics if m["category"] == category]
    return metrics

@app.get("/api/v1/analytics/dashboard")
async def get_analytics_dashboard():
    """Get supply chain analytics dashboard"""
    # Calculate summary metrics
    total_suppliers = len(supplier_profiles)
    high_risk_suppliers = len([s for s in supplier_profiles if s["risk_level"] == "High"])
    active_shipments = len([s for s in shipment_tracking if s["status"] == "In Transit"])
    delayed_shipments = len([s for s in shipment_tracking if s["delays"]])
    
    # Performance overview
    avg_supplier_score = round(sum(s["performance_score"] for s in supplier_profiles) / max(total_suppliers, 1), 1)
    on_time_delivery = round((len([s for s in shipment_tracking if s["status"] == "Delivered" and not s["delays"]]) / 
                            max(len([s for s in shipment_tracking if s["status"] == "Delivered"]), 1)) * 100, 1)
    
    # Inventory insights
    high_risk_items = len([i for i in inventory_optimization if i["stockout_risk"] > 50])
    reorder_needed = len([i for i in inventory_optimization if i["current_stock"] < i["reorder_point"]])
    
    # Risk summary
    critical_risks = len([r for r in risk_assessments if r["risk_score"] > 75])
    avg_risk_score = round(sum(r["risk_score"] for r in risk_assessments) / max(len(risk_assessments), 1), 1)
    
    return {
        "summary": {
            "total_suppliers": total_suppliers,
            "high_risk_suppliers": high_risk_suppliers,
            "active_shipments": active_shipments,
            "delayed_shipments": delayed_shipments
        },
        "performance": {
            "average_supplier_score": avg_supplier_score,
            "on_time_delivery_rate": on_time_delivery,
            "quality_score_avg": round(sum(s["quality_score"] for s in supplier_profiles) / max(total_suppliers, 1), 1),
            "cost_competitiveness_avg": round(sum(s["cost_competitiveness"] for s in supplier_profiles) / max(total_suppliers, 1), 1)
        },
        "inventory": {
            "total_items_tracked": len(inventory_optimization),
            "high_risk_items": high_risk_items,
            "reorder_needed": reorder_needed,
            "optimization_opportunities": random.randint(5, 15)
        },
        "risk_management": {
            "total_risks": len(risk_assessments),
            "critical_risks": critical_risks,
            "average_risk_score": avg_risk_score,
            "mitigation_plans_active": random.randint(8, 20)
        },
        "recent_activities": [
            "New supplier onboarded: Global Manufacturing Co",
            "Risk assessment completed for Asia-Pacific region",
            "Inventory optimization identified 12% cost savings",
            "3 shipments delivered ahead of schedule"
        ]
    }

@app.get("/api/v1/reports/supply-chain")
async def generate_supply_chain_report():
    """Generate comprehensive supply chain report"""
    # Supplier analysis
    supplier_breakdown = {}
    for supplier in supplier_profiles:
        location = supplier["location"].split(", ")[-1]  # Get country
        if location not in supplier_breakdown:
            supplier_breakdown[location] = {"count": 0, "avg_performance": 0, "total_value": 0}
        supplier_breakdown[location]["count"] += 1
        supplier_breakdown[location]["total_value"] += supplier["contract_value"]
    
    # Calculate averages
    for location, data in supplier_breakdown.items():
        location_suppliers = [s for s in supplier_profiles if location in s["location"]]
        data["avg_performance"] = round(sum(s["performance_score"] for s in location_suppliers) / len(location_suppliers), 1)
    
    # Shipment analysis
    shipment_analysis = {
        "total_shipments": len(shipment_tracking),
        "in_transit": len([s for s in shipment_tracking if s["status"] == "In Transit"]),
        "delivered": len([s for s in shipment_tracking if s["status"] == "Delivered"]),
        "delayed": len([s for s in shipment_tracking if s["delays"]]),
        "total_value": sum(s["value"] for s in shipment_tracking)
    }
    
    return {
        "report_id": f"supply_chain_{datetime.now().strftime('%Y%m%d')}",
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "suppliers": total_suppliers,
            "geographic_diversification": len(supplier_breakdown),
            "total_contract_value": sum(s["contract_value"] for s in supplier_profiles),
            "operational_efficiency": round(random.uniform(78, 92), 1)
        },
        "supplier_breakdown": supplier_breakdown,
        "shipment_analysis": shipment_analysis,
        "top_performers": sorted(supplier_profiles, key=lambda x: x["performance_score"], reverse=True)[:5],
        "improvement_opportunities": [
            "Diversify supplier base in high-risk regions",
            "Implement predictive analytics for demand forecasting",
            "Optimize inventory levels to reduce carrying costs",
            "Develop strategic partnerships with top-performing suppliers"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Supply Chain Intelligence Platform", "timestamp": datetime.now().isoformat()}

# Serve dashboard
@app.get("/supply_chain_dashboard.html", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the Supply Chain Intelligence dashboard"""
    dashboard_path = "supply_chain_dashboard.html"
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Supply Chain Intelligence Dashboard</h1><p>Dashboard file not found.</p>")

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    # Generate sample suppliers
    for _ in range(10):
        supplier = generate_supplier_profile()
        supplier_profiles.append(supplier.dict())
        
        # Generate shipments for each supplier
        for _ in range(random.randint(1, 4)):
            shipment = generate_shipment_tracking(supplier.supplier_id)
            shipment_tracking.append(shipment.dict())
    
    # Generate inventory optimization data
    for _ in range(15):
        inventory = generate_inventory_optimization()
        inventory_optimization.append(inventory.dict())
    
    # Generate risk assessments
    for _ in range(8):
        risk = generate_risk_assessment()
        risk_assessments.append(risk.dict())
    
    # Generate performance metrics
    metric_categories = ["Supplier Performance", "Logistics Efficiency", "Inventory Management", "Cost Optimization"]
    metric_names = {
        "Supplier Performance": ["On-time Delivery", "Quality Score", "Reliability Rating"],
        "Logistics Efficiency": ["Transit Time", "Cost per Shipment", "Tracking Accuracy"],
        "Inventory Management": ["Stock Turnover", "Carrying Cost", "Stockout Rate"],
        "Cost Optimization": ["Total Cost Savings", "Contract Efficiency", "Process Improvement"]
    }
    
    for category, names in metric_names.items():
        for name in names:
            current_val = random.uniform(70, 95)
            target_val = current_val + random.uniform(2, 10)
            
            metric = PerformanceMetrics(
                metric_id=f"metric_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                metric_name=name,
                category=category,
                current_value=round(current_val, 1),
                target_value=round(target_val, 1),
                trend=random.choice(["improving", "stable", "declining"]),
                benchmark=round(random.uniform(75, 90), 1),
                improvement_potential=round(target_val - current_val, 1),
                factors=random.sample([
                    "Process optimization", "Technology adoption", "Supplier collaboration",
                    "Market conditions", "Operational efficiency"
                ], k=2),
                action_items=random.sample([
                    "Implement new tracking system", "Negotiate better contract terms",
                    "Increase supplier diversity", "Optimize delivery routes"
                ], k=2)
            )
            performance_metrics.append(metric.dict())
    
    print("🔗 Supply Chain Intelligence Platform started successfully!")
    print("📍 API available at: http://localhost:8021")
    print("📊 Dashboard available at: http://localhost:8021/supply_chain_dashboard.html")
    print("📚 API docs available at: http://localhost:8021/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8021)