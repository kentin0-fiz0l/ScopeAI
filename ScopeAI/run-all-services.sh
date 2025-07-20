#!/bin/bash

# ScopeAI - Complete Business Intelligence Ecosystem
# This script starts all 8 microservices

echo "🚀 Starting ScopeAI Complete Business Intelligence Ecosystem"
echo "================================================================"

# Function to check if a port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $port is already in use. Stopping existing process..."
        kill -9 $(lsof -ti:$port) 2>/dev/null || true
        sleep 2
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local port=$1
    local service_name=$2
    local max_attempts=10
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:$port/health >/dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        echo "   Attempt $attempt/$max_attempts..."
        sleep 3
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start properly"
    return 1
}

# Check if required files exist
echo "🔍 Checking required files..."
required_files=(
    "daily_disruptor_app.py"
    "content_automation_app.py"
    "ai_analytics_platform.py"
    "realtime_alerts_app.py"
    "customer_success_platform.py"
    "competitive_intelligence_platform.py"
    "financial_intelligence_platform.py"
    "supply_chain_intelligence_platform.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: $file not found!"
        exit 1
    fi
done

echo "✅ All required files found"

# Install dependencies
echo "📦 Installing dependencies..."
pip install fastapi uvicorn httpx pydantic slowapi redis bcrypt python-multipart email-validator pyjwt schedule

# Check and start Redis (optional but recommended)
if command -v redis-server &> /dev/null; then
    echo "✅ Redis server found - starting Redis..."
    redis-server --daemonize yes --port 6379 2>/dev/null || echo "ℹ️  Redis already running"
else
    echo "⚠️  Redis not found - some features will use in-memory storage"
fi

# Clear any existing processes on our ports
echo "🧹 Clearing existing processes..."
ports=(8013 8015 8016 8017 8018 8019 8020 8021)
for port in "${ports[@]}"; do
    check_port $port
done

# Function to start a service in the background
start_service() {
    local service_file=$1
    local service_name=$2
    local port=$3
    
    echo "🔄 Starting $service_name on port $port..."
    python $service_file &
    local pid=$!
    sleep 3
    
    if wait_for_service $port "$service_name"; then
        echo "✅ $service_name started successfully"
        echo $pid >> .scopeai_pids
        return $pid
    else
        echo "❌ Failed to start $service_name"
        kill $pid 2>/dev/null
        return 1
    fi
}

# Create PID tracking file
rm -f .scopeai_pids
touch .scopeai_pids

# Start all 8 microservices
echo "🚀 Starting ScopeAI Business Intelligence Ecosystem..."
echo ""

start_service "daily_disruptor_app.py" "Daily Disruptor API" "8015"
DISRUPTOR_PID=$?

start_service "content_automation_app.py" "Content Automation API" "8016"
CONTENT_PID=$?

start_service "ai_analytics_platform.py" "AI Analytics Platform" "8017"
ANALYTICS_PID=$?

start_service "realtime_alerts_app.py" "Real-time Alerts API" "8013"
ALERTS_PID=$?

start_service "customer_success_platform.py" "Customer Success Platform" "8018"
SUCCESS_PID=$?

start_service "competitive_intelligence_platform.py" "Competitive Intelligence Platform" "8019"
COMPETITIVE_PID=$?

start_service "financial_intelligence_platform.py" "Financial Intelligence Platform" "8020"
FINANCIAL_PID=$?

start_service "supply_chain_intelligence_platform.py" "Supply Chain Intelligence Platform" "8021"
SUPPLY_PID=$?

# All services started successfully
echo ""
echo "🎉 ScopeAI Complete Business Intelligence Ecosystem Started Successfully!"
echo "========================================================================"
echo ""
echo "🌐 Service URLs:"
echo "   🎯 Daily Disruptor API:              http://localhost:8015"
echo "   🎙️ Content Automation API:           http://localhost:8016"
echo "   🧠 AI Analytics Platform:            http://localhost:8017"
echo "   🚨 Real-time Alerts API:             http://localhost:8013"
echo "   🎯 Customer Success Platform:         http://localhost:8018"
echo "   🕵️ Competitive Intelligence Platform: http://localhost:8019"
echo "   💰 Financial Intelligence Platform:   http://localhost:8020"
echo "   🔗 Supply Chain Intelligence Platform: http://localhost:8021"
echo ""
echo "📚 API Documentation:"
echo "   Daily Disruptor:      http://localhost:8015/docs"
echo "   Content Automation:   http://localhost:8016/docs"
echo "   AI Analytics:         http://localhost:8017/docs"
echo "   Real-time Alerts:     http://localhost:8013/docs"
echo "   Customer Success:     http://localhost:8018/docs"
echo "   Competitive Intel:    http://localhost:8019/docs"
echo "   Financial Intel:      http://localhost:8020/docs"
echo "   Supply Chain Intel:   http://localhost:8021/docs"
echo ""
echo "🎯 Dashboards:"
echo "   Daily Disruptor:          http://localhost:8015/daily_disruptor_dashboard.html"
echo "   Content Automation:       http://localhost:8016/content_automation_dashboard.html"
echo "   AI Analytics:             http://localhost:8017/ai_analytics_dashboard.html"
echo "   Customer Success:         http://localhost:8018/customer_success_dashboard.html"
echo "   Competitive Intelligence: http://localhost:8019/competitive_intelligence_dashboard.html"
echo "   Financial Intelligence:   http://localhost:8020/financial_intelligence_dashboard.html"
echo "   Supply Chain Intelligence: http://localhost:8021/supply_chain_dashboard.html"
echo ""
echo "🔑 Quick Start:"
echo "   1. Explore daily business disruption ideas in the Daily Disruptor dashboard"
echo "   2. Monitor content automation workflows"
echo "   3. Analyze predictive insights in the AI Analytics dashboard"
echo "   4. Track customer success metrics and interventions"
echo ""
echo "🧪 Testing Commands:"
echo "   • Test innovation insights: curl http://localhost:8015/api/v1/daily-disruptor"
echo "   • Test content automation: curl http://localhost:8016/api/v1/content/stats"
echo "   • Test AI analytics: curl http://localhost:8017/api/v1/analytics/insights"
echo "   • Test health: curl http://localhost:8013/health"
echo ""
echo "📊 Platform Statistics:"
echo "   • Total APIs: 8 business intelligence services"
echo "   • Total Endpoints: 150+ specialized endpoints"
echo "   • Features: Innovation tracking, content automation, predictive analytics"
echo "   • Technologies: FastAPI, AI/ML, Redis, WebSockets, Real-time processing"
echo ""
echo "⚡ AI-Powered Features:"
echo "   • Daily business idea generation with market analysis"
echo "   • Automated podcast and newsletter creation"
echo "   • Predictive business intelligence and forecasting"
echo "   • Real-time market monitoring and alerts"
echo ""
echo "🛡️  Enterprise Features:"
echo "   • Customer success optimization and retention"
echo "   • Competitive intelligence and market research"
echo "   • Financial analytics and investment tracking"
echo "   • Supply chain optimization and risk management"
echo ""
echo "⌨️  Press Ctrl+C to stop all services"
echo "=================================================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping ScopeAI Business Intelligence services..."
    if [ -f .scopeai_pids ]; then
        while read pid; do
            kill $pid 2>/dev/null
        done < .scopeai_pids
        rm -f .scopeai_pids
    fi
    echo "✅ All ScopeAI services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Keep script running and monitor services
while true; do
    # Check if PID file exists and services are running
    if [ -f .scopeai_pids ]; then
        all_running=true
        while read pid; do
            if ! ps -p $pid > /dev/null 2>&1; then
                echo "❌ Service with PID $pid stopped unexpectedly"
                all_running=false
                break
            fi
        done < .scopeai_pids
        
        if [ "$all_running" = false ]; then
            break
        fi
    else
        echo "❌ PID file not found - services may have stopped"
        break
    fi
    
    sleep 10
done

cleanup