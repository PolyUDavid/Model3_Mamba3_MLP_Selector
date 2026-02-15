#!/bin/bash
# D²TL: Start All Microservices
# Author: NOK KO

echo "================================================"
echo "D²TL: Physics-Aware Dual-Path Coverage Predictor"
echo "================================================"
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$SCRIPT_DIR/.."

cd "$BASE_DIR"

# Kill any existing processes on our ports
echo "Cleaning up old processes..."
lsof -ti :8001 | xargs kill -9 2>/dev/null
lsof -ti :8002 | xargs kill -9 2>/dev/null
lsof -ti :8000 | xargs kill -9 2>/dev/null
lsof -ti :8501 | xargs kill -9 2>/dev/null
sleep 1

# Start MLP Service (Port 8001)
echo ""
echo "[1/4] Starting MLP Service (Port 8001)..."
python3 -m uvicorn d2tl.mlp_service.api:app --port 8001 --host 0.0.0.0 &
MLP_PID=$!
sleep 2

# Start Mamba Service (Port 8002)
echo "[2/4] Starting Mamba Service (Port 8002)..."
python3 -m uvicorn d2tl.mamba_service.api:app --port 8002 --host 0.0.0.0 &
MAMBA_PID=$!
sleep 3

# Start Selector Brain (Port 8000)
echo "[3/4] Starting Selector Brain (Port 8000)..."
python3 -m uvicorn d2tl.selector_brain.selector:app --port 8000 --host 0.0.0.0 &
SELECTOR_PID=$!
sleep 2

# Start Dashboard (Port 8501)
echo "[4/4] Starting Dashboard (Port 8501)..."
streamlit run d2tl/dashboard/app.py --server.port 8501 &
DASH_PID=$!
sleep 2

echo ""
echo "================================================"
echo "All services started!"
echo "================================================"
echo ""
echo "  MLP Service:     http://localhost:8001/docs"
echo "  Mamba Service:   http://localhost:8002/docs"
echo "  Selector Brain:  http://localhost:8000/docs"
echo "  Dashboard:       http://localhost:8501"
echo ""
echo "  MLP PID:      $MLP_PID"
echo "  Mamba PID:    $MAMBA_PID"
echo "  Selector PID: $SELECTOR_PID"
echo "  Dashboard PID: $DASH_PID"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for all processes
wait
