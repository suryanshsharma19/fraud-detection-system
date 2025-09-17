#!/bin/bash

# Quick script to start both the API and dashboard
# Run this from the project directory

echo "Starting fraud detection system..."

# kill stuff when we exit
cleanup() {
    echo "Stopping services..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null
    fi
    if [ ! -z "$DASHBOARD_PID" ]; then
        kill $DASHBOARD_PID 2>/dev/null
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# make sure we have the basics
if [ ! -f "requirements.txt" ]; then
    echo "No requirements.txt found, are you in the right directory?"
    exit 1
fi

# use virtual env if it exists
if [ -d "venv" ]; then
    echo "Using virtual environment..."
    source venv/bin/activate
else
    echo "No venv found, using system python"
fi

# install packages
echo "Installing requirements..."
pip install -r requirements.txt

# create logs dir
mkdir -p logs

# start the API server in background
echo "Starting API server..."
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
API_PID=$!

echo "API server started on port 8000"

# start dashboard 
echo "Starting dashboard..."
streamlit run src/dashboard/app.py --server.port=8501 --server.headless=true > logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!

echo "Dashboard started on port 8501"
sleep 3

echo ""
echo "Both services should be running now:"
echo "  Dashboard: http://localhost:8501"
echo "  API: http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop everything"

# wait for user to stop
while true; do
    sleep 5
    # basic check if processes still alive
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "API stopped"
        break
    fi
    if ! kill -0 $DASHBOARD_PID 2>/dev/null; then
        echo "Dashboard stopped"  
        break
    fi
done
