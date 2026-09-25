#!/bin/bash
set -e

echo "=========================================================="
echo " Starting StreamPulse Music Analytics Platform"
echo "=========================================================="

echo "Starting FastAPI Backend Microservice on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

echo "Waiting for API to initialize..."
sleep 2

echo "Starting Streamlit Web Dashboard on port 8501..."
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
