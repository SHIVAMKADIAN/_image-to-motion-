#!/bin/bash
# Image Motion Studio — Node.js Startup Script
set -e

cd "$(dirname "$0")"

# Start Python depth micro-service in background
echo "[Start] Starting Python depth service on port 8001..."
source backend/.venv/bin/activate
cd backend
DEPTH_SERVICE_PORT=8001 python depth_service.py &
DEPTH_PID=$!
cd ..

# Wait for depth service to be ready
echo "[Start] Waiting for depth service..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:8001/health > /dev/null 2>&1; then
    echo "[Start] Depth service ready!"
    break
  fi
  sleep 1
done

# Start Node.js server
echo "[Start] Starting Node.js server on port 8000..."
cd backend-node
node server.js

# Cleanup on exit
trap "kill $DEPTH_PID 2>/dev/null" EXIT
