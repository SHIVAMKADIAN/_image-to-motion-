#!/bin/bash
# Image Motion Studio — Start backend API
set -e

cd "$(dirname "$0")"
source backend/.venv/bin/activate
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
