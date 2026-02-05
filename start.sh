#!/bin/bash
echo "Starting FastAPI server..."
python -m uvicorn serve:app --host 0.0.0.0 --port 8000
