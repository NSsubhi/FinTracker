#!/bin/bash

# Start both services
# Backend on port from Railway (default 8000)
# Frontend on port 8501 (Streamlit default)

# Start backend in background
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} &

# Start frontend
streamlit run app/frontend.py --server.port 8501 --server.address 0.0.0.0

