# 🎯 Project #1: Smart Expense Categorizer with ML

## Overview
An intelligent expense categorization system that uses Machine Learning and NLP to automatically categorize financial transactions, predict spending patterns, detect anomalies, and provide personalized budget recommendations.

## 🚀 Features

### Core ML Features
- **Auto Categorization**: ML-powered transaction categorization using TF-IDF + Random Forest
- **NLP Processing**: Advanced text processing for transaction descriptions
- **Spending Prediction**: Time series forecasting for future expenses
- **Anomaly Detection**: Identify unusual spending patterns using Isolation Forest
- **Smart Recommendations**: Personalized budget suggestions based on spending history

### Web Application Features
- 📊 Interactive Dashboard with visualizations
- 📤 CSV Upload & Processing
- 📈 Real-time Analytics
- 🎯 Category Management
- 📱 Responsive Design
- 🔒 Data Privacy (client-side processing option)

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: React/Streamlit (Python)
- **ML Models**: scikit-learn, Prophet (forecasting)
- **Visualization**: Plotly, Matplotlib
- **Deployment**: Vercel (Frontend) + Railway (Backend API)

## 📋 Installation

```bash
cd projects/project-01-smart-expense-categorizer
pip install -r requirements.txt
```

## 🚀 Running Locally

```bash
# Start API Server
uvicorn app.main:app --reload --port 8000

# Start Frontend (Streamlit)
streamlit run app/frontend.py --server.port 8501
```

## 📊 API Endpoints

- `POST /api/upload` - Upload CSV file
- `POST /api/categorize` - Categorize single transaction
- `POST /api/batch-categorize` - Categorize multiple transactions
- `GET /api/analytics` - Get spending analytics
- `GET /api/predict` - Get spending predictions
- `POST /api/train` - Retrain model with new data

## 🎓 ML Components

1. **Text Classification Model**: TF-IDF Vectorization + Random Forest
2. **Time Series Forecasting**: Prophet for spending prediction
3. **Anomaly Detection**: Isolation Forest for fraud detection
4. **Feature Engineering**: Date features, amount bins, text embeddings

## 📈 Resume Points

- ✅ Built end-to-end ML pipeline from data ingestion to predictions
- ✅ Implemented NLP-based text classification achieving 85%+ accuracy
- ✅ Developed real-time anomaly detection system
- ✅ Created interactive dashboard with Plotly visualizations
- ✅ Deployed scalable API using FastAPI on Railway
- ✅ Integrated multiple ML models (classification, forecasting, anomaly detection)

