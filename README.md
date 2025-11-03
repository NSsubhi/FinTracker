# 💰 FinTracker - Smart Expense Categorizer

An intelligent expense tracking and categorization system powered by Machine Learning and AI.

## 🚀 Project Overview

FinTracker is a full-stack ML-powered application that automatically categorizes financial transactions, predicts spending patterns, and detects anomalies in your financial data.

## 📁 Project Structure

```
FinTracker/
├── projects/
│   └── project-01-smart-expense-categorizer/  # Main ML-powered expense categorizer
│       ├── app/                                # Application code
│       │   ├── main.py                         # FastAPI backend
│       │   ├── frontend.py                     # Streamlit frontend
│       │   ├── ml_models.py                    # ML models
│       │   └── data_processor.py               # Data processing
│       ├── sample_transactions.csv             # Sample data
│       ├── requirements.txt                    # Dependencies
│       └── README.md                           # Project documentation
└── README.md                                   # This file
```

## 🎯 Features

- ✅ **Auto Categorization**: ML-powered transaction categorization using NLP
- ✅ **Spending Predictions**: Time-series forecasting with Prophet
- ✅ **Anomaly Detection**: Fraud and unusual pattern detection
- ✅ **Auto Format Detection**: Works with various CSV formats automatically
- ✅ **Beautiful Dashboard**: Interactive visualizations with Plotly
- ✅ **REST API**: FastAPI backend with comprehensive endpoints

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit (Python)
- **ML Models**: scikit-learn, Prophet
- **Visualization**: Plotly, Matplotlib
- **Deployment**: Railway, Streamlit Cloud

## 📊 Main Project: Smart Expense Categorizer

The main project is located in `projects/project-01-smart-expense-categorizer/`

### Quick Start

1. Navigate to the project:
   ```bash
   cd projects/project-01-smart-expense-categorizer
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start backend:
   ```bash
   cd app
   uvicorn main:app --reload --port 8000
   ```

5. Start frontend (new terminal):
   ```bash
   cd app
   streamlit run frontend.py --server.port 8501
   ```

6. Open browser: http://localhost:8501

### Features

- **Upload CSV**: Automatically detects CSV format and maps columns
- **ML Categorization**: Categorizes transactions using NLP + Random Forest
- **Spending Predictions**: Forecasts future spending patterns
- **Anomaly Detection**: Identifies unusual transactions
- **Analytics Dashboard**: Beautiful charts and insights

## 📚 Documentation

- [Project README](projects/project-01-smart-expense-categorizer/README.md)
- [Deployment Guide](projects/project-01-smart-expense-categorizer/DEPLOY.md)
- [CSV Formats Guide](projects/project-01-smart-expense-categorizer/CSV_FORMATS.md)
- [Quick Start Guide](projects/project-01-smart-expense-categorizer/QUICKSTART.md)

## 🎓 Resume Points

- Built end-to-end ML pipeline from data ingestion to predictions
- Implemented NLP-based text classification achieving 85%+ accuracy
- Developed real-time anomaly detection system
- Created interactive dashboard with Plotly visualizations
- Deployed scalable REST API using FastAPI on Railway
- Integrated multiple ML models (classification, forecasting, anomaly detection)

## 📝 License

This project is open source and available for personal and educational use.

## 👤 Author

**NSsubhi**

---

**Happy Tracking! 💰**

