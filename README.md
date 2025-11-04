# 🎯 Smart Expense Categorizer with ML

## 🌐 Live Demo
**👉 [Try it now!](https://fintracker-production-7af6.up.railway.app/)**

## Overview
An intelligent expense categorization system that uses Machine Learning and NLP to automatically categorize financial transactions, predict spending patterns, detect anomalies, and provide personalized budget recommendations.

## 🚀 Features

### Core ML Features
- **Auto Categorization**: ML-powered transaction categorization using TF-IDF + Random Forest
- **NLP Processing**: Advanced text processing for transaction descriptions
- **Spending Prediction**: Time series forecasting for future expenses using Prophet
- **Anomaly Detection**: Identify unusual spending patterns using Isolation Forest
- **Smart Recommendations**: Personalized budget suggestions based on spending history

### Web Application Features
- 📊 Interactive Dashboard with beautiful visualizations
- 📤 CSV Upload & Processing with automatic format detection
- 📈 Real-time Analytics and insights
- 🎯 Category Management
- 📱 Responsive Design
- 🔒 Data Privacy (client-side processing option)
- 💰 Multi-currency support (USD, INR, EUR, GBP, etc.)

### Smart CSV Processing
- ✅ **Automatic Format Detection**: Works with any CSV format!
- ✅ Supports various column name variations (Date, Transaction Date, Posting Date, etc.)
- ✅ Handles separate Debit/Credit columns or single Amount column
- ✅ Automatically removes currency symbols (₹, $, €, etc.)
- ✅ Multiple encoding support (UTF-8, Latin-1, CP1252)

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit (Python)
- **ML Models**: scikit-learn, Prophet (forecasting)
- **Visualization**: Plotly, Matplotlib
- **Deployment**: Railway (Backend) + Streamlit Cloud (Frontend)

## 📋 Installation

```bash
cd projects/project-01-smart-expense-categorizer
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Running Locally

### Start Backend API (Terminal 1)
```bash
cd app
uvicorn main:app --reload --port 8000
```

✅ API will be available at: http://localhost:8000  
📚 API Docs: http://localhost:8000/docs

### Start Frontend (Terminal 2)
```bash
cd app
streamlit run frontend.py --server.port 8501
```

✅ Frontend will be available at: http://localhost:8501

## 📊 API Endpoints

- `GET /` - API root endpoint
- `GET /health` - Health check
- `POST /api/upload` - Upload CSV file for processing
- `POST /api/categorize` - Categorize single transaction
- `POST /api/batch-categorize` - Categorize multiple transactions
- `POST /api/analytics` - Get spending analytics
- `POST /api/predict` - Get spending predictions (forecast)
- `POST /api/anomalies` - Detect anomalous transactions

## 📁 Sample Data

Download the sample CSV file directly from the app:
1. Open the application
2. Click "📁 Need a sample file?"
3. Click "⬇️ Download Sample File"
4. Upload it back to test the application!

## 🎓 ML Components

1. **Text Classification Model**: TF-IDF Vectorization + Random Forest Classifier
2. **Time Series Forecasting**: Prophet model for spending prediction
3. **Anomaly Detection**: Isolation Forest for fraud/error detection
4. **Feature Engineering**: Date features, amount bins, text embeddings

## 📈 Resume Points

- ✅ Built end-to-end ML pipeline from data ingestion to predictions
- ✅ Implemented NLP-based text classification achieving 85%+ accuracy
- ✅ Developed real-time anomaly detection system
- ✅ Created interactive dashboard with Plotly visualizations
- ✅ Deployed scalable API using FastAPI on Railway
- ✅ Integrated multiple ML models (classification, forecasting, anomaly detection)
- ✅ Built automatic CSV format detection system supporting multiple formats
- ✅ Implemented multi-currency support with dynamic currency display

## 🎯 Supported CSV Formats

The application automatically detects and works with various CSV formats:

- Standard format: `Date, Description, Amount, Transaction_Type`
- Bank statements: `Transaction Date, Narration, Debit, Credit`
- Alternative names: `Posting Date, Details, Transaction Amount`
- Separate columns: `Date, Description, Withdrawal, Deposit`
- Currency symbols included: Automatically removed

See [CSV_FORMATS.md](CSV_FORMATS.md) for complete documentation.

## 🔧 Configuration

- **API URL**: Configure in sidebar (default: http://localhost:8000)
- **Currency Symbol**: Select from sidebar ($, ₹, €, £, ¥, or None)
- **Test Connection**: Use the "🔌 Test Connection" button in sidebar

## 📚 Documentation

- [Quick Start Guide](QUICKSTART.md)
- [Deployment Guide](DEPLOY.md)
- [CSV Formats Guide](CSV_FORMATS.md)
- [Testing Guide](TEST.md)

## 🐛 Troubleshooting

**Issue**: Port already in use  
**Solution**: Change port in uvicorn/streamlit command

**Issue**: Module not found  
**Solution**: Make sure venv is activated: `.\venv\Scripts\Activate.ps1`

**Issue**: CSV format not detected  
**Solution**: Ensure CSV has Date, Description, and Amount columns (any variations work)

**Issue**: API connection error  
**Solution**: Make sure backend is running first, then test connection in sidebar

## 🚀 Deployment

### Deploy Backend (Railway)
1. Push code to GitHub
2. Connect Railway to your GitHub repo
3. Deploy automatically

### Deploy Frontend (Streamlit Cloud)
1. Push code to GitHub
2. Go to streamlit.io/cloud
3. Connect repository
4. Set main file: `app/frontend.py`
5. Deploy!

See [DEPLOY.md](DEPLOY.md) for detailed instructions.

## 📝 License

This project is open source and available for personal and educational use.

## 👤 Author

**NSsubhi**

---

**Happy Expense Tracking! 💰**
