# 🚀 How to Run the Application

## Step 1: Activate Virtual Environment

```powershell
cd projects/project-01-smart-expense-categorizer
.\venv\Scripts\Activate.ps1
```

## Step 2: Start Backend API (Terminal 1)

```powershell
cd app
uvicorn main:app --reload --port 8000
```

✅ Backend running at: http://localhost:8000
📚 API Docs: http://localhost:8000/docs

## Step 3: Start Frontend (Terminal 2)

Open a NEW PowerShell window:

```powershell
cd "D:\UDEMY Resources\competetiveCPP\sub\FinTracker\projects\project-01-smart-expense-categorizer"
.\venv\Scripts\Activate.ps1
cd app
streamlit run frontend.py --server.port 8501
```

✅ Frontend running at: http://localhost:8501

## Step 4: Use the Application

1. Open http://localhost:8501 in your browser
2. Click "📁 Need a sample file?" to download sample CSV
3. Upload the CSV file
4. Click "🚀 Process & Categorize Transactions"
5. Explore the Analytics Dashboard, Predictions, and Anomaly Detection tabs!

## Features Available

- ✅ **Upload & Process**: Upload CSV, auto-categorize transactions
- ✅ **Analytics Dashboard**: Beautiful charts and spending insights
- ✅ **Predictions**: AI-powered spending forecasts
- ✅ **Anomaly Detection**: Detect unusual transactions

## Troubleshooting

**Issue**: Port 8000 already in use
**Solution**: Change port in uvicorn command: `--port 8001`

**Issue**: Frontend can't connect to API
**Solution**: Make sure backend is running first, then check API URL in sidebar

**Issue**: Module not found errors
**Solution**: Make sure venv is activated and all packages installed:
```powershell
pip install -r requirements.txt
```

---

**Enjoy your Smart Expense Categorizer! 🎉**

