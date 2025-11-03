# 🧪 Testing Guide

## Quick Test Steps

### Step 1: Activate Virtual Environment (if not already activated)
```powershell
cd "D:\UDEMY Resources\competetiveCPP\sub\FinTracker\projects\project-01-smart-expense-categorizer"
.\venv\Scripts\Activate.ps1
```

### Step 2: Start Backend API (Terminal 1)
```powershell
cd app
uvicorn main:app --reload --port 8000
```

**✅ Backend is running when you see:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
```

**🌐 URLs:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Step 3: Test Backend API (Quick Test)

Open a NEW terminal and run:
```powershell
# Test health endpoint
curl http://localhost:8000/health

# Test categorization
curl -X POST "http://localhost:8000/api/categorize" -H "Content-Type: application/json" -d "{\"description\": \"Payment to Zomato\", \"amount\": 500, \"date\": \"2024-01-15\", \"transaction_type\": \"DEBIT\"}"
```

**Expected response:**
```json
{
  "category": "Food",
  "confidence": 0.85,
  "suggestions": []
}
```

### Step 4: Start Frontend (Terminal 2)

Open ANOTHER PowerShell window:
```powershell
cd "D:\UDEMY Resources\competetiveCPP\sub\FinTracker\projects\project-01-smart-expense-categorizer"
.\venv\Scripts\Activate.ps1
cd app
streamlit run frontend.py --server.port 8501
```

**✅ Frontend is running when you see:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://localhost:8501
```

### Step 5: Test in Browser

1. **Open browser**: http://localhost:8501
2. **Click "⬇️ Download Sample CSV"** to get test data
3. **Upload the CSV file** (or the sample you downloaded)
4. **Click "🚀 Process & Categorize Transactions"**
5. **Check results**: You should see categorized transactions!

## Visual Testing Checklist

✅ **Upload & Process Tab:**
- [ ] Can download sample CSV
- [ ] Can upload CSV file
- [ ] Can categorize transactions
- [ ] Shows category distribution chart
- [ ] Quick categorize works

✅ **Analytics Dashboard Tab:**
- [ ] Shows financial overview metrics
- [ ] Displays spending by category chart
- [ ] Shows monthly trends
- [ ] Shows income vs expenses
- [ ] Shows daily spending pattern

✅ **Predictions Tab:**
- [ ] Can generate predictions
- [ ] Shows forecast chart
- [ ] Displays trend information
- [ ] Shows prediction table

✅ **Anomaly Detection Tab:**
- [ ] Can detect anomalies
- [ ] Shows risk score gauge
- [ ] Displays anomaly table
- [ ] Shows risk assessment

## Python Test Script (Optional)

Create `test_api.py` in project root:

```python
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("1. Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")

def test_categorize():
    print("2. Testing Categorization...")
    response = requests.post(
        f"{BASE_URL}/api/categorize",
        json={
            "description": "Payment to Zomato",
            "amount": 500,
            "date": "2024-01-15",
            "transaction_type": "DEBIT"
        }
    )
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}\n")

def test_batch_categorize():
    print("3. Testing Batch Categorization...")
    response = requests.post(
        f"{BASE_URL}/api/batch-categorize",
        json={
            "transactions": [
                {"description": "Payment to Zomato", "amount": 500, "date": "2024-01-15", "transaction_type": "DEBIT"},
                {"description": "Salary Credit", "amount": 50000, "date": "2024-01-01", "transaction_type": "CREDIT"}
            ]
        }
    )
    print(f"   Status: {response.status_code}")
    print(f"   Total: {response.json()['total']}\n")

if __name__ == "__main__":
    try:
        test_health()
        test_categorize()
        test_batch_categorize()
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure the backend API is running on http://localhost:8000")
```

Run it:
```powershell
.\venv\Scripts\Activate.ps1
python test_api.py
```

## Troubleshooting

**❌ Backend won't start:**
- Check if port 8000 is already in use
- Make sure venv is activated
- Install dependencies: `pip install -r requirements.txt`

**❌ Frontend can't connect:**
- Make sure backend is running first
- Check API URL in sidebar (should be http://localhost:8000)
- Click "🔌 Test Connection" in sidebar

**❌ Import errors:**
- Make sure you're in the correct directory
- Activate venv: `.\venv\Scripts\Activate.ps1`
- Install all dependencies

**❌ File not found errors:**
- Make sure sample_transactions.csv exists in project root
- Check file paths

---

**Happy Testing! 🚀**

