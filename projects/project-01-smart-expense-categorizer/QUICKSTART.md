# 🚀 Quick Start Guide

## Local Setup (5 minutes)

### 1. Install Dependencies
```bash
cd projects/project-01-smart-expense-categorizer
pip install -r requirements.txt
```

### 2. Start Backend API
```bash
cd app
uvicorn main:app --reload --port 8000
```
API will be available at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### 3. Start Frontend (in a new terminal)
```bash
streamlit run frontend.py --server.port 8501
```
Frontend will be available at: `http://localhost:8501`

## Test the API

### Using curl:
```bash
# Categorize a transaction
curl -X POST "http://localhost:8000/api/categorize" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Payment to Zomato",
    "amount": 500,
    "date": "2024-01-15",
    "transaction_type": "DEBIT"
  }'

# Batch categorize
curl -X POST "http://localhost:8000/api/batch-categorize" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"description": "Payment to Zomato", "amount": 500, "date": "2024-01-15", "transaction_type": "DEBIT"},
      {"description": "Salary Credit", "amount": 50000, "date": "2024-01-01", "transaction_type": "CREDIT"}
    ]
  }'
```

### Using Python:
```python
import requests

# Categorize
response = requests.post(
    "http://localhost:8000/api/categorize",
    json={
        "description": "Payment to Zomato",
        "amount": 500,
        "date": "2024-01-15",
        "transaction_type": "DEBIT"
    }
)
print(response.json())
```

## Sample CSV Format

Create a CSV file with these columns:
- `Date`: Transaction date (YYYY-MM-DD)
- `Description`: Transaction description
- `Amount`: Transaction amount
- `Transaction_Type`: DEBIT or CREDIT

Example:
```csv
Date,Description,Amount,Transaction_Type
2024-01-15,Payment to Zomato,500,DEBIT
2024-01-16,Salary Credit,50000,CREDIT
2024-01-17,Petrol Pump,1000,DEBIT
```

## Next Steps

1. ✅ Test locally with your own data
2. ✅ Deploy backend to Railway (see DEPLOY.md)
3. ✅ Deploy frontend to Streamlit Cloud (see DEPLOY.md)
4. ✅ Update API_URL in frontend after deployment
5. ✅ Add to your resume!

## Common Issues

### Issue: Module not found
**Solution**: Make sure you're in the project directory and dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: Port already in use
**Solution**: Use a different port
```bash
uvicorn main:app --reload --port 8001
streamlit run frontend.py --server.port 8502
```

### Issue: API connection error
**Solution**: Make sure backend is running first, then update API_URL in frontend.py

---

**Ready to deploy?** Check `DEPLOY.md` for Railway and Streamlit Cloud deployment instructions!

