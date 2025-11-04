# Quick Deploy Guide

## 🚀 Fastest Way to Deploy

### 1. Deploy Backend (Railway) - 5 minutes

1. Go to https://railway.app
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `NSsubhi/FinTracker`
5. Railway auto-detects `railway.json` and deploys!
6. Copy your backend URL (e.g., `https://your-app.up.railway.app`)

### 2. Update Frontend with Backend URL

Edit `app/frontend.py`:
```python
API_URL = "https://your-app.up.railway.app"  # Your Railway URL
```

### 3. Deploy Frontend (Streamlit Cloud) - 3 minutes

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select repo: `NSsubhi/FinTracker`
5. Main file: `app/frontend.py`
6. Branch: `main`
7. Click "Deploy"

**Done!** Your app is live! 🎉

---

## 📍 URLs After Deployment

- **Backend API**: https://your-app.up.railway.app/docs
- **Frontend**: https://your-app.streamlit.app

---

## ✅ Test Your Deployment

1. Visit backend: `/health` endpoint
2. Visit frontend: Upload a CSV and test categorization
3. Check API docs: `/docs` endpoint

