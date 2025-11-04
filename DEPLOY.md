# Deployment Guide - Smart Expense Categorizer

## 🚀 Deployment Options

### Option 1: Railway (Backend) + Streamlit Cloud (Frontend) - **Recommended**
- **Backend (FastAPI)**: Railway.app
- **Frontend (Streamlit)**: Streamlit Cloud (free tier available)

### Option 2: Railway (Backend) + Vercel (Frontend)
- **Backend (FastAPI)**: Railway.app
- **Frontend (Streamlit)**: Vercel (requires custom setup)

---

## 📦 Option 1: Railway + Streamlit Cloud (Recommended)

### Step 1: Deploy Backend to Railway

1. **Go to Railway.app**
   - Visit: https://railway.app
   - Sign up/Login with GitHub

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your `FinTracker` repository

3. **Configure Service**
   - Railway will auto-detect `railway.json`
   - Set environment variables if needed:
     ```
     PORT=8000
     ```
   - Railway will automatically:
     - Install dependencies from `requirements.txt`
     - Run `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

4. **Get Backend URL**
   - Railway will provide a URL like: `https://your-app.up.railway.app`
   - Copy this URL

### Step 2: Deploy Frontend to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Sign in with GitHub

2. **Deploy App**
   - Click "New app"
   - Select repository: `NSsubhi/FinTracker`
   - Main file path: `app/frontend.py`
   - Branch: `main`
   - Click "Deploy"

3. **Update API URL**
   - In `app/frontend.py`, update `API_URL` to your Railway backend URL:
   ```python
   API_URL = "https://your-app.up.railway.app"
   ```
   - Commit and push changes
   - Streamlit Cloud will auto-redeploy

### Step 3: Verify Deployment

- **Backend**: Visit `https://your-app.up.railway.app/docs` (FastAPI docs)
- **Frontend**: Visit your Streamlit Cloud URL

---

## 📦 Option 2: Railway + Vercel

### Step 1: Deploy Backend to Railway
(Same as Option 1, Step 1)

### Step 2: Deploy Frontend to Vercel

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Deploy**
   ```bash
   vercel
   ```
   - Follow prompts
   - Link to your GitHub repository

3. **Or Deploy via GitHub**
   - Go to https://vercel.com
   - Import GitHub repository
   - Vercel will auto-detect `vercel.json`
   - Deploy

---

## 🔧 Environment Variables

### Backend (Railway)
```
PORT=8000
```

### Frontend (Streamlit Cloud)
```
API_URL=https://your-backend.up.railway.app
```

---

## 📝 Quick Deployment Commands

### Railway (Backend)
```bash
# Already configured via railway.json
# Just connect repo on Railway dashboard
```

### Streamlit Cloud
```bash
# Deploy via dashboard at share.streamlit.io
# Select: app/frontend.py
```

### Vercel (Frontend)
```bash
vercel
# or
vercel --prod
```

---

## ✅ Post-Deployment Checklist

- [ ] Backend API accessible at Railway URL
- [ ] Frontend accessible at Streamlit Cloud/Vercel URL
- [ ] API URL updated in frontend code
- [ ] Health check endpoint working (`/health`)
- [ ] API docs accessible (`/docs`)
- [ ] CSV upload functionality working
- [ ] ML predictions working

---

## 🐛 Troubleshooting

### Backend Issues
- Check Railway logs for errors
- Verify `requirements.txt` has all dependencies
- Ensure `railway.json` is correct

### Frontend Issues
- Check Streamlit Cloud logs
- Verify `API_URL` is correct
- Check CORS settings in backend

---

## 🔗 Deployment Links

After deployment, your app will be available at:
- **Backend API**: `https://your-app.up.railway.app`
- **Frontend**: `https://your-app.streamlit.app` or `https://your-app.vercel.app`

