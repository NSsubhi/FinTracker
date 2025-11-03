# 🚀 Deployment Guide

## Deploy Backend API (Railway)

1. **Create Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Deploy Backend**
   ```bash
   cd projects/project-01-smart-expense-categorizer
   
   # Install Railway CLI (optional)
   npm i -g @railway/cli
   
   # Login
   railway login
   
   # Initialize project
   railway init
   
   # Deploy
   railway up
   ```

3. **Environment Variables** (if needed)
   - Add in Railway Dashboard → Variables
   - No environment variables required for basic setup

4. **Get API URL**
   - Railway will provide a URL like: `https://your-app.up.railway.app`
   - Update `API_URL` in frontend

## Deploy Frontend (Vercel)

### Option 1: Streamlit on Streamlit Cloud (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Smart Expense Categorizer"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with GitHub
   - Click "New app"
   - Select repository and branch
   - Main file path: `projects/project-01-smart-expense-categorizer/app/frontend.py`
   - Click "Deploy"

3. **Update API URL**
   - In Streamlit Cloud → Settings → Secrets
   - Add your Railway API URL

### Option 2: Vercel (Alternative)

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Deploy**
   ```bash
   cd projects/project-01-smart-expense-categorizer
   vercel
   ```

## Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
cd app
uvicorn main:app --reload --port 8000

# In another terminal, start frontend
streamlit run frontend.py --server.port 8501
```

## Production Checklist

- [ ] Update CORS origins in `app/main.py`
- [ ] Add environment variables for API keys (if needed)
- [ ] Set up error monitoring (Sentry)
- [ ] Configure database for model persistence (optional)
- [ ] Add authentication (optional)
- [ ] Set up CI/CD pipeline

