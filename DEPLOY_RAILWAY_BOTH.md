# Deploy Both Backend & Frontend on Railway

## 🚀 Single Railway Deployment (Both Services)

Railway can host both your FastAPI backend and Streamlit frontend!

---

## 📦 Option 1: Two Services in One Project (Recommended)

### Step 1: Deploy Backend Service

1. Go to https://railway.app
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `NSsubhi/FinTracker`
5. Railway will auto-create a service
6. In the service settings:
   - **Root Directory**: `/` (root)
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Railway auto-detects `railway.json` ✅
7. Railway will assign a URL like: `https://your-app-backend.up.railway.app`
8. Copy this URL!

### Step 2: Add Frontend Service (Same Project)

1. In your Railway project, click **"+ New"** → **"Service"**
2. Select **"GitHub Repo"** → Choose `NSsubhi/FinTracker`
3. In the new service settings:
   - **Root Directory**: `/` (root)
   - **Start Command**: `streamlit run app/frontend.py --server.port $PORT --server.address 0.0.0.0`
     - **IMPORTANT**: Override the start command in Railway settings! Railway will default to `railway.json` which has the backend command.
     - Go to service settings → Deploy → Start Command
     - Replace with: `streamlit run app/frontend.py --server.port $PORT --server.address 0.0.0.0`
   - **Environment Variables**:
     ```
     API_URL=https://your-app-backend.up.railway.app
     ```
4. Railway will assign a separate URL like: `https://your-app-frontend.up.railway.app`

### Step 3: Update Frontend API URL

The frontend has an input field in the sidebar where users can enter the API URL. For production, you can also set it as an environment variable.

**Update `app/frontend.py` to use environment variable:**

```python
import os

# Get API URL from environment or default
API_URL = os.getenv("API_URL", "http://localhost:8000")

# In sidebar, make it editable but default to env var
with st.sidebar:
    api_url_input = st.text_input(
        "🌐 API Server URL", 
        value=API_URL,
        help="Enter your API server URL"
    )
    API_URL = api_url_input or API_URL
```

---

## 📦 Option 2: Single Service with Both (Alternative)

### Setup Script

1. Use the `start.sh` script to run both services
2. Update `railway.json` to use `bash start.sh`

**Note**: This approach runs both on the same port/container, which is less ideal but works.

---

## 🔧 Environment Variables

### Backend Service
```
PORT=8000  (auto-set by Railway)
```

### Frontend Service
```
PORT=8501  (or Railway's assigned port)
API_URL=https://your-backend-service.up.railway.app
```

---

## ✅ After Deployment

### Backend URLs
- **API**: `https://your-backend.up.railway.app`
- **Docs**: `https://your-backend.up.railway.app/docs`
- **Health**: `https://your-backend.up.railway.app/health`

### Frontend URL
- **App**: `https://your-frontend.up.railway.app`

---

## 🎯 Quick Steps Summary

1. **Deploy Backend**:
   - New Project → GitHub Repo
   - Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Copy backend URL

2. **Add Frontend Service**:
   - Same project → New Service
   - Start: `streamlit run app/frontend.py --server.port $PORT --server.address 0.0.0.0`
   - Set env var: `API_URL=<backend-url>`

3. **Test**:
   - Visit frontend URL
   - Enter backend URL in sidebar (or it uses env var)
   - Upload CSV and test!

---

## 💡 Pro Tips

- **Custom Domains**: Railway allows custom domains for both services
- **Monitoring**: Check Railway dashboard for logs and metrics
- **Scaling**: Railway auto-scales based on traffic
- **Free Tier**: Railway offers $5/month free credit

---

## 🐛 Troubleshooting

### Frontend can't connect to backend
- Check `API_URL` environment variable is set correctly
- Verify backend URL is accessible (visit `/health` endpoint)
- Check CORS settings in backend (already configured ✅)

### Port conflicts
- Railway sets `$PORT` automatically - use it!
- Don't hardcode ports

### Build failures
- Check `requirements.txt` has all dependencies
- View Railway build logs for errors

