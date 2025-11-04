# Quick Deploy Both on Railway

## 🚀 Deploy Backend & Frontend on Railway

### Step 1: Deploy Backend Service

1. Go to https://railway.app
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `NSsubhi/FinTracker`
5. Railway auto-detects `railway.json` ✅
6. **Service Name**: Rename to "backend" (optional)
7. Copy backend URL (e.g., `https://your-backend.up.railway.app`)

**✅ Backend will auto-deploy with:**
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- From `railway.json`

---

### Step 2: Add Frontend Service

1. In your Railway project, click **"+ New"** → **"Service"**
2. Select **"GitHub Repo"** → Choose `NSsubhi/FinTracker`
3. **IMPORTANT**: Configure the service manually:

   **a) Root Directory**: `/` (root)
   
   **b) Start Command** (CRITICAL - Override default!):
   ```
   streamlit run app/frontend.py --server.port $PORT --server.address 0.0.0.0
   ```
   - Go to: Service Settings → Deploy → Start Command
   - **Delete** the default command (from railway.json)
   - **Paste** the Streamlit command above
   
   **c) Environment Variables**:
   - Click "Variables" tab
   - Add: `API_URL` = `https://your-backend.up.railway.app`
   - (Use your backend URL from Step 1)

4. **Service Name**: Rename to "frontend" (optional)

5. Railway will deploy the frontend!

---

## ✅ Verify Deployment

### Backend
- Visit: `https://your-backend.up.railway.app/docs`
- Should show FastAPI docs ✅

### Frontend  
- Visit: `https://your-frontend.up.railway.app`
- Should show Streamlit app ✅
- Upload CSV and test!

---

## 🐛 Troubleshooting

### Both services show backend
**Problem**: Both services are using `railway.json` (backend config)

**Solution**: 
1. Go to frontend service settings
2. Deploy → Start Command
3. **Delete** the default command
4. **Add**: `streamlit run app/frontend.py --server.port $PORT --server.address 0.0.0.0`
5. Redeploy

### Frontend can't connect to backend
**Problem**: `API_URL` not set correctly

**Solution**:
1. Go to frontend service → Variables
2. Set `API_URL` = your backend URL (with `https://`)
3. Make sure no trailing slash
4. Redeploy

---

## 📝 Quick Reference

**Backend Service:**
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Config: `railway.json` (auto-detected)

**Frontend Service:**
- Start: `streamlit run app/frontend.py --server.port $PORT --server.address 0.0.0.0`
- Config: **Override in Railway settings!**
- Env Var: `API_URL=https://your-backend.up.railway.app`

