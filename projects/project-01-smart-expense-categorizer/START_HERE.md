# 🚀 QUICK START - Copy & Paste These Commands

## Current Issue: You're in the Wrong Directory!

You're currently in: `FinTracker\app`  
You need to be in: `FinTracker\projects\project-01-smart-expense-categorizer\app`

---

## ✅ SOLUTION: Copy & Paste These Commands

### Step 1: Navigate to Project Directory

**Open a NEW PowerShell terminal and run:**

```powershell
cd "D:\UDEMY Resources\competetiveCPP\sub\FinTracker\projects\project-01-smart-expense-categorizer"
```

### Step 2: Activate Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Start Backend API (Terminal 1)

```powershell
cd app
uvicorn main:app --reload --port 8000
```

**✅ Wait until you see:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**DO NOT CLOSE THIS TERMINAL! Keep it running.**

---

### Step 4: Start Frontend (Terminal 2)

**Open a NEW PowerShell terminal and run:**

```powershell
cd "D:\UDEMY Resources\competetiveCPP\sub\FinTracker\projects\project-01-smart-expense-categorizer"
.\venv\Scripts\Activate.ps1
cd app
streamlit run frontend.py --server.port 8501
```

**✅ Wait until you see:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

---

### Step 5: Open Browser

**Open:** http://localhost:8501

---

## 🔍 Quick Verification

**To check if you're in the right directory, run:**
```powershell
pwd
```

**Should show:**
```
D:\UDEMY Resources\competetiveCPP\sub\FinTracker\projects\project-01-smart-expense-categorizer
```

---

## ❌ Common Mistakes

1. **Wrong Directory**: Make sure you're in `projects\project-01-smart-expense-categorizer`
2. **Venv Not Activated**: You should see `(venv)` in your prompt
3. **Backend Not Running**: Frontend needs backend on port 8000

---

## 📝 All-in-One Command (Copy Everything)

**Terminal 1 (Backend):**
```powershell
cd "D:\UDEMY Resources\competetiveCPP\sub\FinTracker\projects\project-01-smart-expense-categorizer"
.\venv\Scripts\Activate.ps1
cd app
uvicorn main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```powershell
cd "D:\UDEMY Resources\competetiveCPP\sub\FinTracker\projects\project-01-smart-expense-categorizer"
.\venv\Scripts\Activate.ps1
cd app
streamlit run frontend.py --server.port 8501
```

---

**Now you should be able to open http://localhost:8501 and see the app! 🎉**

