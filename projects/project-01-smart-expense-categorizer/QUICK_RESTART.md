# 🔄 Quick Restart Guide

## When to Restart

**You DON'T need to restart if:**
- ✅ Streamlit is running and shows "Rerun" button
- ✅ Browser refreshes automatically
- ✅ You see the new sample file buttons

**You DO need to restart if:**
- ❌ Changes don't appear after 30 seconds
- ❌ You see errors in the Streamlit terminal
- ❌ The app seems stuck

## Quick Restart Steps

### Option 1: Browser Refresh (Easiest)
Just press `Ctrl + R` or `F5` in your browser - this usually works!

### Option 2: Streamlit Auto-Reload
Click the "Rerun" button (⤴️) at the top of Streamlit - or press `R` key

### Option 3: Full Restart (If needed)

**In the Streamlit terminal (Terminal 2):**
1. Press `Ctrl + C` to stop Streamlit
2. Run again:
   ```powershell
   cd app
   streamlit run frontend.py --server.port 8501
   ```

**Backend (Terminal 1) - Usually NO restart needed**
- Backend auto-reloads with `--reload` flag
- Only restart if you changed `main.py` or `ml_models.py`

## Check if it worked:

After refresh/restart, you should see:
- ✅ Two sample file buttons: "Download India Sample (₹)" and "Download US Sample ($)"
- ✅ Currency selector in sidebar: "$" selected by default
- ✅ All amounts showing with $ symbol

---

**Most of the time, just refresh your browser! 🔄**

