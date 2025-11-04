"""
Smart Expense Categorizer - Beautiful Application UI
A modern, polished expense tracking and ML-powered categorization app
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import models (for standalone mode)
try:
    # Try relative imports first
    from .ml_models import ExpenseCategorizer, SpendingPredictor, AnomalyDetector
    from .data_processor import DataProcessor
    STANDALONE_MODE = False
except ImportError:
    try:
        # Try absolute imports
        from ml_models import ExpenseCategorizer, SpendingPredictor, AnomalyDetector
        from data_processor import DataProcessor
        STANDALONE_MODE = False
    except ImportError:
        STANDALONE_MODE = True
        ExpenseCategorizer = None
        SpendingPredictor = None
        AnomalyDetector = None
        DataProcessor = None

# Page configuration
st.set_page_config(
    page_title="Smart Expense Categorizer | ML-Powered Finance Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #333;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to convert DataFrame to JSON-serializable format
def df_to_json_serializable(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to JSON-serializable list of dicts"""
    df_copy = df.copy()
    
    # Convert Timestamp columns to strings
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype(str)
        elif pd.api.types.is_period_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype(str)
    
    # Convert to dict
    records = df_copy.to_dict(orient='records')
    
    # Ensure all values are JSON serializable
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, (pd.Timestamp, pd.Period)):
                record[key] = str(value)
            elif hasattr(value, 'item'):  # Handle numpy types
                try:
                    record[key] = value.item()
                except:
                    record[key] = str(value)
    
    return records

# Initialize session state
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'api_available' not in st.session_state:
    st.session_state.api_available = True

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API URL input
    API_URL = st.text_input("🌐 API Server URL", value="http://localhost:8000", 
                            help="Enter your API server URL (leave as default for local)")
    
    # Currency selection
    currency_symbol = st.selectbox(
        "💰 Currency Symbol",
        ["$", "₹", "€", "£", "¥", "None"],
        index=0,
        help="Select currency symbol for display"
    )
    
    # Test API connection
    if st.button("🔌 Test Connection"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ Connected!")
                st.session_state.api_available = True
            else:
                st.warning("⚠️ API not responding correctly")
                st.session_state.api_available = False
        except:
            st.error("❌ Cannot connect to API. Using standalone mode.")
            st.session_state.api_available = False
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df
        total = len(df)
        categories = df['Category'].nunique() if 'Category' in df.columns else 0
        st.metric("Transactions", total)
        st.metric("Categories", categories)
    else:
        st.info("Upload data to see stats")
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.info("""
    **Smart Expense Categorizer**
    
    ML-powered expense tracking with:
    - Auto categorization
    - Spending predictions
    - Anomaly detection
    - Beautiful analytics
    """)

# Main header
st.markdown('<div class="main-header">💰 Smart Expense Categorizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Personal Finance Tracker & Analytics Dashboard</div>', unsafe_allow_html=True)

# Tab navigation
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Analytics Dashboard", "🔮 Predictions", "🚨 Anomaly Detection"])

# ==================== TAB 1: UPLOAD & PROCESS ====================
with tab1:
    st.header("📤 Upload & Process Transactions")
    
    main_col1, main_col2 = st.columns([2, 1])
    
    with main_col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file to upload",
            type=['csv'],
            help="Upload your transaction CSV file. Should contain: Date, Description, Amount, Transaction_Type"
        )
        
        # Sample file download
        st.markdown("### 📁 Need a sample file?")
        
        # Try multiple paths to find the sample file
        current_file = os.path.abspath(__file__)
        app_dir = os.path.dirname(current_file)
        project_root = os.path.dirname(app_dir)
        
        # Try different possible paths
        possible_paths = [
            os.path.join(project_root, "sample_transactions_us.csv"),
            os.path.join(app_dir, "..", "sample_transactions_us.csv"),
            os.path.join(os.getcwd(), "sample_transactions_us.csv"),
            os.path.join(os.getcwd(), "..", "sample_transactions_us.csv"),
            "sample_transactions_us.csv"
        ]
        
        sample_file_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                sample_file_path = abs_path
                break
        
        if sample_file_path:
            try:
                with open(sample_file_path, "rb") as file:
                    file_data = file.read()
                    st.download_button(
                        label="⬇️ Download Sample File",
                        data=file_data,
                        file_name="sample_transactions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error loading sample file: {e}")
        else:
            st.warning("Sample file not found. Please make sure sample_transactions_us.csv exists in the project directory.")
    
    with main_col2:
        st.markdown("### 💡 Quick Categorize")
        st.markdown("Test single transaction categorization:")
        
        test_description = st.text_input("Description", placeholder="e.g., Payment to Zomato")
        test_amount = st.number_input("Amount", min_value=0.0, value=500.0, step=10.0)
        
        if st.button("🎯 Categorize Now"):
            if test_description:
                try:
                    if st.session_state.api_available:
                        response = requests.post(
                            f"{API_URL}/api/categorize",
                            json={
                                "description": test_description,
                                "amount": test_amount,
                                "date": datetime.now().isoformat(),
                                "transaction_type": "DEBIT"
                            },
                            timeout=5
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"**Category:** {result['category']} | **Confidence:** {result['confidence']:.1%}")
                        else:
                            st.error("API Error")
                    else:
                        # Standalone mode
                        if ExpenseCategorizer:
                            categorizer = ExpenseCategorizer()
                            result = categorizer.categorize(test_description, test_amount)
                            st.success(f"**Category:** {result['category']} | **Confidence:** {result['confidence']:.1%}")
                        else:
                            st.error("Standalone mode not available. Please start the API server.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a description")
    
    # Process uploaded file
    if uploaded_file is not None:
        st.markdown("---")
        
        # Read and display preview
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} transactions")
            
            with st.expander("👁️ Preview Data (First 10 rows)", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Process button
            if st.button("🚀 Process & Categorize Transactions", use_container_width=True):
                with st.spinner("🤖 Processing transactions with AI..."):
                    try:
                        if st.session_state.api_available:
                            # Use API
                            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                            response = requests.post(f"{API_URL}/api/upload", files=files, timeout=30)
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                # Create DataFrame from results
                                processed_df = pd.DataFrame(result['data'])
                                processed_df['Date'] = pd.to_datetime(processed_df['Date'])
                                
                                st.session_state.processed_df = processed_df
                                
                                st.success("✅ Processing complete!")
                                
                                # Show summary
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Transactions", result['total_transactions'])
                                with col2:
                                    st.metric("Categories Found", len(result['categories']))
                                with col3:
                                    st.metric("Anomalies", result['anomalies_detected'])
                                with col4:
                                    total_amt = processed_df['Amount'].sum()
                                    st.metric("Total Amount", f"{currency_symbol if currency_symbol != 'None' else ''}{total_amt:,.2f}")
                                
                                # Category pie chart
                                st.subheader("📊 Category Distribution")
                                category_df = pd.DataFrame(
                                    list(result['categories'].items()),
                                    columns=['Category', 'Count']
                                )
                                fig = px.pie(
                                    category_df, 
                                    values='Count', 
                                    names='Category',
                                    title="Transaction Categories",
                                    color_discrete_sequence=px.colors.qualitative.Set3
                                )
                                fig.update_traces(textposition='inside', textinfo='percent+label')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.balloons()
                            else:
                                st.error(f"API Error: {response.text}")
                        else:
                            # Standalone processing
                            if DataProcessor and ExpenseCategorizer:
                                processor = DataProcessor()
                                categorizer = ExpenseCategorizer()
                                
                                # Process data
                                processed_df = processor.process_dataframe(df)
                                processed_df = categorizer.categorize_batch(processed_df)
                            else:
                                st.error("Standalone mode not available. Please start the API server first.")
                                st.stop()
                            
                            st.session_state.processed_df = processed_df
                            
                            st.success("✅ Processing complete!")
                            
                            # Summary
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Transactions", len(processed_df))
                            with col2:
                                st.metric("Categories", processed_df['Category'].nunique())
                            with col3:
                                st.metric("Date Range", f"{(processed_df['Date'].max() - processed_df['Date'].min()).days} days")
                            with col4:
                                total_amt = processed_df['Amount'].sum()
                                st.metric("Total Amount", f"{currency_symbol if currency_symbol != 'None' else ''}{total_amt:,.2f}")
                            
                            # Category distribution
                            st.subheader("📊 Category Distribution")
                            category_counts = processed_df['Category'].value_counts()
                            fig = px.pie(
                                values=category_counts.values,
                                names=category_counts.index,
                                title="Transaction Categories",
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.balloons()
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
                        st.exception(e)
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ==================== TAB 2: ANALYTICS DASHBOARD ====================
with tab2:
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Summary metrics
        st.subheader("💰 Financial Overview")
        total_spending = df[df['Transaction_Type'] == 'DEBIT']['Amount'].sum()
        total_income = df[df['Transaction_Type'] == 'CREDIT']['Amount'].sum()
        net_savings = total_income - total_spending
        savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            currency_prefix = currency_symbol if currency_symbol != 'None' else ''
            st.metric("💵 Total Spending", f"{currency_prefix}{total_spending:,.2f}", 
                     delta=f"-{currency_prefix}{total_spending:,.2f}" if total_spending > 0 else None)
        with col2:
            currency_prefix = currency_symbol if currency_symbol != 'None' else ''
            st.metric("💰 Total Income", f"{currency_prefix}{total_income:,.2f}",
                     delta=f"+{currency_prefix}{total_income:,.2f}" if total_income > 0 else None)
        with col3:
            currency_prefix = currency_symbol if currency_symbol != 'None' else ''
            st.metric("💸 Net Savings", f"{currency_prefix}{net_savings:,.2f}",
                     delta=f"{savings_rate:.1f}%" if savings_rate > 0 else None)
        with col4:
            avg_trans = df['Amount'].mean()
            currency_prefix = currency_symbol if currency_symbol != 'None' else ''
            st.metric("📊 Avg Transaction", f"{currency_prefix}{avg_trans:,.2f}")
        
        # Category spending breakdown
        st.subheader("📈 Spending by Category")
        category_spending = df[df['Transaction_Type'] == 'DEBIT'].groupby('Category')['Amount'].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                x=category_spending.index,
                y=category_spending.values,
                labels={'x': 'Category', f'y': 'Amount ({currency_symbol if currency_symbol != "None" else ""})'},
                title="Total Spending by Category",
                color=category_spending.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Top Categories")
            for idx, (cat, amount) in enumerate(category_spending.head(5).items(), 1):
                st.markdown(f"**{idx}.** {cat}<br>₹{amount:,.2f}", unsafe_allow_html=True)
        
        # Monthly trends
        st.subheader("📅 Monthly Spending Trends")
        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        monthly_data = df[df['Transaction_Type'] == 'DEBIT'].groupby('Month').agg({
            'Amount': ['sum', 'mean', 'count']
        }).reset_index()
        monthly_data.columns = ['Month', 'Total', 'Average', 'Count']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_data['Month'],
            y=monthly_data['Total'],
            mode='lines+markers',
            name='Total Spending',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="Monthly Spending Trend",
            xaxis_title="Month",
            yaxis_title=f"Amount ({currency_symbol if currency_symbol != 'None' else ''})",
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Income vs Expenses
        st.subheader("💹 Income vs Expenses")
        income_expense = df.groupby(['Month', 'Transaction_Type'])['Amount'].sum().unstack(fill_value=0)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=income_expense.index,
            y=income_expense.get('CREDIT', pd.Series(0, index=income_expense.index)),
            name='Income',
            marker_color='#28a745'
        ))
        fig.add_trace(go.Bar(
            x=income_expense.index,
            y=-income_expense.get('DEBIT', pd.Series(0, index=income_expense.index)),
            name='Expenses',
            marker_color='#dc3545'
        ))
        fig.update_layout(
            title="Monthly Income vs Expenses",
            xaxis_title="Month",
            yaxis_title=f"Amount ({currency_symbol if currency_symbol != 'None' else ''})",
            barmode='group',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily pattern
        st.subheader("📆 Spending Pattern by Day of Week")
        df['DayOfWeek'] = df['Date'].dt.day_name()
        daily_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_pattern = df[df['Transaction_Type'] == 'DEBIT'].groupby('DayOfWeek')['Amount'].mean().reindex(
            [d for d in daily_order if d in df['DayOfWeek'].values], fill_value=0
        )
        
        currency_label = currency_symbol if currency_symbol != 'None' else ''
        fig = px.bar(
            x=daily_pattern.index,
            y=daily_pattern.values,
            labels={'x': 'Day of Week', 'y': f'Average Amount ({currency_label})'},
            title="Average Spending by Day of Week",
            color=daily_pattern.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("👆 Please upload and process a file first from the 'Upload & Process' tab")

# ==================== TAB 3: PREDICTIONS ====================
with tab3:
    st.header("🔮 Spending Predictions")
    
    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df.copy()
        
        st.markdown("### 📊 Forecast Configuration")
        col1, col2 = st.columns(2)
        with col1:
            days = st.slider("Forecast Period (days)", 7, 90, 30)
        with col2:
            confidence_level = st.selectbox("Confidence Level", [0.8, 0.9, 0.95], index=1)
        
        if st.button("🔮 Generate Predictions", use_container_width=True):
            with st.spinner("🧠 AI is analyzing patterns and generating predictions..."):
                try:
                    if st.session_state.api_available:
                        # Convert DataFrame to JSON-serializable format
                        data_to_send = df_to_json_serializable(df)
                        
                        response = requests.post(
                            f"{API_URL}/api/predict",
                            params={"days": days},
                            json=data_to_send,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            predictions = result['predictions']
                            trend = result.get('trend', 'unknown')
                        else:
                            st.error("Prediction API failed")
                            st.stop()
                    else:
                        # Standalone mode
                        if SpendingPredictor:
                            predictor = SpendingPredictor()
                            predictions = predictor.predict(df, days=days)
                            trend = predictor.get_trend(df)
                        else:
                            st.error("Standalone mode not available. Please start the API server.")
                            st.stop()
                    
                    # Display predictions
                    pred_df = pd.DataFrame(predictions)
                    pred_df['date'] = pd.to_datetime(pred_df['date'])
                    
                    st.success("✅ Predictions generated!")
                    
                    # Summary metrics
                    avg_predicted = pred_df['predicted'].mean()
                    total_predicted = pred_df['predicted'].sum()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        currency_prefix = currency_symbol if currency_symbol != 'None' else ''
                        st.metric("📊 Avg Daily Spending", f"{currency_prefix}{avg_predicted:,.2f}")
                    with col2:
                        st.metric("💰 Total Forecasted", f"{currency_prefix}{total_predicted:,.2f}")
                    with col3:
                        trend_emoji = "📈" if trend == "increasing" else "📉" if trend == "decreasing" else "➡️"
                        st.metric("📈 Trend", f"{trend_emoji} {trend.replace('_', ' ').title()}")
                    
                    # Prediction chart
                    fig = go.Figure()
                    
                    # Predicted line
                    fig.add_trace(go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['predicted'],
                        mode='lines+markers',
                        name='Predicted Spending',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=6)
                    ))
                    
                    # Confidence intervals if available
                    if 'lower_bound' in pred_df.columns and 'upper_bound' in pred_df.columns:
                        fig.add_trace(go.Scatter(
                            x=pred_df['date'],
                            y=pred_df['upper_bound'],
                            mode='lines',
                            name='Upper Bound',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=pred_df['date'],
                            y=pred_df['lower_bound'],
                            mode='lines',
                            name='Lower Bound',
                            line=dict(width=0),
                            fillcolor='rgba(102, 126, 234, 0.2)',
                            fill='tonexty',
                            showlegend=False
                        ))
                    
                    currency_label = currency_symbol if currency_symbol != 'None' else ''
                    fig.update_layout(
                        title="Spending Forecast",
                        xaxis_title="Date",
                        yaxis_title=f"Predicted Amount ({currency_label})",
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction table
                    with st.expander("📋 View Detailed Predictions"):
                        display_df = pred_df[['date', 'predicted']].copy()
                        currency_prefix = currency_symbol if currency_symbol != 'None' else ''
                        display_df.columns = ['Date', f'Predicted Amount ({currency_prefix})']
                        display_df[f'Predicted Amount ({currency_prefix})'] = display_df[f'Predicted Amount ({currency_prefix})'].apply(lambda x: f"{currency_prefix}{float(x):,.2f}" if isinstance(x, (int, float, str)) else x)
                        st.dataframe(display_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error generating predictions: {e}")
                    st.exception(e)
    else:
        st.info("👆 Please upload and process a file first")

# ==================== TAB 4: ANOMALY DETECTION ====================
with tab4:
    st.header("🚨 Anomaly Detection")
    
    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df.copy()
        
        st.markdown("### 🔍 Fraud & Anomaly Detection")
        st.info("Our AI analyzes your spending patterns to detect unusual transactions that may indicate fraud or errors.")
        
        if st.button("🚨 Detect Anomalies", use_container_width=True):
            with st.spinner("🔍 Analyzing transactions for anomalies..."):
                try:
                    if st.session_state.api_available:
                        # Convert DataFrame to JSON-serializable format
                        data_to_send = df_to_json_serializable(df)
                        
                        response = requests.post(
                            f"{API_URL}/api/anomalies",
                            json=data_to_send,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                        else:
                            st.error("Anomaly detection API failed")
                            st.stop()
                    else:
                        # Standalone mode
                        if AnomalyDetector:
                            anomaly_detector = AnomalyDetector()
                            anomalies = anomaly_detector.detect(df)
                            risk_score = anomaly_detector.calculate_risk_score(df)
                        else:
                            st.error("Standalone mode not available. Please start the API server.")
                            st.stop()
                        
                        result = {
                            'anomalies_detected': len(anomalies),
                            'anomalies': anomalies.to_dict(orient='records') if len(anomalies) > 0 else [],
                            'risk_score': risk_score
                        }
                    
                    risk_score = result['risk_score']
                    anomalies = result['anomalies']
                    
                    # Risk assessment
                    st.subheader("📊 Risk Assessment")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric("Risk Score", f"{risk_score:.1f}/100")
                        
                        if risk_score < 30:
                            st.success("✅ **Low Risk**")
                            st.markdown("Your spending patterns look normal.")
                        elif risk_score < 70:
                            st.warning("⚠️ **Moderate Risk**")
                            st.markdown("Some unusual patterns detected. Review transactions.")
                        else:
                            st.error("🚨 **High Risk**")
                            st.markdown("Multiple anomalies detected. Immediate review recommended.")
                    
                    with col2:
                        # Risk gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = risk_score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Risk Level"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Anomalies table
                    if len(anomalies) > 0:
                        st.subheader(f"🚨 Detected Anomalies ({len(anomalies)})")
                        
                        anomaly_df = pd.DataFrame(anomalies)
                        if 'Date' in anomaly_df.columns:
                            anomaly_df['Date'] = pd.to_datetime(anomaly_df['Date'])
                        
                        # Display table
                        display_cols = ['Date', 'Description', 'Amount', 'Category']
                        available_cols = [col for col in display_cols if col in anomaly_df.columns]
                        currency_prefix = currency_symbol if currency_symbol != 'None' else ''
                        st.dataframe(
                            anomaly_df[available_cols].style.format({
                                'Amount': lambda x: f"{currency_prefix}{x:,.2f}" if isinstance(x, (int, float)) else x,
                                'Date': lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else x
                            }),
                            use_container_width=True
                        )
                        
                        # Anomaly analysis
                        st.markdown("### 💡 Anomaly Analysis")
                        st.warning(f"""
                        **{len(anomalies)}** unusual transactions detected.
                        
                        These transactions may be:
                        - Unusually large amounts
                        - Transactions on unusual days/times
                        - Spending patterns that deviate from your normal behavior
                        
                        Please review these transactions carefully.
                        """)
                    else:
                        st.success("✅ **No Anomalies Detected!**")
                        st.balloons()
                        st.markdown("Your spending patterns are normal. Great job! 🎉")
                    
                except Exception as e:
                    st.error(f"Error detecting anomalies: {e}")
                    st.exception(e)
    else:
        st.info("👆 Please upload and process a file first")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>💰 <strong>Smart Expense Categorizer</strong> | Powered by ML & AI</p>
        <p>Built with ❤️ using FastAPI, Streamlit & scikit-learn</p>
    </div>
    """,
    unsafe_allow_html=True
)
