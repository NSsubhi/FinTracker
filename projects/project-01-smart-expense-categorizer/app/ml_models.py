"""
ML Models for Expense Categorization, Prediction, and Anomaly Detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import logging
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ExpenseCategorizer:
    """
    ML-powered expense categorizer using NLP and classification
    """
    
    def __init__(self):
        self.categories = [
            'Food', 'Shopping', 'Transport', 'Bills', 'Entertainment',
            'Healthcare', 'Salary', 'Investment', 'Education', 'Travel', 'Other'
        ]
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.is_trained = False
        
        # Rule-based keywords as fallback
        self.keyword_rules = {
            'Food': ['restaurant', 'food', 'lunch', 'dinner', 'zomato', 'swiggy', 'grocery', 'cafe'],
            'Shopping': ['amazon', 'flipkart', 'purchase', 'shopping', 'store', 'mall'],
            'Transport': ['uber', 'ola', 'petrol', 'fuel', 'taxi', 'bus', 'metro', 'train'],
            'Bills': ['electricity', 'phone', 'internet', 'bill', 'recharge', 'utility'],
            'Entertainment': ['netflix', 'spotify', 'movie', 'theatre', 'game', 'concert'],
            'Healthcare': ['medical', 'hospital', 'pharmacy', 'doctor', 'medicine'],
            'Salary': ['salary', 'income', 'payment', 'credit', 'bonus'],
            'Investment': ['investment', 'mutual fund', 'stocks', 'sip', 'trading'],
            'Education': ['education', 'course', 'tuition', 'school', 'college'],
            'Travel': ['flight', 'hotel', 'travel', 'booking', 'trip', 'vacation']
        }
    
    def train(self, df: pd.DataFrame, target_col: str = 'Category'):
        """Train the categorization model"""
        try:
            # Prepare data
            df_filtered = df[df[target_col].notna() & (df[target_col] != '')]
            
            if len(df_filtered) < 10:
                logger.warning("Insufficient training data, using rule-based only")
                self.is_trained = False
                return
            
            X = self.vectorizer.fit_transform(df_filtered['Description'].fillna(''))
            y = df_filtered[target_col].values
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.classifier.fit(X_train, y_train)
            
            # Evaluate
            accuracy = accuracy_score(y_test, self.classifier.predict(X_test))
            logger.info(f"Model trained with accuracy: {accuracy:.3f}")
            
            self.is_trained = True
            return accuracy
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.is_trained = False
    
    def categorize(self, description: str, amount: float = 0.0) -> Dict:
        """Categorize a single transaction"""
        description_lower = str(description).lower()
        
        # Try ML model first if trained
        if self.is_trained:
            try:
                X = self.vectorizer.transform([description])
                category = self.classifier.predict(X)[0]
                confidence = max(self.classifier.predict_proba(X)[0])
                
                # Fall back to rule-based if confidence is low
                if confidence < 0.3:
                    category = self._rule_based_categorize(description_lower)
                    confidence = 0.5
                
                return {
                    "category": category,
                    "confidence": float(confidence),
                    "method": "ml" if confidence >= 0.3 else "rule-based"
                }
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}, using rule-based")
        
        # Rule-based fallback
        category = self._rule_based_categorize(description_lower)
        return {
            "category": category,
            "confidence": 0.5,
            "method": "rule-based"
        }
    
    def _rule_based_categorize(self, description: str) -> str:
        """Rule-based categorization using keywords"""
        scores = {}
        for category, keywords in self.keyword_rules.items():
            score = sum(1 for keyword in keywords if keyword in description)
            if score > 0:
                scores[category] = score
        
        if scores:
            return max(scores, key=scores.get)
        return 'Other'
    
    def categorize_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize multiple transactions"""
        df_copy = df.copy()
        
        predictions = []
        for _, row in df_copy.iterrows():
            result = self.categorize(row['Description'], row.get('Amount', 0))
            predictions.append(result['category'])
        
        df_copy['Category'] = predictions
        return df_copy


class SpendingPredictor:
    """
    Time series forecasting for spending predictions
    """
    
    def __init__(self):
        self.model = None
    
    def predict(self, df: pd.DataFrame, days: int = 30) -> List[Dict]:
        """Predict future spending"""
        try:
            # Prepare time series data
            df_expenses = df[df['Transaction_Type'] == 'DEBIT'].copy()
            df_expenses['Date'] = pd.to_datetime(df_expenses['Date'])
            daily_spending = df_expenses.groupby(df_expenses['Date'].dt.date)['Amount'].sum().reset_index()
            daily_spending.columns = ['ds', 'y']
            
            if len(daily_spending) < 7:
                # Not enough data, return simple average
                avg_daily = daily_spending['y'].mean()
                return [{"date": f"Day {i+1}", "predicted": float(avg_daily)} for i in range(days)]
            
            # Train Prophet model
            model = Prophet(daily_seasonality=True, weekly_seasonality=True)
            model.fit(daily_spending)
            
            # Make predictions
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            # Get predictions for future dates
            predictions = forecast.tail(days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
            return [
                {
                    "date": row['ds'].strftime('%Y-%m-%d'),
                    "predicted": float(row['yhat']),
                    "lower_bound": float(row['yhat_lower']),
                    "upper_bound": float(row['yhat_upper'])
                }
                for _, row in predictions.iterrows()
            ]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Fallback to simple average
            avg_daily = df[df['Transaction_Type'] == 'DEBIT']['Amount'].mean()
            return [{"date": f"Day {i+1}", "predicted": float(avg_daily)} for i in range(days)]
    
    def get_trend(self, df: pd.DataFrame) -> str:
        """Get spending trend (increasing/decreasing/stable)"""
        try:
            df_expenses = df[df['Transaction_Type'] == 'DEBIT'].copy()
            df_expenses['Date'] = pd.to_datetime(df_expenses['Date'])
            monthly = df_expenses.groupby(df_expenses['Date'].dt.to_period('M'))['Amount'].sum()
            
            if len(monthly) < 2:
                return "insufficient_data"
            
            recent_avg = monthly.tail(3).mean()
            previous_avg = monthly.tail(6).head(3).mean() if len(monthly) >= 6 else monthly.head(3).mean()
            
            if recent_avg > previous_avg * 1.1:
                return "increasing"
            elif recent_avg < previous_avg * 0.9:
                return "decreasing"
            else:
                return "stable"
        except:
            return "unknown"


class AnomalyDetector:
    """
    Detect anomalous transactions using Isolation Forest
    """
    
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalous transactions"""
        try:
            df_copy = df.copy()
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
            
            # Feature engineering
            features = pd.DataFrame()
            features['amount'] = df_copy['Amount']
            features['amount_normalized'] = (df_copy['Amount'] - df_copy['Amount'].mean()) / df_copy['Amount'].std()
            features['day_of_week'] = df_copy['Date'].dt.dayofweek
            features['day_of_month'] = df_copy['Date'].dt.day
            features['is_weekend'] = df_copy['Date'].dt.dayofweek.isin([5, 6]).astype(int)
            
            # Fill NaN
            features = features.fillna(0)
            
            # Detect anomalies
            anomalies = self.model.fit_predict(features)
            df_copy['is_anomaly'] = anomalies == -1
            
            # Return only anomalies
            return df_copy[df_copy['is_anomaly']].drop(columns=['is_anomaly'])
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return pd.DataFrame()
    
    def calculate_risk_score(self, df: pd.DataFrame) -> float:
        """Calculate overall risk score (0-100)"""
        try:
            anomalies = self.detect(df)
            anomaly_ratio = len(anomalies) / len(df) if len(df) > 0 else 0
            risk_score = min(100, anomaly_ratio * 1000)  # Scale to 0-100
            return float(risk_score)
        except:
            return 0.0

