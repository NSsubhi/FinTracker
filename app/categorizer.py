import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionCategorizer:
    """
    Automatically categorizes financial transactions using rule-based and ML approaches.
    
    This class provides:
    - Rule-based categorization using keyword matching
    - Machine learning-based categorization using TF-IDF and Random Forest
    - Training and prediction capabilities
    - Model persistence and loading
    """
    
    def __init__(self):
        """Initialize the categorizer with default categories and rules."""
        self.categories = [
            'Food', 'Shopping', 'Transport', 'Bills', 'Entertainment', 
            'Healthcare', 'Salary', 'Investment', 'Education', 'Travel'
        ]
        
        # Rule-based keyword mappings
        self.keyword_rules = {
            'Food': [
                'zomato', 'swiggy', 'restaurant', 'food', 'lunch', 'dinner', 
                'breakfast', 'groceries', 'vegetables', 'mcdonalds', 'pizza',
                'burger', 'cafe', 'coffee', 'tea', 'snacks', 'bakery'
            ],
            'Shopping': [
                'amazon', 'flipkart', 'clothing', 'electronics', 'books',
                'shopping', 'purchase', 'buy', 'store', 'mall', 'market',
                'apparel', 'accessories', 'home', 'furniture', 'appliances'
            ],
            'Transport': [
                'uber', 'ola', 'petrol', 'fuel', 'gas', 'transport', 'ride',
                'taxi', 'bus', 'metro', 'train', 'parking', 'toll', 'maintenance'
            ],
            'Bills': [
                'electricity', 'phone', 'internet', 'water', 'gas', 'bill',
                'recharge', 'subscription', 'netflix', 'spotify', 'utility'
            ],
            'Entertainment': [
                'netflix', 'spotify', 'movie', 'theatre', 'bookmyshow',
                'entertainment', 'game', 'concert', 'show', 'ticket'
            ],
            'Healthcare': [
                'medical', 'medicine', 'hospital', 'doctor', 'pharmacy',
                'healthcare', 'consultation', 'treatment', 'therapy'
            ],
            'Salary': [
                'salary', 'credit', 'income', 'payment', 'wage', 'bonus',
                'commission', 'refund', 'reimbursement'
            ],
            'Investment': [
                'investment', 'mutual fund', 'stocks', 'bonds', 'sip',
                'trading', 'portfolio', 'dividend', 'interest'
            ],
            'Education': [
                'education', 'course', 'training', 'tuition', 'school',
                'college', 'university', 'books', 'stationery'
            ],
            'Travel': [
                'travel', 'flight', 'hotel', 'booking', 'trip', 'vacation',
                'airline', 'accommodation', 'tour', 'tourism'
            ]
        }
        
        # Initialize ML components
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def rule_based_categorize(self, description: str) -> str:
        """
        Categorize transaction using rule-based keyword matching.
        
        Args:
            description (str): Transaction description
            
        Returns:
            str: Predicted category
        """
        description_lower = description.lower()
        
        # Count matches for each category
        category_scores = {}
        for category, keywords in self.keyword_rules.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            category_scores[category] = score
        
        # Return category with highest score, or 'Other' if no matches
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        else:
            return 'Other'
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for ML model.
        
        Args:
            df (pd.DataFrame): Dataframe with 'Description' and 'Category' columns
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels
        """
        # Filter out rows without categories
        df_filtered = df[df['Category'].notna() & (df['Category'] != '')].copy()
        
        if df_filtered.empty:
            raise ValueError("No labeled data found for training")
        
        # Prepare features (descriptions)
        descriptions = df_filtered['Description'].fillna('').astype(str)
        X = self.vectorizer.fit_transform(descriptions)
        
        # Prepare labels (categories)
        y = df_filtered['Category'].values
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train the ML categorization model.
        
        Args:
            df (pd.DataFrame): Training data with 'Description' and 'Category' columns
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Dict: Training results and metrics
        """
        logger.info("Training ML categorization model...")
        
        # Prepare training data
        X, y = self.prepare_training_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.classifier.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.is_trained = True
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'categories': list(set(y))
        }
        
        logger.info(f"Model training complete. Accuracy: {accuracy:.3f}")
        return results
    
    def predict_category(self, description: str, use_ml: bool = True) -> str:
        """
        Predict category for a transaction description.
        
        Args:
            description (str): Transaction description
            use_ml (bool): Whether to use ML model (if trained) or rule-based
            
        Returns:
            str: Predicted category
        """
        if use_ml and self.is_trained:
            # Use ML model
            X = self.vectorizer.transform([description])
            prediction = self.classifier.predict(X)[0]
            confidence = max(self.classifier.predict_proba(X)[0])
            
            # If confidence is low, fall back to rule-based
            if confidence < 0.3:
                logger.info(f"Low ML confidence ({confidence:.3f}), using rule-based")
                return self.rule_based_categorize(description)
            
            return prediction
        else:
            # Use rule-based approach
            return self.rule_based_categorize(description)
    
    def categorize_transactions(self, df: pd.DataFrame, use_ml: bool = True) -> pd.DataFrame:
        """
        Categorize all transactions in a dataframe.
        
        Args:
            df (pd.DataFrame): Dataframe with transaction data
            use_ml (bool): Whether to use ML model for categorization
            
        Returns:
            pd.DataFrame: Dataframe with predicted categories
        """
        logger.info(f"Categorizing {len(df)} transactions...")
        
        df_copy = df.copy()
        
        # Predict categories for each transaction
        predictions = []
        for idx, row in df_copy.iterrows():
            description = row['Description']
            predicted_category = self.predict_category(description, use_ml)
            predictions.append(predicted_category)
        
        df_copy['Predicted_Category'] = predictions
        
        # Update Category column if it doesn't exist or is empty
        if 'Category' not in df_copy.columns:
            df_copy['Category'] = predictions
        else:
            # Fill empty categories with predictions
            mask = df_copy['Category'].isna() | (df_copy['Category'] == '')
            df_copy.loc[mask, 'Category'] = df_copy.loc[mask, 'Predicted_Category']
        
        logger.info("Transaction categorization complete")
        return df_copy
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            logger.warning("No trained model to save")
            return
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'categories': self.categories,
            'keyword_rules': self.keyword_rules
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.categories = model_data['categories']
        self.keyword_rules = model_data['keyword_rules']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_category_insights(self, df: pd.DataFrame) -> Dict:
        """
        Generate insights about transaction categories.
        
        Args:
            df (pd.DataFrame): Dataframe with categorized transactions
            
        Returns:
            Dict: Category insights and statistics
        """
        if 'Category' not in df.columns:
            return {}
        
        insights = {
            'category_distribution': df['Category'].value_counts().to_dict(),
            'category_amounts': df.groupby('Category')['Amount'].sum().to_dict(),
            'avg_amount_by_category': df.groupby('Category')['Amount'].mean().to_dict(),
            'total_transactions': len(df),
            'unique_categories': df['Category'].nunique()
        }
        
        return insights 