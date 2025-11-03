import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data cleaning, preprocessing, and standardization of financial transaction data.
    
    This class provides methods to:
    - Clean and standardize transaction data
    - Handle missing values and duplicates
    - Normalize transaction descriptions
    - Validate data integrity
    """
    
    def __init__(self):
        """Initialize the DataProcessor with default settings."""
        self.required_columns = ['Date', 'Description', 'Amount', 'Transaction_Type']
        self.amount_pattern = re.compile(r'[^\d.-]')
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load transaction data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded transaction data
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} transactions")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataframe contains required columns and basic data integrity.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        logger.info("Validating data structure...")
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for empty dataframe
        if df.empty:
            logger.error("Dataframe is empty")
            return False
        
        logger.info("Data validation passed")
        return True
    
    def clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize date column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with cleaned dates
        """
        logger.info("Cleaning date column...")
        
        # Convert to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Remove rows with invalid dates
        invalid_dates = df['Date'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Removing {invalid_dates} rows with invalid dates")
            df = df.dropna(subset=['Date'])
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"Date cleaning complete. Date range: {df['Date'].min()} to {df['Date'].max()}")
        return df
    
    def clean_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize amount column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with cleaned amounts
        """
        logger.info("Cleaning amount column...")
        
        # Convert to numeric, removing any non-numeric characters
        df['Amount'] = df['Amount'].astype(str).str.replace(self.amount_pattern, '', regex=True)
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # Remove rows with invalid amounts
        invalid_amounts = df['Amount'].isna().sum()
        if invalid_amounts > 0:
            logger.warning(f"Removing {invalid_amounts} rows with invalid amounts")
            df = df.dropna(subset=['Amount'])
        
        # Ensure amounts are positive
        df['Amount'] = df['Amount'].abs()
        
        logger.info(f"Amount cleaning complete. Amount range: ₹{df['Amount'].min():.2f} to ₹{df['Amount'].max():.2f}")
        return df
    
    def clean_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize transaction descriptions.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with cleaned descriptions
        """
        logger.info("Cleaning description column...")
        
        # Convert to string and handle NaN
        df['Description'] = df['Description'].astype(str)
        
        # Remove extra whitespace and convert to lowercase
        df['Description'] = df['Description'].str.strip().str.lower()
        
        # Remove special characters but keep spaces and basic punctuation
        df['Description'] = df['Description'].str.replace(r'[^\w\s\-\.]', '', regex=True)
        
        # Remove multiple spaces
        df['Description'] = df['Description'].str.replace(r'\s+', ' ', regex=True)
        
        logger.info("Description cleaning complete")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate transactions based on date, description, and amount.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with duplicates removed
        """
        logger.info("Removing duplicate transactions...")
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=['Date', 'Description', 'Amount'], keep='first')
        final_count = len(df)
        
        removed_count = initial_count - final_count
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate transactions")
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add useful derived features for analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        logger.info("Adding derived features...")
        
        # Extract year, month, day
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.day_name()
        
        # Add month-year for grouping
        df['MonthYear'] = df['Date'].dt.to_period('M')
        
        # Add transaction type (income/expense)
        df['Is_Expense'] = df['Transaction_Type'] == 'DEBIT'
        df['Is_Income'] = df['Transaction_Type'] == 'CREDIT'
        
        # Add amount categories for analysis
        df['Amount_Category'] = pd.cut(
            df['Amount'], 
            bins=[0, 100, 500, 1000, 5000, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        logger.info("Derived features added successfully")
        return df
    
    def process_data(self, file_path: str) -> pd.DataFrame:
        """
        Complete data processing pipeline.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Cleaned and processed transaction data
        """
        logger.info("Starting data processing pipeline...")
        
        # Load data
        df = self.load_data(file_path)
        
        # Validate data
        if not self.validate_data(df):
            raise ValueError("Data validation failed")
        
        # Clean data
        df = self.clean_dates(df)
        df = self.clean_amounts(df)
        df = self.clean_descriptions(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Add derived features
        df = self.add_derived_features(df)
        
        logger.info(f"Data processing complete. Final dataset: {len(df)} transactions")
        return df
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the processed data.
        
        Args:
            df (pd.DataFrame): Processed dataframe
            
        Returns:
            Dict: Summary statistics
        """
        stats = {
            'total_transactions': len(df),
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d'),
                'end': df['Date'].max().strftime('%Y-%m-%d')
            },
            'total_income': df[df['Is_Income']]['Amount'].sum(),
            'total_expenses': df[df['Is_Expense']]['Amount'].sum(),
            'net_savings': df[df['Is_Income']]['Amount'].sum() - df[df['Is_Expense']]['Amount'].sum(),
            'avg_transaction_amount': df['Amount'].mean(),
            'unique_categories': df['Category'].nunique() if 'Category' in df.columns else 0
        }
        
        return stats 