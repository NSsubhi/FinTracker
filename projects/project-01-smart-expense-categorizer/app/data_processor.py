"""
Smart Data Processor - Automatically detects and maps various CSV formats
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and clean transaction data with automatic format detection"""
    
    def __init__(self):
        # Common column name variations
        self.date_variations = [
            'date', 'transaction_date', 'trans_date', 'date/time', 'date_time',
            'posting_date', 'value_date', 'txn_date', 'transactiondate'
        ]
        self.description_variations = [
            'description', 'transaction_description', 'details', 'narration',
            'memo', 'remarks', 'transaction_details', 'particulars', 'notes',
            'transaction_narration', 'description/narration', 'payment_details'
        ]
        self.amount_variations = [
            'amount', 'transaction_amount', 'txn_amount', 'value', 'sum',
            'transaction_value', 'amt', 'transaction_amt'
        ]
        self.debit_variations = ['debit', 'withdrawal', 'paid', 'out', 'expense', 'spent']
        self.credit_variations = ['credit', 'deposit', 'received', 'in', 'income', 'earned']
        self.type_variations = [
            'type', 'transaction_type', 'txn_type', 'transactiontype',
            'category', 'transaction_category'
        ]
    
    def detect_column_mapping(self, df: pd.DataFrame) -> dict:
        """Automatically detect column mappings"""
        df.columns = df.columns.str.strip().str.lower()
        
        mapping = {
            'date': None,
            'description': None,
            'amount': None,
            'debit': None,
            'credit': None,
            'type': None
        }
        
        # Find date column
        for col in df.columns:
            if any(variation in col for variation in self.date_variations):
                mapping['date'] = col
                break
        
        # Find description column
        for col in df.columns:
            if any(variation in col for variation in self.description_variations):
                mapping['description'] = col
                break
        
        # Find amount column
        for col in df.columns:
            if any(variation in col for variation in self.amount_variations):
                mapping['amount'] = col
                break
        
        # Find debit/credit columns
        for col in df.columns:
            col_lower = col.lower()
            if any(variation in col_lower for variation in self.debit_variations):
                mapping['debit'] = col
            elif any(variation in col_lower for variation in self.credit_variations):
                mapping['credit'] = col
        
        # Find type column
        for col in df.columns:
            if any(variation in col for variation in self.type_variations):
                mapping['type'] = col
                break
        
        # Fallback: try to guess from column names
        if not mapping['date']:
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    mapping['date'] = col
                    break
        
        if not mapping['description']:
            for col in df.columns:
                if any(word in col.lower() for word in ['desc', 'detail', 'narration', 'memo', 'note', 'particular']):
                    mapping['description'] = col
                    break
        
        if not mapping['amount']:
            for col in df.columns:
                if any(word in col.lower() for word in ['amount', 'amt', 'value', 'sum', 'total']):
                    # Don't use if it's clearly balance
                    if 'balance' not in col.lower():
                        mapping['amount'] = col
                        break
        
        return mapping
    
    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize dataframe with automatic column detection"""
        df_copy = df.copy()
        
        # Detect column mapping
        mapping = self.detect_column_mapping(df_copy)
        
        logger.info(f"Detected column mapping: {mapping}")
        
        # Create normalized columns
        normalized_df = pd.DataFrame()
        
        # Map date
        if mapping['date']:
            normalized_df['Date'] = df_copy[mapping['date']]
        else:
            # Try to find any date-like column
            for col in df_copy.columns:
                try:
                    # Try to parse as date
                    test_date = pd.to_datetime(df_copy[col].head(5), errors='coerce')
                    if test_date.notna().sum() >= 3:  # If at least 3 valid dates
                        normalized_df['Date'] = df_copy[col]
                        logger.info(f"Auto-detected date column: {col}")
                        break
                except:
                    pass
        
        if 'Date' not in normalized_df.columns:
            raise ValueError("Could not find date column. Please ensure your CSV has a date column.")
        
        # Map description
        if mapping['description']:
            normalized_df['Description'] = df_copy[mapping['description']]
        else:
            # Use first text column as fallback
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object' and col.lower() not in ['date', 'amount', 'debit', 'credit']:
                    normalized_df['Description'] = df_copy[col]
                    logger.info(f"Auto-detected description column: {col}")
                    break
        
        if 'Description' not in normalized_df.columns:
            raise ValueError("Could not find description column.")
        
        # Map amount - handle debit/credit columns or single amount column
        if mapping['debit'] and mapping['credit']:
            # Has separate debit and credit columns
            debit_col = mapping['debit']
            credit_col = mapping['credit']
            
            # Combine debit (negative) and credit (positive)
            normalized_df['Amount'] = df_copy[credit_col].fillna(0) - df_copy[debit_col].fillna(0)
            normalized_df['Transaction_Type'] = normalized_df['Amount'].apply(
                lambda x: 'CREDIT' if x > 0 else 'DEBIT'
            )
        elif mapping['amount']:
            # Single amount column
            normalized_df['Amount'] = df_copy[mapping['amount']]
            
            # Determine transaction type
            if mapping['type']:
                # Use type column if available
                type_col = df_copy[mapping['type']].astype(str).str.upper()
                normalized_df['Transaction_Type'] = type_col.map({
                    'DEBIT': 'DEBIT', 'D': 'DEBIT', 'WITHDRAWAL': 'DEBIT',
                    'CREDIT': 'CREDIT', 'C': 'CREDIT', 'DEPOSIT': 'CREDIT',
                    'INCOME': 'CREDIT', 'EXPENSE': 'DEBIT'
                }).fillna('DEBIT')
            else:
                # Infer from amount sign
                normalized_df['Transaction_Type'] = normalized_df['Amount'].apply(
                    lambda x: 'CREDIT' if x > 0 else 'DEBIT'
                )
        elif mapping['debit']:
            # Only debit column
            normalized_df['Amount'] = df_copy[mapping['debit']]
            normalized_df['Transaction_Type'] = 'DEBIT'
        elif mapping['credit']:
            # Only credit column
            normalized_df['Amount'] = df_copy[mapping['credit']]
            normalized_df['Transaction_Type'] = 'CREDIT'
        else:
            # Try to find any numeric column that could be amount
            for col in df_copy.columns:
                if col.lower() not in ['balance', 'available']:
                    try:
                        test_amt = pd.to_numeric(df_copy[col].head(10), errors='coerce')
                        if test_amt.notna().sum() >= 5:
                            normalized_df['Amount'] = df_copy[col]
                            normalized_df['Transaction_Type'] = normalized_df['Amount'].apply(
                                lambda x: 'CREDIT' if x > 0 else 'DEBIT'
                            )
                            logger.info(f"Auto-detected amount column: {col}")
                            break
                    except:
                        pass
        
        if 'Amount' not in normalized_df.columns:
            raise ValueError("Could not find amount column. Please ensure your CSV has an amount, debit, or credit column.")
        
        return normalized_df
    
    def clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize date column"""
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
        
        # Remove rows with invalid dates
        invalid_dates = df['Date'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Removing {invalid_dates} rows with invalid dates")
            df = df.dropna(subset=['Date'])
        
        return df
    
    def clean_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize amount column"""
        # Convert to string first to handle currency symbols
        df['Amount'] = df['Amount'].astype(str)
        
        # Remove currency symbols, commas, and other non-numeric characters (except minus)
        df['Amount'] = df['Amount'].str.replace(r'[^\d.-]', '', regex=True)
        
        # Convert to numeric
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # Remove rows with invalid amounts
        invalid_amounts = df['Amount'].isna().sum()
        if invalid_amounts > 0:
            logger.warning(f"Removing {invalid_amounts} rows with invalid amounts")
            df = df.dropna(subset=['Amount'])
        
        # Ensure amounts are positive (sign handled by Transaction_Type)
        df['Amount'] = df['Amount'].abs()
        
        return df
    
    def clean_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize description column"""
        df['Description'] = df['Description'].astype(str)
        df['Description'] = df['Description'].str.strip()
        # Keep original case for better ML classification
        
        return df
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete data processing pipeline with automatic format detection"""
        logger.info(f"Processing dataframe with {len(df)} rows and columns: {list(df.columns)}")
        
        # Step 1: Normalize column names and detect mapping
        normalized_df = self.normalize_dataframe(df)
        
        # Step 2: Clean data
        normalized_df = self.clean_dates(normalized_df)
        normalized_df = self.clean_amounts(normalized_df)
        normalized_df = self.clean_descriptions(normalized_df)
        
        # Ensure Transaction_Type exists
        if 'Transaction_Type' not in normalized_df.columns:
            normalized_df['Transaction_Type'] = 'DEBIT'
        
        # Sort by date
        normalized_df = normalized_df.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"Processing complete. Final dataset: {len(normalized_df)} transactions")
        return normalized_df
