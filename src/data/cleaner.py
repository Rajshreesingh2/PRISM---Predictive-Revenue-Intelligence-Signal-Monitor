"""
Data cleaner module for PRISM
"""

import logging
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess data"""
    
    def __init__(self, config):
        """Initialize data cleaner"""
        self.config = config
    
    def clean_data(self, df):
        """Execute complete cleaning pipeline"""
        logger.info("Starting data cleaning...")
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle outliers
        df = self._handle_outliers(df)
        
        # Standardize types
        df = self._standardize_types(df)
        
        logger.info(f"✅ Data cleaning complete: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        
        return df
    
    def _remove_duplicates(self, df):
        """Remove duplicate rows"""
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        
        if removed > 0:
            logger.info(f"  Removed {removed} duplicate rows")
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values"""
        # Check missing values
        missing = df.isnull().sum()
        
        if missing.sum() > 0:
            logger.info(f"  Found {missing.sum()} missing values:")
            for col in missing[missing > 0].index:
                # Drop rows with missing TotalCharges
                if col == 'TotalCharges':
                    df = df.dropna(subset=[col])
                    logger.info(f"    - {col}: {missing[col]} rows dropped")
                # Forward fill for other columns
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
                    logger.info(f"    - {col}: filled with mode")
        
        return df
    
    def _handle_outliers(self, df):
        """Handle outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                logger.info(f"  {col}: {len(outliers)} outliers detected (capping at bounds)")
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def _standardize_types(self, df):
        """Standardize column data types"""
        # Convert TotalCharges to numeric
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Convert Churn to binary
        if 'Churn' in df.columns:
            df['Churn'] = (df['Churn'] == 'Yes').astype(int)
        
        # Convert yes/no columns to binary
        yes_no_cols = df.select_dtypes(include=['object']).columns
        for col in yes_no_cols:
            if df[col].isin(['Yes', 'No']).all():
                df[col] = (df[col] == 'Yes').astype(int)
        
        logger.info("  Data types standardized")
        
        return df
