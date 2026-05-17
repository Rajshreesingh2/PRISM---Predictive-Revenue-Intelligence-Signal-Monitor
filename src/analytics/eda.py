"""Exploratory Data Analysis module for PRISM"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


class ExploratoryAnalysis:
    """Perform exploratory data analysis"""
    
    def __init__(self, config):
        """Initialize EDA"""
        self.config = config
        sns.set_style('whitegrid')
    
    def analyze(self, df):
        """Execute complete EDA pipeline"""
        logger.info("Starting Exploratory Data Analysis...")
        
        # Basic statistics
        self._basic_statistics(df)
        
        # Churn analysis
        self._churn_analysis(df)
        
        # Feature correlations
        self._correlation_analysis(df)
        
        logger.info("✅ EDA complete")
        
        return {
            'basic_stats': df.describe(),
            'churn_rate': (df['Churn'].sum() / len(df)) if 'Churn' in df.columns else 0
        }
    
    def _basic_statistics(self, df):
        """Calculate basic statistics"""
        logger.info("  Calculating basic statistics...")
        
        logger.info(f"    Dataset shape: {df.shape}")
        logger.info(f"    Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info(f"    Missing values: {df.isnull().sum().sum()}")
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df) > 0:
            logger.info(f"    Mean values:")
            for col in numeric_df.columns[:3]:
                logger.info(f"      {col}: {numeric_df[col].mean():.2f}")
    
    def _churn_analysis(self, df):
        """Analyze churn patterns"""
        if 'Churn' not in df.columns:
            return
        
        logger.info("  Analyzing churn patterns...")
        
        churn_rate = df['Churn'].mean()
        logger.info(f"    Overall churn rate: {churn_rate:.2%}")
        
        # Churn by contract type
        if 'Contract' in df.columns:
            churn_by_contract = df.groupby('Contract')['Churn'].agg(['mean', 'count'])
            logger.info(f"    Churn by contract type:")
            for contract, row in churn_by_contract.iterrows():
                logger.info(f"      {contract}: {row['mean']:.2%} (n={int(row['count'])})")
    
    def _correlation_analysis(self, df):
        """Analyze feature correlations"""
        logger.info("  Analyzing feature correlations...")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if 'Churn' in numeric_df.columns:
            correlations = numeric_df.corr()['Churn'].sort_values(ascending=False)
            logger.info(f"    Top features correlated with churn:")
            for feature, corr in correlations.head(5).items():
                if feature != 'Churn':
                    logger.info(f"      {feature}: {corr:.4f}")
