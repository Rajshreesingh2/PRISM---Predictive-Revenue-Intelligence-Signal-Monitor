"""Cohort analysis module for PRISM"""

import logging
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class CohortAnalysis:
    """Perform cohort-based analysis"""
    
    def __init__(self, config):
        """Initialize cohort analysis"""
        self.config = config
    
    def analyze(self, df):
        """Execute cohort analysis"""
        logger.info("Starting Cohort Analysis...")
        
        # Create cohorts based on tenure
        df = df.copy()
        
        if 'tenure' in df.columns:
            df['cohort'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 72], labels=['0-6m', '6-12m', '12-24m', '24m+'])
            
            # Retention by cohort
            cohort_retention = df.groupby('cohort').agg({
                'Churn': ['sum', 'count'],
                'MonthlyCharges': 'mean',
                'TotalCharges': 'mean'
            })
            
            logger.info("  Retention metrics by cohort:")
            for cohort in ['0-6m', '6-12m', '12-24m', '24m+']:
                if cohort in cohort_retention.index:
                    churned = int(cohort_retention.loc[cohort, ('Churn', 'sum')])
                    total = int(cohort_retention.loc[cohort, ('Churn', 'count')])
                    retention = (1 - churned / total) if total > 0 else 0
                    logger.info(f"    {cohort}: {retention:.2%} retention (n={total})")
        
        logger.info("✅ Cohort analysis complete")
        
        return df
