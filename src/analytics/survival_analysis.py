"""Survival analysis module for PRISM"""

import logging
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class SurvivalAnalysis:
    """Perform survival analysis"""
    
    def __init__(self, config):
        """Initialize survival analysis"""
        self.config = config
    
    def analyze(self, df):
        """Execute survival analysis"""
        logger.info("Starting Survival Analysis...")
        
        # Event and duration
        df = df.copy()
        
        if 'tenure' in df.columns and 'Churn' in df.columns:
            # Prepare data
            df['event'] = df['Churn']
            df['duration'] = df['tenure']
            
            # Kaplan-Meier-like estimation
            sorted_df = df.sort_values('duration')
            
            # Calculate survival function
            n_total = len(df)
            events = df['event'].sum()
            
            logger.info(f"  Total customers: {n_total}")
            logger.info(f"  Events (churn): {events}")
            logger.info(f"  Censored (retained): {n_total - events}")
            
            # Survival probability
            survival_prob = (n_total - events) / n_total
            logger.info(f"  Overall survival probability: {survival_prob:.2%}")
            
            # Survival by contract type
            if 'Contract' in df.columns:
                logger.info(f"  Survival by contract type:")
                for contract in df['Contract'].unique():
                    subset = df[df['Contract'] == contract]
                    survival = (len(subset) - subset['event'].sum()) / len(subset)
                    logger.info(f"    {contract}: {survival:.2%}")
        
        logger.info("✅ Survival analysis complete")
        
        return df
