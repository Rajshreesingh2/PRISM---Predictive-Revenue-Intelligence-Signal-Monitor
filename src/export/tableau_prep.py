"""Tableau data preparation module for PRISM"""

import logging
import pandas as pd
from pathlib import Path


logger = logging.getLogger(__name__)


class TableauPrep:
    """Prepare data for Tableau"""
    
    def __init__(self, config):
        """Initialize Tableau prep"""
        self.config = config
    
    def prepare_data(self, df, predictions):
        """Prepare all Tableau data sources"""
        logger.info("Preparing Tableau data sources...")
        
        output_dir = self.config.data_dir / 'export'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data source 1: Customers with risk scores
        customers_df = df[['customerID', 'gender', 'SeniorCitizen', 'Contract', 'InternetService']].copy()
        customers_df['churn_probability'] = predictions
        customers_df['risk_level'] = pd.cut(predictions, bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])
        customers_df.to_csv(output_dir / 'tableau_customers.csv', index=False)
        logger.info("  ✓ Customers data source created")
        
        # Data source 2: Services usage
        services_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        available_services = [col for col in services_cols if col in df.columns]
        
        if available_services:
            services_df = df[['customerID', 'Contract'] + available_services].copy()
            services_df.to_csv(output_dir / 'tableau_services.csv', index=False)
            logger.info("  ✓ Services data source created")
        
        # Data source 3: Cohort analysis
        cohorts_df = df[['customerID', 'tenure', 'MonthlyCharges', 'Churn', 'Contract']].copy()
        cohorts_df['tenure_cohort'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 72], labels=['0-6m', '6-12m', '12-24m', '24m+'])
        cohorts_df.to_csv(output_dir / 'tableau_cohorts.csv', index=False)
        logger.info("  ✓ Cohorts data source created")
        
        # Data source 4: Segment performance
        segments_df = df.groupby('Contract').agg({
            'customerID': 'count',
            'Churn': 'mean',
            'MonthlyCharges': 'mean',
            'tenure': 'mean'
        }).reset_index()
        segments_df.columns = ['Contract', 'total_customers', 'churn_rate', 'avg_monthly_charges', 'avg_tenure']
        segments_df.to_csv(output_dir / 'tableau_segments.csv', index=False)
        logger.info("  ✓ Segments data source created")
        
        logger.info("✅ Tableau data preparation complete")
        
        return {
            'customers': output_dir / 'tableau_customers.csv',
            'services': output_dir / 'tableau_services.csv',
            'cohorts': output_dir / 'tableau_cohorts.csv',
            'segments': output_dir / 'tableau_segments.csv'
        }
