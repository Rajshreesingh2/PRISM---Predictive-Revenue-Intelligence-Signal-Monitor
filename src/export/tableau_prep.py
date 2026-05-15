"""
Tableau data preparation
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


class TableauDataPreparer:
    """Prepare data for Tableau"""
    
    def __init__(self, config):
        self.config = config
    
    def prepare_customers_table(self, df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        """Prepare customer dimension table"""
        logger.info("Preparing customers table...")
        
        # Select key columns
        cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
               'tenure', 'Contract', 'InternetService', 'MonthlyCharges', 'Churn']
        cols = [col for col in cols if col in df.columns]
        
        customers = df[cols].copy()
        
        # Join predictions
        if 'customerID' in predictions.columns and 'churn_probability' in predictions.columns:
            customers = customers.merge(
                predictions[['customerID', 'churn_probability', 'churn_risk_tier']],
                on='customerID',
                how='left'
            )
        
        logger.info(f"✅ Customers table: {len(customers)} records")
        
        return customers
    
    def prepare_services_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare services dimension table"""
        logger.info("Preparing services table...")
        
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        service_cols = [col for col in service_cols if col in df.columns]
        
        services = df[['customerID'] + service_cols].copy()
        
        # Melt
        services = services.melt(
            id_vars=['customerID'],
            var_name='service_type',
            value_name='adopted'
        )
        
        logger.info(f"✅ Services table: {len(services)} records")
        
        return services
    
    def prepare_cohorts_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare cohort analysis table"""
        logger.info("Preparing cohorts table...")
        
        df_cohort = df.copy()
        df_cohort['cohort'] = pd.cut(df_cohort['tenure'], bins=12)
        
        cohorts = df_cohort.groupby('cohort').agg({
            'customerID': 'count',
            'Churn': ['sum', 'mean'],
            'MonthlyCharges': 'mean'
        }).reset_index()
        
        cohorts.columns = ['cohort', 'size', 'churned', 'churn_rate', 'avg_monthly_charges']
        
        logger.info(f"✅ Cohorts table: {len(cohorts)} records")
        
        return cohorts
    
    def prepare_segments_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare segment metrics table"""
        logger.info("Preparing segments table...")
        
        segments = df.groupby('Contract').agg({
            'customerID': 'count',
            'Churn': ['sum', 'mean'],
            'MonthlyCharges': 'mean',
            'tenure': 'mean'
        }).reset_index()
        
        segments.columns = ['contract_type', 'customers', 'churned', 'churn_rate', 
                          'avg_monthly_charges', 'avg_tenure']
        
        logger.info(f"✅ Segments table: {len(segments)} records")
        
        return segments
