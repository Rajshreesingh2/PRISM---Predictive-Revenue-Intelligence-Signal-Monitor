"""Feature engineering module for PRISM"""

import logging
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create and transform features"""
    
    def __init__(self, config):
        """Initialize feature engineer"""
        self.config = config
    
    def engineer_features(self, df):
        """Execute complete feature engineering pipeline"""
        logger.info("Starting feature engineering...")
        
        df = df.copy()
        
        # Numeric features
        df = self._create_charge_ratios(df)
        df = self._create_tenure_features(df)
        
        # Categorical features
        df = self._encode_categorical(df)
        
        # Service adoption
        df = self._create_service_features(df)
        
        # Interaction features
        df = self._create_interaction_features(df)
        
        logger.info(f"✅ Feature engineering complete: {df.shape[1]} features created")
        
        return df
    
    def _create_charge_ratios(self, df):
        """Create charge-based features"""
        if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns:
            df['MonthlyCharges_log'] = np.log1p(df['MonthlyCharges'])
            df['TotalCharges_log'] = np.log1p(df['TotalCharges'])
            df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)
            logger.info("  Created charge ratio features")
        
        return df
    
    def _create_tenure_features(self, df):
        """Create tenure-based features"""
        if 'tenure' in df.columns:
            df['tenure_log'] = np.log1p(df['tenure'])
            df['tenure_months_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 60, 72], labels=['0-6', '6-12', '12-24', '24-60', '60+'])
            df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
            df['is_at_risk'] = (df['tenure'] <= 12).astype(int)
            logger.info("  Created tenure-based features")
        
        return df
    
    def _encode_categorical(self, df):
        """Encode categorical features"""
        # Target encoding for Contract
        if 'Contract' in df.columns:
            contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
            df['Contract_encoded'] = df['Contract'].map(contract_map)
        
        # Internet Service encoding
        if 'InternetService' in df.columns:
            internet_map = {'Fiber optic': 2, 'DSL': 1, 'No': 0}
            df['InternetService_encoded'] = df['InternetService'].map(internet_map)
        
        logger.info("  Encoded categorical features")
        
        return df
    
    def _create_service_features(self, df):
        """Create service adoption features"""
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Count active services
        active_services = []
        for col in service_cols:
            if col in df.columns:
                active_services.append((df[col] == 'Yes').astype(int))
        
        if active_services:
            df['num_services'] = sum(active_services)
            logger.info("  Created service adoption features")
        
        return df
    
    def _create_interaction_features(self, df):
        """Create interaction features"""
        if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
            df['tenure_x_charges'] = df['tenure'] * df['MonthlyCharges']
        
        if 'num_services' in df.columns and 'tenure' in df.columns:
            df['services_x_tenure'] = df['num_services'] * df['tenure']
        
        logger.info("  Created interaction features")
        
        return df
