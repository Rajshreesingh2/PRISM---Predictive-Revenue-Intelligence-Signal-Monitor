"""Customer segmentation module for PRISM"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)


class Segmentation:
    """Perform customer segmentation"""
    
    def __init__(self, config):
        """Initialize segmentation"""
        self.config = config
    
    def analyze(self, df, features=['tenure', 'MonthlyCharges', 'TotalCharges']):
        """Execute segmentation analysis"""
        logger.info("Starting Customer Segmentation...")
        
        df = df.copy()
        
        # Filter available features
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) > 0:
            # Prepare data
            X = df[available_features].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            df['segment'] = kmeans.fit_predict(X_scaled)
            
            logger.info("  Segment characteristics:")
            for segment in range(4):
                segment_data = df[df['segment'] == segment]
                logger.info(f"    Segment {segment}: {len(segment_data)} customers")
                
                if 'Churn' in df.columns:
                    churn_rate = segment_data['Churn'].mean()
                    logger.info(f"      Churn rate: {churn_rate:.2%}")
                
                if 'MonthlyCharges' in segment_data.columns:
                    avg_charges = segment_data['MonthlyCharges'].mean()
                    logger.info(f"      Avg monthly charges: ${avg_charges:.2f}")
        
        logger.info("✅ Segmentation complete")
        
        return df
