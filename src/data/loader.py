"""Data loader module for PRISM"""

import logging
import pandas as pd
from pathlib import Path


logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate data"""
    
    def __init__(self, config):
        """Initialize data loader"""
        self.config = config
    
    def load_telco_data(self):
        """Load Telco Customer Churn dataset"""
        # Check if CSV exists, if not create sample data
        data_file = self.config.data_dir / 'raw' / 'telco_customer_churn.csv'
        
        if data_file.exists():
            logger.info(f"Loading data from {data_file}")
            df = pd.read_csv(data_file)
        else:
            logger.warning(f"Data file not found at {data_file}")
            logger.info("Generating sample Telco data...")
            df = self._generate_sample_telco_data()
        
        logger.info(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        logger.info(f"✅ Columns: {', '.join(df.columns.tolist()[:8])}...")
        
        return df
    
    def _generate_sample_telco_data(self):
        """Generate sample Telco dataset for testing"""
        import numpy as np
        
        np.random.seed(42)
        n_samples = 7043
        
        data = {
            'customerID': [f'ID-{i:05d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(0, 72, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'InternetService': np.random.choice(['Fiber optic', 'DSL', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
            'MonthlyCharges': np.random.uniform(20, 150, n_samples).round(2),
            'TotalCharges': np.random.uniform(100, 8000, n_samples).round(2),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.265, 0.735])
        }
        
        df = pd.DataFrame(data)
        
        # Create output directory
        output_file = self.config.data_dir / 'raw' / 'telco_customer_churn.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Generated sample data: {output_file}")
        
        return df
