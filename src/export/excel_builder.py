"""Excel export module for PRISM"""

import logging
import pandas as pd
from pathlib import Path


logger = logging.getLogger(__name__)


class ExcelBuilder:
    """Build professional Excel workbooks"""
    
    def __init__(self, config):
        """Initialize Excel builder"""
        self.config = config
    
    def build_executive_summary(self, df, predictions, results):
        """Build executive summary workbook"""
        logger.info("Building executive summary workbook...")
        
        output_path = self.config.data_dir / 'export' / 'PRISM_Executive_Summary.xlsx'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: KPIs
            kpis = pd.DataFrame({
                'Metric': ['Total Customers', 'Churn Rate', 'At-Risk Customers', 'Revenue at Risk'],
                'Value': [
                    len(df),
                    f"{df['Churn'].mean():.2%}",
                    (predictions >= 0.5).sum(),
                    f"${(predictions * df['MonthlyCharges'].sum() / 100):.0f}"
                ]
            })
            kpis.to_excel(writer, sheet_name='KPIs', index=False)
            
            # Sheet 2: Churn by segment
            if 'Contract' in df.columns:
                churn_by_contract = df.groupby('Contract').agg({
                    'Churn': ['count', 'sum', 'mean']
                }).round(3)
                churn_by_contract.to_excel(writer, sheet_name='Churn Analysis')
            
            # Sheet 3: At-risk customers
            at_risk = df.copy()
            at_risk['churn_probability'] = predictions
            at_risk_top = at_risk.nlargest(100, 'churn_probability')[['customerID', 'Contract', 'MonthlyCharges', 'churn_probability']]
            at_risk_top.to_excel(writer, sheet_name='At-Risk Customers', index=False)
        
        logger.info(f"✅ Executive summary saved: {output_path}")
        
        return output_path
    
    def build_detailed_analysis(self, df, predictions):
        """Build detailed analysis workbook"""
        logger.info("Building detailed analysis workbook...")
        
        output_path = self.config.data_dir / 'export' / 'PRISM_Detailed_Analysis.xlsx'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: All predictions
            results_df = df.copy()
            results_df['churn_probability'] = predictions
            results_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Sheet 2: Summary statistics
            summary = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Churn Probability': [
                    predictions.mean(),
                    pd.Series(predictions).median(),
                    predictions.std(),
                    predictions.min(),
                    predictions.max()
                ]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"✅ Detailed analysis saved: {output_path}")
        
        return output_path
