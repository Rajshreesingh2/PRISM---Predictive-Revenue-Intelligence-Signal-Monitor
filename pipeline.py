"""Complete PRISM Pipeline Orchestrator"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.config import Config
from utils.logger import setup_logging
from data.loader import DataLoader
from data.cleaner import DataCleaner
from ml.feature_engineering import FeatureEngineer
from ml.modeling import MLModeler
from analytics.eda import ExploratoryAnalysis
from analytics.cohort_analysis import CohortAnalysis
from analytics.survival_analysis import SurvivalAnalysis
from analytics.segmentation import Segmentation
from sql.duckdb_queries import DuckDBAnalytics
from export.excel_builder import ExcelBuilder
from export.tableau_prep import TableauPrep


def run_pipeline():
    """Execute complete PRISM pipeline"""
    
    # Setup
    config = Config('config.yaml')
    logger = setup_logging('PRISM')
    
    logger.info("="*80)
    logger.info("🚀 PRISM - Predictive Revenue Intelligence Signal Monitor")
    logger.info(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    try:
        # Stage 1: Data Loading
        logger.info("\n📦 STAGE 1: Data Loading")
        loader = DataLoader(config)
        df = loader.load_telco_data()
        
        # Stage 2: Data Cleaning
        logger.info("\n🧹 STAGE 2: Data Cleaning")
        cleaner = DataCleaner(config)
        df = cleaner.clean_data(df)
        
        # Stage 3: Feature Engineering
        logger.info("\n⚙️  STAGE 3: Feature Engineering")
        engineer = FeatureEngineer(config)
        df = engineer.engineer_features(df)
        
        # Stage 4: EDA
        logger.info("\n📊 STAGE 4: Exploratory Data Analysis")
        eda = ExploratoryAnalysis(config)
        eda_results = eda.analyze(df)
        
        # Stage 5: Cohort Analysis
        logger.info("\n👥 STAGE 5: Cohort Analysis")
        cohort = CohortAnalysis(config)
        df = cohort.analyze(df)
        
        # Stage 6: Survival Analysis
        logger.info("\n⏱️  STAGE 6: Survival Analysis")
        survival = SurvivalAnalysis(config)
        df = survival.analyze(df)
        
        # Stage 7: Segmentation
        logger.info("\n🧬 STAGE 7: Customer Segmentation")
        segmentation = Segmentation(config)
        df = segmentation.analyze(df)
        
        # Stage 8: ML Modeling
        logger.info("\n🤖 STAGE 8: Machine Learning Modeling")
        modeler = MLModeler(config)
        
        # Prepare features for modeling
        feature_cols = [col for col in df.columns if col not in ['customerID', 'Churn', 'cohort', 'segment']]
        X = df[feature_cols].fillna(0)
        y = df['Churn']
        
        best_model, X_test, y_test = modeler.train_models(X, y)
        
        # Generate predictions
        _, predictions, risk_tiers = modeler.generate_predictions(best_model, X)
        
        # Stage 9: SQL Analytics
        logger.info("\n🗄️  STAGE 9: SQL Analytics (DuckDB)")
        sql_analytics = DuckDBAnalytics(config)
        sql_results = sql_analytics.run_analytics(df)
        
        # Stage 10: Excel Export
        logger.info("\n📊 STAGE 10: Excel Workbook Generation")
        excel = ExcelBuilder(config)
        excel.build_executive_summary(df, predictions, modeler.results)
        excel.build_detailed_analysis(df, predictions)
        
        # Stage 11: Tableau Prep
        logger.info("\n📊 STAGE 11: Tableau Data Preparation")
        tableau = TableauPrep(config)
        tableau_files = tableau.prepare_data(df, predictions)
        
        # Save processed data
        processed_path = config.data_dir / 'processed' / 'features_engineered.csv'
        df.to_csv(processed_path, index=False)
        logger.info(f"  Processed data saved: {processed_path}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("\n📂 Output Files:")
        logger.info(f"  ✓ Processed Data: {processed_path}")
        logger.info(f"  ✓ Executive Summary: {config.data_dir / 'export' / 'PRISM_Executive_Summary.xlsx'}")
        logger.info(f"  ✓ Detailed Analysis: {config.data_dir / 'export' / 'PRISM_Detailed_Analysis.xlsx'}")
        logger.info(f"  ✓ Tableau Data Sources: {config.data_dir / 'export'}/tableau_*.csv")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    run_pipeline()
