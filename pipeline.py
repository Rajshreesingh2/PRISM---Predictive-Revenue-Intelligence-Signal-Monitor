"""
PRISM Data Analyst Edition - Complete Analysis Pipeline
========================================================

Execute the full data science pipeline:
1. Data loading & cleaning
2. Feature engineering
3. ML model training
4. SQL analytics queries
5. Excel workbook generation
6. Tableau data prep
7. Dashboard deployment

Usage:
    python pipeline.py --full
    python pipeline.py --stage eda
    python pipeline.py --stage ml
    python pipeline.py --stage export
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.loader import DataLoader
from data.cleaner import DataCleaner
from ml.feature_engineering import FeatureEngineer
from ml.modeling import ModelPipeline
from analytics.eda import ExploratorDataAnalysis
from analytics.cohort_analysis import CohortAnalysis
from analytics.survival_analysis import SurvivalAnalysis
from analytics.segmentation import ChurnSegmentation
from sql.duckdb_queries import DuckDBAnalytics
from export.excel_builder import ExcelReportBuilder
from export.tableau_prep import TableauDataPreparer
from utils.config import Config
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class PRISMPipeline:
    """Main PRISM execution pipeline"""
    
    def __init__(self, config_path='configs/analysis_config.yaml'):
        """Initialize pipeline with configuration"""
        self.config = Config(config_path)
        self.start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("PRISM DATA ANALYST EDITION - COMPLETE PIPELINE")
        logger.info("=" * 80)
        
    def stage_data_loading(self):
        """Stage 1: Load and validate data"""
        logger.info("\n📦 STAGE 1: DATA LOADING & VALIDATION")
        logger.info("-" * 80)
        
        loader = DataLoader(self.config)
        df = loader.load_telco_data()
        
        logger.info(f"✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        logger.info(f"✅ Columns: {', '.join(df.columns.tolist()[:5])}...")
        logger.info(f"✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
    
    def stage_data_cleaning(self, df):
        """Stage 2: Clean and preprocess data"""
        logger.info("\n🧹 STAGE 2: DATA CLEANING & PREPROCESSING")
        logger.info("-" * 80)
        
        cleaner = DataCleaner(self.config)
        df_clean = cleaner.clean_data(df)
        
        logger.info(f"✅ Data cleaned: {df_clean.shape[0]:,} rows")
        logger.info(f"✅ Missing values: {df_clean.isnull().sum().sum()}")
        logger.info(f"✅ Duplicates removed: {(df.shape[0] - df_clean.shape[0])}")
        
        # Save cleaned data
        clean_path = self.config.data_dir / 'processed' / 'telco_cleaned.csv'
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(clean_path, index=False)
        logger.info(f"✅ Saved cleaned data: {clean_path}")
        
        return df_clean
    
    def stage_feature_engineering(self, df):
        """Stage 3: Engineer features"""
        logger.info("\n⚙️  STAGE 3: FEATURE ENGINEERING")
        logger.info("-" * 80)
        
        engineer = FeatureEngineer(self.config)
        df_features = engineer.create_features(df)
        
        logger.info(f"✅ Features engineered: {df_features.shape[1]:,} total columns")
        logger.info(f"✅ New features created: {df_features.shape[1] - df.shape[1]}")
        
        # Save engineered features
        features_path = self.config.data_dir / 'processed' / 'features_engineered.csv'
        df_features.to_csv(features_path, index=False)
        logger.info(f"✅ Saved engineered features: {features_path}")
        
        return df_features
    
    def stage_exploratory_analysis(self, df):
        """Stage 4: Exploratory Data Analysis"""
        logger.info("\n📊 STAGE 4: EXPLORATORY DATA ANALYSIS (EDA)")
        logger.info("-" * 80)
        
        eda = ExploratorDataAnalysis(self.config)
        
        # Distribution analysis
        logger.info("📈 Analyzing distributions...")
        dist_report = eda.distribution_analysis(df)
        
        # Correlation analysis
        logger.info("🔗 Computing correlations...")
        corr_report = eda.correlation_analysis(df)
        
        # Churn analysis
        logger.info("🎯 Analyzing churn patterns...")
        churn_analysis = eda.churn_analysis(df)
        
        logger.info(f"✅ Overall churn rate: {churn_analysis['churn_rate']:.2%}")
        logger.info(f"✅ Top churn driver: {churn_analysis['top_driver']['feature']}")
        logger.info(f"✅ Churn rate spread: {churn_analysis['top_driver']['churn_rate']:.2%}")
        
        # Save EDA results
        eda_path = self.config.data_dir / 'analytics' / 'eda_results.pkl'
        eda_path.parent.mkdir(parents=True, exist_ok=True)
        
        return {
            'distributions': dist_report,
            'correlations': corr_report,
            'churn': churn_analysis
        }
    
    def stage_cohort_analysis(self, df):
        """Stage 5: Cohort retention analysis"""
        logger.info("\n👥 STAGE 5: COHORT ANALYSIS")
        logger.info("-" * 80)
        
        cohort = CohortAnalysis(self.config)
        
        # Create cohorts
        logger.info("🔍 Creating cohorts...")
        cohort_retention = cohort.retention_heatmap(df)
        cohort_ltv = cohort.ltv_by_cohort(df)
        
        logger.info(f"✅ Cohorts created: {cohort_retention.shape[0]}")
        logger.info(f"✅ Retention range: {cohort_retention.min().min():.2%} - {cohort_retention.max().max():.2%}")
        
        # Save cohort analysis
        cohort_path = self.config.data_dir / 'analytics' / 'cohort_retention.csv'
        cohort_retention.to_csv(cohort_path)
        logger.info(f"✅ Saved cohort analysis: {cohort_path}")
        
        return {
            'retention': cohort_retention,
            'ltv': cohort_ltv
        }
    
    def stage_survival_analysis(self, df):
        """Stage 6: Survival analysis"""
        logger.info("\n⏱️  STAGE 6: SURVIVAL ANALYSIS")
        logger.info("-" * 80)
        
        survival = SurvivalAnalysis(self.config)
        
        logger.info("📊 Fitting Cox Proportional Hazards model...")
        cox_model = survival.fit_cox_model(df)
        
        logger.info("📈 Computing Kaplan-Meier curves...")
        km_curves = survival.kaplan_meier_curves(df)
        
        logger.info(f"✅ Cox model fit: {cox_model.summary.shape[0]} features")
        logger.info(f"✅ KM curves computed: {len(km_curves)} segments")
        
        return {
            'cox': cox_model,
            'km_curves': km_curves
        }
    
    def stage_segmentation(self, df):
        """Stage 7: Customer segmentation & archetypes"""
        logger.info("\n🧬 STAGE 7: CUSTOMER SEGMENTATION & ARCHETYPES")
        logger.info("-" * 80)
        
        segmentation = ChurnSegmentation(self.config)
        
        # Churn archetypes
        logger.info("🔍 Identifying churn archetypes...")
        archetypes = segmentation.identify_archetypes(df)
        
        # RFM segmentation
        logger.info("📊 Performing RFM segmentation...")
        rfm_segments = segmentation.rfm_segmentation(df)
        
        logger.info(f"✅ Archetypes identified: {len(archetypes)}")
        logger.info(f"✅ Largest archetype: {archetypes.iloc[0]['name']} ({archetypes.iloc[0]['count']} customers)")
        
        return {
            'archetypes': archetypes,
            'rfm_segments': rfm_segments
        }
    
    def stage_ml_modeling(self, df):
        """Stage 8: Machine Learning model training"""
        logger.info("\n🤖 STAGE 8: MACHINE LEARNING MODELING")
        logger.info("-" * 80)
        
        ml_pipeline = ModelPipeline(self.config)
        
        # Train models
        logger.info("🚀 Training 5 models...")
        models = ml_pipeline.train_all_models(df)
        
        logger.info(f"✅ Models trained: {len(models)}")
        
        # Best model
        best_model_name = max(models.items(), key=lambda x: x[1]['metrics']['roc_auc'])[0]
        best_roc_auc = models[best_model_name]['metrics']['roc_auc']
        logger.info(f"✅ Best model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
        
        # Generate predictions
        logger.info("🎯 Generating predictions...")
        predictions = ml_pipeline.predict(df, best_model_name)
        
        # Save model
        model_path = self.config.models_dir / f'{best_model_name}_model.pkl'
        ml_pipeline.save_model(best_model_name)
        logger.info(f"✅ Model saved: {model_path}")
        
        return {
            'models': models,
            'predictions': predictions,
            'best_model': best_model_name
        }
    
    def stage_sql_analytics(self, df):
        """Stage 9: SQL analytics queries"""
        logger.info("\n🗄️  STAGE 9: SQL ANALYTICS (DuckDB)")
        logger.info("-" * 80)
        
        # Save to parquet for DuckDB
        parquet_path = self.config.data_dir / 'processed' / 'telco.parquet'
        df.to_parquet(parquet_path)
        
        db = DuckDBAnalytics(str(parquet_path))
        
        queries = [
            ('Churn Rate by Segment', db.churn_rate_by_segment),
            ('Revenue Concentration', db.revenue_concentration),
            ('Cohort Retention', db.cohort_retention_analysis),
            ('Feature Adoption Impact', db.feature_adoption_impact),
            ('Payment Method Analysis', db.payment_method_analysis),
            ('High-Risk Customers', db.high_risk_customers),
            ('Revenue at Risk', db.revenue_at_risk_by_month),
        ]
        
        results = {}
        for query_name, query_func in queries:
            logger.info(f"  📊 Running: {query_name}...")
            try:
                results[query_name] = query_func()
                logger.info(f"    ✅ {len(results[query_name])} rows returned")
            except Exception as e:
                logger.warning(f"    ⚠️  Query failed: {str(e)}")
        
        return results
    
    def stage_export_excel(self, df, predictions, analytics_results):
        """Stage 10: Generate Excel workbooks"""
        logger.info("\n📊 STAGE 10: EXCEL WORKBOOK GENERATION")
        logger.info("-" * 80)
        
        excel_builder = ExcelReportBuilder(self.config)
        
        # Executive Summary Workbook
        logger.info("📝 Building Executive Summary workbook...")
        exec_summary_path = excel_builder.build_executive_summary(df, predictions, analytics_results)
        logger.info(f"✅ Executive Summary: {exec_summary_path}")
        
        # Detailed Analysis Workbook
        logger.info("📝 Building Detailed Analysis workbook...")
        detailed_path = excel_builder.build_detailed_analysis(df, predictions)
        logger.info(f"✅ Detailed Analysis: {detailed_path}")
        
        return {
            'executive_summary': exec_summary_path,
            'detailed_analysis': detailed_path
        }
    
    def stage_tableau_prep(self, df, predictions):
        """Stage 11: Prepare data for Tableau"""
        logger.info("\n📊 STAGE 11: TABLEAU DATA PREPARATION")
        logger.info("-" * 80)
        
        tableau_prep = TableauDataPreparer(self.config)
        
        # Prepare data sources
        logger.info("🔄 Creating Tableau data sources...")
        
        customers_table = tableau_prep.prepare_customers_table(df, predictions)
        services_table = tableau_prep.prepare_services_table(df)
        cohorts_table = tableau_prep.prepare_cohorts_table(df)
        segments_table = tableau_prep.prepare_segments_table(df)
        
        # Save to CSV
        export_dir = self.config.data_dir / 'export'
        export_dir.mkdir(parents=True, exist_ok=True)
        
        customers_table.to_csv(export_dir / 'tableau_customers.csv', index=False)
        services_table.to_csv(export_dir / 'tableau_services.csv', index=False)
        cohorts_table.to_csv(export_dir / 'tableau_cohorts.csv', index=False)
        segments_table.to_csv(export_dir / 'tableau_segments.csv', index=False)
        
        logger.info(f"✅ Customers table: {customers_table.shape[0]:,} records")
        logger.info(f"✅ Services table: {services_table.shape[0]:,} records")
        logger.info(f"✅ Cohorts table: {cohorts_table.shape[0]:,} records")
        logger.info(f"✅ Segments table: {segments_table.shape[0]:,} records")
        
        return {
            'customers': customers_table,
            'services': services_table,
            'cohorts': cohorts_table,
            'segments': segments_table
        }
    
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        logger.info("\n🚀 EXECUTING FULL PIPELINE")
        
        try:
            # Data loading
            df = self.stage_data_loading()
            
            # Data cleaning
            df = self.stage_data_cleaning(df)
            
            # Feature engineering
            df = self.stage_feature_engineering(df)
            
            # EDA
            eda_results = self.stage_exploratory_analysis(df)
            
            # Cohort analysis
            cohort_results = self.stage_cohort_analysis(df)
            
            # Survival analysis
            survival_results = self.stage_survival_analysis(df)
            
            # Segmentation
            segmentation_results = self.stage_segmentation(df)
            
            # ML Modeling
            ml_results = self.stage_ml_modeling(df)
            
            # SQL Analytics
            sql_results = self.stage_sql_analytics(df)
            
            # Excel Export
            excel_results = self.stage_export_excel(df, ml_results['predictions'], sql_results)
            
            # Tableau Prep
            tableau_results = self.stage_tableau_prep(df, ml_results['predictions'])
            
            # Print summary
            self._print_completion_summary()
            
            return {
                'data': df,
                'eda': eda_results,
                'cohort': cohort_results,
                'survival': survival_results,
                'segmentation': segmentation_results,
                'ml': ml_results,
                'sql': sql_results,
                'excel': excel_results,
                'tableau': tableau_results
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def run_stage(self, stage):
        """Run specific pipeline stage"""
        stages = {
            'load': self.stage_data_loading,
            'clean': lambda: self.stage_data_cleaning(self.stage_data_loading()),
            'features': lambda: self.stage_feature_engineering(
                self.stage_data_cleaning(self.stage_data_loading())
            ),
            'eda': lambda: self.stage_exploratory_analysis(
                self.stage_feature_engineering(
                    self.stage_data_cleaning(self.stage_data_loading())
                )
            ),
        }
        
        if stage not in stages:
            logger.error(f"Unknown stage: {stage}")
            logger.info(f"Available stages: {', '.join(stages.keys())}")
            return
        
        logger.info(f"Running stage: {stage}")
        return stages[stage]()
    
    def _print_completion_summary(self):
        """Print pipeline completion summary"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total execution time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        logger.info("\n📊 Output Files Generated:")
        logger.info("  • data/processed/telco_cleaned.csv")
        logger.info("  • data/processed/features_engineered.csv")
        logger.info("  • data/export/PRISM_Executive_Summary.xlsx")
        logger.info("  • data/export/PRISM_Detailed_Analysis.xlsx")
        logger.info("  • data/export/tableau_customers.csv")
        logger.info("  • data/export/tableau_services.csv")
        logger.info("  • data/export/tableau_cohorts.csv")
        logger.info("  • data/export/tableau_segments.csv")
        logger.info("  • models/xgboost_model.pkl")
        logger.info("\n🚀 Next Steps:")
        logger.info("  1. Review Excel workbooks in data/export/")
        logger.info("  2. Import CSV files to Tableau")
        logger.info("  3. Run: streamlit run dashboards/streamlit_app.py")
        logger.info("  4. Check logs for detailed insights")
        logger.info("=" * 80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='PRISM Data Analyst Edition - Analytics Pipeline'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full pipeline'
    )
    parser.add_argument(
        '--stage',
        choices=['load', 'clean', 'features', 'eda'],
        help='Run specific pipeline stage'
    )
    
    args = parser.parse_args()
    
    pipeline = PRISMPipeline()
    
    if args.full:
        pipeline.run_full_pipeline()
    elif args.stage:
        pipeline.run_stage(args.stage)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
