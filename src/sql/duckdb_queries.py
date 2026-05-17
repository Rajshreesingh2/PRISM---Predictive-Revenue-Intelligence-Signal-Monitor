"""DuckDB SQL queries for PRISM"""

import logging
import duckdb
import pandas as pd


logger = logging.getLogger(__name__)


class DuckDBAnalytics:
    """Execute SQL analytics using DuckDB"""
    
    def __init__(self, config):
        """Initialize DuckDB analytics"""
        self.config = config
        self.conn = duckdb.connect(':memory:')
    
    def run_analytics(self, df):
        """Execute all analytics queries"""
        logger.info("Starting SQL Analytics with DuckDB...")
        
        # Register dataframe
        self.conn.register('customers', df)
        
        results = {}
        
        # Query 1: Churn rate by segment
        try:
            churn_query = """
            SELECT 
                Contract,
                COUNT(*) as total_customers,
                SUM(CAST(Churn AS INT)) as churned,
                ROUND(100.0 * SUM(CAST(Churn AS INT)) / COUNT(*), 2) as churn_rate_pct
            FROM customers
            GROUP BY Contract
            ORDER BY churn_rate_pct DESC
            """
            results['churn_by_contract'] = self.conn.execute(churn_query).fetchall()
            logger.info("  ✓ Query 1: Churn by contract type")
        except Exception as e:
            logger.warning(f"  ✗ Query 1 failed: {e}")
        
        # Query 2: Revenue metrics
        try:
            revenue_query = """
            SELECT 
                Contract,
                COUNT(*) as customers,
                ROUND(AVG(MonthlyCharges), 2) as avg_monthly,
                ROUND(AVG(TotalCharges), 2) as avg_total,
                ROUND(SUM(TotalCharges), 0) as total_revenue
            FROM customers
            GROUP BY Contract
            ORDER BY total_revenue DESC
            """
            results['revenue_metrics'] = self.conn.execute(revenue_query).fetchall()
            logger.info("  ✓ Query 2: Revenue metrics by contract")
        except Exception as e:
            logger.warning(f"  ✗ Query 2 failed: {e}")
        
        # Query 3: Customer value distribution
        try:
            ltv_query = """
            SELECT 
                CASE 
                    WHEN TotalCharges < 500 THEN 'Low'
                    WHEN TotalCharges < 2000 THEN 'Medium'
                    ELSE 'High'
                END as value_segment,
                COUNT(*) as customers,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct_of_total,
                ROUND(AVG(CAST(Churn AS FLOAT)), 3) as churn_rate
            FROM customers
            GROUP BY value_segment
            ORDER BY customers DESC
            """
            results['ltv_segments'] = self.conn.execute(ltv_query).fetchall()
            logger.info("  ✓ Query 3: Customer value segments")
        except Exception as e:
            logger.warning(f"  ✗ Query 3 failed: {e}")
        
        logger.info("✅ SQL analytics complete")
        
        return results
