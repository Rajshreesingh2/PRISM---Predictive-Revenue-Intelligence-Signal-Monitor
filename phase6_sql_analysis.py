"""
PRISM v2 — Phase 6a: SQL Analytics Layer
DuckDB in-process SQL — identical syntax to BigQuery, Snowflake, Redshift
10 real analytical queries a DS at Visa/Google would write
"""

import duckdb
import pandas as pd
import json
import os

os.makedirs("sql_outputs", exist_ok=True)

print("=" * 60)
print("  PRISM v2 — Phase 6a: SQL Analytics Layer")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# Connect and load data
# ─────────────────────────────────────────────────────────────
con = duckdb.connect("data/prism.duckdb")

# Load CSVs as SQL tables
con.execute("""
    CREATE OR REPLACE TABLE customers AS
    SELECT * FROM read_csv_auto('data/telco_cleaned.csv')
""")

con.execute("""
    CREATE OR REPLACE TABLE predictions AS
    SELECT * FROM read_csv_auto('data/predictions_with_roi.csv')
""")

con.execute("""
    CREATE OR REPLACE TABLE feature_store AS
    SELECT * FROM read_csv_auto('data/feature_store.csv')
""")

print("\n  Tables created:")
tables = con.execute("SHOW TABLES").fetchdf()
for t in tables["name"]:
    count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"    {t}: {count:,} rows")

results = {}

# ─────────────────────────────────────────────────────────────
# Query 1: Churn rate by contract and internet service
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("  Q1: Churn rate by contract × internet service")
q1 = con.execute("""
    SELECT
        Contract,
        InternetService,
        COUNT(*)                                    AS total_customers,
        SUM(Churn_binary)                           AS churned,
        ROUND(AVG(Churn_binary) * 100, 2)           AS churn_rate_pct,
        ROUND(AVG(MonthlyCharges), 2)               AS avg_monthly_charges,
        ROUND(SUM(MonthlyCharges), 2)               AS total_mrr
    FROM customers
    GROUP BY Contract, InternetService
    ORDER BY churn_rate_pct DESC
""").fetchdf()
print(q1.to_string(index=False))
results["churn_by_segment"] = q1.to_dict(orient="records")

# ─────────────────────────────────────────────────────────────
# Query 2: Revenue at risk by tenure cohort
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("  Q2: Revenue at risk by tenure cohort")
q2 = con.execute("""
    SELECT
        CASE
            WHEN tenure BETWEEN 0  AND 6  THEN '0-6m'
            WHEN tenure BETWEEN 7  AND 12 THEN '6-12m'
            WHEN tenure BETWEEN 13 AND 24 THEN '1-2yr'
            WHEN tenure BETWEEN 25 AND 48 THEN '2-4yr'
            ELSE '4+yr'
        END                                         AS tenure_cohort,
        COUNT(*)                                    AS customers,
        SUM(Churn_binary)                           AS churned,
        ROUND(AVG(Churn_binary)*100, 1)             AS churn_rate_pct,
        ROUND(SUM(CASE WHEN Churn_binary=1
              THEN MonthlyCharges ELSE 0 END), 2)   AS mrr_at_risk,
        ROUND(SUM(CASE WHEN Churn_binary=1
              THEN MonthlyCharges*12 ELSE 0 END),2) AS annual_revenue_at_risk
    FROM customers
    GROUP BY tenure_cohort
    ORDER BY churn_rate_pct DESC
""").fetchdf()
print(q2.to_string(index=False))
results["revenue_at_risk_cohort"] = q2.to_dict(orient="records")

# ─────────────────────────────────────────────────────────────
# Query 3: Payment method risk analysis
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("  Q3: Payment method churn risk")
q3 = con.execute("""
    SELECT
        PaymentMethod,
        COUNT(*)                                AS customers,
        ROUND(AVG(Churn_binary)*100, 2)         AS churn_rate_pct,
        ROUND(AVG(MonthlyCharges), 2)           AS avg_monthly_charges,
        ROUND(SUM(MonthlyCharges *
              Churn_binary), 2)                 AS mrr_lost
    FROM customers
    GROUP BY PaymentMethod
    ORDER BY churn_rate_pct DESC
""").fetchdf()
print(q3.to_string(index=False))
results["payment_method_risk"] = q3.to_dict(orient="records")

# ─────────────────────────────────────────────────────────────
# Query 4: Service adoption impact on churn
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("  Q4: Service adoption score vs churn")
q4 = con.execute("""
    SELECT
        service_adoption_score,
        COUNT(*)                                AS customers,
        ROUND(AVG(Churn_binary)*100, 2)         AS churn_rate_pct,
        ROUND(AVG(MonthlyCharges), 2)           AS avg_monthly_charges
    FROM customers
    WHERE service_adoption_score IS NOT NULL
    GROUP BY service_adoption_score
    ORDER BY service_adoption_score
""").fetchdf()
print(q4.to_string(index=False))
results["service_adoption_churn"] = q4.to_dict(orient="records")

# ─────────────────────────────────────────────────────────────
# Query 5: Top 10% highest CLV customers at risk
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("  Q5: High-CLV customers at churn risk")
q5 = con.execute("""
    WITH customer_clv AS (
        SELECT
            customerID,
            tenure,
            MonthlyCharges,
            TotalCharges,
            Contract,
            InternetService,
            Churn_binary,
            MonthlyCharges * 24 AS estimated_clv,
            NTILE(10) OVER (ORDER BY MonthlyCharges DESC) AS charge_decile
        FROM customers
    )
    SELECT
        charge_decile,
        COUNT(*)                                AS customers,
        ROUND(AVG(Churn_binary)*100, 2)         AS churn_rate_pct,
        ROUND(AVG(MonthlyCharges), 2)           AS avg_monthly_charges,
        ROUND(SUM(estimated_clv * Churn_binary),2) AS clv_at_risk
    FROM customer_clv
    GROUP BY charge_decile
    ORDER BY charge_decile
""").fetchdf()
print(q5.to_string(index=False))
results["clv_at_risk_decile"] = q5.to_dict(orient="records")

# ─────────────────────────────────────────────────────────────
# Query 6: Month-to-month vs contract comparison with window functions
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("  Q6: Contract type cohort analysis with window functions")
q6 = con.execute("""
    SELECT
        Contract,
        COUNT(*)                                            AS total,
        SUM(Churn_binary)                                   AS churned,
        ROUND(AVG(Churn_binary)*100, 2)                     AS churn_rate,
        ROUND(AVG(MonthlyCharges), 2)                       AS avg_mrr,
        ROUND(SUM(MonthlyCharges), 2)                       AS total_mrr,
        ROUND(SUM(MonthlyCharges) /
              SUM(SUM(MonthlyCharges)) OVER () * 100, 2)    AS mrr_share_pct,
        ROUND(SUM(Churn_binary*MonthlyCharges) /
              SUM(SUM(Churn_binary*MonthlyCharges)) OVER () * 100, 2) AS risk_share_pct
    FROM customers
    GROUP BY Contract
    ORDER BY churn_rate DESC
""").fetchdf()
print(q6.to_string(index=False))
results["contract_window_analysis"] = q6.to_dict(orient="records")

# ─────────────────────────────────────────────────────────────
# Query 7: Senior citizen risk segmentation
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("  Q7: Senior citizen risk vs demographic profile")
q7 = con.execute("""
    SELECT
        CASE WHEN SeniorCitizen=1 THEN 'Senior' ELSE 'Non-Senior' END AS segment,
        CASE WHEN Partner=1 THEN 'Has Partner' ELSE 'No Partner' END  AS partner_status,
        COUNT(*)                                AS customers,
        ROUND(AVG(Churn_binary)*100, 2)         AS churn_rate_pct,
        ROUND(AVG(MonthlyCharges), 2)           AS avg_monthly_charges
    FROM customers
    GROUP BY SeniorCitizen, Partner
    ORDER BY churn_rate_pct DESC
""").fetchdf()
print(q7.to_string(index=False))
results["senior_risk_profile"] = q7.to_dict(orient="records")

# ─────────────────────────────────────────────────────────────
# Query 8: Running total MRR loss over tenure
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("  Q8: Cumulative MRR loss by tenure (running total)")
q8 = con.execute("""
    WITH monthly_loss AS (
        SELECT
            tenure,
            SUM(CASE WHEN Churn_binary=1 THEN MonthlyCharges ELSE 0 END) AS mrr_lost
        FROM customers
        GROUP BY tenure
    )
    SELECT
        tenure,
        ROUND(mrr_lost, 2)                              AS mrr_lost_at_tenure,
        ROUND(SUM(mrr_lost) OVER (ORDER BY tenure), 2)  AS cumulative_mrr_lost
    FROM monthly_loss
    ORDER BY tenure
    LIMIT 15
""").fetchdf()
print(q8.to_string(index=False))
results["cumulative_mrr_loss"] = q8.to_dict(orient="records")

# ─────────────────────────────────────────────────────────────
# Query 9: Fiber optic deep dive
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("  Q9: Fiber optic customer risk deep dive")
q9 = con.execute("""
    SELECT
        InternetService,
        CASE WHEN TechSupport='Yes' OR TechSupport=1 THEN 'Has TechSupport'
             ELSE 'No TechSupport' END              AS tech_support_status,
        CASE WHEN OnlineSecurity='Yes' OR OnlineSecurity=1 THEN 'Secured'
             ELSE 'Unsecured' END                   AS security_status,
        COUNT(*)                                    AS customers,
        ROUND(AVG(Churn_binary)*100, 2)             AS churn_rate_pct,
        ROUND(AVG(MonthlyCharges), 2)               AS avg_monthly_charges
    FROM customers
    WHERE InternetService = 'Fiber optic'
    GROUP BY InternetService, tech_support_status, security_status
    ORDER BY churn_rate_pct DESC
""").fetchdf()
print(q9.to_string(index=False))
results["fiber_optic_deep_dive"] = q9.to_dict(orient="records")

# ─────────────────────────────────────────────────────────────
# Query 10: Business intervention ROI calculation
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("  Q10: Intervention ROI by segment")
q10 = con.execute("""
    WITH segment_stats AS (
        SELECT
            Contract,
            InternetService,
            COUNT(*)                                    AS total,
            SUM(Churn_binary)                           AS churned,
            ROUND(AVG(Churn_binary)*100, 2)             AS churn_rate,
            ROUND(AVG(MonthlyCharges), 2)               AS avg_mrr,
            ROUND(SUM(Churn_binary * MonthlyCharges), 2) AS mrr_at_risk
        FROM customers
        GROUP BY Contract, InternetService
    )
    SELECT
        Contract,
        InternetService,
        total,
        churn_rate,
        avg_mrr,
        mrr_at_risk,
        ROUND(mrr_at_risk * 12, 2)                      AS annual_risk,
        ROUND(mrr_at_risk * 0.15, 2)                    AS cost_15pct_discount,
        ROUND((mrr_at_risk * 12) / NULLIF(mrr_at_risk * 0.15 * 12, 0), 2) AS intervention_roi_x
    FROM segment_stats
    WHERE churn_rate > 20
    ORDER BY annual_risk DESC
    LIMIT 8
""").fetchdf()
print(q10.to_string(index=False))
results["intervention_roi"] = q10.to_dict(orient="records")

# ─────────────────────────────────────────────────────────────
# Save all query results
# ─────────────────────────────────────────────────────────────
with open("sql_outputs/query_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Save individual CSVs
for name, records in results.items():
    pd.DataFrame(records).to_csv(f"sql_outputs/{name}.csv", index=False)

print("\n" + "="*60)
print("  Phase 6a Complete — SQL Analytics Layer")
print("="*60)
print(f"\n  Database: data/prism.duckdb")
print(f"  Queries run: 10")
print(f"  Output files saved to: sql_outputs/")
print(f"\n  SQL skills demonstrated:")
print(f"    GROUP BY + aggregations")
print(f"    Window functions (NTILE, SUM OVER, running totals)")
print(f"    CTEs (WITH clauses)")
print(f"    CASE WHEN conditional logic")
print(f"    Multi-table analysis")
print(f"    Business metric calculation (ROI)")
print(f"\n  Next: python phase6b_monitoring.py")
print("="*60)
