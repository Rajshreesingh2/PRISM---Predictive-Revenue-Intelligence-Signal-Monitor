<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-2.0-FF6600?style=flat-square&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-5.19-3F4F75?style=flat-square&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/DuckDB-0.9-FFF000?style=flat-square&logo=duckdb&logoColor=black"/>
<img src="https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Deployed-Streamlit Cloud-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>

<br><br>

# PRISM

### Predictive Revenue & Intelligence Signal Monitor

Catch churn before it happens. Know who is leaving, why, and when.

**[Live Dashboard](https://prism---predictive-revenue-intelligence-signal-monitor-jevgfv9.streamlit.app/) · [GitHub](https://github.com/Rajshreesingh2/PRISM---Predictive-Revenue-Intelligence-Signal-Monitor)**

</div>

---

## What This Is

PRISM is a production-grade data science project that predicts customer churn 30 to 90 days before it happens, quantifies how much revenue is at risk, and surfaces the exact behavioral signals driving it — deployed as a live interactive dashboard.

Most companies only react to churn after the customer is already gone. The data to predict it exists inside every product. PRISM connects it.

---

## Live Dashboard

**[prism---predictive-revenue-intelligence-signal-monitor-jevgfv9.streamlit.app](https://prism---predictive-revenue-intelligence-signal-monitor-jevgfv9.streamlit.app/)**

12 interactive pages covering executive overview, customer risk scoring, churn archetypes, 3D segmentation, survival analysis, A/B testing, model intelligence, macro signals, behavioral analytics, cohort retention, revenue analytics, and RFM segmentation.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Machine Learning | XGBoost, LightGBM, scikit-learn, Optuna, SHAP |
| Statistics | SciPy, lifelines (Cox PH, Kaplan-Meier) |
| Data Engineering | Pandas, NumPy, DuckDB, SQLite |
| Visualization | Plotly, Streamlit |
| Experiment Tracking | MLflow |
| Deployment | Streamlit Cloud, GitHub |
| APIs | World Bank, Alpha Vantage, Google Trends |

---

## Project Phases

### Phase 1 — Data Engineering & Feature Store

Built a feature factory on the Kaggle Telco Customer Churn dataset enriched with three live API sources. Engineered 100+ features using SQL-style window functions, rolling aggregations, and cohort transformations. The feature store mirrors what a data engineer would build in BigQuery or Snowflake.

Key outputs: cleaned dataset, feature store, macro signal enrichment, API integration layer.

### Phase 2 — Exploratory Data Analysis

Hypothesis-driven EDA across five analytical dimensions: churn by contract and channel, cohort retention heatmaps, engagement velocity distributions, revenue at risk over time, and feature adoption rates between churned and retained customers.

Key finding: engagement velocity (month-over-month usage change) is a 5x stronger predictor than raw activity level.

### Phase 2b — Survival Analysis

Cox Proportional Hazards model estimating time-to-churn rather than binary churn probability. Kaplan-Meier curves by contract type reveal the exact month where each segment hits 50% survival. Median survival time is used as a direct input to CLV calculation.

Month-to-month customers: median survival 8 months. Two-year contract customers: 48+ months. 6x CLV difference.

### Phase 2c — Churn Archetypes

K-Means clustering on churned customers identifies four behavioral archetypes: Price Refugee, Early Dropout, Tech Dissatisfied, and Lifecycle Leaver. Each archetype has a distinct intervention strategy with quantified ROI.

### Phase 3 — ML Modeling & Explainability

Five models trained: XGBoost, LightGBM, Random Forest, Logistic Regression, and an Optuna-tuned XGBoost. Evaluated on ROC-AUC, F1, precision-recall, and a custom business metric (revenue recovered at different operating thresholds).

Temporal train/test split used throughout — no data leakage. SHAP values explain every prediction at the individual customer level.

Best model: XGBoost with ROC-AUC 0.79 on honest temporal holdout.

### Phase 4 — A/B Testing & Causal Inference

Simulated retention intervention with randomized control group. Implemented power analysis, two-sample t-test, chi-square test, and Difference-in-Differences estimator. DiD isolates the true causal effect of the intervention by controlling for pre-existing group differences.

### Phase 5 — FastAPI Inference API

REST API serving real-time churn predictions. Accepts customer feature vectors, returns churn probability, risk tier, intervention priority, and top SHAP drivers. Designed for integration with CRM and CS tooling.

### Phase 6a — SQL Analytics Layer

Ten DuckDB queries covering churn rate by segment, revenue concentration, cohort retention, feature adoption impact, payment method analysis, CLV by contract type, and high-risk customer identification. Each query maps directly to a business decision.

### Phase 6b — Model Monitoring

Drift detection pipeline comparing feature distributions between training data and live scoring data. Flags when input distributions shift beyond threshold, triggering model retraining alerts.

### Phase 6c — Airflow Pipeline

Eight-task DAG orchestrating the full weekly pipeline: data pull, feature engineering, model scoring, archetype assignment, drift check, dashboard refresh, and alert dispatch.

---

## Key Findings

**Contract type is the strongest predictor.** Month-to-month customers churn at 42.7% — nearly 7x the rate of two-year contract customers. Converting a single customer from month-to-month to annual is worth $1,040 in additional CLV at average charges.

**Engagement velocity beats raw activity.** Users declining month-over-month churn at 37% vs 7% for stable users. A power user who goes quiet is more at risk than a consistently low-activity user.

**Service adoption is a retention moat.** Each additional service adopted reduces churn by 4-5 percentage points. Zero-service customers churn at 35%. Six-service customers churn at under 10%.

**Auto-pay reduces churn by 2x.** Electronic check customers churn at 45%. Automatic payment customers at 15-17%. Payment method nudges are one of the cheapest retention interventions available.

**Revenue loss is concentrated, not distributed.** Fiber optic customers represent the largest MRR loss despite being fewer in number. They pay premium prices and churn at premium rates — the highest-value intervention target.

---

## Project Structure

```
PRISM/
│
├── data/
│   ├── telco_cleaned.csv
│   ├── predictions_with_roi.csv
│   ├── archetype_summary.json
│   ├── macro_signals.json
│   ├── ab_test_results.json
│   └── feature_importance_mi.csv
│
├── models/
│   └── model_results.json
│
├── phase1_data_engineering.py
├── phase2_eda.py
├── phase2b_survival.py
├── phase2c_clustering.py
├── phase3_modeling.py
├── phase4_ab_testing.py
├── phase5_api.py
├── phase6_sql_analysis.py
├── phase6b_monitoring.py
├── phase6c_pipeline.py
├── dashboard.py
└── requirements.txt
```

---

## How to Run Locally

```bash
git clone https://github.com/Rajshreesingh2/PRISM---Predictive-Revenue-Intelligence-Signal-Monitor.git
cd PRISM---Predictive-Revenue-Intelligence-Signal-Monitor
pip install -r requirements.txt
streamlit run dashboard.py
```

---

## Why This Project Is Built This Way

Most data science portfolios load a Kaggle CSV, train a model, plot a confusion matrix, and stop.

At any company doing data science at scale, the work looks different: raw data from multiple sources gets cleaned into a feature store, models are tracked with experiment management tooling, every stakeholder output is explainable, interventions are validated statistically, and everything is reproducible and deployable.

PRISM demonstrates all of that. Each phase maps to a real part of the DS workflow. Each tool is chosen because it is used in production, not because it was easy to implement.

---

<div align="center">

Built by **Rajshree Singh**

[Live Dashboard](https://prism---predictive-revenue-intelligence-signal-monitor-jevgfv9.streamlit.app/) · [GitHub](https://github.com/Rajshreesingh2/PRISM---Predictive-Revenue-Intelligence-Signal-Monitor)

</div>