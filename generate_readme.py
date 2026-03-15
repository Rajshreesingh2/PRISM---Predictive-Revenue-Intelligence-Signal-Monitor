"""
PRISM v2 — Generate final README.md
"""

readme = """# 🔮 PRISM — Predictive Revenue & Intelligence Signal Monitor

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-Live_API-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **End-to-end ML pipeline for banking and telecom customer churn prediction** — integrating live economic signals, survival analysis, causal inference, and a real-time prediction API.

---

## 🎯 What PRISM Does

Most churn models answer: *"Will this customer churn?"*

PRISM answers four harder questions:

- **WHEN** will this customer churn? *(Cox Proportional Hazards survival model)*
- **WHY** are they leaving? *(K-Means clustering → 4 behavioral archetypes)*
- **What should we do?** *(Per-archetype intervention playbook with ROI calculation)*
- **Did our intervention work?** *(A/B testing with Difference-in-Differences causal inference)*

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  Kaggle Telco (7,043 customers) + World Bank API +          │
│  Alpha Vantage API + Google Trends API                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                 FEATURE ENGINEERING                         │
│  100+ features: charges, tenure, services, contracts,       │
│  demographics, polynomial interactions, macro signals       │
│  → Mutual Information selection → VIF collinearity filter   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              ANALYTICAL LAYER (SQL)                         │
│  DuckDB in-process SQL — 10 analytical queries              │
│  Window functions, CTEs, ROI calculations                   │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌───────────────┐ ┌─────────────┐ ┌─────────────────┐
│  SURVIVAL     │ │  CLUSTERING │ │  ML MODELS      │
│  ANALYSIS     │ │             │ │                 │
│ Kaplan-Meier  │ │ K-Means     │ │ Logistic Reg.   │
│ Cox PH Model  │ │ UMAP/PCA    │ │ Random Forest   │
│ Time-to-churn │ │ 4 Archetypes│ │ XGBoost+Optuna  │
└───────────────┘ └─────────────┘ │ LightGBM        │
                                  └────────┬────────┘
                                           │
┌──────────────────────────────────────────▼────────────────┐
│                  EVALUATION                                │
│  ROC-AUC · F1 · Lift Curve · CLV Impact                   │
│  Temporal split (no leakage) · SHAP explainability         │
│  A/B test + Difference-in-Differences causal inference     │
└──────────────────────────────────────────┬────────────────┘
                                           │
        ┌──────────────────────────────────┤
        ▼                                  ▼
┌───────────────────┐          ┌─────────────────────────┐
│   FastAPI         │          │   Streamlit Dashboard   │
│   /predict        │          │   12 pages              │
│   /predict/batch  │          │   Arctic Clean design   │
│   /model/info     │          │   3D visualizations     │
│   /archetypes     │          │   Live metrics          │
└───────────────────┘          └─────────────────────────┘
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────┐
        │  MLOPS LAYER             │
        │  Evidently AI monitoring │
        │  Airflow daily pipeline  │
        │  DuckDB SQL analytics    │
        │  MLflow experiment track │
        └──────────────────────────┘
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Dataset | 7,043 real Telco customers (IBM) |
| Features engineered | 100+ (from 21 raw) |
| Best model ROC-AUC | 0.794 (XGBoost + Optuna) |
| Split method | Temporal (no data leakage) |
| MRR at risk identified | $139,131/month |
| High-risk customers | 2,005 (RiskScore ≥ 6) |
| Churn archetypes discovered | 4 behavioral segments |
| API endpoints | 4 (predict, batch, info, archetypes) |
| Dashboard pages | 12 |
| SQL analytical queries | 10 |

---

## 🔬 What Makes This Different

**Survival analysis** — Cox PH model predicts *when* customers will churn, not just if. Median survival time varies from 8 months (month-to-month) to 72+ months (two-year contracts).

**Churn archetypes** — Unsupervised K-Means on churned customers only discovers 4 behavioral segments: Price Refugee, Early Dropout, Tech Dissatisfied, and Lifecycle Leaver. Each gets a different intervention strategy.

**Macro signal enrichment** — Every prediction is enriched with live GDP, CPI, and unemployment data from the World Bank API. A consumer stress index modulates baseline churn probability. No standard churn model includes this.

**Temporal train/test split** — No data leakage. Training on shorter-tenure customers, testing on longer-tenure customers mirrors real production deployment. Honest 0.794 AUC vs inflated 0.92+ from random splits.

**Causal inference** — A/B test framework with Difference-in-Differences estimation measures the true causal effect of retention interventions, not just correlation.

**Production API** — FastAPI inference endpoint returns churn probability, archetype, top risk factors, CLV at risk, and a specific intervention recommendation in a single API call.

---

## 🚀 Quick Start

```bash
git clone https://github.com/Rajshreesingh2/PRISM
cd PRISM
pip install -r requirements.txt

# Run the full pipeline
python phase1_data_engineering.py    # Feature factory
python phase2_eda.py                  # Deep EDA
python phase2b_survival.py           # Survival analysis
python phase2c_clustering.py         # Churn archetypes
python phase3_modeling.py            # Train models
python phase4_ab_testing.py          # A/B framework
python phase5_api.py                  # FastAPI server
python phase6_sql_analysis.py        # SQL analytics
python phase6b_monitoring.py         # Model monitoring
python phase6c_pipeline.py           # Automated pipeline

# Launch dashboard
streamlit run dashboard.py

# Launch API
python phase5_api.py
# Visit: http://localhost:8000/docs
```

---

## 📁 Project Structure

```
PRISM/
├── data/
│   ├── telco_cleaned.csv          # 7,043 real customers
│   ├── feature_store.csv          # 100+ engineered features
│   ├── survival_predictions.csv   # Cox model output
│   ├── churn_archetypes.csv       # K-Means cluster labels
│   ├── predictions_with_roi.csv   # Model predictions + CLV
│   ├── macro_signals.json         # Live API data
│   └── archetype_summary.json     # Archetype profiles
│
├── models/
│   ├── best_model.pkl             # XGBoost (Optuna-tuned)
│   ├── scaler.pkl                 # Feature scaler
│   └── model_results.json         # All model metrics
│
├── monitoring/
│   ├── drift_report.csv           # Feature drift analysis
│   └── monitoring_summary.json    # Evidently AI output
│
├── sql_outputs/                   # DuckDB query results
├── pipeline_logs/                 # Airflow run logs
│
├── phase1_data_engineering.py     # Feature factory (100+ features)
├── phase2_eda.py                  # Hypothesis-driven EDA
├── phase2b_survival.py            # Kaplan-Meier + Cox PH
├── phase2c_clustering.py          # K-Means + UMAP
├── phase3_modeling.py             # XGBoost + LightGBM + Optuna
├── phase4_ab_testing.py           # A/B test + DiD
├── phase5_api.py                  # FastAPI inference service
├── phase6_sql_analysis.py         # DuckDB SQL analytics
├── phase6b_monitoring.py          # Evidently AI monitoring
├── phase6c_pipeline.py            # Airflow pipeline simulation
├── airflow_dag.py                 # Production Airflow DAG
├── dashboard.py                   # Streamlit dashboard (12 pages)
├── config.py                      # API keys (gitignored)
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data | Pandas, NumPy, DuckDB SQL |
| ML | Scikit-learn, XGBoost, LightGBM, Lifelines |
| Optimization | Optuna (Bayesian hyperparameter search) |
| Explainability | SHAP |
| Statistics | SciPy, Statsmodels |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Monitoring | Evidently AI, custom drift detection |
| Pipeline | Apache Airflow (DAG included) |
| Database | DuckDB (in-process SQL) |
| APIs | World Bank, Alpha Vantage, Google Trends |
| Version Control | Git, GitHub |

---

## 📈 Model Performance

| Model | ROC-AUC | F1 | Avg Precision |
|-------|---------|-----|--------------|
| XGBoost (Optuna-tuned) | **0.794** | 0.180 | 0.210 |
| Logistic Regression | 0.777 | 0.061 | 0.205 |
| Random Forest | 0.777 | 0.000 | 0.201 |
| XGBoost (default) | 0.766 | 0.038 | 0.173 |
| LightGBM | 0.752 | 0.018 | 0.162 |

*All models evaluated on temporal test split (no data leakage). F1 is low due to class imbalance in the test set (6.6% churn rate) — expected and correct.*

---

## 🧬 Churn Archetypes

| Archetype | Trigger | Intervention |
|-----------|---------|-------------|
| Price Refugee | High charges + month-to-month | 15-20% loyalty discount |
| Early Dropout | 0-6 month tenure | Proactive onboarding call |
| Tech Dissatisfied | Fiber + no tech support | Free TechSupport upgrade |
| Lifecycle Leaver | Long tenure + contract expiry | Renewal incentive |

---

## 🌍 Macro Signal Integration

Live data pulled from 3 APIs at pipeline runtime:

- **World Bank API** — US GDP growth, CPI inflation, unemployment rate, GNI per capita
- **Alpha Vantage** — Telecom sector performance
- **Google Trends** — "cancel phone plan" and "switch carrier" search intent

These signals enrich every customer prediction with economic context — a consumer stress index that modulates baseline churn probability across the entire customer base.

---

## 📡 API Reference

```bash
# Health check
GET http://localhost:8000/health

# Single prediction
POST http://localhost:8000/predict
{
  "tenure": 3,
  "MonthlyCharges": 85.5,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  ...
}

# Response includes:
# churn_probability, risk_tier, archetype,
# top_risk_factors, intervention, clv_at_risk_12m,
# macro_context, predicted_at
```

---

## 👤 Author

**Rajshree Singh**
[GitHub](https://github.com/Rajshreesingh2/PRISM)

---

*Built as a DS portfolio project targeting roles at Visa, Google, Amazon, and Mastercard.*
*This project demonstrates end-to-end ML engineering, not just model training.*
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)

print("README.md generated successfully")
print(f"Length: {len(readme.split(chr(10)))} lines")
