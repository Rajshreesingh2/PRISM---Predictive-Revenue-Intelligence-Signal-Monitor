<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-2.1-150458?style=flat-square&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/DuckDB-0.9-FFF000?style=flat-square&logo=duckdb&logoColor=black"/>
<img src="https://img.shields.io/badge/XGBoost-2.0-FF6600?style=flat-square&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/Excel-OpenPyXL-217346?style=flat-square&logo=microsoft-excel&logoColor=white"/>
<img src="https://img.shields.io/badge/Tableau-Ready-E97627?style=flat-square&logo=tableau&logoColor=white"/>
<img src="https://img.shields.io/badge/Edition-Data%20Analyst-FF6B6B?style=flat-square&logo=analytics&logoColor=white"/>

<br><br>

# PRISM: Data Analyst Edition

### Predictive Revenue & Intelligence Signal Monitor
### 🎯 Complete Analytics Stack for Revenue Risk Prediction

Catch churn before it happens. Know who is leaving, why, and when.  
**Full Python EDA + ML + SQL Analytics + Excel Workbook + Tableau Integration in One Package**

**[Live Dashboard](https://prism---predictive-revenue-intelligence-signal-monitor-jevgfv9.streamlit.app/) · [GitHub](https://github.com/Rajshreesingh2/PRISM---Predictive-Revenue-Intelligence-Signal-Monitor) · [Data Analyst Guide](./docs/ANALYST_GUIDE.md)**

</div>

---

## 📊 What This Is

**PRISM: Data Analyst Edition** is an enterprise-grade, end-to-end analytics system that:

- ✅ Predicts customer churn **30–90 days early** with 79% ROC-AUC
- ✅ **Quantifies revenue at risk** with CLV impact analysis
- ✅ Surfaces **exact behavioral signals** driving churn (SHAP explainability)
- ✅ Delivers **production-ready Excel workbooks** for business stakeholders
- ✅ Provides **complete SQL analytics layer** (DuckDB) for ad-hoc exploration
- ✅ Generates **Tableau-ready datasets** for interactive BI dashboards
- ✅ Includes **survival analysis**, **cohort analysis**, and **archetype segmentation**
- ✅ Runs **A/B testing** and causal inference for retention interventions

Built for **data analysts, BI teams, and business intelligence professionals** who need production-grade insights without DevOps overhead.

---

## 🏗️ Project Architecture

```
PRISM-Data-Analyst-Edition/
│
├── 📁 data/
│   ├── raw/
│   │   └── telco_customer_churn.csv          # Source data
│   ├── processed/
│   │   ├── telco_cleaned.csv                 # Cleaned dataset
│   │   ├── features_engineered.csv           # 100+ features
│   │   └── predictions_with_revenue.csv      # Model outputs
│   ├── analytics/
│   │   ├── cohort_retention.csv              # Cohort analysis
│   │   ├── segment_metrics.csv               # Segment metrics
│   │   └── survival_curves.csv               # Survival analysis
│   └── export/
│       ├── PRISM_Executive_Summary.xlsx      # Stakeholder workbook
│       ├── PRISM_Detailed_Analysis.xlsx      # Deep-dive workbook
│       └── tableau_prep.csv                  # Tableau data source
│
├── 📁 models/
│   ├── xgboost_churn_model.pkl               # Trained model
│   ├── feature_importance.pkl                # SHAP values
│   └── model_metrics.json                    # Performance metrics
│
├── 📁 notebooks/
│   ├── 01_EDA_Initial_Exploration.ipynb      # Exploratory analysis
│   ├── 02_Feature_Engineering.ipynb          # Feature creation
│   ├── 03_ML_Modeling.ipynb                  # Model training
│   └── 04_Insights_Generation.ipynb          # Business insights
│
├── 📁 src/
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── eda.py                            # EDA functions
│   │   ├── survival_analysis.py              # Survival curves
│   │   ├── cohort_analysis.py                # Cohort analysis
│   │   └── segmentation.py                   # Clustering & archetypes
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── preprocessing.py                  # Data prep
│   │   ├── feature_engineering.py            # Feature factory
│   │   ├── modeling.py                       # Model training
│   │   └── explainability.py                 # SHAP analysis
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                         # Data loading
│   │   ├── cleaner.py                        # Data cleaning
│   │   └── enricher.py                       # API enrichment
│   ├── sql/
│   │   ├── __init__.py
│   │   ├── duckdb_queries.py                 # SQL analytics
│   │   └── schemas.py                        # Data schemas
│   ├── export/
│   │   ├── __init__.py
│   │   ├── excel_builder.py                  # Excel generation
│   │   ├── tableau_prep.py                   # Tableau datasets
│   │   └── report_generator.py               # PDF/HTML reports
│   └── utils/
│       ├── __init__.py
│       ├── config.py                         # Configuration
│       ├── logger.py                         # Logging
│       └── metrics.py                        # Custom metrics
│
├── 📁 sql/
│   ├── analytics_queries.sql                 # DuckDB queries
│   └── schemas.sql                           # Table definitions
│
├── 📁 dashboards/
│   ├── streamlit_app.py                      # Streamlit dashboard
│   └── pages/
│       ├── 1_Executive_Overview.py
│       ├── 2_Customer_Risk_Scoring.py
│       ├── 3_Cohort_Analysis.py
│       ├── 4_Segment_Insights.py
│       ├── 5_Survival_Analysis.py
│       ├── 6_Feature_Analysis.py
│       ├── 7_Model_Explainability.py
│       └── 8_Excel_Report_Generator.py
│
├── 📁 configs/
│   ├── analysis_config.yaml                  # Analysis settings
│   ├── ml_config.yaml                        # ML hyperparameters
│   └── export_config.yaml                    # Export templates
│
├── 📁 tests/
│   ├── test_data_quality.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   └── test_exports.py
│
├── 📁 docs/
│   ├── ANALYST_GUIDE.md                      # Analyst user guide
│   ├── SETUP_GUIDE.md                        # Installation guide
│   ├── SQL_REFERENCE.md                      # SQL query library
│   ├── EXCEL_WORKBOOK_GUIDE.md               # Workbook structure
│   └── TABLEAU_INTEGRATION.md                # Tableau setup
│
├── requirements.txt                           # Python dependencies
├── setup.py                                   # Package setup
├── pipeline.py                                # Main execution pipeline
└── README.md                                  # This file
```

---

## 🚀 Key Features

### 1. **Python EDA (Exploratory Data Analysis)**
- **Distribution analysis**: Feature distributions, outlier detection
- **Correlation analysis**: Pearson, Spearman, mutual information
- **Cohort analysis**: Retention heatmaps by sign-up cohort
- **Engagement velocity**: Month-over-month usage trends
- **Revenue at risk**: CLV-weighted customer segments
- **Interactive plots**: 50+ visualizations for business context

### 2. **Advanced ML Modeling**
- **5 model architectures**: XGBoost, LightGBM, Random Forest, Logistic Regression, Neural Network
- **Hyperparameter tuning**: Optuna optimization on business metrics
- **Temporal validation**: Proper train/test split to prevent data leakage
- **SHAP explainability**: Individual customer-level prediction drivers
- **Business metrics**: ROC-AUC, F1, precision-recall, custom revenue recovery metric

### 3. **Statistical Analysis**
- **Survival analysis**: Cox Proportional Hazards, Kaplan-Meier curves
- **Churn archetypes**: K-Means clustering on behavior patterns
- **A/B testing**: Power analysis, t-tests, chi-square, Difference-in-Differences
- **Cohort retention**: LTV evolution by sign-up month

### 4. **SQL Analytics Layer (DuckDB)**
10+ pre-built queries for:
- Churn rate by segment, channel, contract type
- Revenue concentration analysis
- Cohort retention trends
- Feature adoption impact
- High-risk customer identification
- Payment method analysis
- CLV by contract type
- Churn prediction by month
- Service adoption funnel
- Intervention ROI tracking

### 5. **Excel Workbook Generation**
**Executive Summary Workbook** (`PRISM_Executive_Summary.xlsx`):
- 📊 KPI dashboard (churn rate, revenue at risk, intervention ROI)
- 📈 Churn trends by segment
- 🎯 Top 100 at-risk customers with revenue impact
- 💰 Revenue recovery potential
- 🔍 Intervention recommendations

**Detailed Analysis Workbook** (`PRISM_Detailed_Analysis.xlsx`):
- 📋 Full customer scoring dataset (model outputs)
- 🔬 Feature importance rankings
- 📊 Cohort retention tables
- 🧬 Churn archetypes breakdown
- 📉 Survival curves by segment
- 💡 Key insights & findings

### 6. **Tableau Data Prep**
Ready-to-import CSV files for:
- Customer risk scores with dimensions
- Cohort retention matrix
- Segment performance metrics
- Feature adoption trends
- Revenue at risk by category

Includes:
- Dimension tables (Customer, Contract, Service)
- Fact tables (Monthly activity, Churn events)
- Pre-calculated metrics for fast dashboard builds

### 7. **Interactive Streamlit Dashboard**
12 pages of exploration:
- Executive overview with KPIs
- Customer risk scoring interface
- Cohort retention analysis
- 3D segment visualization
- Survival curves by contract type
- Feature importance & SHAP plots
- A/B testing results
- Model performance metrics
- Macro signal exploration
- Excel report generator
- SQL query builder

---

## 📊 Key Insights Generated

✅ **Contract type is 7x more predictive than raw activity**
- Month-to-month: 42.7% churn | Two-year: 6.1% churn
- Converting 1 customer saves $1,800 annual revenue

✅ **Engagement velocity beats raw activity**
- Declining users: 37% churn | Stable users: 7% churn
- Early warning system for at-risk customers

✅ **Service adoption is a retention moat**
- Zero services: 35% churn | Six services: 10% churn
- +4-5% retention per additional service adopted

✅ **Payment method is a 2x churn lever**
- Electronic check: 45% churn | Auto-pay: 15% churn
- Cheapest, fastest retention intervention

✅ **Revenue loss is concentrated**
- Fiber optic (10% of base) = 25% of churn revenue
- Premium segment = premium churn rates

---

## 🛠️ Tech Stack

| Component | Tools |
|-----------|-------|
| **Data Processing** | Pandas, NumPy, Polars |
| **Analytics** | SciPy, StatsModels, lifelines |
| **Machine Learning** | XGBoost, LightGBM, scikit-learn, Optuna |
| **Explainability** | SHAP, permutation importance |
| **SQL Analytics** | DuckDB, SQL |
| **Excel Export** | OpenPyXL, XlsxWriter |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Notebooks** | Jupyter, nbconvert |
| **Testing** | Pytest, Great Expectations |
| **Configuration** | PyYAML |
| **Logging** | Python logging |

---

## ⚡ Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Rajshreesingh2/PRISM---Predictive-Revenue-Intelligence-Signal-Monitor.git
cd PRISM---Predictive-Revenue-Intelligence-Signal-Monitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Execute complete analysis pipeline
python pipeline.py

# Output:
# ✅ Data cleaning & feature engineering
# ✅ Model training & evaluation
# ✅ SQL analytics queries executed
# ✅ Excel workbooks generated
# ✅ Tableau prep files created
# ✅ Reports available in data/export/
```

### Launch Interactive Dashboard

```bash
streamlit run dashboards/streamlit_app.py
```

### Generate Excel Reports

```bash
python -c "from src.export.excel_builder import generate_workbooks; generate_workbooks()"

# Outputs:
# - data/export/PRISM_Executive_Summary.xlsx
# - data/export/PRISM_Detailed_Analysis.xlsx
```

### Run SQL Analytics

```python
from src.sql.duckdb_queries import DuckDBAnalytics

analytics = DuckDBAnalytics("data/processed/telco_cleaned.csv")

# Run pre-built queries
churn_by_segment = analytics.churn_rate_by_segment()
revenue_at_risk = analytics.revenue_concentration()
cohort_retention = analytics.cohort_retention_analysis()
```

---

## 📈 Analysis Workflow

### Phase 1: Data Engineering & Feature Store
```
Raw CSV → Data Cleaning → 100+ Features → Feature Store
```
- Missing value handling
- Outlier detection
- Feature normalization
- Temporal feature creation
- API enrichment (World Bank, Alpha Vantage, Google Trends)

### Phase 2: Exploratory Data Analysis
```
Feature Store → Statistical Analysis → Business Insights
```
- Distribution analysis
- Correlation analysis
- Cohort retention heatmaps
- Engagement velocity trends
- Revenue at risk segmentation

### Phase 3: ML Modeling
```
Features → Train/Test Split → 5 Models → SHAP Explainability
```
- Hyperparameter tuning (Optuna)
- Cross-validation
- Business metric optimization
- Temporal validation

### Phase 4: Analytics & Insights
```
Model + Features → SQL Queries → Excel Reports → Tableau Prep
```
- DuckDB SQL queries
- Excel workbook generation
- Stakeholder reports
- BI data source preparation

### Phase 5: Dashboard & Monitoring
```
All Outputs → Streamlit Dashboard → Model Monitoring
```
- Interactive exploration
- Real-time scoring
- Drift detection
- Alert triggers

---

## 📊 Excel Workbooks

### Executive Summary Workbook
**Audience**: C-Suite, Product, Marketing

**Sheets**:
1. **Dashboard**: KPIs, churn trends, revenue at risk
2. **Risk Tier Breakdown**: Distribution of customers by risk level
3. **Top 100 At-Risk**: Specific customers with revenue impact
4. **Retention ROI**: Intervention cost vs. CLV savings
5. **Segment Analysis**: Churn by contract, channel, service
6. **Cohort Trends**: New customer retention by sign-up month
7. **Key Insights**: Executive summary + recommendations

### Detailed Analysis Workbook
**Audience**: Data team, Analysts, Revenue team

**Sheets**:
1. **Full Scores**: All customers with risk score & drivers
2. **Feature Importance**: Top 20 features by SHAP
3. **Cohort Retention**: Retention % by month for each cohort
4. **Archetypes**: Churn archetypes with intervention strategies
5. **Survival Curves**: Kaplan-Meier curves by segment
6. **Monthly Metrics**: Time-series churn, revenue, volumes
7. **Model Performance**: ROC curve, precision-recall, confusion matrix
8. **Data Dictionary**: All column definitions & calculations

---

## 🔍 SQL Analytics Library

Pre-built DuckDB queries in `src/sql/duckdb_queries.py`:

```python
from src.sql.duckdb_queries import DuckDBAnalytics

db = DuckDBAnalytics("telco_cleaned.csv")

# Segment analysis
db.churn_rate_by_segment()                    # Churn % by service/channel
db.revenue_concentration()                    # Pareto analysis
db.cohort_retention_analysis()                # Retention by sign-up month

# Feature analysis
db.feature_adoption_impact()                  # Service adoption → retention
db.payment_method_analysis()                  # Payment type churn rates
db.contract_clv_analysis()                    # CLV by contract type

# High-value customers
db.high_risk_customers()                      # Top 100 at-risk by revenue
db.revenue_at_risk_by_month()                 # Revenue loss forecasting

# Churn prediction
db.churn_prediction_by_month()                # Month-level churn forecast
db.intervention_roi_tracking()                # Intervention effectiveness
```

---

## 📱 Tableau Integration

### Data Sources (Auto-Generated)

1. **Customers Table** (`tableau_prep/customers.csv`)
   - Dimensions: customer_id, age, tenure, location, contract_type
   - Measures: churn_risk_score, clv, monthly_charges, revenue_at_risk

2. **Services Table** (`tableau_prep/services.csv`)
   - Services adopted by customer
   - Service adoption timeline

3. **Cohort Table** (`tableau_prep/cohorts.csv`)
   - Cohort retention by month
   - Lifetime value evolution

4. **Segment Table** (`tableau_prep/segments.csv`)
   - Segment metrics
   - Churn trends

### Recommended Tableau Dashboards

- **Executive Dashboard**: KPIs, trends, at-risk customers
- **Segment Deep-Dive**: Performance by contract/channel/service
- **Cohort Performance**: LTV and retention by acquisition month
- **Churn Drivers**: Feature importance, top churn reasons
- **Intervention Tracker**: ROI on retention campaigns

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Specific test suites
pytest tests/test_data_quality.py          # Data validation
pytest tests/test_feature_engineering.py   # Feature tests
pytest tests/test_models.py                # Model tests
pytest tests/test_exports.py               # Export validation
```

---

## 📚 Documentation

- **[Analyst User Guide](docs/ANALYST_GUIDE.md)** - How to use this system
- **[Setup Guide](docs/SETUP_GUIDE.md)** - Installation & configuration
- **[SQL Reference](docs/SQL_REFERENCE.md)** - Query library & examples
- **[Excel Workbook Guide](docs/EXCEL_WORKBOOK_GUIDE.md)** - Workbook structure
- **[Tableau Integration](docs/TABLEAU_INTEGRATION.md)** - BI dashboard setup

---

## 🎯 Who This Is For

✅ **Data Analysts** - Exploratory analysis, insights generation, ad-hoc queries  
✅ **Business Analysts** - Executive reports, KPI dashboards, ROI analysis  
✅ **BI Engineers** - Data prep, Tableau source building, metric definition  
✅ **Data Scientists** - ML modeling, feature engineering, explainability  
✅ **Product Teams** - Churn drivers, cohort retention, intervention testing  
✅ **Revenue Teams** - CLV analysis, at-risk customers, intervention ROI  

---

## 💼 Business Impact

- 🎯 **30-90 day lead time** on customer churn predictions
- 💰 **$100K+ annual revenue** recovery potential (based on test cohort)
- ⚡ **2x faster** intervention response vs. reactive retention
- 🔍 **100% explainability** on individual customer predictions
- 📊 **Weekly automated** scoring & reporting

---

## 🔄 Deployment Options

- **Local**: Run `python pipeline.py` + `streamlit run dashboards/streamlit_app.py`
- **Airflow**: DAG-based weekly automation (`airflow/dags/prism_weekly.py`)
- **Scheduled Tasks**: Cron jobs for daily scoring
- **FastAPI**: REST API for real-time predictions (Phase 5)
- **Cloud**: Deploy to AWS, GCP, or Azure

---

## 📄 License

MIT License - See LICENSE file

---

## 👤 Author

**Rajshree Singh**

[GitHub](https://github.com/Rajshreesingh2) · [Live Dashboard](https://prism---predictive-revenue-intelligence-signal-monitor-jevgfv9.streamlit.app/)

---

<div align="center">

**⭐ Found this useful? Star the repo!**

[Live Dashboard](https://prism---predictive-revenue-intelligence-signal-monitor-jevgfv9.streamlit.app/) · [GitHub](https://github.com/Rajshreesingh2/PRISM---Predictive-Revenue-Intelligence-Signal-Monitor) · [Analyst Guide](./docs/ANALYST_GUIDE.md)

</div>
