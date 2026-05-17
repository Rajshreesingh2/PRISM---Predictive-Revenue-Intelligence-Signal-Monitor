# PRISM - Predictive Revenue Intelligence Signal Monitor

## Overview

PRISM is a **production-grade, end-to-end Data Science system** that detects customer churn risk 30-90 days early, forecasts revenue impact, and surfaces actionable retention insights.

### 🎯 Key Capabilities

✅ **Python EDA** - Comprehensive exploratory data analysis with statistical insights  
✅ **ML Pipeline** - 4 production models (Logistic Regression, Random Forest, XGBoost, LightGBM)  
✅ **SQL Analytics** - 7 pre-built DuckDB queries for business intelligence  
✅ **Excel Export** - Executive summary & detailed analysis workbooks  
✅ **Tableau Integration** - 4 ready-to-import data sources  
✅ **Survival Analysis** - Cox proportional hazards & Kaplan-Meier curves  
✅ **Cohort Analysis** - Retention tracking & LTV evolution  
✅ **Customer Segmentation** - RFM scoring & behavioral archetypes  

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Setup

```bash
# Clone repository
git clone <repo-url>
cd PRISM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Run Complete Pipeline

```bash
python pipeline.py
```

This executes all 11 stages:

1. 📦 **Data Loading** - Loads Telco CSV or generates sample data
2. 🧹 **Data Cleaning** - Removes duplicates, handles missing values, detects outliers
3. ⚙️ **Feature Engineering** - Creates 100+ features (numeric, categorical, temporal, interactions)
4. 📊 **EDA Analysis** - Distribution, correlation, churn pattern analysis
5. 👥 **Cohort Analysis** - Retention heatmaps, LTV tracking
6. ⏱️ **Survival Analysis** - Cox models, Kaplan-Meier curves
7. 🧬 **Segmentation** - Churn archetypes, RFM clustering
8. 🤖 **ML Modeling** - 4 models trained with ROC-AUC, F1, precision, recall
9. 🗄️ **SQL Analytics** - DuckDB queries for BI dashboards
10. 📊 **Excel Export** - 2 professional workbooks generated
11. 📊 **Tableau Prep** - 4 CSV data sources created

---

## 📊 Output Files

After running the pipeline, outputs are available in:

```
data/
├── processed/
│   ├── telco_cleaned.csv              # Clean dataset
│   ├── features_engineered.csv        # 100+ features
│   └── telco.parquet                  # DuckDB optimized
├── export/
│   ├── PRISM_Executive_Summary.xlsx   # KPIs, trends, at-risk customers
│   ├── PRISM_Detailed_Analysis.xlsx   # Full scores, feature importance
│   ├── tableau_customers.csv          # Customer dimensions + risk scores
│   ├── tableau_services.csv           # Service adoption tracking
│   ├── tableau_cohorts.csv            # Retention metrics
│   └── tableau_segments.csv           # Segment performance

models/
└── best_model.pkl                  # Trained XGBoost model

logs/
└── prism_YYYYMMDD_HHMMSS.log     # Complete execution log
```

---

## 🏗️ Project Structure

```
PRISM/
├── src/
│   ├── data/
│   │   ├── loader.py          # Data loading with sample generation
│   │   └── cleaner.py         # Data cleaning & preprocessing
│   ├── ml/
│   │   ├── feature_engineering.py   # 100+ feature creation
│   │   └── modeling.py             # 4 ML models
│   ├── analytics/
│   │   ├── eda.py                  # Exploratory analysis
│   │   ├── cohort_analysis.py      # Retention & LTV
│   │   ├── survival_analysis.py    # Cox models, Kaplan-Meier
│   │   └── segmentation.py        # RFM & clustering
│   ├── sql/
│   │   └── duckdb_queries.py      # 7 pre-built SQL queries
│   ├── export/
│   │   ├── excel_builder.py       # Excel workbooks
│   │   └── tableau_prep.py        # Tableau data sources
│   └── utils/
│       ├── config.py              # Configuration management
│       └── logger.py              # Logging setup
├── pipeline.py                    # Main orchestrator
├── config.yaml                    # Configuration file
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 📈 Key Analytics

### Exploratory Data Analysis
- Dataset shape, data types, memory usage
- Missing value analysis
- Feature distribution analysis
- Correlation matrix with churn
- Churn rate by contract type

### Machine Learning
- **Logistic Regression** - Baseline interpretable model
- **Random Forest** - Feature importance extraction
- **XGBoost** - Production model (best performer)
- **LightGBM** - Fast gradient boosting alternative

Metrics reported:
- ROC-AUC score
- F1-Score
- Precision & Recall
- Confusion Matrix

### Advanced Analytics
- **Survival Analysis**: Estimate customer lifetime probability
- **Cohort Analysis**: Track retention by customer generation
- **Segmentation**: RFM scoring & behavioral clustering
- **SQL Queries**: Pareto analysis, revenue forecasting, etc.

---

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  export_dir: data/export

ml:
  test_size: 0.2
  random_state: 42
  models:
    - logistic_regression
    - random_forest
    - xgboost
    - lightgbm

analytics:
  enable_eda: true
  enable_cohort_analysis: true
  enable_survival_analysis: true
```

---

## 📝 Example Usage

### Python API

```python
from src.utils.config import Config
from src.data.loader import DataLoader
from src.ml.modeling import MLModeler

# Initialize
config = Config('config.yaml')

# Load data
loader = DataLoader(config)
df = loader.load_telco_data()

# Train models
modeler = MLModeler(config)
model, X_test, y_test = modeler.train_models(X, y)

# Generate predictions
predictions = model.predict_proba(X_test)[:, 1]
```

---

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

## 📄 License

MIT License - See LICENSE file for details

---

## 📞 Support

For issues or questions, please open a GitHub issue.

---

**Last Updated**: May 17, 2026  
**Version**: 1.0.0
