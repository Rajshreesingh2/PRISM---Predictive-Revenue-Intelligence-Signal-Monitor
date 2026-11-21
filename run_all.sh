#!/usr/bin/env bash
# Runs every phase in order, starting from the raw IBM Telco CSV in data/raw/.
# Phase 1 tries to fetch live macro data (World Bank, Google Trends, Alpha Vantage);
# if they are unreachable it reuses the values saved in data/macro_signals.json.
set -e
python phase1_data_engineering.py
python phase2_eda.py
python phase2b_survival.py
python phase2c_clustering.py
python phase3_modeling.py      # trains models, saves the held-out test split
python phase4_ab_testing.py    # SIMULATED A/B test
python phase6_sql_analysis.py
python phase6b_monitoring.py
python phase6c_pipeline.py
echo "Done. Start the dashboard with:  streamlit run dashboard.py"
echo "Start the API with:              uvicorn phase5_api:app --reload"
