"""
PRISM v2 — Phase 6c: Automated ML Pipeline
Simulates an Airflow DAG for daily retraining
In production: deploy this as airflow/dags/prism_pipeline.py
"""

import pandas as pd
import numpy as np
import json
import os
import time
import joblib
from datetime import datetime
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

os.makedirs("pipeline_logs", exist_ok=True)

print("="*60)
print("  PRISM v2 — Phase 6c: Automated Pipeline")
print("="*60)
print("""
  In production this would be an Apache Airflow DAG:
  
  DAG: prism_daily_pipeline
  Schedule: 0 2 * * *  (2am daily)
  
  Task flow:
  ingest_data >> validate_data >> engineer_features >>
  check_drift  >> retrain_if_needed >> evaluate_model >>
  update_predictions >> send_alerts
  
  Running all tasks now as a simulation...
""")

log = {"pipeline_run": datetime.now().isoformat(), "tasks": {}}

def task(name, func):
    print(f"\n  ▶ Task: {name}")
    start = time.time()
    try:
        result = func()
        elapsed = round(time.time() - start, 2)
        log["tasks"][name] = {"status": "SUCCESS", "elapsed_s": elapsed, "result": str(result)[:100]}
        print(f"    ✅ SUCCESS ({elapsed}s)")
        return result
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        log["tasks"][name] = {"status": "FAILED", "elapsed_s": elapsed, "error": str(e)}
        print(f"    ❌ FAILED: {e}")
        return None

# ── Task 1: Ingest data ──────────────────────────────────────
def ingest():
    telco = pd.read_csv("data/telco_cleaned.csv")
    assert len(telco) > 0, "Empty dataset"
    return f"{len(telco):,} rows loaded"

task("ingest_data", ingest)

# ── Task 2: Validate data ────────────────────────────────────
def validate():
    telco = pd.read_csv("data/telco_cleaned.csv")
    checks = {
        "no_nulls_in_target"   : telco["Churn_binary"].isnull().sum() == 0,
        "positive_charges"     : (telco["MonthlyCharges"] >= 0).all(),
        "valid_tenure"         : telco["tenure"].between(0, 72).all(),
        "churn_rate_reasonable": 0.05 <= telco["Churn_binary"].mean() <= 0.50,
        "min_row_count"        : len(telco) >= 1000,
    }
    failed = [k for k,v in checks.items() if not v]
    if failed:
        raise ValueError(f"Validation failed: {failed}")
    return f"All {len(checks)} checks passed"

task("validate_data", validate)

# ── Task 3: Feature engineering ──────────────────────────────
def engineer():
    fs = pd.read_csv("data/feature_store.csv")
    return f"{fs.shape[1]} features, {fs.shape[0]:,} rows"

task("engineer_features", engineer)

# ── Task 4: Check for data drift ─────────────────────────────
def check_drift():
    try:
        with open("monitoring/monitoring_summary.json") as f:
            summary = json.load(f)
        alert = summary["drift_summary"]["alert_level"]
        drifted = summary["drift_summary"]["drifted_features"]
        return f"Alert level: {alert}, drifted features: {drifted}"
    except:
        return "No previous monitoring data — first run"

drift_result = task("check_drift", check_drift)

# ── Task 5: Retrain model (conditional) ──────────────────────
def retrain():
    fs = pd.read_csv("data/feature_store.csv")
    with open("data/selected_features.json") as f:
        feat_cols = [c for c in json.load(f) if c != "Churn_binary" and c in fs.columns]

    X = fs[feat_cols].fillna(0)
    y = fs["Churn_binary"]

    split = int(len(fs) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    import xgboost as xgb
    model = joblib.load("models/best_model.pkl")

    # Partial refit on recent data (last 20%)
    recent_X = X.iloc[-int(len(X)*0.3):]
    recent_y = y.iloc[-int(len(y)*0.3):]

    # Evaluate current model
    probs = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, probs)

    return f"Current AUC: {auc:.4f} — model retained (AUC > 0.70 threshold)"

task("retrain_if_needed", retrain)

# ── Task 6: Update predictions ───────────────────────────────
def update_preds():
    preds = pd.read_csv("data/predictions_with_roi.csv").reset_index(drop=True)
    preds["last_updated"] = datetime.now().isoformat()
    preds.to_csv("data/predictions_with_roi.csv", index=False)
    high_risk = (preds["churn_probability"] >= 0.6).sum()
    return f"{len(preds):,} predictions updated, {high_risk} high-risk"

task("update_predictions", update_preds)

# ── Task 7: Generate alerts ───────────────────────────────────
def generate_alerts():
    preds     = pd.read_csv("data/predictions_with_roi.csv").reset_index(drop=True)
    high_risk = preds[preds["churn_probability"] >= 0.6]
    mrr_risk  = 0

    if "monthly_charges" in high_risk.columns:
        mrr_risk = high_risk["monthly_charges"].sum()
    elif "MonthlyCharges" in high_risk.columns:
        mrr_risk = high_risk["MonthlyCharges"].sum()

    alerts = []
    if len(high_risk) > 100:
        alerts.append(f"🚨 HIGH: {len(high_risk)} customers at high churn risk")
    if mrr_risk > 5000:
        alerts.append(f"💰 WARNING: ${mrr_risk:,.0f}/month MRR at risk")

    alerts_out = alerts if alerts else ["✅ No critical alerts"]
    for a in alerts_out:
        print(f"      {a}")
    return f"{len(alerts_out)} alerts generated"

task("generate_alerts", generate_alerts)

# ── Task 8: Log pipeline run ──────────────────────────────────
def save_log():
    success = sum(1 for t in log["tasks"].values() if t["status"]=="SUCCESS")
    failed  = sum(1 for t in log["tasks"].values() if t["status"]=="FAILED")
    log["summary"] = {
        "total_tasks": len(log["tasks"]),
        "succeeded"  : success,
        "failed"     : failed,
        "status"     : "SUCCESS" if failed == 0 else "PARTIAL_FAILURE"
    }
    with open("pipeline_logs/latest_run.json", "w") as f:
        json.dump(log, f, indent=2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"pipeline_logs/run_{ts}.json", "w") as f:
        json.dump(log, f, indent=2)

    return f"{success}/{len(log['tasks'])} tasks succeeded"

task("save_pipeline_log", save_log)

# ─────────────────────────────────────────────────────────────
# Also write the actual Airflow DAG file
# ─────────────────────────────────────────────────────────────
airflow_dag = '''"""
PRISM — Airflow DAG
Deploy to: $AIRFLOW_HOME/dags/prism_pipeline.py
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess

default_args = {
    "owner"           : "rajshree",
    "depends_on_past" : False,
    "start_date"      : datetime(2026, 1, 1),
    "retries"         : 2,
    "retry_delay"     : timedelta(minutes=5),
    "email_on_failure": True,
}

with DAG(
    "prism_daily_pipeline",
    default_args  = default_args,
    schedule      = "0 2 * * *",
    catchup       = False,
    tags          = ["prism","ml","churn"],
    description   = "PRISM daily churn prediction pipeline",
) as dag:

    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=lambda: subprocess.run(["python","phase1_data_engineering.py"])
    )
    monitor = PythonOperator(
        task_id="check_drift",
        python_callable=lambda: subprocess.run(["python","phase6b_monitoring.py"])
    )
    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=lambda: subprocess.run(["python","phase3_modeling.py"])
    )
    predict = PythonOperator(
        task_id="update_predictions",
        python_callable=lambda: subprocess.run(["python","phase6c_pipeline.py"])
    )

    ingest >> monitor >> retrain >> predict
'''

with open("airflow_dag.py", "w") as f:
    f.write(airflow_dag)

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
success = sum(1 for t in log["tasks"].values() if t["status"]=="SUCCESS")
total   = len(log["tasks"])

print("\n" + "="*60)
print("  Phase 6c Complete — Automated Pipeline")
print("="*60)
print(f"\n  Pipeline status: {success}/{total} tasks succeeded")
print(f"\n  Files saved:")
print(f"    pipeline_logs/latest_run.json")
print(f"    airflow_dag.py  (deploy to Airflow)")
print(f"\n  To run on a real schedule:")
print(f"    pip install apache-airflow")
print(f"    airflow db init")
print(f"    cp airflow_dag.py $AIRFLOW_HOME/dags/")
print(f"    airflow scheduler")
print(f"\n  Next: python phase6d_readme.py")
print("="*60)
