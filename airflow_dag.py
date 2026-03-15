"""
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
