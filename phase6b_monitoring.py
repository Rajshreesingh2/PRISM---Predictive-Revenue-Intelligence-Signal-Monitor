"""
PRISM v2 — Phase 6b: Model Monitoring with Evidently AI
Data drift detection + model performance tracking
Production ML systems monitor for when data distribution shifts
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("monitoring", exist_ok=True)

print("="*60)
print("  PRISM v2 — Phase 6b: Model Monitoring")
print("="*60)

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
fs      = pd.read_csv("data/feature_store.csv")
telco   = pd.read_csv("data/telco_cleaned.csv")
preds   = pd.read_csv("data/predictions_with_roi.csv").reset_index(drop=True)

# Simulate reference (training) vs current (production) data
# In production: reference = last month, current = this month
split   = int(len(fs) * 0.7)
reference_data = fs.iloc[:split].copy()
current_data   = fs.iloc[split:].copy()

print(f"\n  Reference data (training period): {len(reference_data):,} rows")
print(f"  Current data  (production period): {len(current_data):,} rows")

# ─────────────────────────────────────────────────────────────
# Try Evidently — fallback to manual drift detection
# ─────────────────────────────────────────────────────────────
evidently_available = False
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
    from evidently.metrics import *
    evidently_available = True
    print("  Evidently AI: available")
except ImportError:
    print("  Evidently AI: not installed — using manual drift detection")
    print("  (Run: pip install evidently  to enable full Evidently reports)")

# ─────────────────────────────────────────────────────────────
# Manual drift detection (always runs)
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("  Running statistical drift detection...")

numeric_cols = fs.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "Churn_binary"][:15]

from scipy import stats

drift_results = []
for col in numeric_cols:
    ref_vals = reference_data[col].dropna()
    cur_vals = current_data[col].dropna()
    if len(ref_vals) < 10 or len(cur_vals) < 10:
        continue

    # KS test for distribution drift
    ks_stat, ks_p = stats.ks_2samp(ref_vals, cur_vals)

    # Mean shift
    mean_shift = abs(cur_vals.mean() - ref_vals.mean()) / (ref_vals.std() + 1e-8)

    drift_detected = ks_p < 0.05 or mean_shift > 0.1

    drift_results.append({
        "feature"         : col,
        "ref_mean"        : round(ref_vals.mean(), 4),
        "cur_mean"        : round(cur_vals.mean(), 4),
        "mean_shift_sigma": round(mean_shift, 4),
        "ks_statistic"    : round(ks_stat, 4),
        "ks_p_value"      : round(ks_p, 4),
        "drift_detected"  : drift_detected,
        "severity"        : "HIGH" if ks_p < 0.01 else "MEDIUM" if ks_p < 0.05 else "LOW"
    })

drift_df = pd.DataFrame(drift_results).sort_values("ks_statistic", ascending=False)

drifted   = drift_df[drift_df["drift_detected"]==True]
no_drift  = drift_df[drift_df["drift_detected"]==False]

print(f"\n  Features checked: {len(drift_df)}")
print(f"  Drift detected:   {len(drifted)} features")
print(f"  Stable features:  {len(no_drift)} features")

if len(drifted) > 0:
    print(f"\n  {'Feature':<35} {'KS Stat':>8} {'p-value':>10} {'Severity':>10}")
    print("  " + "─"*65)
    for _, row in drifted.iterrows():
        print(f"  {row['feature']:<35} {row['ks_statistic']:>8.4f} {row['ks_p_value']:>10.4f} {row['severity']:>10}")

# ─────────────────────────────────────────────────────────────
# Model performance monitoring
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("  Model performance on reference vs current data...")

import joblib
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

try:
    model  = joblib.load("models/best_model.pkl")
    with open("data/selected_features.json") as f:
        feat_cols = [c for c in json.load(f) if c != "Churn_binary"]

    perf_results = {}
    for name, dataset in [("reference", reference_data), ("current", current_data)]:
        cols = [c for c in feat_cols if c in dataset.columns]
        X = dataset[cols].fillna(0)
        y = dataset["Churn_binary"]
        if len(y.unique()) < 2:
            continue
        probs = model.predict_proba(X)[:, 1]
        preds_bin = (probs >= 0.5).astype(int)
        perf_results[name] = {
            "roc_auc"  : round(roc_auc_score(y, probs), 4),
            "f1"       : round(f1_score(y, preds_bin), 4),
            "precision": round(precision_score(y, preds_bin, zero_division=0), 4),
            "recall"   : round(recall_score(y, preds_bin, zero_division=0), 4),
            "n_samples": len(y),
            "churn_rate": round(y.mean(), 4),
        }

    if perf_results:
        print(f"\n  {'Metric':<15} {'Reference':>12} {'Current':>12} {'Delta':>10}")
        print("  " + "─"*52)
        ref = perf_results.get("reference", {})
        cur = perf_results.get("current", {})
        for metric in ["roc_auc","f1","precision","recall"]:
            r = ref.get(metric, 0)
            c = cur.get(metric, 0)
            delta = c - r
            flag = " ⚠️" if abs(delta) > 0.05 else ""
            print(f"  {metric:<15} {r:>12.4f} {c:>12.4f} {delta:>+10.4f}{flag}")

    performance_report = perf_results
except Exception as e:
    print(f"  Performance monitoring skipped: {e}")
    performance_report = {}

# ─────────────────────────────────────────────────────────────
# Prediction distribution monitoring
# ─────────────────────────────────────────────────────────────
print("\n" + "─"*50)
print("  Prediction distribution analysis...")

if len(preds) > 0 and "churn_probability" in preds.columns:
    prob_col = preds["churn_probability"]
    dist_report = {
        "mean_probability"  : round(prob_col.mean(), 4),
        "median_probability": round(prob_col.median(), 4),
        "std_probability"   : round(prob_col.std(), 4),
        "pct_high_risk"     : round((prob_col >= 0.6).mean() * 100, 2),
        "pct_medium_risk"   : round(((prob_col >= 0.3) & (prob_col < 0.6)).mean() * 100, 2),
        "pct_low_risk"      : round((prob_col < 0.3).mean() * 100, 2),
    }
    print(f"\n  Mean churn probability:  {dist_report['mean_probability']:.4f}")
    print(f"  High risk (>60%):        {dist_report['pct_high_risk']:.1f}%")
    print(f"  Medium risk (30-60%):    {dist_report['pct_medium_risk']:.1f}%")
    print(f"  Low risk (<30%):         {dist_report['pct_low_risk']:.1f}%")
else:
    dist_report = {}

# ─────────────────────────────────────────────────────────────
# Evidently full report (if available)
# ─────────────────────────────────────────────────────────────
if evidently_available:
    try:
        print("\n" + "─"*50)
        print("  Generating Evidently AI report...")

        feat_sample = numeric_cols[:10]
        ref_ev = reference_data[feat_sample + ["Churn_binary"]].fillna(0).head(500)
        cur_ev = current_data[feat_sample + ["Churn_binary"]].fillna(0).head(200)

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=ref_ev, current_data=cur_ev)
        report.save_html("monitoring/evidently_report.html")
        print("  Evidently report saved: monitoring/evidently_report.html")
    except Exception as e:
        print(f"  Evidently report failed: {e}")

# ─────────────────────────────────────────────────────────────
# Save monitoring results
# ─────────────────────────────────────────────────────────────
monitoring_summary = {
    "generated_at"        : pd.Timestamp.now().isoformat(),
    "drift_summary": {
        "features_checked": len(drift_df),
        "drifted_features": len(drifted),
        "stable_features" : len(no_drift),
        "alert_level"     : "HIGH" if len(drifted) > 5 else "MEDIUM" if len(drifted) > 2 else "LOW",
    },
    "drifted_features"    : drift_df[drift_df["drift_detected"]].to_dict(orient="records"),
    "performance"         : performance_report,
    "prediction_dist"     : dist_report,
}

with open("monitoring/monitoring_summary.json", "w") as f:
    json.dump(monitoring_summary, f, indent=2, default=str)

drift_df.to_csv("monitoring/drift_report.csv", index=False)

print("\n" + "="*60)
print("  Phase 6b Complete — Model Monitoring")
print("="*60)
print(f"\n  Drift alert level: {monitoring_summary['drift_summary']['alert_level']}")
print(f"  Drifted features:  {len(drifted)}/{len(drift_df)}")
print(f"  Files saved:")
print(f"    monitoring/monitoring_summary.json")
print(f"    monitoring/drift_report.csv")
if evidently_available:
    print(f"    monitoring/evidently_report.html")
print(f"\n  Next: python phase6c_airflow_pipeline.py")
print("="*60)
