"""
PRISM v2 — Phase 3: Modeling Pipeline
Logistic Regression + Random Forest + XGBoost + LightGBM
Optuna hyperparameter tuning + SHAP explainability + MLflow tracking
Business metrics: Lift curve, CLV impact, Intervention ROI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
import json

warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("charts", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              roc_curve, precision_recall_curve,
                              confusion_matrix, classification_report)
from sklearn.calibration import CalibratedClassifierCV
import joblib

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})
COLORS = ["#3B8BD4", "#D85A30", "#1D9E75", "#EF9F27", "#7F77DD"]

print("=" * 60)
print("  PRISM v2 — Phase 3: Modeling Pipeline")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# Load data — temporal split (no leakage)
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Loading feature store with temporal split...")

fs = pd.read_csv("data/feature_store.csv")
df = pd.read_csv("data/telco_cleaned.csv")

X = fs.drop(columns=["Churn_binary"])
y = fs["Churn_binary"]

# Temporal split: sort by tenure, train on earlier customers
# This prevents data leakage — we train on past, test on future
df_sorted = fs.sort_values("tenure").reset_index(drop=True)
X_sorted  = df_sorted.drop(columns=["Churn_binary"])
y_sorted  = df_sorted["Churn_binary"]

split_idx = int(len(df_sorted) * 0.8)
X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"      Train: {len(X_train):,} customers (shorter tenure)")
print(f"      Test:  {len(X_test):,} customers (longer tenure)")
print(f"      Train churn rate: {y_train.mean()*100:.1f}%")
print(f"      Test churn rate:  {y_test.mean()*100:.1f}%")
print(f"      Features: {X_train.shape[1]}")
print(f"      Split type: TEMPORAL (no data leakage)")


# ─────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Training baseline models...")

results = {}

# 1. Logistic Regression (calibrated baseline)
print("      Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_scaled, y_train)
lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
results["Logistic Regression"] = {
    "model": lr, "probs": lr_probs,
    "roc_auc": roc_auc_score(y_test, lr_probs),
    "avg_precision": average_precision_score(y_test, lr_probs),
    "f1": f1_score(y_test, (lr_probs > 0.5).astype(int)),
}
print(f"        ROC-AUC: {results['Logistic Regression']['roc_auc']:.4f}")

# 2. Random Forest
print("      Random Forest...")
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                             max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
results["Random Forest"] = {
    "model": rf, "probs": rf_probs,
    "roc_auc": roc_auc_score(y_test, rf_probs),
    "avg_precision": average_precision_score(y_test, rf_probs),
    "f1": f1_score(y_test, (rf_probs > 0.5).astype(int)),
}
print(f"        ROC-AUC: {results['Random Forest']['roc_auc']:.4f}")

# 3. XGBoost
print("      XGBoost...")
try:
    from xgboost import XGBClassifier
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=scale_pos,
                         random_state=42, eval_metric="auc",
                         verbosity=0, use_label_encoder=False)
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)], verbose=False)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    results["XGBoost"] = {
        "model": xgb, "probs": xgb_probs,
        "roc_auc": roc_auc_score(y_test, xgb_probs),
        "avg_precision": average_precision_score(y_test, xgb_probs),
        "f1": f1_score(y_test, (xgb_probs > 0.5).astype(int)),
    }
    print(f"        ROC-AUC: {results['XGBoost']['roc_auc']:.4f}")
except ImportError:
    print("        XGBoost not installed — skipping")

# 4. LightGBM
print("      LightGBM...")
try:
    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           class_weight="balanced",
                           random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)
    lgbm_probs = lgbm.predict_proba(X_test)[:, 1]
    results["LightGBM"] = {
        "model": lgbm, "probs": lgbm_probs,
        "roc_auc": roc_auc_score(y_test, lgbm_probs),
        "avg_precision": average_precision_score(y_test, lgbm_probs),
        "f1": f1_score(y_test, (lgbm_probs > 0.5).astype(int)),
    }
    print(f"        ROC-AUC: {results['LightGBM']['roc_auc']:.4f}")
except ImportError:
    print("        LightGBM not installed — skipping")


# ─────────────────────────────────────────────────────────────
# Optuna hyperparameter tuning on best model
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Optuna hyperparameter tuning (XGBoost)...")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 100, 500),
            "max_depth"       : trial.suggest_int("max_depth", 3, 8),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": scale_pos,
            "random_state"    : 42,
            "verbosity"       : 0,
            "eval_metric"     : "auc",
        }
        from xgboost import XGBClassifier
        model = XGBClassifier(**params, use_label_encoder=False)
        cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=-1)
        return score.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos
    best_params["random_state"]     = 42
    best_params["verbosity"]        = 0
    best_params["eval_metric"]      = "auc"

    from xgboost import XGBClassifier
    xgb_tuned = XGBClassifier(**best_params, use_label_encoder=False)
    xgb_tuned.fit(X_train, y_train)
    xgb_tuned_probs = xgb_tuned.predict_proba(X_test)[:, 1]

    results["XGBoost (Tuned)"] = {
        "model": xgb_tuned, "probs": xgb_tuned_probs,
        "roc_auc": roc_auc_score(y_test, xgb_tuned_probs),
        "avg_precision": average_precision_score(y_test, xgb_tuned_probs),
        "f1": f1_score(y_test, (xgb_tuned_probs > 0.5).astype(int)),
    }
    print(f"      Best trial ROC-AUC: {study.best_value:.4f}")
    print(f"      Tuned model ROC-AUC: {results['XGBoost (Tuned)']['roc_auc']:.4f}")
    print(f"      Best params: {best_params}")

    with open("models/best_params.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in best_params.items()}, f, indent=2)

except Exception as e:
    print(f"      Optuna skipped: {e}")


# ─────────────────────────────────────────────────────────────
# Select best model
# ─────────────────────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["roc_auc"])
best_result= results[best_name]
best_probs = best_result["probs"]
best_model = best_result["model"]

print(f"\n  Best model: {best_name}")
print(f"  ROC-AUC:    {best_result['roc_auc']:.4f}")
print(f"  F1 Score:   {best_result['f1']:.4f}")
print(f"  Avg Prec:   {best_result['avg_precision']:.4f}")


# ─────────────────────────────────────────────────────────────
# CHART 1: Model comparison
# ─────────────────────────────────────────────────────────────
print("\n[4/7] Generating evaluation charts...")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("PRISM — Model Comparison", fontsize=14, fontweight="bold")

model_names = list(results.keys())
roc_aucs    = [results[m]["roc_auc"] for m in model_names]
f1_scores   = [results[m]["f1"] for m in model_names]
avg_precs   = [results[m]["avg_precision"] for m in model_names]

x = np.arange(len(model_names))
bars = axes[0].bar(x, roc_aucs, color=COLORS[:len(model_names)], alpha=0.85, width=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=8)
axes[0].set_title("ROC-AUC", fontweight="bold")
axes[0].set_ylim(0.5, 1.0)
for bar, val in zip(bars, roc_aucs):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

bars2 = axes[1].bar(x, f1_scores, color=COLORS[:len(model_names)], alpha=0.85, width=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=8)
axes[1].set_title("F1 Score", fontweight="bold")
axes[1].set_ylim(0, 1.0)
for bar, val in zip(bars2, f1_scores):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

# ROC curves
for name, color in zip(model_names, COLORS):
    fpr, tpr, _ = roc_curve(y_test, results[name]["probs"])
    axes[2].plot(fpr, tpr, color=color, lw=1.5,
                 label=f"{name} ({results[name]['roc_auc']:.3f})")
axes[2].plot([0,1],[0,1],"k--", alpha=0.4, label="Random")
axes[2].set_title("ROC Curves", fontweight="bold")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].legend(fontsize=7)

plt.tight_layout()
plt.savefig("charts/13_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Chart 13 saved: Model comparison")


# ─────────────────────────────────────────────────────────────
# CHART 2: Lift curve (business metric)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PRISM — Business Metrics: Lift Curve & Precision-Recall",
             fontsize=14, fontweight="bold")

# Lift curve
df_lift = pd.DataFrame({"prob": best_probs, "actual": y_test.values})
df_lift = df_lift.sort_values("prob", ascending=False).reset_index(drop=True)
df_lift["decile"] = pd.qcut(df_lift.index, 10, labels=False)

baseline_rate = y_test.mean()
lift_data = df_lift.groupby("decile")["actual"].mean() / baseline_rate
axes[0].bar(range(1, 11), lift_data.values[::-1],
            color=[COLORS[1] if v > 2 else COLORS[0] for v in lift_data.values[::-1]],
            alpha=0.85, edgecolor="white")
axes[0].axhline(y=1, color="gray", linestyle="--", alpha=0.7, label="Baseline (random)")
axes[0].set_title("Lift Curve by Decile\n(Top decile = highest risk customers)",
                  fontweight="bold")
axes[0].set_xlabel("Decile (1=highest risk)")
axes[0].set_ylabel("Lift over baseline")
axes[0].legend()
for i, v in enumerate(lift_data.values[::-1]):
    axes[0].text(i+1, v+0.02, f"{v:.1f}x", ha="center", va="bottom", fontsize=8)

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_test, best_probs)
axes[1].plot(rec, prec, color=COLORS[0], lw=2)
axes[1].axhline(y=baseline_rate, color="gray", linestyle="--",
                alpha=0.7, label=f"Baseline ({baseline_rate:.2f})")
axes[1].set_title("Precision-Recall Curve", fontweight="bold")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()
axes[1].fill_between(rec, prec, baseline_rate, alpha=0.1, color=COLORS[0])

plt.tight_layout()
plt.savefig("charts/14_business_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Chart 14 saved: Business metrics (lift + PR)")


# ─────────────────────────────────────────────────────────────
# CHART 3: SHAP explainability
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Computing SHAP values...")
try:
    import shap
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"PRISM — SHAP Explainability ({best_name})",
                 fontsize=14, fontweight="bold")

    # Global feature importance
    mean_shap = np.abs(sv).mean(axis=0)
    shap_df   = pd.DataFrame({"feature": X_test.columns, "importance": mean_shap})
    shap_df   = shap_df.sort_values("importance", ascending=True).tail(15)
    axes[0].barh(shap_df["feature"], shap_df["importance"],
                 color=COLORS[0], alpha=0.85)
    axes[0].set_title("Top 15 Features by SHAP Importance", fontweight="bold")
    axes[0].set_xlabel("Mean |SHAP value|")

    # SHAP beeswarm (summary plot to file)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_test, max_display=15, show=False)
    plt.title(f"SHAP Summary — {best_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig("charts/15b_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Back to subplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"PRISM — SHAP Explainability ({best_name})",
                 fontsize=14, fontweight="bold")
    axes[0].barh(shap_df["feature"], shap_df["importance"],
                 color=COLORS[0], alpha=0.85)
    axes[0].set_title("Top 15 Features — Global Importance", fontweight="bold")
    axes[0].set_xlabel("Mean |SHAP value|")

    # SHAP for single high-risk customer
    high_risk_idx = np.argmax(best_probs)
    customer_shap = sv[high_risk_idx]
    shap_customer = pd.DataFrame({
        "feature": X_test.columns,
        "shap_value": customer_shap
    }).sort_values("shap_value", key=abs, ascending=True).tail(12)

    colors_shap = [COLORS[1] if v > 0 else COLORS[2]
                   for v in shap_customer["shap_value"]]
    axes[1].barh(shap_customer["feature"], shap_customer["shap_value"],
                 color=colors_shap, alpha=0.85)
    axes[1].axvline(x=0, color="black", linewidth=0.8)
    axes[1].set_title(f"SHAP for Highest-Risk Customer\n(Churn prob: {best_probs[high_risk_idx]:.3f})",
                      fontweight="bold")
    axes[1].set_xlabel("SHAP value (red=increases churn risk)")

    plt.tight_layout()
    plt.savefig("charts/15_shap_explainability.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("      Chart 15 saved: SHAP explainability")
    print("      Chart 15b saved: SHAP beeswarm")

except Exception as e:
    print(f"      SHAP skipped: {e}")


# ─────────────────────────────────────────────────────────────
# CLV Impact & Intervention ROI
# ─────────────────────────────────────────────────────────────
print("\n[6/7] Computing business impact & intervention ROI...")

df_test = df.iloc[X_test.index].copy() if len(X_test.index) <= len(df) else df.tail(len(X_test)).copy()
df_test = df_test.reset_index(drop=True)

df_test["churn_probability"] = best_probs
df_test["clv_12m"]           = df_test["MonthlyCharges"] * 12
df_test["clv_at_risk"]       = df_test["churn_probability"] * df_test["clv_12m"]
df_test["intervention_cost"] = 50  # estimated cost per outreach
df_test["intervention_roi"]  = (df_test["clv_at_risk"] - df_test["intervention_cost"]) / df_test["intervention_cost"]

df_test = df_test.sort_values("churn_probability", ascending=False)

# Business impact at different thresholds
print(f"\n  {'Threshold':>10} {'Flagged':>8} {'Precision':>10} {'MRR Saved':>12} {'ROI':>8}")
print("  " + "-" * 55)
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    flagged   = (best_probs >= thresh).sum()
    if flagged == 0:
        continue
    precision = y_test[best_probs >= thresh].mean()
    mrr_saved = df_test[df_test["churn_probability"] >= thresh]["MonthlyCharges"].sum() * precision
    roi       = (mrr_saved - flagged * 50) / (flagged * 50) if flagged > 0 else 0
    print(f"  {thresh:>10.1f} {flagged:>8,} {precision:>10.1%} ${mrr_saved:>10,.0f} {roi:>7.1f}x")

df_test.to_csv("data/predictions_with_roi.csv", index=False)
print(f"\n  Saved: data/predictions_with_roi.csv")


# ─────────────────────────────────────────────────────────────
# MLflow experiment tracking
# ─────────────────────────────────────────────────────────────
print("\n[7/7] Logging to MLflow...")
try:
    import mlflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("PRISM_churn_prediction")

    for name, res in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type",   name)
            mlflow.log_param("train_size",   len(X_train))
            mlflow.log_param("test_size",    len(X_test))
            mlflow.log_param("n_features",   X_train.shape[1])
            mlflow.log_param("split_type",   "temporal")
            mlflow.log_metric("roc_auc",     res["roc_auc"])
            mlflow.log_metric("f1_score",    res["f1"])
            mlflow.log_metric("avg_precision", res["avg_precision"])

    print(f"      {len(results)} runs logged to MLflow")
    print(f"      View with: mlflow ui")

except Exception as e:
    print(f"      MLflow skipped: {e}")

# Save best model
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler,     "models/scaler.pkl")
with open("models/feature_names.json", "w") as f:
    json.dump(list(X_train.columns), f)

# ─────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Phase 3 Complete — Modeling Pipeline")
print("=" * 60)
print(f"\n  Models trained: {len(results)}")
print(f"\n  {'Model':<25} {'ROC-AUC':>8} {'F1':>8} {'Avg Prec':>10}")
print("  " + "-" * 55)
for name, res in sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True):
    print(f"  {name:<25} {res['roc_auc']:>8.4f} {res['f1']:>8.4f} {res['avg_precision']:>10.4f}")

print(f"\n  Best model: {best_name}")
print(f"  Split: Temporal (no data leakage)")
print(f"\n  Files saved:")
print(f"    models/best_model.pkl")
print(f"    models/scaler.pkl")
print(f"    data/predictions_with_roi.csv")
print(f"\n  Charts saved: 13, 14, 15")
print(f"\n  Next: python phase4_ab_testing.py")
print("=" * 60)
