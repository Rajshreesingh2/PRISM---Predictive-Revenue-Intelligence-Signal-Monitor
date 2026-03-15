"""
PRISM v2 — Phase 3: Modeling Pipeline
XGBoost + LightGBM + Logistic Regression + Random Forest
Optuna tuning + SHAP explainability + Business metrics
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, roc_curve, precision_recall_curve,
                              average_precision_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import shap
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
# STEP 1: Load & Temporal Split
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Loading feature store with temporal split...")

fs = pd.read_csv("data/feature_store.csv")
telco = pd.read_csv("data/telco_cleaned.csv")

# Add tenure back from telco for temporal split
fs["tenure"] = telco["tenure"].values

# Temporal split — sort by tenure, 80/20
fs_sorted = fs.sort_values("tenure").reset_index(drop=True)
split_idx  = int(len(fs_sorted) * 0.8)
train_df   = fs_sorted.iloc[:split_idx]
test_df    = fs_sorted.iloc[split_idx:]

feature_cols = [c for c in fs.columns if c not in ["Churn_binary", "tenure"]]

X_train = train_df[feature_cols].fillna(0)
y_train = train_df["Churn_binary"]
X_test  = test_df[feature_cols].fillna(0)
y_test  = test_df["Churn_binary"]

print(f"      Train: {len(X_train):,} customers (temporal 80%)")
print(f"      Test:  {len(X_test):,} customers (temporal 20%)")
print(f"      Features: {len(feature_cols)}")
print(f"      Train churn rate: {y_train.mean()*100:.1f}%")
print(f"      Test churn rate:  {y_test.mean()*100:.1f}%")

scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ─────────────────────────────────────────────────────────────
# STEP 2: Baseline — Logistic Regression
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Training models...")
print("      Logistic Regression (baseline)...")

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train_sc, y_train)
lr_probs = lr.predict_proba(X_test_sc)[:, 1]
lr_preds = (lr_probs >= 0.5).astype(int)

print(f"        ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 3: Random Forest
# ─────────────────────────────────────────────────────────────
print("      Random Forest...")

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                             class_weight="balanced", n_jobs=-1)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_preds = (rf_probs >= 0.5).astype(int)

print(f"        ROC-AUC: {roc_auc_score(y_test, rf_probs):.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 4: XGBoost with Optuna tuning
# ─────────────────────────────────────────────────────────────
print("      XGBoost + Optuna tuning (50 trials)...")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    def xgb_objective(trial):
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 100, 500),
            "max_depth"       : trial.suggest_int("max_depth", 3, 8),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": scale_pos,
            "random_state"    : 42,
            "eval_metric"     : "auc",
            "verbosity"       : 0,
        }
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(xgb_objective, n_trials=50, show_progress_bar=False)

    best_xgb_params = study.best_params
    best_xgb_params["scale_pos_weight"] = scale_pos
    best_xgb_params["random_state"]     = 42
    best_xgb_params["verbosity"]        = 0
    print(f"        Best CV AUC: {study.best_value:.4f}")

except ImportError:
    print("        Optuna not found — using default params")
    best_xgb_params = {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": (y_train==0).sum()/(y_train==1).sum(),
        "random_state": 42, "verbosity": 0
    }

xgb_model = xgb.XGBClassifier(**best_xgb_params)
xgb_model.fit(X_train, y_train)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_preds = (xgb_probs >= 0.5).astype(int)
print(f"        ROC-AUC (test): {roc_auc_score(y_test, xgb_probs):.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 5: LightGBM
# ─────────────────────────────────────────────────────────────
print("      LightGBM...")

lgbm = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            class_weight="balanced", random_state=42,
                            verbose=-1)
lgbm.fit(X_train, y_train)
lgbm_probs = lgbm.predict_proba(X_test)[:, 1]
lgbm_preds = (lgbm_probs >= 0.5).astype(int)
print(f"        ROC-AUC: {roc_auc_score(y_test, lgbm_probs):.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 6: Model Comparison
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Comparing models...")

models = {
    "Logistic Regression": (lr_probs,  lr_preds),
    "Random Forest"      : (rf_probs,  rf_preds),
    "XGBoost"            : (xgb_probs, xgb_preds),
    "LightGBM"           : (lgbm_probs,lgbm_preds),
}

results = {}
print(f"\n  {'Model':<22} {'ROC-AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Avg-PR':>8}")
print("  " + "-" * 70)
for name, (probs, preds) in models.items():
    auc  = roc_auc_score(y_test, probs)
    f1   = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)
    apr  = average_precision_score(y_test, probs)
    results[name] = {"auc": auc, "f1": f1, "precision": prec,
                     "recall": rec, "avg_pr": apr}
    print(f"  {name:<22} {auc:>8.4f} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f} {apr:>8.4f}")

best_model_name = max(results, key=lambda x: results[x]["auc"])
print(f"\n  Best model: {best_model_name} (ROC-AUC={results[best_model_name]['auc']:.4f})")

best_probs = models[best_model_name][0]
best_model = {"Logistic Regression": lr, "Random Forest": rf,
              "XGBoost": xgb_model, "LightGBM": lgbm}[best_model_name]


# ─────────────────────────────────────────────────────────────
# CHART 1: ROC + PR curves
# ─────────────────────────────────────────────────────────────
print("\n[4/7] Generating evaluation charts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("PRISM — Model Evaluation", fontsize=14, fontweight="bold")

for (name, (probs, _)), color in zip(models.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = results[name]["auc"]
    axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")

axes[0].plot([0,1],[0,1], "k--", alpha=0.3, lw=1)
axes[0].set_title("ROC Curves", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(fontsize=9)

for (name, (probs, _)), color in zip(models.items(), COLORS):
    prec_c, rec_c, _ = precision_recall_curve(y_test, probs)
    apr = results[name]["avg_pr"]
    axes[1].plot(rec_c, prec_c, color=color, lw=2, label=f"{name} (AP={apr:.3f})")

axes[1].set_title("Precision-Recall Curves", fontweight="bold")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("charts/13_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Chart 13 saved: ROC + PR curves")


# ─────────────────────────────────────────────────────────────
# CHART 2: Lift curve (business metric)
# ─────────────────────────────────────────────────────────────
print("      Computing lift curve...")

telco_test = telco.iloc[test_df.index] if len(telco) == len(fs) else telco.iloc[-len(X_test):]
monthly_charges = telco_test["MonthlyCharges"].values if len(telco_test) == len(X_test) else np.full(len(X_test), 65)

lift_df = pd.DataFrame({
    "prob"           : best_probs,
    "actual"         : y_test.values,
    "monthly_charges": monthly_charges
}).sort_values("prob", ascending=False).reset_index(drop=True)

lift_df["cumulative_churn"]    = lift_df["actual"].cumsum()
lift_df["cumulative_pct"]      = (lift_df.index + 1) / len(lift_df)
lift_df["cumulative_mrr_saved"]= (lift_df["actual"] * lift_df["monthly_charges"]).cumsum()

total_churners = lift_df["actual"].sum()
lift_df["lift"] = (lift_df["cumulative_churn"] / (lift_df.index + 1)) / (total_churners / len(lift_df))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PRISM — Business Impact Metrics", fontsize=14, fontweight="bold")

axes[0].plot(lift_df["cumulative_pct"]*100, lift_df["lift"],
             color=COLORS[0], lw=2.5, label=f"{best_model_name}")
axes[0].axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
axes[0].fill_between(lift_df["cumulative_pct"]*100, 1, lift_df["lift"],
                      alpha=0.15, color=COLORS[0])
axes[0].set_title("Lift Curve — Model vs Random Targeting", fontweight="bold")
axes[0].set_xlabel("% of Customers Contacted")
axes[0].set_ylabel("Lift (x better than random)")
axes[0].legend()

top20_mrr = lift_df.iloc[:int(len(lift_df)*0.2)]["cumulative_mrr_saved"].max()
axes[1].plot(lift_df["cumulative_pct"]*100, lift_df["cumulative_mrr_saved"],
             color=COLORS[1], lw=2.5)
axes[1].axvline(x=20, color="gray", linestyle="--", alpha=0.5)
axes[1].text(21, top20_mrr*0.8, f"Top 20%:\n${top20_mrr:,.0f} MRR\nsaved",
             fontsize=9, color=COLORS[1])
axes[1].set_title("Cumulative MRR Saved by Intervention", fontweight="bold")
axes[1].set_xlabel("% of Customers Contacted")
axes[1].set_ylabel("Cumulative MRR Saved ($)")

plt.tight_layout()
plt.savefig("charts/14_lift_business_impact.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Chart 14 saved: Lift + business impact")


# ─────────────────────────────────────────────────────────────
# CHART 3: SHAP explainability
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Computing SHAP values...")

try:
    explainer   = shap.TreeExplainer(best_model if best_model_name != "Logistic Regression"
                                     else xgb_model)
    shap_model  = best_model if best_model_name != "Logistic Regression" else xgb_model
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("PRISM — SHAP Explainability", fontsize=14, fontweight="bold")

    # Global feature importance
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df   = pd.DataFrame({"feature": feature_cols, "importance": mean_shap})
    shap_df   = shap_df.sort_values("importance", ascending=True).tail(15)

    axes[0].barh(shap_df["feature"], shap_df["importance"],
                 color=COLORS[0], alpha=0.85)
    axes[0].set_title("Top 15 Features by SHAP Importance", fontweight="bold")
    axes[0].set_xlabel("Mean |SHAP value|")

    # SHAP summary dot plot (manual)
    top_features = shap_df["feature"].tolist()[-10:]
    top_idx      = [list(feature_cols).index(f) for f in top_features]

    for i, (feat, idx) in enumerate(zip(top_features, top_idx)):
        feat_vals  = X_test.iloc[:, idx].values
        shap_vals  = shap_values[:, idx]
        feat_norm  = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-8)
        colors_dot = plt.cm.RdYlBu_r(feat_norm)
        axes[1].scatter(shap_vals, np.full_like(shap_vals, i),
                        c=colors_dot, alpha=0.4, s=8)

    axes[1].set_yticks(range(len(top_features)))
    axes[1].set_yticklabels(top_features, fontsize=9)
    axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("SHAP Value Distribution (top 10)", fontweight="bold")
    axes[1].set_xlabel("SHAP value (impact on churn probability)")

    plt.tight_layout()
    plt.savefig("charts/15_shap_explainability.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("      Chart 15 saved: SHAP explainability")

except Exception as e:
    print(f"      SHAP chart skipped: {e}")


# ─────────────────────────────────────────────────────────────
# STEP 7: Per-customer output with archetype
# ─────────────────────────────────────────────────────────────
print("\n[6/7] Building per-customer prediction output...")

archetypes_df = pd.read_csv("data/churn_archetypes.csv", index_col=0)
survival_df   = pd.read_csv("data/survival_predictions.csv")

output_df = test_df[["Churn_binary"]].copy()
output_df["churn_probability"]  = best_probs
output_df["risk_tier"]          = pd.cut(best_probs,
                                          bins=[0, 0.3, 0.6, 1.0],
                                          labels=["Low", "Medium", "High"])
output_df["monthly_charges"]    = monthly_charges

# Merge survival predictions
if "churn_prob_30d" in survival_df.columns:
    output_df["churn_prob_30d"] = survival_df.loc[output_df.index, "churn_prob_30d"].values \
                                   if len(survival_df) == len(fs) else np.nan
    output_df["months_remaining"] = survival_df.loc[output_df.index, "expected_months_remaining"].values \
                                     if len(survival_df) == len(fs) else np.nan

# Merge archetype
if len(archetypes_df) > 0:
    output_df["archetype"] = archetypes_df.reindex(output_df.index)["archetype_name"].values

# CLV at risk
output_df["clv_at_risk_12m"]     = output_df["monthly_charges"] * 12 * output_df["churn_probability"]
output_df["intervention_priority"]= output_df["clv_at_risk_12m"].rank(ascending=False).astype(int)

output_df.to_csv("data/customer_predictions.csv")

print(f"      Saved: data/customer_predictions.csv")
print(f"\n      Sample — top 5 highest priority customers:")
top5 = output_df.nsmallest(5, "intervention_priority")[
    ["churn_probability", "risk_tier", "monthly_charges",
     "clv_at_risk_12m", "intervention_priority"]
]
print(top5.to_string())


# ─────────────────────────────────────────────────────────────
# Save models + results
# ─────────────────────────────────────────────────────────────
print("\n[7/7] Saving models and results...")

joblib.dump(xgb_model, "models/xgboost_model.pkl")
joblib.dump(lgbm,      "models/lightgbm_model.pkl")
joblib.dump(scaler,    "models/scaler.pkl")

with open("models/model_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Business impact summary
high_risk   = output_df[output_df["risk_tier"] == "High"]
total_mrr   = output_df["monthly_charges"].sum()
hr_mrr      = high_risk["monthly_charges"].sum()
top20_count = int(len(output_df) * 0.2)
top20_mrr_v = output_df.nsmallest(top20_count, "intervention_priority")["monthly_charges"].sum()

print("\n" + "=" * 60)
print("  Phase 3 Complete — Modeling Pipeline")
print("=" * 60)
print(f"\n  Best model    : {best_model_name}")
print(f"  ROC-AUC       : {results[best_model_name]['auc']:.4f}")
print(f"  F1 Score      : {results[best_model_name]['f1']:.4f}")
print(f"  Precision     : {results[best_model_name]['precision']:.4f}")
print(f"  Recall        : {results[best_model_name]['recall']:.4f}")
print(f"\n  Business Impact (test set):")
print(f"  Total MRR             : ${total_mrr:,.0f}/month")
print(f"  High-risk MRR         : ${hr_mrr:,.0f}/month")
print(f"  Top 20% targeting MRR : ${top20_mrr_v:,.0f}/month recoverable")
print(f"\n  Charts saved  : 13, 14, 15")
print(f"  Models saved  : models/")
print(f"  Predictions   : data/customer_predictions.csv")
print(f"\n  Next: python phase2c_clustering.py (if not done)")
print(f"  Then: build Streamlit dashboard")
print("=" * 60)