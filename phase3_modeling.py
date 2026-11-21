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
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_val_predict, train_test_split)
from sklearn.base import clone
from sklearn.pipeline import Pipeline
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
RANDOM_STATE = 42
# Business ASSUMPTIONS (not measured from data) used for the ROI columns:
INTERVENTION_COST = 50     # $ per retention offer
ASSUMED_SAVE_RATE = 0.15   # share of at-risk customers an offer is assumed to keep
COLORS = ["#3B8BD4", "#D85A30", "#1D9E75", "#EF9F27", "#7F77DD"]

print("=" * 60)
print("  PRISM v2 — Phase 3: Modeling Pipeline")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load & Stratified Train/Test Split
# ─────────────────────────────────────────────────────────────
# WHY stratified random: the Telco data has no dates, so a "time" split is
# not possible. Sorting by tenure (the old approach) put the loyal, rarely-
# churning customers in the test set. A stratified split keeps the churn
# rate the same in train and test, so the test score is a fair grade.
print("\n[1/7] Loading feature store with stratified split...")

fs    = pd.read_csv("data/feature_store.csv")
telco = pd.read_csv("data/telco_cleaned.csv")

# Same row order in both files, so tenure can be added back safely
fs["tenure"] = telco["tenure"].values

feature_cols = [c for c in fs.columns if c != "Churn_binary"]

# train_test_split keeps each row's ORIGINAL index, which we rely on below
# to line rows up with telco / survival / archetype files.
train_df, test_df = train_test_split(
    fs, test_size=0.2, stratify=fs["Churn_binary"], random_state=RANDOM_STATE
)

X_train = train_df[feature_cols].fillna(0)
y_train = train_df["Churn_binary"]
X_test  = test_df[feature_cols].fillna(0)
y_test  = test_df["Churn_binary"]

print(f"      Train: {len(X_train):,} customers (stratified 80%)")
print(f"      Test:  {len(X_test):,} customers (stratified 20%)")
print(f"      Features: {len(feature_cols)}")
print(f"      Train churn rate: {y_train.mean()*100:.1f}%")
print(f"      Test churn rate:  {y_test.mean()*100:.1f}%")

# Save which rows are the held-out test set so every later script
# evaluates ONLY on rows the model never trained on.
with open("data/split_indices.json", "w") as f:
    json.dump({"random_state": RANDOM_STATE,
               "train_index": [int(i) for i in train_df.index],
               "test_index":  [int(i) for i in test_df.index]}, f)


def pick_threshold(model, X_tr, y_tr):
    """Pick the probability cut-off with the best F1 using cross-validated
    predictions on TRAIN data only (the test set is never used to tune it)."""
    oof = cross_val_predict(clone(model), X_tr, y_tr,
                            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                            method="predict_proba")[:, 1]
    grid = np.arange(0.10, 0.91, 0.01)
    f1s  = [f1_score(y_tr, oof >= t) for t in grid]
    return float(round(grid[int(np.argmax(f1s))], 2))


scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ─────────────────────────────────────────────────────────────
# STEP 2: Baseline — Logistic Regression
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Training models...")
print("      Logistic Regression (baseline)...")

# Scaling is bundled inside the model (a Pipeline), so the saved model works on
# raw feature values wherever it is used (API, monitoring, pipeline).
lr = Pipeline([("scale", StandardScaler()),
               ("model", LogisticRegression(max_iter=1000, random_state=42,
                                            class_weight="balanced"))])
lr.fit(X_train, y_train)
lr_probs = lr.predict_proba(X_test)[:, 1]
lr_thr   = pick_threshold(lr, X_train, y_train)
lr_preds = (lr_probs >= lr_thr).astype(int)

print(f"        ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 3: Random Forest
# ─────────────────────────────────────────────────────────────
print("      Random Forest...")

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                             class_weight="balanced", n_jobs=-1)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_thr   = pick_threshold(rf, X_train, y_train)
rf_preds = (rf_probs >= rf_thr).astype(int)

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

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
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
xgb_thr   = pick_threshold(xgb_model, X_train, y_train)
xgb_preds = (xgb_probs >= xgb_thr).astype(int)
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
lgbm_thr   = pick_threshold(lgbm, X_train, y_train)
lgbm_preds = (lgbm_probs >= lgbm_thr).astype(int)
print(f"        ROC-AUC: {roc_auc_score(y_test, lgbm_probs):.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 6: Model Comparison
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Comparing models...")

thresholds = {"Logistic Regression": lr_thr, "Random Forest": rf_thr,
              "XGBoost": xgb_thr, "LightGBM": lgbm_thr}

models = {
    "Logistic Regression": (lr_probs,  lr_preds),
    "Random Forest"      : (rf_probs,  rf_preds),
    "XGBoost"            : (xgb_probs, xgb_preds),
    "LightGBM"           : (lgbm_probs,lgbm_preds),
}

# Choose the best model using 5-fold cross-validation on the TRAINING data only,
# so the test set is used once, to report the final score.
cv_models = {"Logistic Regression": lr, "Random Forest": rf,
             "XGBoost": xgb_model, "LightGBM": lgbm}
cv5 = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
cv_auc = {n: float(cross_val_score(clone(m), X_train, y_train, cv=cv5,
                                   scoring="roc_auc", n_jobs=-1).mean())
          for n, m in cv_models.items()}

results = {}
print(f"\n  {'Model':<22} {'ROC-AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Avg-PR':>8}")
print("  " + "-" * 70)
for name, (probs, preds) in models.items():
    auc  = roc_auc_score(y_test, probs)
    f1   = f1_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds)
    apr  = average_precision_score(y_test, probs)
    results[name] = {"auc": round(auc, 4), "f1": round(f1, 4),
                     "precision": round(prec, 4), "recall": round(rec, 4),
                     "avg_pr": round(apr, 4), "threshold": thresholds[name],
                     "cv_auc": round(cv_auc[name], 4)}
    print(f"  {name:<22} {auc:>8.4f} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f} {apr:>8.4f}  (cut-off {thresholds[name]:.2f})")

best_model_name = max(results, key=lambda x: results[x]["cv_auc"])
print(f"\n  Best model (highest cross-validated AUC on training data): {best_model_name}")
print(f"  Test ROC-AUC: {results[best_model_name]['auc']:.4f}  (cross-validated: {results[best_model_name]['cv_auc']:.4f})")

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

# ── Full per-customer file used by the SQL analysis, dashboard, monitoring ──
# Rows = the held-out TEST customers only (model never saw them in training).
roi_df = telco.loc[test_df.index].copy()
roi_df["churn_probability"]  = best_probs
roi_df["clv_12m"]            = roi_df["MonthlyCharges"] * 12
roi_df["intervention_cost"]  = INTERVENTION_COST
# Expected 12-month revenue kept = revenue at risk x assumed save rate.
# ROI = (kept revenue - cost) / cost.  ASSUMED_SAVE_RATE is an assumption.
roi_df["intervention_roi"]   = (roi_df["clv_12m"] * roi_df["churn_probability"]
                                * ASSUMED_SAVE_RATE - INTERVENTION_COST) / INTERVENTION_COST
roi_df["last_updated"]       = pd.Timestamp.now().isoformat()
roi_df = roi_df.sort_values("churn_probability", ascending=False)
roi_df.to_csv("data/predictions_with_roi.csv", index=False)
print("      Saved: data/predictions_with_roi.csv (held-out test customers)")


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
joblib.dump(best_model, "models/best_model.pkl")      # used by API, monitoring, pipeline
with open("models/feature_names.json", "w") as f:
    json.dump(feature_cols, f, indent=2)
with open("models/best_params.json", "w") as f:
    json.dump(best_xgb_params, f, indent=2, default=float)
with open("models/evaluation_meta.json", "w") as f:
    json.dump({"split_method": "stratified random 80/20",
               "random_state": RANDOM_STATE,
               "n_train": int(len(X_train)), "n_test": int(len(X_test)),
               "train_churn_rate": round(float(y_train.mean()), 4),
               "test_churn_rate": round(float(y_test.mean()), 4),
               "best_model": best_model_name,
               "threshold_chosen_on": "cross-validated predictions on train data",
               "model_chosen_by": "5-fold cross-validated AUC on train data",
               "note": "The test set is used once, to report the final score."},
              f, indent=2)
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
print(f"  Predictions   : data/predictions_with_roi.csv")
print(f"\n  Next: python phase4_ab_testing.py, then streamlit run dashboard.py")
print("=" * 60)