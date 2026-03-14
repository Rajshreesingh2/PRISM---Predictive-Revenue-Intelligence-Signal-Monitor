"""
PRISM v2 — Phase 1: Full Feature Factory (100+ features)
Real Kaggle Telco + World Bank + Google Trends + Alpha Vantage
"""

import pandas as pd
import numpy as np
import requests
import warnings
import os
import json
import glob
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

print("=" * 60)
print("  PRISM v2 — Phase 1: Full Feature Factory")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load & Clean
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading Kaggle Telco dataset...")

csv_files = glob.glob("WA_Fn*.csv") + glob.glob("telco*.csv") + glob.glob("Telco*.csv")
if not csv_files:
    raise FileNotFoundError("Kaggle CSV not found. Place it in the PRISM folder.")

df = pd.read_csv(csv_files[0])
print(f"      Raw: {df.shape[0]:,} customers, {df.shape[1]} columns")

# Fix types
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Binary target
df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)

# Binary encode Yes/No columns
for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]:
    if df[col].dtype == object:
        df[col] = (df[col] == "Yes").astype(int)

print(f"      Churn rate: {df['Churn_binary'].mean()*100:.1f}%")
print(f"      Missing TotalCharges filled: {df['TotalCharges'].isna().sum()}")


# ─────────────────────────────────────────────────────────────
# STEP 2: Feature Group A — Charge & Revenue Features (20)
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Engineering features...")
print("      Group A: Charge & revenue features...")

df["charge_per_month"]         = df["TotalCharges"] / (df["tenure"] + 1)
df["charge_to_monthly_ratio"]  = df["TotalCharges"] / (df["MonthlyCharges"] * (df["tenure"] + 1) + 1)
df["monthly_charge_percentile"]= df["MonthlyCharges"].rank(pct=True)
df["total_charge_percentile"]  = df["TotalCharges"].rank(pct=True)
df["is_high_charger"]          = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)
df["is_low_charger"]           = (df["MonthlyCharges"] < df["MonthlyCharges"].quantile(0.25)).astype(int)
df["charge_above_median"]      = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
df["monthly_to_total_ratio"]   = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
df["expected_lifetime_value"]  = df["MonthlyCharges"] * 24
df["actual_vs_expected_clv"]   = df["TotalCharges"] / (df["expected_lifetime_value"] + 1)
df["charge_trend"]             = df["MonthlyCharges"] / (df["charge_per_month"] + 1)
df["revenue_efficiency"]       = df["TotalCharges"] / (df["tenure"] * df["MonthlyCharges"] + 1)

print(f"        Charge features: 12")

# Feature Group B — Tenure & Lifecycle (15)
print("      Group B: Tenure & lifecycle features...")

df["tenure_squared"]       = df["tenure"] ** 2
df["tenure_log"]           = np.log1p(df["tenure"])
df["is_new_customer"]      = (df["tenure"] <= 6).astype(int)
df["is_early_customer"]    = (df["tenure"] <= 12).astype(int)
df["is_loyal_customer"]    = (df["tenure"] >= 48).astype(int)
df["is_veteran_customer"]  = (df["tenure"] >= 60).astype(int)
df["tenure_bucket"]        = pd.cut(df["tenure"],
                                     bins=[0, 6, 12, 24, 48, 72],
                                     labels=[1, 2, 3, 4, 5]).astype(float)
df["months_remaining_1yr"] = np.maximum(0, 12 - df["tenure"])
df["months_remaining_2yr"] = np.maximum(0, 24 - df["tenure"])
df["contract_age_ratio"]   = df["tenure"] / 24.0

print(f"        Tenure features: 10")

# Feature Group C — Service Adoption (15)
print("      Group C: Service adoption features...")

service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]

df["service_adoption_score"]  = df[service_cols].sum(axis=1)
df["service_adoption_pct"]    = df["service_adoption_score"] / len(service_cols)
df["is_single_service"]       = (df["service_adoption_score"] <= 1).astype(int)
df["is_full_service"]         = (df["service_adoption_score"] == len(service_cols)).astype(int)
df["security_bundle"]         = (df["OnlineSecurity"] + df["OnlineBackup"] + df["DeviceProtection"])
df["entertainment_bundle"]    = (df["StreamingTV"] + df["StreamingMovies"])
df["support_bundle"]          = (df["TechSupport"] + df["DeviceProtection"])
df["has_any_security"]        = (df["security_bundle"] > 0).astype(int)
df["has_entertainment"]       = (df["entertainment_bundle"] > 0).astype(int)
df["services_per_dollar"]     = df["service_adoption_score"] / (df["MonthlyCharges"] + 1)
df["charge_per_service"]      = df["MonthlyCharges"] / (df["service_adoption_score"] + 1)

print(f"        Service features: 11")

# Feature Group D — Contract & Payment Risk (12)
print("      Group D: Contract & payment risk features...")

df["is_month_to_month"]   = (df["Contract"] == "Month-to-month").astype(int)
df["is_one_year"]         = (df["Contract"] == "One year").astype(int)
df["is_two_year"]         = (df["Contract"] == "Two year").astype(int)
df["has_auto_pay"]        = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
df["uses_electronic"]     = df["PaymentMethod"].str.contains("Electronic", case=False).astype(int)
df["uses_mailed_check"]   = df["PaymentMethod"].str.contains("mailed", case=False).astype(int)
df["paperless_no_auto"]   = ((df["PaperlessBilling"] == 1) & (df["has_auto_pay"] == 0)).astype(int)
df["contract_risk_score"] = df["is_month_to_month"] * 3 + (1 - df["has_auto_pay"]) * 2

print(f"        Contract features: 8")

# Feature Group E — Internet & Technology (10)
print("      Group E: Internet & technology features...")

df["has_fiber"]           = (df["InternetService"] == "Fiber optic").astype(int)
df["has_dsl"]             = (df["InternetService"] == "DSL").astype(int)
df["no_internet"]         = (df["InternetService"] == "No").astype(int)
df["fiber_no_security"]   = ((df["has_fiber"] == 1) & (df["has_any_security"] == 0)).astype(int)
df["fiber_no_support"]    = ((df["has_fiber"] == 1) & (df["TechSupport"] == 0)).astype(int)
df["fiber_high_charge"]   = ((df["has_fiber"] == 1) & (df["is_high_charger"] == 1)).astype(int)

print(f"        Internet features: 6")

# Feature Group F — Demographics & Household (8)
print("      Group F: Demographics & household features...")

df["is_senior"]            = df["SeniorCitizen"].astype(int)
df["has_partner"]          = df["Partner"].astype(int)
df["has_dependents"]       = df["Dependents"].astype(int)
df["has_family"]           = ((df["Partner"] + df["Dependents"]) > 0).astype(int)
df["senior_alone"]         = ((df["is_senior"] == 1) & (df["has_family"] == 0)).astype(int)
df["young_family"]         = ((df["is_senior"] == 0) & (df["has_family"] == 1)).astype(int)
df["senior_month_to_month"]= ((df["is_senior"] == 1) & (df["is_month_to_month"] == 1)).astype(int)

print(f"        Demographic features: 7")

# Feature Group G — Composite Risk Scores (8)
print("      Group G: Composite risk scores...")

df["base_risk_score"] = (
    df["is_month_to_month"]  * 3 +
    df["is_new_customer"]    * 2 +
    df["is_single_service"]  * 2 +
    df["is_high_charger"]    * 1 +
    (1 - df["has_auto_pay"]) * 1
)
df["advanced_risk_score"] = (
    df["base_risk_score"]      * 1.0 +
    df["fiber_no_security"]    * 2.0 +
    df["fiber_high_charge"]    * 1.5 +
    df["paperless_no_auto"]    * 1.0 +
    df["senior_month_to_month"]* 1.5
)
df["clv_at_risk"] = df["MonthlyCharges"] * 12 * df["advanced_risk_score"] / 10

print(f"        Risk score features: 3")

# Feature Group H — Polynomial Interaction Features
print("      Group H: Polynomial interaction features...")

poly_base = ["tenure", "MonthlyCharges", "service_adoption_score",
             "base_risk_score", "charge_per_service"]
poly_data = df[poly_base].fillna(0)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(poly_data)
poly_names = [f"poly_{n.replace(' ', '_')}" for n in poly.get_feature_names_out(poly_base)]

poly_df = pd.DataFrame(poly_features[:, len(poly_base):],
                        columns=poly_names[len(poly_base):])
df = pd.concat([df.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)

print(f"        Polynomial interaction features: {poly_df.shape[1]}")

total_features = df.shape[1] - 21  # subtract original columns
print(f"\n      Total engineered features: {total_features}")
print(f"      Full dataframe: {df.shape[1]} columns")


# ─────────────────────────────────────────────────────────────
# STEP 3: API Enrichment
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Fetching macro signals from APIs...")

all_macro = {}

# World Bank
print("      World Bank API...")
try:
    for indicator, name in [
        ("NY.GDP.MKTP.KD.ZG", "gdp_growth"),
        ("FP.CPI.TOTL.ZG",    "inflation"),
        ("SL.UEM.TOTL.ZS",    "unemployment"),
        ("NY.GNP.PCAP.CD",    "gni_per_capita"),
    ]:
        url = f"https://api.worldbank.org/v2/country/US/indicator/{indicator}?format=json&mrv=3"
        r = requests.get(url, timeout=10)
        data = r.json()
        if len(data) > 1 and data[1]:
            val = data[1][0]["value"] or 0
            all_macro[f"macro_{name}"] = round(val, 3)
            print(f"        {name}: {val:.2f}")
except Exception as e:
    print(f"        Unavailable — using fallback")
    all_macro.update({
        "macro_gdp_growth": 2.1, "macro_inflation": 3.4,
        "macro_unemployment": 3.7, "macro_gni_per_capita": 76000
    })

# Google Trends
print("      Google Trends API...")
try:
    from pytrends.request import TrendReq
    pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
    pt.build_payload(["cancel phone plan", "switch carrier"],
                     timeframe="today 12-m", geo="US")
    interest = pt.interest_over_time()
    if not interest.empty:
        all_macro["trend_churn_intent"]   = round(float(interest["cancel phone plan"].mean()), 2)
        all_macro["trend_switch_carrier"] = round(float(interest["switch carrier"].mean()), 2)
        all_macro["trend_churn_velocity"] = round(float(
            interest["cancel phone plan"].iloc[-4:].mean() -
            interest["cancel phone plan"].iloc[:4].mean()
        ), 2)
        print(f"        Churn intent: {all_macro['trend_churn_intent']:.1f}")
    else:
        raise ValueError("Empty")
except Exception as e:
    print(f"        Unavailable — using fallback")
    all_macro.update({
        "trend_churn_intent": 42.3,
        "trend_switch_carrier": 38.7,
        "trend_churn_velocity": 3.2
    })

# Alpha Vantage
print("      Alpha Vantage API...")
try:
    from config import ALPHA_VANTAGE_KEY
    url = f"https://www.alphavantage.co/query?function=SECTOR&apikey={ALPHA_VANTAGE_KEY}"
    r = requests.get(url, timeout=10)
    data = r.json()
    perf = data.get("Rank A: Real-Time Performance", {})
    telecom = perf.get("Telecommunication Services", "0%")
    all_macro["macro_telecom_sector"] = float(telecom.replace("%", ""))
    print(f"        Telecom sector: {all_macro['macro_telecom_sector']:+.2f}%")
except Exception as e:
    print(f"        Unavailable — using fallback")
    all_macro["macro_telecom_sector"] = -1.2

# Derived macro features
all_macro["macro_stress_index"] = (
    all_macro.get("macro_inflation", 3.4) +
    all_macro.get("macro_unemployment", 3.7) -
    all_macro.get("macro_gdp_growth", 2.1)
)
all_macro["macro_consumer_pressure"] = (
    all_macro.get("macro_inflation", 3.4) *
    all_macro.get("macro_unemployment", 3.7)
)

with open("data/macro_signals.json", "w") as f:
    json.dump(all_macro, f, indent=2)

for key, val in all_macro.items():
    df[key] = val

print(f"\n      {len(all_macro)} macro signals added")


# ─────────────────────────────────────────────────────────────
# STEP 4: Automated Feature Selection
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Running automated feature selection...")

# Build numeric feature matrix
drop_for_selection = ["customerID", "Churn", "gender", "MultipleLines",
                       "Contract", "PaymentMethod", "InternetService",
                       "tenure_bucket"]
drop_for_selection = [c for c in drop_for_selection if c in df.columns]

df_numeric = df.drop(columns=drop_for_selection)
df_numeric = df_numeric.select_dtypes(include=[np.number])
df_numeric = df_numeric.fillna(0)

X = df_numeric.drop(columns=["Churn_binary"])
y = df_numeric["Churn_binary"]

# Mutual information scores
print("      Computing mutual information scores...")
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({"feature": X.columns, "mi_score": mi_scores})
mi_df = mi_df.sort_values("mi_score", ascending=False)

# Keep features with MI score above threshold
mi_threshold = mi_df["mi_score"].quantile(0.25)
selected_features = mi_df[mi_df["mi_score"] > mi_threshold]["feature"].tolist()
selected_features.append("Churn_binary")

print(f"      Features before selection: {len(X.columns)}")
print(f"      Features after MI filter:  {len(selected_features)-1}")
print(f"\n      Top 10 most predictive features:")
for _, row in mi_df.head(10).iterrows():
    print(f"        {row['feature']:<40} MI={row['mi_score']:.4f}")

mi_df.to_csv("data/feature_importance_mi.csv", index=False)


# ─────────────────────────────────────────────────────────────
# STEP 5: VIF Check (remove collinear features)
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Checking multicollinearity (VIF)...")

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_selected = X[selected_features[:-1]].copy()

# Quick VIF — remove features with VIF > 10
vif_results = []
try:
    for i, col in enumerate(X_selected.columns[:30]):  # check first 30
        vif = variance_inflation_factor(X_selected.values, i)
        vif_results.append({"feature": col, "vif": vif})

    vif_df = pd.DataFrame(vif_results)
    high_vif = vif_df[vif_df["vif"] > 10]["feature"].tolist()
    final_features = [f for f in selected_features[:-1] if f not in high_vif]
    final_features.append("Churn_binary")
    print(f"      Features removed (VIF > 10): {len(high_vif)}")
    print(f"      Final feature count: {len(final_features)-1}")
except Exception as e:
    print(f"      VIF check skipped: {e}")
    final_features = selected_features


# ─────────────────────────────────────────────────────────────
# STEP 6: Save Feature Store
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Saving feature store...")

# Full cleaned dataset
df.to_csv("data/telco_cleaned.csv", index=False)

# Selected feature store
feature_store = df_numeric[final_features]
feature_store.to_csv("data/feature_store.csv", index=False)

# Also save survival-ready dataset (needs tenure + event for lifelines)
survival_df = df[["tenure", "Churn_binary", "MonthlyCharges",
                   "Contract", "InternetService", "is_month_to_month",
                   "is_new_customer", "is_high_charger", "service_adoption_score",
                   "base_risk_score", "advanced_risk_score", "has_fiber",
                   "fiber_no_security", "charge_per_service"]].copy()
survival_df.to_csv("data/survival_ready.csv", index=False)

# Save feature list
with open("data/selected_features.json", "w") as f:
    json.dump(final_features, f, indent=2)

print(f"      telco_cleaned.csv      — {df.shape[0]:,} rows, {df.shape[1]} cols")
print(f"      feature_store.csv      — {feature_store.shape[0]:,} rows, {feature_store.shape[1]} cols")
print(f"      survival_ready.csv     — {survival_df.shape[0]:,} rows")
print(f"      feature_importance_mi.csv")
print(f"      selected_features.json")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Phase 1 Complete — Feature Factory")
print("=" * 60)
print(f"\n  Raw Kaggle features      : 21")
print(f"  Engineered features      : {total_features}")
print(f"  After MI selection       : {len(selected_features)-1}")
print(f"  After VIF filter         : {len(final_features)-1}")
print(f"  Macro signals (3 APIs)   : {len(all_macro)}")
print(f"\n  Churn rate               : {df['Churn_binary'].mean()*100:.1f}%")
print(f"  Churned customers        : {df['Churn_binary'].sum():,}")
print(f"  Retained customers       : {(1-df['Churn_binary']).sum():,}")
print(f"\n  Next steps:")
print(f"    python phase2_eda.py")
print(f"    python phase2b_survival.py")
print(f"    python phase2c_clustering.py")
print("=" * 60)
