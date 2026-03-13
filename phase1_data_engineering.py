"""
PRISM v2 — Phase 1: Data Ingestion & Feature Engineering
Real Kaggle data + Google Trends + World Bank API + Alpha Vantage
"""

import pandas as pd
import numpy as np
import requests
import warnings
import os
import json
import glob

warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

print("=" * 60)
print("  PRISM v2 — Phase 1: Data Ingestion Pipeline")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load Kaggle Telco Dataset
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading Kaggle Telco dataset...")

csv_files = glob.glob("WA_Fn*.csv") + glob.glob("telco*.csv") + glob.glob("Telco*.csv")
if not csv_files:
    raise FileNotFoundError(
        "Kaggle CSV not found in PRISM folder.\n"
        "Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
        "Place it in your PRISM folder and run again."
    )

df = pd.read_csv(csv_files[0])
print(f"      Loaded: {df.shape[0]:,} customers, {df.shape[1]} columns")
print(f"      Churn rate: {(df['Churn']=='Yes').mean()*100:.1f}%")


# ─────────────────────────────────────────────────────────────
# STEP 2: Data Cleaning & Validation
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Cleaning and validating data...")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print(f"      Missing values found:")
    for col, n in missing.items():
        print(f"        {col}: {n} ({n/len(df)*100:.1f}%) — filling with median")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
else:
    print("      No critical missing values")

df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)

binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in binary_cols:
    df[col] = (df[col] == "Yes").astype(int)

print(f"      Cleaned: {df.shape[0]:,} rows")
print(f"      Churned: {df['Churn_binary'].sum():,} ({df['Churn_binary'].mean()*100:.1f}%)")
print(f"      Retained: {(1-df['Churn_binary']).sum():,} ({(1-df['Churn_binary']).mean()*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# STEP 3: Deep Feature Engineering
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Engineering features...")

# Charge features
df["ChargePerMonth"]        = df["TotalCharges"] / (df["tenure"] + 1)
df["ChargeToMonthly_ratio"] = df["TotalCharges"] / (df["MonthlyCharges"] * (df["tenure"] + 1) + 1)
df["HighCharger"]           = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)

# Tenure features
df["TenureBucket"]  = pd.cut(df["tenure"], bins=[0,6,12,24,48,72],
                              labels=["0-6m","6-12m","1-2yr","2-4yr","4+yr"])
df["NewCustomer"]   = (df["tenure"] <= 6).astype(int)
df["LoyalCustomer"] = (df["tenure"] >= 48).astype(int)

# Service adoption score
service_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies"]
for col in service_cols:
    df[col + "_bin"] = (df[col] == "Yes").astype(int)

service_bin_cols = [c for c in df.columns if c.endswith("_bin")]
df["ServiceAdoptionScore"] = df[service_bin_cols].sum(axis=1)
df["SingleServiceUser"]    = (df["ServiceAdoptionScore"] <= 1).astype(int)

# Contract and payment
df["MonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
df["LongContract"]  = (df["Contract"] == "Two year").astype(int)
df["AutoPay"]       = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)

# Internet
df["HasFiber"]   = (df["InternetService"] == "Fiber optic").astype(int)
df["NoInternet"] = (df["InternetService"] == "No").astype(int)

# Composite risk score
df["RiskScore"] = (
    df["MonthToMonth"]      * 3 +
    df["NewCustomer"]       * 2 +
    df["SingleServiceUser"] * 2 +
    df["HighCharger"]       * 1 +
    (1 - df["AutoPay"])     * 1
)

print(f"      Engineered 13 new features")
print(f"      RiskScore range: {df['RiskScore'].min()} — {df['RiskScore'].max()}")
print(f"      High-risk customers (score >= 6): {(df['RiskScore']>=6).sum():,}")


# ─────────────────────────────────────────────────────────────
# STEP 4: API Enrichment
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Fetching macro signals from APIs...")

all_macro = {}

# World Bank API
print("      World Bank API...")
try:
    for indicator, name in [
        ("NY.GDP.MKTP.KD.ZG", "gdp_growth"),
        ("FP.CPI.TOTL.ZG",    "inflation"),
        ("SL.UEM.TOTL.ZS",    "unemployment")
    ]:
        url = f"https://api.worldbank.org/v2/country/US/indicator/{indicator}?format=json&mrv=3"
        r = requests.get(url, timeout=10)
        data = r.json()
        if len(data) > 1 and data[1]:
            val = data[1][0]["value"] or 0
            all_macro[f"latest_{name}"] = round(val, 3)
            print(f"        {name}: {val:.2f}%")
except Exception as e:
    print(f"        Unavailable ({e}) — using fallback values")
    all_macro.update({"latest_gdp_growth": 2.1,
                      "latest_inflation": 3.4,
                      "latest_unemployment": 3.7})

# Google Trends
print("      Google Trends API...")
try:
    from pytrends.request import TrendReq
    pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
    pt.build_payload(["cancel phone plan", "switch carrier"],
                     timeframe="today 12-m", geo="US")
    interest = pt.interest_over_time()
    if not interest.empty:
        all_macro["churn_intent_avg"]   = round(float(interest["cancel phone plan"].mean()), 2)
        all_macro["switch_carrier_avg"] = round(float(interest["switch carrier"].mean()), 2)
        all_macro["churn_intent_trend"] = round(float(
            interest["cancel phone plan"].iloc[-4:].mean() -
            interest["cancel phone plan"].iloc[:4].mean()
        ), 2)
        print(f"        Churn intent avg: {all_macro['churn_intent_avg']}")
        print(f"        Trend direction: {all_macro['churn_intent_trend']:+.1f}")
    else:
        raise ValueError("Empty response")
except Exception as e:
    print(f"        Unavailable ({e}) — using fallback values")
    all_macro.update({"churn_intent_avg": 42.3,
                      "switch_carrier_avg": 38.7,
                      "churn_intent_trend": 3.2})

# Alpha Vantage
print("      Alpha Vantage API...")
try:
    from config import ALPHA_VANTAGE_KEY
    url = f"https://www.alphavantage.co/query?function=SECTOR&apikey={ALPHA_VANTAGE_KEY}"
    r = requests.get(url, timeout=10)
    data = r.json()
    perf = data.get("Rank A: Real-Time Performance", {})
    telecom = perf.get("Telecommunication Services", "0%")
    all_macro["telecom_sector_perf"] = float(telecom.replace("%", ""))
    print(f"        Telecom sector: {all_macro['telecom_sector_perf']:+.2f}%")
except Exception as e:
    print(f"        Unavailable ({e}) — using fallback value")
    all_macro["telecom_sector_perf"] = -1.2

with open("data/macro_signals.json", "w") as f:
    json.dump(all_macro, f, indent=2)

for key, val in all_macro.items():
    df[key] = val

print(f"\n      {len(all_macro)} macro signals added to every customer row")


# ─────────────────────────────────────────────────────────────
# STEP 5: Final Feature Store
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Building final feature store...")

cat_cols = ["InternetService", "Contract", "PaymentMethod", "TenureBucket"]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

drop_cols = ["customerID", "Churn", "gender", "MultipleLines",
             "OnlineSecurity", "OnlineBackup", "DeviceProtection",
             "TechSupport", "StreamingTV", "StreamingMovies"]
drop_cols = [c for c in drop_cols if c in df_encoded.columns]
df_encoded.drop(columns=drop_cols, inplace=True)

for col in df_encoded.columns:
    if df_encoded[col].dtype == bool:
        df_encoded[col] = df_encoded[col].astype(int)

df.to_csv("data/telco_cleaned.csv", index=False)
df_encoded.to_csv("data/feature_store.csv", index=False)

print(f"      Feature store: {df_encoded.shape[0]:,} rows x {df_encoded.shape[1]} features")

print("\n" + "=" * 60)
print("  Phase 1 Complete")
print("=" * 60)
print(f"\n  Customers      : {df.shape[0]:,} real Telco customers")
print(f"  Total features : {df_encoded.shape[1]}")
print(f"  Macro signals  : {len(all_macro)} indicators")
print(f"  Churn rate     : {df_encoded['Churn_binary'].mean()*100:.1f}%")
print(f"\n  Saved:")
print(f"    data/telco_cleaned.csv")
print(f"    data/feature_store.csv")
print(f"    data/macro_signals.json")
print("\n  Next: python phase2_eda.py")
print("=" * 60)