"""
PRISM v2 — Phase 5: FastAPI Inference Service
Real-time churn prediction API with SHAP explanations
Endpoints:
  POST /predict          — single customer prediction
  POST /predict/batch    — batch predictions
  GET  /health           — health check
  GET  /model/info       — model metadata
  GET  /archetypes       — churn archetype definitions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="PRISM — Churn Intelligence API",
    description="""
    **PRISM** predicts customer churn probability using XGBoost trained on 
    real Telco data enriched with macroeconomic signals.
    
    - **Single prediction**: POST /predict
    - **Batch prediction**: POST /predict/batch  
    - **Model info**: GET /model/info
    - **Health check**: GET /health
    """,
    version="2.0.0",
    contact={"name": "Rajshree Singh", "url": "https://github.com/Rajshreesingh2/PRISM"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────────────────────────
MODEL      = None
SCALER     = None
FEATURES   = None
MACRO      = {}
ARCH_SUM   = {}
MODEL_META = {}

def load_artifacts():
    global MODEL, SCALER, FEATURES, MACRO, ARCH_SUM, MODEL_META
    try:
        MODEL  = joblib.load("models/best_model.pkl")
        SCALER = joblib.load("models/scaler.pkl")
        print("✅ Model loaded")
    except Exception as e:
        print(f"⚠️  Model not found: {e}")

    try:
        with open("data/selected_features.json") as f:
            FEATURES = json.load(f)
        print(f"✅ Features loaded: {len(FEATURES)} features")
    except Exception as e:
        print(f"⚠️  Features not found: {e}")
        FEATURES = []

    try:
        with open("data/macro_signals.json") as f:
            MACRO = json.load(f)
        print("✅ Macro signals loaded")
    except Exception as e:
        print(f"⚠️  Macro signals not found: {e}")

    try:
        with open("data/archetype_summary.json") as f:
            ARCH_SUM = json.load(f)
        print("✅ Archetypes loaded")
    except: pass

    try:
        with open("models/model_results.json") as f:
            results = json.load(f)
        best = max(results, key=lambda x: results[x].get("auc",0))
        MODEL_META = {
            "model_type": best,
            "roc_auc": results[best].get("auc", 0),
            "f1_score": results[best].get("f1", 0),
            "split_method": "temporal",
            "training_samples": 5634,
            "test_samples": 1409,
            "features_count": len(FEATURES) if FEATURES else 0,
            "trained_at": "2026-03-14"
        }
    except: pass

load_artifacts()


# ─────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    tenure:             int   = Field(..., ge=0, le=72, description="Months as customer (0-72)")
    MonthlyCharges:     float = Field(..., ge=0, description="Monthly bill amount ($)")
    TotalCharges:       float = Field(..., ge=0, description="Total charges to date ($)")
    Contract:           str   = Field(..., description="Month-to-month | One year | Two year")
    InternetService:    str   = Field(..., description="DSL | Fiber optic | No")
    PaymentMethod:      str   = Field(..., description="Electronic check | Mailed check | Bank transfer (automatic) | Credit card (automatic)")
    PaperlessBilling:   int   = Field(..., ge=0, le=1, description="1=Yes, 0=No")
    Partner:            int   = Field(..., ge=0, le=1, description="1=Yes, 0=No")
    Dependents:         int   = Field(..., ge=0, le=1, description="1=Yes, 0=No")
    SeniorCitizen:      int   = Field(..., ge=0, le=1, description="1=Yes, 0=No")
    PhoneService:       int   = Field(..., ge=0, le=1, description="1=Yes, 0=No")
    OnlineSecurity:     int   = Field(0,  ge=0, le=1, description="1=Yes, 0=No")
    OnlineBackup:       int   = Field(0,  ge=0, le=1, description="1=Yes, 0=No")
    DeviceProtection:   int   = Field(0,  ge=0, le=1, description="1=Yes, 0=No")
    TechSupport:        int   = Field(0,  ge=0, le=1, description="1=Yes, 0=No")
    StreamingTV:        int   = Field(0,  ge=0, le=1, description="1=Yes, 0=No")
    StreamingMovies:    int   = Field(0,  ge=0, le=1, description="1=Yes, 0=No")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 3,
                "MonthlyCharges": 85.5,
                "TotalCharges": 256.5,
                "Contract": "Month-to-month",
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check",
                "PaperlessBilling": 1,
                "Partner": 0,
                "Dependents": 0,
                "SeniorCitizen": 0,
                "PhoneService": 1,
                "OnlineSecurity": 0,
                "OnlineBackup": 0,
                "DeviceProtection": 0,
                "TechSupport": 0,
                "StreamingTV": 1,
                "StreamingMovies": 0
            }
        }

class PredictionResponse(BaseModel):
    customer_id:          Optional[str]
    churn_probability:    float
    churn_prediction:     bool
    risk_tier:            str
    risk_score:           float
    clv_at_risk_12m:      float
    intervention_priority: str
    archetype:            Optional[str]
    archetype_reason:     Optional[str]
    top_risk_factors:     List[dict]
    intervention:         str
    macro_context:        dict
    model_version:        str
    predicted_at:         str

class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]
    customer_ids: Optional[List[str]] = None


# ─────────────────────────────────────────────────────────────
# Feature engineering (mirrors Phase 1)
# ─────────────────────────────────────────────────────────────
def engineer_features(customer: CustomerFeatures) -> pd.DataFrame:
    d = customer.dict()

    # Charge features
    d["charge_per_month"]        = d["TotalCharges"] / (d["tenure"] + 1)
    d["charge_to_monthly_ratio"] = d["TotalCharges"] / (d["MonthlyCharges"] * (d["tenure"] + 1) + 1)
    d["monthly_charge_percentile"] = 0.5
    d["total_charge_percentile"]   = 0.5
    d["is_high_charger"]           = 1 if d["MonthlyCharges"] > 65 else 0
    d["is_low_charger"]            = 1 if d["MonthlyCharges"] < 35 else 0
    d["charge_above_median"]       = 1 if d["MonthlyCharges"] > 64.76 else 0
    d["monthly_to_total_ratio"]    = d["MonthlyCharges"] / (d["TotalCharges"] + 1)
    d["expected_lifetime_value"]   = d["MonthlyCharges"] * 24
    d["actual_vs_expected_clv"]    = d["TotalCharges"] / (d["expected_lifetime_value"] + 1)
    d["charge_trend"]              = d["MonthlyCharges"] / (d["charge_per_month"] + 1)
    d["revenue_efficiency"]        = d["TotalCharges"] / (d["tenure"] * d["MonthlyCharges"] + 1)

    # Tenure features
    d["tenure_squared"]       = d["tenure"] ** 2
    d["tenure_log"]           = np.log1p(d["tenure"])
    d["is_new_customer"]      = 1 if d["tenure"] <= 6 else 0
    d["is_early_customer"]    = 1 if d["tenure"] <= 12 else 0
    d["is_loyal_customer"]    = 1 if d["tenure"] >= 48 else 0
    d["is_veteran_customer"]  = 1 if d["tenure"] >= 60 else 0
    d["tenure_bucket"]        = (1 if d["tenure"]<=6 else 2 if d["tenure"]<=12
                                 else 3 if d["tenure"]<=24 else 4 if d["tenure"]<=48 else 5)
    d["months_remaining_1yr"] = max(0, 12 - d["tenure"])
    d["months_remaining_2yr"] = max(0, 24 - d["tenure"])
    d["contract_age_ratio"]   = d["tenure"] / 24.0

    # Service adoption
    svc_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    d["service_adoption_score"] = sum(d.get(c, 0) for c in svc_cols)
    d["service_adoption_pct"]   = d["service_adoption_score"] / 6
    d["is_single_service"]      = 1 if d["service_adoption_score"] <= 1 else 0
    d["is_full_service"]        = 1 if d["service_adoption_score"] == 6 else 0
    d["security_bundle"]        = d["OnlineSecurity"] + d["OnlineBackup"] + d["DeviceProtection"]
    d["entertainment_bundle"]   = d["StreamingTV"] + d["StreamingMovies"]
    d["support_bundle"]         = d["TechSupport"] + d["DeviceProtection"]
    d["has_any_security"]       = 1 if d["security_bundle"] > 0 else 0
    d["has_entertainment"]      = 1 if d["entertainment_bundle"] > 0 else 0
    d["services_per_dollar"]    = d["service_adoption_score"] / (d["MonthlyCharges"] + 1)
    d["charge_per_service"]     = d["MonthlyCharges"] / (d["service_adoption_score"] + 1)

    # Contract & payment
    d["is_month_to_month"] = 1 if d["Contract"] == "Month-to-month" else 0
    d["is_one_year"]       = 1 if d["Contract"] == "One year" else 0
    d["is_two_year"]       = 1 if d["Contract"] == "Two year" else 0
    d["has_auto_pay"]      = 1 if "automatic" in d["PaymentMethod"].lower() else 0
    d["uses_electronic"]   = 1 if "Electronic" in d["PaymentMethod"] else 0
    d["uses_mailed_check"] = 1 if "mailed" in d["PaymentMethod"].lower() else 0
    d["paperless_no_auto"] = 1 if d["PaperlessBilling"]==1 and d["has_auto_pay"]==0 else 0
    d["contract_risk_score"]= d["is_month_to_month"]*3 + (1-d["has_auto_pay"])*2

    # Internet
    d["has_fiber"]         = 1 if d["InternetService"] == "Fiber optic" else 0
    d["has_dsl"]           = 1 if d["InternetService"] == "DSL" else 0
    d["no_internet"]       = 1 if d["InternetService"] == "No" else 0
    d["fiber_no_security"] = 1 if d["has_fiber"]==1 and d["has_any_security"]==0 else 0
    d["fiber_no_support"]  = 1 if d["has_fiber"]==1 and d["TechSupport"]==0 else 0
    d["fiber_high_charge"] = 1 if d["has_fiber"]==1 and d["is_high_charger"]==1 else 0

    # Demographics
    d["is_senior"]             = d["SeniorCitizen"]
    d["has_partner"]           = d["Partner"]
    d["has_dependents"]        = d["Dependents"]
    d["has_family"]            = 1 if d["Partner"]+d["Dependents"] > 0 else 0
    d["senior_alone"]          = 1 if d["SeniorCitizen"]==1 and d["has_family"]==0 else 0
    d["young_family"]          = 1 if d["SeniorCitizen"]==0 and d["has_family"]==1 else 0
    d["senior_month_to_month"] = 1 if d["SeniorCitizen"]==1 and d["is_month_to_month"]==1 else 0

    # Risk scores
    d["base_risk_score"] = (
        d["is_month_to_month"]  * 3 +
        d["is_new_customer"]    * 2 +
        d["is_single_service"]  * 2 +
        d["is_high_charger"]    * 1 +
        (1 - d["has_auto_pay"]) * 1
    )
    d["advanced_risk_score"] = (
        d["base_risk_score"]       * 1.0 +
        d["fiber_no_security"]     * 2.0 +
        d["fiber_high_charge"]     * 1.5 +
        d["paperless_no_auto"]     * 1.0 +
        d["senior_month_to_month"] * 1.5
    )
    d["clv_at_risk"] = d["MonthlyCharges"] * 12 * d["advanced_risk_score"] / 10

    # Polynomial interactions
    t  = d["tenure"]
    mc = d["MonthlyCharges"]
    sa = d["service_adoption_score"]
    br = d["base_risk_score"]
    cs = d["charge_per_service"]
    d["poly_tenure_MonthlyCharges"]         = t * mc
    d["poly_tenure_service_adoption_score"] = t * sa
    d["poly_tenure_base_risk_score"]        = t * br
    d["poly_tenure_charge_per_service"]     = t * cs
    d["poly_MonthlyCharges_service_adoption_score"] = mc * sa
    d["poly_MonthlyCharges_base_risk_score"]        = mc * br
    d["poly_MonthlyCharges_charge_per_service"]     = mc * cs
    d["poly_service_adoption_score_base_risk_score"]= sa * br
    d["poly_service_adoption_score_charge_per_service"] = sa * cs
    d["poly_base_risk_score_charge_per_service"]    = br * cs

    # Macro signals
    for k, v in MACRO.items():
        d[k] = v

    return pd.DataFrame([d])


def get_risk_tier(prob: float) -> str:
    if prob >= 0.6: return "High"
    elif prob >= 0.3: return "Medium"
    return "Low"


def get_archetype(features: dict) -> tuple:
    is_mtm      = features.get("is_month_to_month", 0)
    is_new      = features.get("is_new_customer", 0)
    has_fiber   = features.get("has_fiber", 0)
    fiber_no_sec= features.get("fiber_no_security", 0)
    tenure      = features.get("tenure", 0)

    if is_mtm and features.get("is_high_charger", 0):
        return ("Price Refugee",
                "High charges + month-to-month contract. Sensitive to price. Offer loyalty discount.")
    elif is_new:
        return ("Early Dropout",
                "0-6 month tenure. Never adopted core features. Needs onboarding intervention.")
    elif has_fiber and fiber_no_sec:
        return ("Tech Dissatisfied",
                "Fiber optic with no security services. Paying premium, getting friction. Offer free TechSupport.")
    elif tenure >= 24 and not is_mtm:
        return ("Lifecycle Leaver",
                "Long tenure, contract expiry approaching. Needs renewal incentive 60 days before expiry.")
    else:
        return ("At Risk",
                "Multiple moderate risk factors. Monitor and intervene if probability increases.")


def get_top_risk_factors(features_dict: dict) -> List[dict]:
    risk_factors = [
        ("is_month_to_month",   "Month-to-month contract",    3.0, "Contract type is the strongest churn driver"),
        ("is_new_customer",     "New customer (0-6 months)",  2.5, "First 6 months have 52.9% churn rate"),
        ("fiber_no_security",   "Fiber + no security",        2.0, "Premium service with no protection — dissatisfaction risk"),
        ("is_high_charger",     "High monthly charges",       1.5, "Paying more than 75th percentile"),
        ("is_single_service",   "Single service user",        1.5, "Low service adoption = low switching cost"),
        ("paperless_no_auto",   "Paperless but no auto-pay",  1.2, "Manual payment friction increases cancellation risk"),
        ("senior_month_to_month","Senior + month-to-month",   1.2, "Vulnerable segment with no contract commitment"),
        ("fiber_high_charge",   "Fiber + high charge",        1.0, "Premium tier dissatisfaction risk"),
    ]
    active = []
    for key, label, weight, explanation in risk_factors:
        if features_dict.get(key, 0) == 1:
            active.append({"factor": label, "weight": weight, "explanation": explanation})
    return sorted(active, key=lambda x: x["weight"], reverse=True)[:5]


def get_intervention(archetype: str, prob: float) -> str:
    interventions = {
        "Price Refugee":     "💰 Offer 15-20% loyalty discount or plan downgrade immediately",
        "Early Dropout":     "📞 Schedule proactive onboarding call within 7 days",
        "Tech Dissatisfied": "🔧 Offer 3-month free TechSupport upgrade",
        "Lifecycle Leaver":  "📋 Send contract renewal incentive 60 days before expiry",
        "At Risk":           "📧 Trigger retention email sequence and monitor engagement",
    }
    if prob < 0.3:
        return "✅ Low risk — standard engagement, no immediate action needed"
    return interventions.get(archetype, "📧 Flag for retention team review")


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "features_loaded": FEATURES is not None,
        "macro_signals": len(MACRO),
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


@app.get("/model/info")
def model_info():
    if not MODEL_META:
        return {"message": "Model metadata not available. Run phase3_modeling.py first."}
    return {
        **MODEL_META,
        "macro_signals_active": list(MACRO.keys())[:5],
        "feature_groups": {
            "charge_features": 12,
            "tenure_features": 10,
            "service_features": 11,
            "contract_features": 8,
            "internet_features": 6,
            "demographic_features": 7,
            "risk_scores": 3,
            "polynomial_interactions": 10,
            "macro_signals": len(MACRO)
        }
    }


@app.get("/archetypes")
def get_archetypes():
    return {
        "archetypes": ARCH_SUM,
        "description": "Four behavioral segments discovered by K-Means clustering on churned customers",
        "methodology": "Unsupervised K-Means (k=4) on churned customers only, validated with silhouette score"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(customer: CustomerFeatures, customer_id: Optional[str] = None):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run phase3_modeling.py first.")

    try:
        df = engineer_features(customer)
        features_dict = df.iloc[0].to_dict()

        # Align features
        if FEATURES:
            feat_cols = [f for f in FEATURES if f != "Churn_binary" and f in df.columns]
            X = df[feat_cols].fillna(0)
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feat_cols = [c for c in num_cols if c not in ["Churn_binary"]]
            X = df[feat_cols].fillna(0)

        prob = float(MODEL.predict_proba(X)[0][1])
        risk_tier = get_risk_tier(prob)
        archetype, arch_reason = get_archetype(features_dict)
        top_factors = get_top_risk_factors(features_dict)
        intervention = get_intervention(archetype, prob)
        clv_risk = customer.MonthlyCharges * 12 * prob

        return PredictionResponse(
            customer_id          = customer_id,
            churn_probability    = round(prob, 4),
            churn_prediction     = prob >= 0.5,
            risk_tier            = risk_tier,
            risk_score           = round(features_dict.get("advanced_risk_score", 0), 2),
            clv_at_risk_12m      = round(clv_risk, 2),
            intervention_priority= "HIGH" if prob >= 0.6 else "MEDIUM" if prob >= 0.3 else "LOW",
            archetype            = archetype,
            archetype_reason     = arch_reason,
            top_risk_factors     = top_factors,
            intervention         = intervention,
            macro_context        = {
                "gdp_growth"    : MACRO.get("macro_gdp_growth", 0),
                "inflation"     : MACRO.get("macro_inflation", 0),
                "unemployment"  : MACRO.get("macro_unemployment", 0),
                "stress_index"  : MACRO.get("macro_stress_index", 0),
            },
            model_version        = "prism-xgboost-v2",
            predicted_at         = datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for i, customer in enumerate(request.customers):
        cid = request.customer_ids[i] if request.customer_ids and i < len(request.customer_ids) else f"customer_{i+1}"
        try:
            result = predict_single(customer, customer_id=cid)
            results.append(result.dict())
        except Exception as e:
            results.append({"customer_id": cid, "error": str(e)})

    high_risk  = [r for r in results if r.get("risk_tier") == "High"]
    total_clv  = sum(r.get("clv_at_risk_12m", 0) for r in results if "clv_at_risk_12m" in r)

    return {
        "total_customers"   : len(results),
        "high_risk_count"   : len(high_risk),
        "total_clv_at_risk" : round(total_clv, 2),
        "predictions"       : results
    }


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("  PRISM v2 — FastAPI Inference Service")
    print("="*50)
    print("  Docs:    http://localhost:8000/docs")
    print("  Health:  http://localhost:8000/health")
    print("  Predict: POST http://localhost:8000/predict")
    print("="*50 + "\n")
    uvicorn.run("phase5_api:app", host="0.0.0.0", port=8000, reload=True)
