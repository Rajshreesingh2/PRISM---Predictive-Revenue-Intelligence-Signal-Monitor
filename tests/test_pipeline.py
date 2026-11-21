"""Basic checks that the saved outputs are consistent.
Run from the project root after the pipeline:  pytest -q
"""
import json
import numpy as np
import pandas as pd
import pytest

SPLIT = json.load(open("data/split_indices.json"))
FS = pd.read_csv("data/feature_store.csv")
RESULTS = json.load(open("models/model_results.json"))
META = json.load(open("models/evaluation_meta.json"))


def test_train_and_test_do_not_overlap_and_cover_everything():
    train, test = set(SPLIT["train_index"]), set(SPLIT["test_index"])
    assert not train & test
    assert len(train | test) == len(FS)


def test_split_has_same_churn_rate_in_train_and_test():
    y = FS["Churn_binary"]
    train_rate = y.loc[SPLIT["train_index"]].mean()
    test_rate = y.loc[SPLIT["test_index"]].mean()
    assert abs(train_rate - test_rate) < 0.01   # stratified split


def test_every_model_has_real_precision_and_recall():
    for name, m in RESULTS.items():
        assert 0.5 < m["auc"] < 1.0, name
        assert m["precision"] > 0 and m["recall"] > 0, name


def test_monitoring_auc_matches_modeling_auc():
    mon = json.load(open("monitoring/monitoring_summary.json"))
    best_auc = RESULTS[META["best_model"]]["auc"]
    assert abs(mon["performance"]["current"]["roc_auc"] - best_auc) < 0.001


def test_predictions_file_contains_only_held_out_customers():
    preds = pd.read_csv("data/predictions_with_roi.csv")
    telco = pd.read_csv("data/telco_cleaned.csv")
    assert len(preds) == len(SPLIT["test_index"])
    test_ids = set(telco.loc[SPLIT["test_index"], "customerID"])
    assert set(preds["customerID"]) == test_ids


def test_roi_formula():
    p = pd.read_csv("data/predictions_with_roi.csv")
    expected = (p["MonthlyCharges"] * 12 * p["churn_probability"] * 0.15 - 50) / 50
    assert np.allclose(p["intervention_roi"], expected, atol=1e-6)


def test_ab_results_are_labelled_as_simulated():
    ab = json.load(open("data/ab_test_results.json"))
    assert ab["SIMULATED"] is True
    assert "mrr_saved_annual" not in ab["results"]


def test_api_predicts():
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient
    import phase5_api
    client = TestClient(phase5_api.app)
    r = client.post("/predict", json={
        "tenure": 3, "MonthlyCharges": 85.5, "TotalCharges": 256.5,
        "Contract": "Month-to-month", "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check", "PaperlessBilling": 1,
        "Partner": 0, "Dependents": 0, "SeniorCitizen": 0, "PhoneService": 1})
    assert r.status_code == 200
    assert 0 <= r.json()["churn_probability"] <= 1


def test_feature_store_has_no_constant_columns():
    constant = [c for c in FS.columns if FS[c].nunique() <= 1]
    assert constant == []


def test_raw_data_matches_cleaned_data():
    raw = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    telco = pd.read_csv("data/telco_cleaned.csv")
    assert raw["customerID"].tolist() == telco["customerID"].tolist()
    assert (raw["Churn"] == "Yes").sum() == telco["Churn_binary"].sum()


def test_no_highly_collinear_features():
    X = FS.drop(columns=["Churn_binary"])
    corr = np.corrcoef(X.values, rowvar=False)
    vif = np.diag(np.linalg.inv(corr + 1e-8 * np.eye(len(corr))))
    assert vif.max() <= 10.01
