<div align="center">

# PRISM
### Predictive Revenue & Intelligence Signal Monitor



PRISM is a full end-to-end Data Science project that predicts customer churn **30 to 90 days before it happens**, quantifies how much revenue is at risk, and surfaces the exact behavioral signals driving it — all packaged into a dashboard a product manager can actually use without needing to understand the model.

It started from a simple observation: most companies only react to churn after the customer is already gone. The data to predict it exists inside every product — login patterns, feature usage, billing trends, support tickets — it just never gets connected properly. PRISM connects it.

---

## The Problem This Solves

Imagine you run a SaaS product. Every month, some percentage of your users cancel. You look at your dashboard after the fact and see the number. But you have no idea which users are *about* to cancel in the next 30 days. You cannot intervene because you do not know who to intervene with.

Your data team runs a report. It shows aggregate churn rates by plan. Not helpful — you already knew free users churn more than enterprise users. What you need is a list of specific accounts trending toward cancellation right now, ranked by how much revenue they represent, with an explanation of what is driving the risk for each one.

That is exactly what PRISM produces.

---

## How It Works

PRISM tracks five behavioral signals for every user across rolling time windows and combines them into a single risk score with a plain-English explanation:

```
Signal 1 — Recency
User hasn't logged in for 18 days
→ recency score: HIGH RISK

Signal 2 — Engagement Velocity
User logged in 10 times last month, only 3 times this month
→ velocity score: HIGH RISK (trending down 70%)

Signal 3 — Feature Adoption Breadth
User only ever uses 1 out of 8 available features
→ adoption score: MEDIUM RISK (low stickiness)

Signal 4 — Revenue Trajectory
User's MRR has been inconsistent over last 3 months
→ revenue score: MEDIUM RISK

Signal 5 — Support Activity
User opened 3 support tickets this week (up from 0)
→ support score: HIGH RISK (frustration signal)

──────────────────────────────────────────────────
PRISM Churn Risk Score:  84%  →  HIGH
Top driver:              Engagement velocity drop
Secondary driver:        Support spike
Recommended action:      Trigger retention outreach within 7 days
Estimated MRR at risk:   $99/month
```

The model does not just output a number. It tells you *why* the number is what it is. That is the SHAP explainability layer — every prediction comes with a ranked list of which features contributed most to the score, so a non-technical stakeholder can read it and act on it without needing to understand gradient boosting.

---

## Key Findings From the Data

These are real outputs from Phase 2 EDA, not hypothetical examples.

**Churn by plan tier**

Free plan users churn at 20%. Basic at around 12%. Pro drops to 1.6%. Enterprise barely registers. This tells you two things: the conversion from free to paid is the single highest-leverage retention move, and pro users have built enough habits in the product that leaving feels costly. The goal of any retention campaign should be accelerating feature adoption early enough that users feel that switching cost before they consider leaving.

**Engagement velocity is the strongest predictor**

Users with declining engagement velocity — meaning they used the product less this month than last month — churn at 37%. Users with stable or growing engagement churn at 7%. That is a 5x difference. This is a stronger signal than raw activity level, which makes sense: a power user who suddenly goes quiet is more at risk than a low-activity user who has been consistently low since day one. The trend matters more than the absolute number.

**Feature adoption breadth beats depth**

A user who uses 5 different features at a moderate level is significantly less likely to churn than a user who uses 1 feature heavily. Single-feature users have one reason to stay. Multi-feature users have five. If onboarding can get a user to their third feature within the first two weeks, churn risk drops substantially.

**Referral beats paid search on retention**

After controlling for plan type, users acquired through referral retain significantly better than users acquired through paid search. Referral users come in with a peer recommendation and a warmer starting point. They move through onboarding faster and reach feature adoption milestones earlier. Paid search users often arrive with vaguer intent and churn at the first point of friction.

**Revenue at risk is concentrated, not distributed**

The $8,800/month in at-risk MRR does not come from 528 users equally. It is concentrated in a small subset of basic and pro plan users whose accounts are large enough to matter. A targeted outreach to 40–50 high-value flagged accounts can recover the majority of at-risk revenue. The model ranks them so you know exactly which 40.

---

## Project Structure

```
prism/
│
├── data/                             # All generated datasets (gitignored)
│   ├── users.csv                     # User master table — plan, region, signup date
│   ├── events.csv                    # Raw behavioral event log — 228k+ timestamped rows
│   ├── revenue.csv                   # Monthly MRR records per user
│   ├── labels.csv                    # Churn ground truth labels with dates
│   └── feature_store.csv            # Final 35-feature ML-ready matrix
│
├── charts/                           # EDA output charts (regenerated by phase2)
│
├── models/                           # Saved model artifacts
│
├── mlruns/                           # MLflow experiment tracking logs
│
├── phase1_data_engineering.py        # Synthetic data generation + feature store
├── phase2_eda.py                     # Exploratory data analysis + business insights
├── phase3_modeling.py                # ML models + hyperparameter tuning + SHAP
├── phase4_ab_testing.py              # A/B testing + causal inference framework
├── phase5_forecasting.py             # 90-day revenue forecasting
├── phase6_dashboard.py               # Streamlit stakeholder dashboard
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack Explained

**Pandas and NumPy** handle all data manipulation. The feature engineering in Phase 1 is written using SQL-style patterns — window functions, cohort queries, rolling aggregations — the same logic you would write in BigQuery or Snowflake at a real company. If you can read Phase 1, you can read a production data pipeline.

**Scikit-learn** provides the baseline models and all preprocessing utilities — train/test split, cross-validation, scaling, and SMOTE for handling class imbalance. It is the backbone of every ML pipeline regardless of which final model gets used.

**XGBoost and LightGBM** are the primary models. Both are gradient boosting frameworks that dominate tabular ML in production systems at companies like Visa and Amazon. They handle missing values, mixed feature types, and class imbalance better than most alternatives. Phase 3 trains both and compares them on ROC-AUC, precision-recall, and a custom business metric — estimated revenue recovered at different decision thresholds.

**Optuna** handles hyperparameter tuning using Bayesian optimization. Instead of exhaustively trying every combination like a grid search, it learns which hyperparameter regions are promising and focuses the search there. This is how tuning is done in production — not brute force.

**SHAP** explains every model prediction. For each flagged user, it produces a ranked list of which features pushed the risk score up or down and by how much. This is what makes the model usable by a product manager or customer success team — they can see exactly why a user was flagged without needing to understand the algorithm behind it.

**SciPy** powers the A/B testing framework. This includes two-sample t-tests, chi-square tests, power analysis to calculate required sample sizes before running an experiment, and proper p-value interpretation. A/B testing is the single most commonly tested topic in DS interviews at Google and Meta — Phase 4 demonstrates it at a practical level, not just a theoretical one.

**Prophet and LSTM** handle time series forecasting. Prophet is an open-source library built for business time series — it handles seasonality, trend changes, and missing data well and is widely used in industry. LSTM adds a deep learning layer that captures more complex temporal patterns. Both are evaluated on the same holdout window so the comparison is honest.

**MLflow** tracks every experiment — hyperparameters, metrics, model artifacts, and run metadata. Every result in the project is reproducible. You can look at the MLflow UI and see exactly which configuration produced which result and why the final model was chosen.

**Streamlit** turns the model output into a stakeholder-facing dashboard. It is designed for a product manager or customer success manager, not a data scientist. It shows at-risk accounts ranked by MRR, the top churn drivers per account, a recommended intervention, and a revenue recovery estimate if the intervention works.

**Docker** packages the full application so it runs anywhere without environment issues. One command to start the dashboard.

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/Rajshreesingh2/prism.git
cd prism
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Phase 1 to generate all datasets and build the feature store:

```bash
python phase1_data_engineering.py
```

This generates 5,000 users, 228,991 behavioral events, monthly revenue records, churn labels, and a 35-feature matrix. Everything is saved to the `data/` folder. The random seed is fixed so results are fully reproducible across machines.

Run Phase 2 to generate EDA charts and print business insights:

```bash
python phase2_eda.py
```

This produces 5 charts saved to `charts/` and prints the key findings to the console.

---

## Phase Breakdown

**Phase 1 — Data Engineering and Feature Store**

The foundation. Generates a synthetic but realistic SaaS dataset and builds a feature store using SQL-style transformations.

The user table contains 5,000 users distributed across free, basic, pro, and enterprise plans with realistic proportions — 40% free, 30% basic, 20% pro, 10% enterprise. Each user has a signup date, region, company size, and acquisition channel.

The event log captures 8 event types: login, feature use, report view, export, API call, settings change, support ticket, and invite sent. Higher-plan users generate more events — a pro user generates roughly 3.6x the events of a free user, which mirrors real product telemetry.

The churn labels are not randomly assigned. They are derived from behavioral signals — a user with low recent activity on a free plan has a genuinely higher probability of being labeled churned. This means the features have real predictive signal and the model is learning something meaningful.

The feature engineering mirrors what a DS would write in BigQuery. Recency is calculated as days since last event. Frequency is event counts across 30, 60, and 90 day windows. Adoption is the number of distinct event types a user has triggered. Revenue features include average MRR, MRR volatility, and 90-day total. Engagement velocity compares the last 30 days to the prior 30 days. Every transformation has a direct SQL CTE equivalent.

**Phase 2 — Exploratory Data Analysis**

Five charts that answer real product questions before modeling begins.

The first chart examines churn rate by plan and by acquisition channel — not just which is highest but the magnitude of the difference and the sample size behind each segment, which matters when deciding where to invest retention resources.

The second chart is a cohort retention heatmap. Each row is a signup cohort by month, each column is months since signup, and each cell is the percentage of that cohort still active. This is the standard retention analysis used at every SaaS company and in every DS interview that touches product analytics.

The third chart compares engagement velocity and days-since-active distributions for churned versus retained users. The separation between these distributions shows whether a feature will be useful in the model — overlapping distributions mean the feature has no predictive power.

The fourth chart examines revenue at risk over time and compares MRR distribution between churned and retained users. This reframes churn from a user count problem to a revenue problem.

The fifth chart compares feature adoption rates between churned and retained users across all 8 event types, directly answering which product features correlate most strongly with retention.

**Phase 3 — ML Modeling and Explainability**

Trains XGBoost, LightGBM, and Logistic Regression on the feature store. Evaluates all three using ROC-AUC, F1, and a custom business metric — estimated revenue recovered at different operating thresholds. Tunes the winning model with Optuna. Generates SHAP summary plots and per-user explanation outputs. All runs tracked in MLflow.

**Phase 4 — A/B Testing Framework**

Simulates a retention intervention: a cohort of high-risk users receives an in-app message, a control group does not. Implements power analysis to determine the required sample size before running, a two-sample t-test to evaluate the result, and a difference-in-differences estimator to control for confounders. This is the statistical validation layer that separates a complete DS project from a modeling exercise.

**Phase 5 — Revenue Forecasting**

90-day MRR forecast using Prophet for the trend and seasonality component, and an LSTM for residual pattern capture. Scheduled as a weekly Airflow job that updates the at-risk account list automatically. Forecast accuracy evaluated on a 90-day holdout using MAE and MAPE.

**Phase 6 — Streamlit Dashboard**

A stakeholder-facing dashboard showing at-risk accounts ranked by MRR exposure, the top SHAP drivers for each account, a recommended intervention, and a revenue recovery estimate. Designed for a product manager or customer success team — no data science knowledge required to operate it. Packaged in Docker for one-command deployment.

---

## Why This Project Is Built This Way

Most DS portfolios are a Jupyter notebook that loads a Kaggle CSV, trains a model, plots a confusion matrix, and stops. That demonstrates understanding of model mechanics. It does not demonstrate how DS work actually happens inside a company.

At Google, Amazon, Visa, or any company doing DS at scale, the work looks like this: raw data from multiple sources gets cleaned and transformed into a feature store. Models are trained, compared, and tracked with experiment management tools. Every output that reaches a stakeholder is explainable. Interventions are validated statistically, not assumed to work. Everything is reproducible, versioned, and deployable.

PRISM is structured to demonstrate all of that — not because it looks impressive, but because that is what the job actually requires. Each phase maps to a real part of the DS workflow. Each tool is chosen because it is used in production, not because it was the easiest to implement.

The git history tells the story phase by phase. A recruiter or hiring manager who looks at the commit log can see how the project was built, not just what it produced.

---

## Connect

**GitHub:** [Rajshreesingh2](https://github.com/Rajshreesingh2)

Open to feedback, questions, and contributions. If something is wrong or could be improved, open an issue.

---

<div align="center">
<sub>Built phase by phase. Each commit, one step closer.</sub>
</div>