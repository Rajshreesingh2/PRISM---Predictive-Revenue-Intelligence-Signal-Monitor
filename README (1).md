# PRISM 🔬
**Predictive Revenue & Intelligence Signal Monitor**

---

I built this because I kept seeing the same problem everywhere — companies only find out a customer is leaving *after* they've left. The data to predict it was always there, just scattered across event logs, billing tables, and support tickets with no one connecting the dots.

PRISM is my attempt to build the system that actually catches it early. It's not a Kaggle notebook. It's built the way I'd build it if I were a DS at Google or Visa — with a real feature store, proper experiment tracking, A/B testing, and a dashboard that a PM could actually use.

---

## What it does

Takes raw user behavior data → spits out who's going to churn in the next 30–90 days, why, and what to do about it.

![Churn by Segment](charts/01_churn_by_segment.png)

The free plan churns at 20%. Pro users barely churn at all (1.6%). That's not surprising — but *why* free users churn, and *which* free users are about to churn, is where it gets interesting.

---

## The signals that actually matter

Most churn models just look at "days since last login." That's lazy. PRISM tracks five things:

**Engagement velocity** — not just how active someone is, but whether they're becoming *more* or *less* active over time. A user who logged in 10 times last month but only 3 times this month is a very different risk profile than someone consistently at 10.

**Feature adoption breadth** — users who only ever use one feature are fragile. One bad experience and they're gone. Users spread across 4–5 features have built habits.

![Feature Adoption](charts/05_feature_adoption.png)

**Revenue trajectory** — MRR volatility is a signal. Accounts that fluctuate a lot are usually debating whether to stay.

**Support ticket spikes** — a sudden burst of support tickets right before churn is one of the clearest signals in the data.

**Cohort retention shape** — some signup cohorts retain way better than others. Knowing *which* cohorts are healthy tells you what acquisition channels to double down on.

![Cohort Retention](charts/02_cohort_retention.png)

---

## The finding that surprised me

Users acquired through referral retain significantly better than paid search — even after controlling for plan type. Referral users come in with someone vouching for the product. They hit onboarding differently. That's not a modeling insight, that's a business decision waiting to happen.

![Behavioral Signals](charts/03_behavioral_signals.png)

---

## Revenue at risk

At any given time, about $8,800/month of MRR is sitting with users who are 60%+ likely to churn in the next 90 days. The model flags them. The dashboard shows you who they are. What you do with that is up to the product team.

![Revenue at Risk](charts/04_revenue_at_risk.png)

---

## How it's built

Nothing exotic. The stack is what you'd actually find at a real company:

```
Data layer      →  pandas, SQL-style CTEs (window functions, cohort queries)
Models          →  XGBoost, LightGBM, Logistic Regression (compared properly)
Tuning          →  Optuna
Explainability  →  SHAP  (so you can tell a PM *why* a user is flagged)
A/B testing     →  scipy.stats  (power analysis, t-tests, the real stuff)
Forecasting     →  Prophet + LSTM
Tracking        →  MLflow
Dashboard       →  Streamlit
```

Each phase is its own script. Run them in order.

---

## Run it yourself

```bash
git clone https://github.com/YOUR_USERNAME/prism.git
cd prism
pip install -r requirements.txt

python phase1_data_engineering.py   # builds the feature store
python phase2_eda.py                # generates all the charts above
# phase 3–6 coming as I build them out
```

The synthetic dataset generates ~229k behavioral events across 5,000 users. It's designed to mimic real SaaS telemetry — plan distribution, activity patterns, revenue records — so the model actually has something meaningful to learn from.

---

## Project status

- [x] Phase 1 — Data engineering & feature store
- [x] Phase 2 — EDA & business insights
- [ ] Phase 3 — ML modeling & SHAP explainability
- [ ] Phase 4 — A/B testing framework
- [ ] Phase 5 — Revenue forecasting
- [ ] Phase 6 — Streamlit dashboard

Building this phase by phase. Each commit = one phase complete.

---

## What I learned

The hardest part wasn't the model. It was building a feature store that actually reflects *how* users behave over time, not just a snapshot. Getting recency, frequency, velocity, and adoption into clean, interpretable features took most of Phase 1.

The A/B testing framework (Phase 4) is what I'm most looking forward to — that's the piece that closes the loop between "here's who will churn" and "here's what intervention actually worked."

---

*Built by [Your Name]. Open to feedback, questions, and collaboration.*
