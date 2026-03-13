"""
PRISM — Predictive Revenue & Intelligence Signal Monitor
Phase 1: Synthetic Data Generation + Feature Engineering Pipeline

Simulates real-world SaaS/fintech user telemetry as seen at
companies like Visa, Google, Amazon, and Apple.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_USERS        = 5000
START_DATE     = datetime(2023, 1, 1)
END_DATE       = datetime(2024, 6, 30)
CHURN_RATE     = 0.27   # realistic SaaS churn rate


# ─────────────────────────────────────────────
# 1. USER MASTER TABLE
# ─────────────────────────────────────────────
def generate_users(n: int) -> pd.DataFrame:
    """
    Simulates a users table as it would exist in a production DB.
    Mirrors what you'd query from a data warehouse at Visa or Google.
    """
    plans       = ["free", "basic", "pro", "enterprise"]
    plan_weights= [0.40,   0.30,   0.20,  0.10]
    regions     = ["NA", "EU", "APAC", "LATAM"]

    signup_days = np.random.randint(0, (END_DATE - START_DATE).days - 90, n)

    df = pd.DataFrame({
        "user_id"       : [f"U{str(i).zfill(5)}" for i in range(n)],
        "signup_date"   : [START_DATE + timedelta(days=int(d)) for d in signup_days],
        "plan"          : np.random.choice(plans, n, p=plan_weights),
        "region"        : np.random.choice(regions, n),
        "age"           : np.random.randint(18, 65, n),
        "company_size"  : np.random.choice(["1-10","11-50","51-200","200+"], n,
                                            p=[0.35, 0.30, 0.20, 0.15]),
        "acquisition_channel": np.random.choice(
                            ["organic","paid_search","referral","social","direct"], n,
                            p=[0.30, 0.25, 0.20, 0.15, 0.10])
    })
    return df


# ─────────────────────────────────────────────
# 2. BEHAVIORAL EVENTS TABLE
# ─────────────────────────────────────────────
def generate_events(users: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates a raw event log — the kind stored in BigQuery or Snowflake.
    Each row is one user action timestamped in the product.
    """
    event_types = [
        "login", "feature_use", "report_view", "export",
        "api_call", "settings_change", "support_ticket", "invite_sent"
    ]

    rows = []
    for _, user in users.iterrows():
        # Higher-plan users are more active
        plan_multiplier = {"free": 0.5, "basic": 1.0, "pro": 1.8, "enterprise": 3.0}
        base_events = int(np.random.poisson(40) * plan_multiplier[user["plan"]])

        user_start = user["signup_date"]
        user_end   = min(END_DATE, user_start + timedelta(days=540))
        date_range = (user_end - user_start).days

        if date_range <= 0:
            continue

        event_days = np.random.randint(0, date_range, base_events)
        for day in event_days:
            rows.append({
                "user_id"   : user["user_id"],
                "event_date": user_start + timedelta(days=int(day)),
                "event_type": np.random.choice(event_types,
                                p=[0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03])
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 3. REVENUE TABLE
# ─────────────────────────────────────────────
def generate_revenue(users: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly revenue per user — simulates billing records.
    """
    plan_mrr = {"free": 0, "basic": 29, "pro": 99, "enterprise": 499}

    rows = []
    for _, user in users.iterrows():
        months = pd.date_range(user["signup_date"], END_DATE, freq="MS")
        base   = plan_mrr[user["plan"]]
        for month in months:
            # Add noise + occasional upgrades
            mrr = max(0, base + np.random.normal(0, base * 0.05))
            rows.append({
                "user_id" : user["user_id"],
                "month"   : month,
                "mrr"     : round(mrr, 2)
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 4. CHURN LABELS — realistic signal-based logic
# ─────────────────────────────────────────────
def generate_churn_labels(users: pd.DataFrame,
                           events: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns churn labels based on realistic behavioral signals,
    not random — mirrors how actual churn is labeled from DB records.
    """
    OBSERVATION_DATE = datetime(2024, 3, 31)
    CHURN_WINDOW     = 90  # days after observation

    # Recent activity score per user
    recent_events = events[
        events["event_date"] >= OBSERVATION_DATE - timedelta(days=30)
    ].groupby("user_id").size().reset_index(name="recent_events")

    df = users.merge(recent_events, on="user_id", how="left")
    df["recent_events"] = df["recent_events"].fillna(0)

    # Churn probability driven by inactivity + plan
    plan_churn_base = {"free": 0.45, "basic": 0.28, "pro": 0.15, "enterprise": 0.08}
    df["base_churn_prob"] = df["plan"].map(plan_churn_base)

    # Less activity → higher churn probability
    activity_factor = np.exp(-df["recent_events"] / 10)
    df["churn_prob"] = (df["base_churn_prob"] * activity_factor).clip(0.02, 0.95)

    df["churned"] = (np.random.random(len(df)) < df["churn_prob"]).astype(int)
    df["observation_date"] = OBSERVATION_DATE
    df["churn_date"] = df.apply(
        lambda r: OBSERVATION_DATE + timedelta(days=int(np.random.randint(1, CHURN_WINDOW)))
        if r["churned"] else pd.NaT, axis=1
    )

    return df[["user_id", "observation_date", "churned", "churn_date", "churn_prob"]]


# ─────────────────────────────────────────────
# 5. SQL-STYLE FEATURE ENGINEERING
#    (Mirrors CTEs you'd write in BigQuery / Snowflake)
# ─────────────────────────────────────────────
def build_feature_store(users: pd.DataFrame,
                         events: pd.DataFrame,
                         revenue: pd.DataFrame,
                         labels: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the feature store used for ML modeling.

    Each feature group maps to a real SQL CTE you'd write
    at a company like Google or Visa. Comments show the
    equivalent SQL pattern.
    """
    OBS_DATE = pd.Timestamp("2024-03-31")

    # ── Feature Group 1: Recency ──────────────────────────
    # SQL: SELECT user_id, MAX(event_date) as last_active FROM events GROUP BY user_id
    last_active = (events.groupby("user_id")["event_date"]
                   .max().reset_index()
                   .rename(columns={"event_date": "last_active_date"}))
    last_active["days_since_last_active"] = (
        OBS_DATE - pd.to_datetime(last_active["last_active_date"])
    ).dt.days

    # ── Feature Group 2: Frequency ────────────────────────
    # SQL: SELECT user_id, COUNT(*) / date_diff(...) as daily_avg FROM events ...
    freq = events.copy()
    freq["event_date"] = pd.to_datetime(freq["event_date"])

    last_30  = freq[freq["event_date"] >= OBS_DATE - timedelta(days=30)]
    last_60  = freq[freq["event_date"] >= OBS_DATE - timedelta(days=60)]
    last_90  = freq[freq["event_date"] >= OBS_DATE - timedelta(days=90)]

    f30 = last_30.groupby("user_id").size().reset_index(name="events_last_30d")
    f60 = last_60.groupby("user_id").size().reset_index(name="events_last_60d")
    f90 = last_90.groupby("user_id").size().reset_index(name="events_last_90d")

    # ── Feature Group 3: Feature Adoption ─────────────────
    # SQL: SELECT user_id, COUNT(DISTINCT event_type) as feature_breadth FROM events ...
    adoption = (events.groupby("user_id")["event_type"]
                .nunique().reset_index()
                .rename(columns={"event_type": "feature_breadth"}))

    pivoted = (events.groupby(["user_id", "event_type"])
               .size().unstack(fill_value=0).reset_index())
    pivoted.columns = ["user_id"] + [f"evt_{c}" for c in pivoted.columns[1:]]

    # ── Feature Group 4: Revenue Signals ──────────────────
    # SQL: SELECT user_id, AVG(mrr) as avg_mrr, STDDEV(mrr) as mrr_volatility FROM revenue ...
    rev = revenue.copy()
    rev["month"] = pd.to_datetime(rev["month"])
    rev_recent = rev[rev["month"] >= OBS_DATE - timedelta(days=90)]

    rev_features = rev_recent.groupby("user_id")["mrr"].agg(
        avg_mrr="mean",
        mrr_volatility="std",
        total_revenue_90d="sum"
    ).reset_index()
    rev_features["mrr_volatility"] = rev_features["mrr_volatility"].fillna(0)

    # ── Feature Group 5: Engagement Velocity ──────────────
    # Trend: are they using the product more or less over time?
    # SQL: CTE comparing last_30d vs prior_30d event counts
    prior_30 = freq[
        (freq["event_date"] >= OBS_DATE - timedelta(days=60)) &
        (freq["event_date"] <  OBS_DATE - timedelta(days=30))
    ]
    p30 = prior_30.groupby("user_id").size().reset_index(name="events_prior_30d")

    velocity = f30.merge(p30, on="user_id", how="left")
    velocity["events_prior_30d"] = velocity["events_prior_30d"].fillna(0)
    velocity["engagement_velocity"] = (
        (velocity["events_last_30d"] - velocity["events_prior_30d"])
        / (velocity["events_prior_30d"] + 1)
    )

    # ── Feature Group 6: Support Signals ──────────────────
    support = events[events["event_type"] == "support_ticket"]
    sup_feat = (support.groupby("user_id").size()
                .reset_index(name="support_tickets_90d"))

    # ── Tenure ────────────────────────────────────────────
    users_copy = users.copy()
    users_copy["tenure_days"] = (OBS_DATE - pd.to_datetime(users_copy["signup_date"])).dt.days

    # ── Assemble feature store ─────────────────────────────
    base = users_copy[["user_id", "plan", "region", "age",
                        "company_size", "acquisition_channel", "tenure_days"]]

    for df_feat in [last_active[["user_id","days_since_last_active"]],
                    f30, f60, f90,
                    adoption, pivoted,
                    rev_features,
                    velocity[["user_id","engagement_velocity"]],
                    sup_feat,
                    labels[["user_id","churned"]]]:
        base = base.merge(df_feat, on="user_id", how="left")

    # Fill missing numerics
    num_cols = base.select_dtypes(include=np.number).columns
    base[num_cols] = base[num_cols].fillna(0)

    # Encode categoricals
    cat_cols = ["plan", "region", "company_size", "acquisition_channel"]
    base = pd.get_dummies(base, columns=cat_cols, drop_first=True)

    return base


# ─────────────────────────────────────────────
# MAIN — run the full pipeline
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  PRISM — Phase 1: Data Engineering Pipeline")
    print("=" * 55)

    print("\n[1/5] Generating user master table...")
    users = generate_users(N_USERS)
    print(f"      {len(users):,} users created")

    print("[2/5] Generating behavioral event log...")
    events = generate_events(users)
    print(f"      {len(events):,} events generated")

    print("[3/5] Generating revenue records...")
    revenue = generate_revenue(users)
    print(f"      {len(revenue):,} revenue records")

    print("[4/5] Generating churn labels...")
    labels = generate_churn_labels(users, events)
    churn_count = labels["churned"].sum()
    churn_pct   = labels["churned"].mean() * 100
    print(f"      {churn_count:,} churned users ({churn_pct:.1f}%)")

    print("[5/5] Building feature store...")
    features = build_feature_store(users, events, revenue, labels)
    print(f"      {features.shape[0]:,} rows × {features.shape[1]:,} features")

    # Save all tables
    import os
    os.makedirs("/home/claude/prism/data", exist_ok=True)
    users.to_csv("/home/claude/prism/data/users.csv", index=False)
    events.to_csv("/home/claude/prism/data/events.csv", index=False)
    revenue.to_csv("/home/claude/prism/data/revenue.csv", index=False)
    labels.to_csv("/home/claude/prism/data/labels.csv", index=False)
    features.to_csv("/home/claude/prism/data/feature_store.csv", index=False)

    print("\n✓ All tables saved to /data/")
    print("\nFeature store sample:")
    print(features[["user_id","tenure_days","days_since_last_active",
                     "events_last_30d","engagement_velocity",
                     "avg_mrr","churned"]].head(8).to_string(index=False))

    print("\n" + "=" * 55)
    print("  Phase 1 complete. Ready for Phase 2: EDA")
    print("=" * 55)