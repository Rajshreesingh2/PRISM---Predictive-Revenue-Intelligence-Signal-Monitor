"""
PRISM — Phase 2: Exploratory Data Analysis
Produces the kind of business-grade EDA a DS at Google,
Amazon, or Visa would present to a product team.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "font.family"       : "DejaVu Sans",
    "font.size"         : 11,
})
COLORS = ["#3B8BD4", "#1D9E75", "#EF9F27", "#D85A30", "#7F77DD", "#888780"]

# ── Load data ─────────────────────────────────────────
users    = pd.read_csv("/home/claude/prism/data/users.csv", parse_dates=["signup_date"])
events   = pd.read_csv("/home/claude/prism/data/events.csv", parse_dates=["event_date"])
revenue  = pd.read_csv("/home/claude/prism/data/revenue.csv", parse_dates=["month"])
labels   = pd.read_csv("/home/claude/prism/data/labels.csv")
features = pd.read_csv("/home/claude/prism/data/feature_store.csv")

df = users.merge(labels[["user_id","churned"]], on="user_id")

import os
os.makedirs("/home/claude/prism/charts", exist_ok=True)

# ─────────────────────────────────────────────────────
# CHART 1: Churn rate by plan & acquisition channel
# ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PRISM — Churn Analysis by Segment", fontsize=14, fontweight="bold", y=1.01)

# By plan
plan_churn = df.groupby("plan")["churned"].agg(["mean","count"]).reset_index()
plan_churn.columns = ["plan","churn_rate","count"]
plan_order = ["free","basic","pro","enterprise"]
plan_churn = plan_churn.set_index("plan").reindex(plan_order).reset_index()

bars = axes[0].bar(plan_churn["plan"], plan_churn["churn_rate"] * 100,
                   color=COLORS[:4], edgecolor="white", linewidth=1.5, width=0.6)
for bar, (_, row) in zip(bars, plan_churn.iterrows()):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{row['churn_rate']*100:.1f}%\n(n={int(row['count']):,})",
                 ha="center", va="bottom", fontsize=9.5)
axes[0].set_title("Churn Rate by Plan", fontweight="bold")
axes[0].set_ylabel("Churn Rate (%)")
axes[0].set_ylim(0, plan_churn["churn_rate"].max() * 130)

# By acquisition channel
ch_churn = df.groupby("acquisition_channel")["churned"].mean().sort_values(ascending=True) * 100
axes[1].barh(ch_churn.index, ch_churn.values, color=COLORS[0], alpha=0.85, edgecolor="white")
for i, v in enumerate(ch_churn.values):
    axes[1].text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9.5)
axes[1].set_title("Churn Rate by Acquisition Channel", fontweight="bold")
axes[1].set_xlabel("Churn Rate (%)")

plt.tight_layout()
plt.savefig("/home/claude/prism/charts/01_churn_by_segment.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 1 saved: Churn by Segment")

# ─────────────────────────────────────────────────────
# CHART 2: Cohort retention heatmap
# ─────────────────────────────────────────────────────
events["event_month"]  = events["event_date"].dt.to_period("M")
users["signup_month"]  = users["signup_date"].dt.to_period("M")

cohort_data = events.merge(users[["user_id","signup_month"]], on="user_id")
cohort_data["cohort_index"] = (
    cohort_data["event_month"] - cohort_data["signup_month"]
).apply(lambda x: x.n)

cohort_counts = cohort_data[cohort_data["cohort_index"] >= 0].groupby(
    ["signup_month","cohort_index"])["user_id"].nunique().reset_index()

cohort_pivot = cohort_counts.pivot(
    index="signup_month", columns="cohort_index", values="user_id")
cohort_size  = cohort_pivot[0]
retention    = cohort_pivot.divide(cohort_size, axis=0).iloc[:12, :8] * 100

fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(retention, annot=True, fmt=".0f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Retention %"},
            annot_kws={"size": 9})
ax.set_title("PRISM — Monthly Cohort Retention Heatmap (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Months Since Signup")
ax.set_ylabel("Signup Cohort")
plt.tight_layout()
plt.savefig("/home/claude/prism/charts/02_cohort_retention.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 2 saved: Cohort Retention Heatmap")

# ─────────────────────────────────────────────────────
# CHART 3: Engagement velocity vs churn
# ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PRISM — Behavioral Signals vs Churn", fontsize=14, fontweight="bold")

churned     = features[features["churned"] == 1]
not_churned = features[features["churned"] == 0]

# Engagement velocity distribution
axes[0].hist(not_churned["engagement_velocity"].clip(-5, 20), bins=40,
             alpha=0.7, color=COLORS[1], label="Retained", density=True)
axes[0].hist(churned["engagement_velocity"].clip(-5, 20), bins=40,
             alpha=0.7, color=COLORS[3], label="Churned", density=True)
axes[0].set_title("Engagement Velocity Distribution", fontweight="bold")
axes[0].set_xlabel("Engagement Velocity (last 30d vs prior 30d)")
axes[0].set_ylabel("Density")
axes[0].legend()

# Days since last active vs churn
axes[1].hist(not_churned["days_since_last_active"].abs().clip(0, 180), bins=40,
             alpha=0.7, color=COLORS[1], label="Retained", density=True)
axes[1].hist(churned["days_since_last_active"].abs().clip(0, 180), bins=40,
             alpha=0.7, color=COLORS[3], label="Churned", density=True)
axes[1].set_title("Days Since Last Active", fontweight="bold")
axes[1].set_xlabel("Days")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.savefig("/home/claude/prism/charts/03_behavioral_signals.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 3 saved: Behavioral Signals")

# ─────────────────────────────────────────────────────
# CHART 4: Revenue at risk by segment
# ─────────────────────────────────────────────────────
rev_risk = features[features["churned"] == 1].copy()
plan_cols = [c for c in features.columns if c.startswith("plan_")]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PRISM — Revenue at Risk", fontsize=14, fontweight="bold")

# MRR distribution: churned vs retained
axes[0].boxplot(
    [not_churned["avg_mrr"].clip(0, 600), churned["avg_mrr"].clip(0, 600)],
    labels=["Retained", "Churned"],
    patch_artist=True,
    boxprops=dict(facecolor=COLORS[0], alpha=0.6),
    medianprops=dict(color="black", linewidth=2)
)
axes[0].set_title("MRR Distribution: Retained vs Churned", fontweight="bold")
axes[0].set_ylabel("Average MRR ($)")

# Monthly revenue at risk trend
rev_monthly = revenue.merge(labels[["user_id","churned"]], on="user_id")
at_risk = rev_monthly[rev_monthly["churned"] == 1].groupby("month")["mrr"].sum()
total   = rev_monthly.groupby("month")["mrr"].sum()
risk_pct = (at_risk / total * 100).dropna()

axes[1].fill_between(risk_pct.index, risk_pct.values, alpha=0.3, color=COLORS[3])
axes[1].plot(risk_pct.index, risk_pct.values, color=COLORS[3], linewidth=2)
axes[1].set_title("Revenue at Risk Over Time (%)", fontweight="bold")
axes[1].set_ylabel("% of Monthly Revenue at Risk")
axes[1].set_xlabel("Month")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("/home/claude/prism/charts/04_revenue_at_risk.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 4 saved: Revenue at Risk")

# ─────────────────────────────────────────────────────
# CHART 5: Feature adoption funnel
# ─────────────────────────────────────────────────────
event_cols = [c for c in features.columns if c.startswith("evt_")]
adoption_rates = {}
for col in event_cols:
    feature_name = col.replace("evt_", "")
    used_retained = (not_churned[col] > 0).mean() * 100
    used_churned  = (churned[col]     > 0).mean() * 100
    adoption_rates[feature_name] = {
        "retained": used_retained,
        "churned" : used_churned
    }

adopt_df = pd.DataFrame(adoption_rates).T.sort_values("retained", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(adopt_df))
w = 0.35
ax.barh(y + w/2, adopt_df["retained"], w, label="Retained", color=COLORS[1], alpha=0.85)
ax.barh(y - w/2, adopt_df["churned"],  w, label="Churned",  color=COLORS[3], alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(adopt_df.index)
ax.set_xlabel("% of Users Who Used Feature")
ax.set_title("PRISM — Feature Adoption: Retained vs Churned Users",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("/home/claude/prism/charts/05_feature_adoption.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 5 saved: Feature Adoption Funnel")

# ─────────────────────────────────────────────────────
# Print key business insights
# ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  PRISM — Key Business Insights from EDA")
print("=" * 55)

total_at_risk_mrr = features[features["churned"]==1]["avg_mrr"].sum()
print(f"\n  Total MRR at risk        : ${total_at_risk_mrr:,.0f}/month")
print(f"  Churned users            : {labels['churned'].sum():,} ({labels['churned'].mean()*100:.1f}%)")

plan_churn_rates = df.groupby("plan")["churned"].mean().sort_values(ascending=False)
print(f"\n  Highest churn plan       : {plan_churn_rates.index[0]} ({plan_churn_rates.iloc[0]*100:.1f}%)")
print(f"  Lowest churn plan        : {plan_churn_rates.index[-1]} ({plan_churn_rates.iloc[-1]*100:.1f}%)")

high_vel  = features[features["engagement_velocity"] > 2]["churned"].mean() * 100
low_vel   = features[features["engagement_velocity"] <= 0]["churned"].mean() * 100
print(f"\n  Churn rate (high engagement velocity) : {high_vel:.1f}%")
print(f"  Churn rate (low/neg engagement velocity): {low_vel:.1f}%")

top_channel = df.groupby("acquisition_channel")["churned"].mean().idxmin()
print(f"\n  Best retention channel   : {top_channel}")
print("\n  5 EDA charts saved to /charts/")
print("=" * 55)
print("  Phase 2 complete. Ready for Phase 3: ML Modeling")
print("=" * 55)