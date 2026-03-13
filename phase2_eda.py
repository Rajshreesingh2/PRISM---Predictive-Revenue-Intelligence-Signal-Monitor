"""
PRISM v2 — Phase 2: Deep Hypothesis-Driven EDA
Real Telco data + macro signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("charts", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.size"        : 11,
})
COLORS = ["#3B8BD4", "#D85A30", "#1D9E75", "#EF9F27", "#7F77DD", "#888780"]

print("=" * 60)
print("  PRISM v2 — Phase 2: Deep EDA")
print("=" * 60)

# Load data
df = pd.read_csv("data/telco_cleaned.csv")
fs = pd.read_csv("data/feature_store.csv")

churned  = df[df["Churn_binary"] == 1]
retained = df[df["Churn_binary"] == 0]

print(f"\n  Loaded: {len(df):,} customers")
print(f"  Churned: {len(churned):,} | Retained: {len(retained):,}")


# ─────────────────────────────────────────────────────────────
# CHART 1: Churn rate by key segments
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("PRISM — Churn Rate by Customer Segment", fontsize=14, fontweight="bold")

# By contract type
contract_churn = df.groupby("Contract")["Churn_binary"].agg(["mean","count"]).reset_index()
contract_churn.columns = ["Contract","churn_rate","count"]
bars = axes[0].bar(contract_churn["Contract"], contract_churn["churn_rate"]*100,
                   color=COLORS[:3], width=0.5, edgecolor="white")
for bar, (_, row) in zip(bars, contract_churn.iterrows()):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{row['churn_rate']*100:.1f}%\n(n={int(row['count']):,})",
                 ha="center", va="bottom", fontsize=9)
axes[0].set_title("By Contract Type", fontweight="bold")
axes[0].set_ylabel("Churn Rate (%)")
axes[0].set_ylim(0, 55)

# By internet service
internet_churn = df.groupby("InternetService")["Churn_binary"].agg(["mean","count"]).reset_index()
internet_churn.columns = ["Service","churn_rate","count"]
bars = axes[1].bar(internet_churn["Service"], internet_churn["churn_rate"]*100,
                   color=COLORS[:3], width=0.5, edgecolor="white")
for bar, (_, row) in zip(bars, internet_churn.iterrows()):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{row['churn_rate']*100:.1f}%\n(n={int(row['count']):,})",
                 ha="center", va="bottom", fontsize=9)
axes[1].set_title("By Internet Service", fontweight="bold")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].set_ylim(0, 55)

# By tenure bucket
df["TenureBucket"] = pd.cut(df["tenure"], bins=[0,6,12,24,48,72],
                             labels=["0-6m","6-12m","1-2yr","2-4yr","4+yr"])
tenure_churn = df.groupby("TenureBucket", observed=True)["Churn_binary"].mean() * 100
axes[2].bar(tenure_churn.index, tenure_churn.values, color=COLORS[0], alpha=0.85, width=0.5)
for i, v in enumerate(tenure_churn.values):
    axes[2].text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
axes[2].set_title("By Tenure", fontweight="bold")
axes[2].set_ylabel("Churn Rate (%)")
axes[2].set_ylim(0, 60)

plt.tight_layout()
plt.savefig("charts/01_churn_by_segment.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Chart 1 saved: Churn by Segment")


# ─────────────────────────────────────────────────────────────
# CHART 2: Monthly charges distribution — churned vs retained
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PRISM — Revenue Signals", fontsize=14, fontweight="bold")

axes[0].hist(retained["MonthlyCharges"], bins=40, alpha=0.7,
             color=COLORS[0], label="Retained", density=True)
axes[0].hist(churned["MonthlyCharges"], bins=40, alpha=0.7,
             color=COLORS[1], label="Churned", density=True)

# KS test
ks_stat, ks_p = stats.ks_2samp(retained["MonthlyCharges"], churned["MonthlyCharges"])
axes[0].set_title(f"Monthly Charges Distribution\nKS stat={ks_stat:.3f}, p={ks_p:.2e}",
                  fontweight="bold")
axes[0].set_xlabel("Monthly Charges ($)")
axes[0].set_ylabel("Density")
axes[0].legend()

# Total charges by churn
axes[1].boxplot(
    [retained["TotalCharges"].dropna(), churned["TotalCharges"].dropna()],
    labels=["Retained", "Churned"],
    patch_artist=True,
    boxprops=dict(facecolor=COLORS[0], alpha=0.6),
    medianprops=dict(color="black", linewidth=2)
)
axes[1].set_title("Total Charges Distribution", fontweight="bold")
axes[1].set_ylabel("Total Charges ($)")

plt.tight_layout()
plt.savefig("charts/02_revenue_signals.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 2 saved: Revenue Signals")


# ─────────────────────────────────────────────────────────────
# CHART 3: Service adoption vs churn (hypothesis test)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PRISM — Service Adoption & Risk Score", fontsize=14, fontweight="bold")

# Service adoption score
adoption_churn = df.groupby("ServiceAdoptionScore")["Churn_binary"].mean() * 100
axes[0].bar(adoption_churn.index, adoption_churn.values,
            color=[COLORS[1] if v > 30 else COLORS[0] for v in adoption_churn.values],
            width=0.6, edgecolor="white")
for i, v in enumerate(adoption_churn.values):
    axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
axes[0].set_title("Churn Rate by Service Adoption Score", fontweight="bold")
axes[0].set_xlabel("Number of Services Used")
axes[0].set_ylabel("Churn Rate (%)")
axes[0].set_xticks(range(len(adoption_churn)))

# Risk score distribution
axes[1].hist(retained["RiskScore"], bins=9, alpha=0.7,
             color=COLORS[0], label="Retained", density=True)
axes[1].hist(churned["RiskScore"], bins=9, alpha=0.7,
             color=COLORS[1], label="Churned", density=True)

ks_stat2, ks_p2 = stats.ks_2samp(retained["RiskScore"], churned["RiskScore"])
axes[1].set_title(f"Risk Score Distribution\nKS stat={ks_stat2:.3f}, p={ks_p2:.2e}",
                  fontweight="bold")
axes[1].set_xlabel("Composite Risk Score")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.savefig("charts/03_service_adoption.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 3 saved: Service Adoption & Risk Score")


# ─────────────────────────────────────────────────────────────
# CHART 4: Payment method & paperless billing
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PRISM — Behavioral Signals", fontsize=14, fontweight="bold")

# Payment method
pay_churn = df.groupby("PaymentMethod")["Churn_binary"].mean().sort_values(ascending=True) * 100
axes[0].barh(pay_churn.index, pay_churn.values,
             color=[COLORS[1] if v > 30 else COLORS[0] for v in pay_churn.values],
             alpha=0.85)
for i, v in enumerate(pay_churn.values):
    axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
axes[0].set_title("Churn Rate by Payment Method", fontweight="bold")
axes[0].set_xlabel("Churn Rate (%)")
axes[0].set_xlim(0, 55)

# Senior citizen vs non-senior
senior_churn = df.groupby("SeniorCitizen")["Churn_binary"].agg(["mean","count"])
labels = ["Non-Senior", "Senior"]
bars = axes[1].bar(labels, senior_churn["mean"]*100,
                   color=COLORS[:2], width=0.4, edgecolor="white")
for bar, (_, row) in zip(bars, senior_churn.iterrows()):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{row['mean']*100:.1f}%\n(n={int(row['count']):,})",
                 ha="center", va="bottom", fontsize=9)

# Chi-square test
contingency = pd.crosstab(df["SeniorCitizen"], df["Churn_binary"])
chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
axes[1].set_title(f"Senior vs Non-Senior Churn\nChi2={chi2:.1f}, p={chi_p:.2e}",
                  fontweight="bold")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].set_ylim(0, 50)

plt.tight_layout()
plt.savefig("charts/04_behavioral_signals.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 4 saved: Behavioral Signals")


# ─────────────────────────────────────────────────────────────
# CHART 5: Tenure vs Monthly Charges — scatter by churn
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(retained["tenure"], retained["MonthlyCharges"],
           alpha=0.3, color=COLORS[0], s=10, label="Retained")
ax.scatter(churned["tenure"], churned["MonthlyCharges"],
           alpha=0.4, color=COLORS[1], s=10, label="Churned")
ax.set_title("PRISM — Tenure vs Monthly Charges by Churn Status",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Monthly Charges ($)")
ax.legend()
plt.tight_layout()
plt.savefig("charts/05_tenure_vs_charges.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 5 saved: Tenure vs Charges")


# ─────────────────────────────────────────────────────────────
# CHART 6: Macro signals correlation with churn segments
# ─────────────────────────────────────────────────────────────
import json
with open("data/macro_signals.json") as f:
    macro = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("PRISM — Macro Economic Context", fontsize=14, fontweight="bold")

macro_names  = ["GDP Growth", "Inflation", "Unemployment"]
macro_values = [macro.get("latest_gdp_growth", 0),
                macro.get("latest_inflation", 0),
                macro.get("latest_unemployment", 0)]
macro_colors = [COLORS[2], COLORS[1], COLORS[3]]

for i, (name, val, color) in enumerate(zip(macro_names, macro_values, macro_colors)):
    axes[i].bar([name], [val], color=color, width=0.4, alpha=0.85)
    axes[i].text(0, val + 0.05, f"{val:.2f}%", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
    axes[i].set_title(f"US {name}", fontweight="bold")
    axes[i].set_ylabel("Rate (%)")
    axes[i].set_ylim(0, max(val * 1.4, 1))
    context = ("Healthy growth" if name == "GDP Growth" and val > 2
               else "High inflation" if name == "Inflation" and val > 4
               else "Low unemployment" if name == "Unemployment" and val < 5
               else "Watch closely")
    axes[i].text(0, val * 0.5, context, ha="center", va="center",
                 fontsize=9, color="white", fontweight="bold")

plt.tight_layout()
plt.savefig("charts/06_macro_context.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 6 saved: Macro Context")


# ─────────────────────────────────────────────────────────────
# HYPOTHESIS TESTS — printed results
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PRISM — Hypothesis Test Results")
print("=" * 60)

# H1: Month-to-month customers churn significantly more
h1_mtm  = df[df["MonthToMonth"]==1]["Churn_binary"].mean()
h1_long = df[df["MonthToMonth"]==0]["Churn_binary"].mean()
ct1 = pd.crosstab(df["MonthToMonth"], df["Churn_binary"])
chi1, p1, _, _ = stats.chi2_contingency(ct1)
print(f"\n  H1: Month-to-month customers churn more")
print(f"      Month-to-month churn: {h1_mtm*100:.1f}%")
print(f"      Long-term churn:      {h1_long*100:.1f}%")
print(f"      Chi2={chi1:.1f}, p={p1:.2e} → {'CONFIRMED' if p1 < 0.05 else 'NOT confirmed'}")

# H2: Single-service users churn more than multi-service
h2_single = df[df["SingleServiceUser"]==1]["Churn_binary"].mean()
h2_multi  = df[df["SingleServiceUser"]==0]["Churn_binary"].mean()
ct2 = pd.crosstab(df["SingleServiceUser"], df["Churn_binary"])
chi2_stat, p2, _, _ = stats.chi2_contingency(ct2)
print(f"\n  H2: Single-service users churn more")
print(f"      Single-service churn: {h2_single*100:.1f}%")
print(f"      Multi-service churn:  {h2_multi*100:.1f}%")
print(f"      Chi2={chi2_stat:.1f}, p={p2:.2e} → {'CONFIRMED' if p2 < 0.05 else 'NOT confirmed'}")

# H3: Fiber optic customers churn more than DSL
h3_fiber = df[df["InternetService"]=="Fiber optic"]["Churn_binary"].mean()
h3_dsl   = df[df["InternetService"]=="DSL"]["Churn_binary"].mean()
ct3 = pd.crosstab(df["InternetService"], df["Churn_binary"])
chi3, p3, _, _ = stats.chi2_contingency(ct3)
print(f"\n  H3: Fiber optic customers churn more than DSL")
print(f"      Fiber optic churn: {h3_fiber*100:.1f}%")
print(f"      DSL churn:         {h3_dsl*100:.1f}%")
print(f"      Chi2={chi3:.1f}, p={p3:.2e} → {'CONFIRMED' if p3 < 0.05 else 'NOT confirmed'}")

# H4: High monthly charges customers churn more
high_charge = df[df["HighCharger"]==1]["Churn_binary"].mean()
low_charge  = df[df["HighCharger"]==0]["Churn_binary"].mean()
ct4 = pd.crosstab(df["HighCharger"], df["Churn_binary"])
chi4, p4, _, _ = stats.chi2_contingency(ct4)
print(f"\n  H4: High-charge customers churn more")
print(f"      High charge churn: {high_charge*100:.1f}%")
print(f"      Low charge churn:  {low_charge*100:.1f}%")
print(f"      Chi2={chi4:.1f}, p={p4:.2e} → {'CONFIRMED' if p4 < 0.05 else 'NOT confirmed'}")

# H5: New customers (0-6 months) are highest risk
new_churn  = df[df["NewCustomer"]==1]["Churn_binary"].mean()
old_churn  = df[df["NewCustomer"]==0]["Churn_binary"].mean()
print(f"\n  H5: New customers (0-6m) are highest churn risk")
print(f"      New customer churn:  {new_churn*100:.1f}%")
print(f"      Older customer churn:{old_churn*100:.1f}%")
print(f"      Difference:          {(new_churn-old_churn)*100:+.1f}pp")

# Business impact
total_monthly = df["MonthlyCharges"].sum()
churned_monthly = churned["MonthlyCharges"].sum()
high_risk = df[df["RiskScore"] >= 6]
high_risk_mrr = high_risk["MonthlyCharges"].sum()

print(f"\n" + "=" * 60)
print(f"  Business Impact Summary")
print(f"=" * 60)
print(f"\n  Total MRR:              ${total_monthly:,.0f}/month")
print(f"  MRR from churned users: ${churned_monthly:,.0f}/month ({churned_monthly/total_monthly*100:.1f}%)")
print(f"  High-risk customers:    {len(high_risk):,} (RiskScore >= 6)")
print(f"  High-risk MRR:          ${high_risk_mrr:,.0f}/month")
print(f"\n  6 EDA charts saved to charts/")
print(f"\n  Next: python phase3_modeling.py")
print("=" * 60)