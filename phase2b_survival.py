"""
PRISM v2 — Phase 2b: Survival Analysis
Kaplan-Meier curves + Cox Proportional Hazards
Answers: WHEN will a customer churn, not just IF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("charts", exist_ok=True)

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
except ImportError:
    raise ImportError("Run: pip install lifelines")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})
COLORS = ["#3B8BD4", "#D85A30", "#1D9E75", "#EF9F27", "#7F77DD", "#D4537E"]

print("=" * 60)
print("  PRISM v2 — Phase 2b: Survival Analysis")
print("=" * 60)

df = pd.read_csv("data/survival_ready.csv")
print(f"\n  Loaded: {len(df):,} customers")
print(f"  Event (churn): {df['Churn_binary'].sum():,} ({df['Churn_binary'].mean()*100:.1f}%)")
print(f"  Censored (retained): {(1-df['Churn_binary']).sum():,}")


# ─────────────────────────────────────────────────────────────
# CHART 1: Kaplan-Meier by Contract Type
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("PRISM — Survival Analysis: When Do Customers Churn?",
             fontsize=14, fontweight="bold")

kmf = KaplanMeierFitter()
contract_types = df["Contract"].unique()

for i, (contract, color) in enumerate(zip(contract_types, COLORS)):
    mask = df["Contract"] == contract
    kmf.fit(df[mask]["tenure"], event_observed=df[mask]["Churn_binary"],
            label=contract)
    kmf.plot_survival_function(ax=axes[0], color=color, ci_show=True, ci_alpha=0.1)

axes[0].set_title("Survival Probability by Contract Type", fontweight="bold")
axes[0].set_xlabel("Tenure (months)")
axes[0].set_ylabel("Survival Probability (staying)")
axes[0].set_ylim(0, 1.05)

# Log-rank test between month-to-month and two-year
mtm  = df[df["Contract"] == "Month-to-month"]
tyr  = df[df["Contract"] == "Two year"]
if len(mtm) > 0 and len(tyr) > 0:
    lr = logrank_test(mtm["tenure"], tyr["tenure"],
                      mtm["Churn_binary"], tyr["Churn_binary"])
    axes[0].text(0.05, 0.1,
                 f"Log-rank p={lr.p_value:.2e}\n(Month-to-month vs Two year)",
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# KM by internet service
for svc, color in zip(df["InternetService"].unique(), COLORS):
    mask = df["InternetService"] == svc
    kmf.fit(df[mask]["tenure"], event_observed=df[mask]["Churn_binary"], label=svc)
    kmf.plot_survival_function(ax=axes[1], color=color, ci_show=True, ci_alpha=0.1)

axes[1].set_title("Survival Probability by Internet Service", fontweight="bold")
axes[1].set_xlabel("Tenure (months)")
axes[1].set_ylabel("Survival Probability")
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("charts/07_kaplan_meier.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Chart 7 saved: Kaplan-Meier survival curves")


# ─────────────────────────────────────────────────────────────
# CHART 2: Cox Proportional Hazards — hazard ratios
# ─────────────────────────────────────────────────────────────
print("\n  Fitting Cox Proportional Hazards model...")

cox_features = ["tenure", "MonthlyCharges", "is_month_to_month",
                "is_new_customer", "is_high_charger", "service_adoption_score",
                "has_fiber", "fiber_no_security", "charge_per_service",
                "Churn_binary"]

cox_df = df[cox_features].dropna().copy()

cph = CoxPHFitter(penalizer=0.1)
cph.fit(cox_df, duration_col="tenure", event_col="Churn_binary")

fig, ax = plt.subplots(figsize=(10, 7))
cph.plot(ax=ax)
ax.set_title("PRISM — Cox PH: Hazard Ratios\n(HR > 1 = increases churn risk)",
             fontsize=13, fontweight="bold")
ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("log(Hazard Ratio) — 95% CI")
plt.tight_layout()
plt.savefig("charts/08_cox_hazard_ratios.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 8 saved: Cox hazard ratios")

# Print Cox summary
print("\n  Cox PH Model Summary:")
print(f"  {'Feature':<30} {'HR':>8} {'p-value':>10}  {'Interpretation'}")
print("  " + "-" * 75)
summary = cph.summary[["exp(coef)", "p"]].copy()
summary.columns = ["hazard_ratio", "p_value"]
summary = summary.sort_values("hazard_ratio", ascending=False)

for feat, row in summary.iterrows():
    hr = row["hazard_ratio"]
    p  = row["p_value"]
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    interp = "↑ churn risk" if hr > 1.1 else "↓ churn risk" if hr < 0.9 else "neutral"
    print(f"  {str(feat):<30} {hr:>8.3f} {p:>10.4f}{sig:3}  {interp}")


# ─────────────────────────────────────────────────────────────
# CHART 3: Expected survival time per segment
# ─────────────────────────────────────────────────────────────
print("\n  Computing median survival time per segment...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PRISM — Median Survival Time by Segment",
             fontsize=14, fontweight="bold")

segments = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["Fiber optic", "DSL", "No"]
}

for ax, (col, groups) in zip(axes, segments.items()):
    medians = []
    labels  = []
    for group in groups:
        mask = df[col] == group
        if mask.sum() < 10:
            continue
        kmf.fit(df[mask]["tenure"], event_observed=df[mask]["Churn_binary"])
        med = kmf.median_survival_time_
        medians.append(med if med != np.inf else 72)
        labels.append(group)

    colors = [COLORS[1] if m < 24 else COLORS[2] for m in medians]
    bars = ax.bar(labels, medians, color=colors, alpha=0.85, width=0.5, edgecolor="white")
    for bar, m in zip(bars, medians):
        label = f"{m:.0f}m" if m < 72 else "72m+"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(f"By {col}", fontweight="bold")
    ax.set_ylabel("Median Survival Time (months)")
    ax.set_ylim(0, 80)

plt.tight_layout()
plt.savefig("charts/09_median_survival_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Chart 9 saved: Median survival time")


# ─────────────────────────────────────────────────────────────
# Add survival predictions back to dataset
# ─────────────────────────────────────────────────────────────
print("\n  Adding survival predictions to dataset...")

cox_input = df[cox_features].dropna().copy()
survival_probs = cph.predict_survival_function(cox_input.drop(columns=["Churn_binary"]))

# Expected remaining tenure = area under survival curve
expected_remaining = survival_probs.sum(axis=0).values
df_with_survival = df.copy()
df_with_survival["expected_months_remaining"] = np.nan
df_with_survival.loc[cox_input.index, "expected_months_remaining"] = expected_remaining

# Predicted 30-day churn probability
pred_30d = 1 - cph.predict_survival_function(
    cox_input.drop(columns=["Churn_binary"]), times=[30]
).values.flatten()
df_with_survival["churn_prob_30d"] = np.nan
df_with_survival.loc[cox_input.index, "churn_prob_30d"] = pred_30d

df_with_survival.to_csv("data/survival_predictions.csv", index=False)

print(f"  Saved: data/survival_predictions.csv")
print(f"\n  Sample predictions:")
sample = df_with_survival[["tenure", "Contract", "InternetService",
                            "MonthlyCharges", "Churn_binary",
                            "expected_months_remaining", "churn_prob_30d"]].head(8)
print(sample.to_string(index=False))


print("\n" + "=" * 60)
print("  Phase 2b Complete — Survival Analysis")
print("=" * 60)
print(f"\n  Model: Cox Proportional Hazards")
print(f"  Concordance index: {cph.concordance_index_:.4f}")
print(f"  (0.5 = random, 1.0 = perfect)")
print(f"\n  Charts saved: 07, 08, 09")
print(f"  Data saved: survival_predictions.csv")
print(f"\n  Next: python phase2c_clustering.py")
print("=" * 60)
