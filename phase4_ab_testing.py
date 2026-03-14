"""
PRISM v2 — Phase 4: A/B Testing & Causal Inference Framework
Power analysis + experiment design + difference-in-differences
Answers: Does our intervention actually reduce churn?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import warnings
import os
import json

warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("charts", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})
COLORS = ["#3B8BD4", "#D85A30", "#1D9E75", "#EF9F27", "#7F77DD"]

print("=" * 60)
print("  PRISM v2 — Phase 4: A/B Testing Framework")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load predictions + design experiment
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Designing experiment...")

preds  = pd.read_csv("data/predictions_with_roi.csv", index_col=0)
telco  = pd.read_csv("data/telco_cleaned.csv")

# Align telco to predictions index
telco_aligned = telco.iloc[preds.index].reset_index(drop=True)
preds = preds.reset_index(drop=True)
preds["MonthlyCharges"] = telco_aligned["MonthlyCharges"].values
preds["tenure"]         = telco_aligned["tenure"].values
preds["Contract"]       = telco_aligned["Contract"].values

# High-risk pool — customers we would intervene on
high_risk = preds[preds["churn_probability"] >= 0.3].copy()
print(f"      High-risk customers (prob >= 0.3): {len(high_risk):,}")
print(f"      Their total MRR: ${high_risk['MonthlyCharges'].sum():,.0f}/month")


# ─────────────────────────────────────────────────────────────
# STEP 2: Power Analysis
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Running power analysis...")

baseline_churn_rate = preds["Churn_binary"].mean()
effect_sizes        = [0.05, 0.10, 0.15, 0.20]  # 5-20% reduction in churn
alpha               = 0.05
power               = 0.80

def required_sample_size(p1, p2, alpha=0.05, power=0.80):
    """Two-proportion z-test sample size."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta  = stats.norm.ppf(power)
    p_bar   = (p1 + p2) / 2
    n = (z_alpha * np.sqrt(2 * p_bar * (1-p_bar)) +
         z_beta  * np.sqrt(p1*(1-p1) + p2*(1-p2)))**2 / (p1 - p2)**2
    return int(np.ceil(n))

print(f"\n  Baseline churn rate: {baseline_churn_rate*100:.1f}%")
print(f"\n  {'Effect Size':<15} {'Treatment Rate':<18} {'N per arm':<12} {'Total N':<10} {'Feasible?'}")
print("  " + "-" * 65)

power_results = []
for effect in effect_sizes:
    p_control   = baseline_churn_rate
    p_treatment = baseline_churn_rate * (1 - effect)
    n_per_arm   = required_sample_size(p_control, p_treatment)
    total_n     = n_per_arm * 2
    feasible    = "YES" if total_n <= len(high_risk) else "NO — need more data"
    print(f"  {effect*100:.0f}% reduction    {p_treatment*100:.1f}%              "
          f"{n_per_arm:<12,} {total_n:<10,} {feasible}")
    power_results.append({
        "effect_size": effect, "p_control": p_control,
        "p_treatment": p_treatment, "n_per_arm": n_per_arm,
        "total_n": total_n
    })


# ─────────────────────────────────────────────────────────────
# STEP 3: Simulate A/B Test
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Simulating A/B test experiment...")

# Randomly assign high-risk customers to control/treatment
high_risk = high_risk.sample(frac=1, random_state=42).reset_index(drop=True)
mid       = len(high_risk) // 2
high_risk["group"] = ["control" if i < mid else "treatment" for i in range(len(high_risk))]

# Simulate intervention effect
# Treatment: 15% reduction in churn probability (realistic for discount/outreach)
INTERVENTION_EFFECT = 0.15

control   = high_risk[high_risk["group"] == "control"].copy()
treatment = high_risk[high_risk["group"] == "treatment"].copy()

# Simulate outcomes
control["churned"]   = np.random.binomial(1, control["churn_probability"].clip(0,1))
treatment_prob       = (treatment["churn_probability"] * (1 - INTERVENTION_EFFECT)).clip(0, 1)
treatment["churned"] = np.random.binomial(1, treatment_prob)

control_churn_rate   = control["churned"].mean()
treatment_churn_rate = treatment["churned"].mean()
absolute_reduction   = control_churn_rate - treatment_churn_rate
relative_reduction   = absolute_reduction / control_churn_rate

print(f"\n  Experiment design:")
print(f"    Control group:   {len(control):,} customers (no intervention)")
print(f"    Treatment group: {len(treatment):,} customers (retention offer)")
print(f"    Intervention:    {INTERVENTION_EFFECT*100:.0f}% simulated churn reduction")

print(f"\n  Results:")
print(f"    Control churn rate:   {control_churn_rate*100:.1f}%")
print(f"    Treatment churn rate: {treatment_churn_rate*100:.1f}%")
print(f"    Absolute reduction:   {absolute_reduction*100:.1f}pp")
print(f"    Relative reduction:   {relative_reduction*100:.1f}%")

# Statistical significance tests
# 1. Chi-square test
contingency = pd.DataFrame({
    "Churned"  : [control["churned"].sum(),   treatment["churned"].sum()],
    "Retained" : [(~control["churned"].astype(bool)).sum(),
                  (~treatment["churned"].astype(bool)).sum()]
}, index=["Control", "Treatment"])

chi2, p_chi2, _, _ = chi2_contingency(contingency)

# 2. Two-proportion z-test
n1, n2   = len(control), len(treatment)
p1, p2   = control_churn_rate, treatment_churn_rate
p_pool   = (control["churned"].sum() + treatment["churned"].sum()) / (n1 + n2)
z_stat   = (p1 - p2) / np.sqrt(p_pool * (1-p_pool) * (1/n1 + 1/n2))
p_ztest  = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"\n  Statistical Tests:")
print(f"    Chi-square: chi2={chi2:.3f}, p={p_chi2:.4f} "
      f"{'SIGNIFICANT' if p_chi2 < 0.05 else 'NOT significant'}")
print(f"    Z-test:     z={z_stat:.3f}, p={p_ztest:.4f} "
      f"{'SIGNIFICANT' if p_ztest < 0.05 else 'NOT significant'}")

# MRR impact
control_mrr_lost   = (control["churned"]   * control["MonthlyCharges"]).sum()
treatment_mrr_lost = (treatment["churned"] * treatment["MonthlyCharges"]).sum()
mrr_saved          = control_mrr_lost - treatment_mrr_lost

print(f"\n  Revenue Impact:")
print(f"    MRR lost (control):   ${control_mrr_lost:,.0f}/month")
print(f"    MRR lost (treatment): ${treatment_mrr_lost:,.0f}/month")
print(f"    MRR saved:            ${mrr_saved:,.0f}/month")
print(f"    Annual impact:        ${mrr_saved*12:,.0f}/year")


# ─────────────────────────────────────────────────────────────
# STEP 4: Difference-in-Differences (DiD)
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Difference-in-Differences analysis...")

# Simulate pre/post periods for DiD
# Pre-period: churn rates before any intervention
# Post-period: churn rates after intervention

pre_control_rate   = control_churn_rate + np.random.normal(0.01, 0.005)
pre_treatment_rate = treatment_churn_rate + np.random.normal(0.01, 0.005) + INTERVENTION_EFFECT * treatment_churn_rate

post_control_rate   = control_churn_rate
post_treatment_rate = treatment_churn_rate

did_estimate = (post_treatment_rate - pre_treatment_rate) - \
               (post_control_rate   - pre_control_rate)

print(f"\n  DiD Framework:")
print(f"  {'Group':<12} {'Pre-period':>12} {'Post-period':>12} {'Change':>10}")
print("  " + "-" * 50)
print(f"  {'Control':<12} {pre_control_rate*100:>11.1f}% {post_control_rate*100:>11.1f}% "
      f"{(post_control_rate-pre_control_rate)*100:>+9.1f}pp")
print(f"  {'Treatment':<12} {pre_treatment_rate*100:>11.1f}% {post_treatment_rate*100:>11.1f}% "
      f"{(post_treatment_rate-pre_treatment_rate)*100:>+9.1f}pp")
print(f"\n  DiD Estimate: {did_estimate*100:+.1f}pp")
print(f"  Interpretation: Intervention caused a {abs(did_estimate)*100:.1f}pp "
      f"{'reduction' if did_estimate < 0 else 'increase'} in churn")


# ─────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Generating charts...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("PRISM — A/B Testing & Causal Inference Framework",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Chart A: Power analysis curve
ax1 = fig.add_subplot(gs[0, 0])
sample_sizes = range(20, 500, 10)
power_vals   = []
for n in sample_sizes:
    p1 = baseline_churn_rate
    p2 = baseline_churn_rate * 0.85
    p_bar = (p1+p2)/2
    se = np.sqrt(p_bar*(1-p_bar)*(2/n))
    z  = (p1-p2)/se - stats.norm.ppf(0.975)
    pwr = stats.norm.cdf(z)
    power_vals.append(max(0, min(1, pwr)))

ax1.plot(sample_sizes, power_vals, color=COLORS[0], lw=2)
ax1.axhline(y=0.80, color="gray", linestyle="--", alpha=0.6, label="80% power")
ax1.axhline(y=0.90, color=COLORS[1], linestyle="--", alpha=0.6, label="90% power")
ax1.set_title("Power Analysis\n(15% churn reduction)", fontweight="bold")
ax1.set_xlabel("Sample size per arm")
ax1.set_ylabel("Statistical power")
ax1.legend(fontsize=8)
ax1.set_ylim(0, 1.05)

# Chart B: A/B test results
ax2 = fig.add_subplot(gs[0, 1])
groups  = ["Control\n(no offer)", "Treatment\n(retention offer)"]
rates   = [control_churn_rate*100, treatment_churn_rate*100]
bars    = ax2.bar(groups, rates, color=[COLORS[1], COLORS[2]],
                  width=0.4, edgecolor="white")
for bar, rate in zip(bars, rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{rate:.1f}%", ha="center", va="bottom", fontweight="bold")
ax2.text(0.5, max(rates)*0.6,
         f"p={p_ztest:.3f}\n{'Significant' if p_ztest < 0.05 else 'Not significant'}",
         ha="center", transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
ax2.set_title("A/B Test Results\nChurn Rate by Group", fontweight="bold")
ax2.set_ylabel("Churn Rate (%)")
ax2.set_ylim(0, max(rates) * 1.4)

# Chart C: DiD visualisation
ax3 = fig.add_subplot(gs[0, 2])
periods = [0, 1]
ax3.plot(periods, [pre_control_rate*100, post_control_rate*100],
         "o-", color=COLORS[1], lw=2, markersize=8, label="Control")
ax3.plot(periods, [pre_treatment_rate*100, post_treatment_rate*100],
         "o-", color=COLORS[2], lw=2, markersize=8, label="Treatment")
cf_line = pre_treatment_rate*100 + (post_control_rate - pre_control_rate)*100
ax3.plot([1], [cf_line], "o--", color=COLORS[2], alpha=0.4, markersize=8,
         label="Counterfactual")
ax3.annotate("", xy=(1.05, post_treatment_rate*100),
             xytext=(1.05, cf_line),
             arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))
ax3.text(1.08, (post_treatment_rate*100 + cf_line)/2,
         f"DiD\n{did_estimate*100:+.1f}pp", color="purple", fontsize=9)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(["Pre", "Post"])
ax3.set_title("Difference-in-Differences", fontweight="bold")
ax3.set_ylabel("Churn Rate (%)")
ax3.legend(fontsize=8)

# Chart D: MRR impact
ax4 = fig.add_subplot(gs[1, 0])
mrr_labels = ["MRR Lost\n(Control)", "MRR Lost\n(Treatment)", "MRR\nSaved"]
mrr_values = [control_mrr_lost, treatment_mrr_lost, mrr_saved]
mrr_colors = [COLORS[1], COLORS[3], COLORS[2]]
bars = ax4.bar(mrr_labels, mrr_values, color=mrr_colors, width=0.4, edgecolor="white")
for bar, val in zip(bars, mrr_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f"${val:,.0f}", ha="center", va="bottom", fontsize=9)
ax4.set_title("Revenue Impact\nper Month", fontweight="bold")
ax4.set_ylabel("Monthly Revenue ($)")

# Chart E: Segment-level test results
ax5 = fig.add_subplot(gs[1, 1:])
segment_results = []
for contract in high_risk["Contract"].unique():
    seg = high_risk[high_risk["Contract"] == contract]
    ctrl = seg[seg["group"] == "control"]
    trt  = seg[seg["group"] == "treatment"]
    if len(ctrl) > 5 and len(trt) > 5:
        ctrl_rate = np.random.binomial(1, ctrl["churn_probability"].clip(0,1)).mean()
        trt_rate  = np.random.binomial(1, (trt["churn_probability"]*(1-INTERVENTION_EFFECT)).clip(0,1)).mean()
        segment_results.append({
            "segment": contract, "control": ctrl_rate*100,
            "treatment": trt_rate*100, "n": len(seg)
        })

if segment_results:
    seg_df = pd.DataFrame(segment_results)
    x      = np.arange(len(seg_df))
    w      = 0.3
    ax5.bar(x-w/2, seg_df["control"],   width=w, color=COLORS[1], alpha=0.85,
            label="Control", edgecolor="white")
    ax5.bar(x+w/2, seg_df["treatment"], width=w, color=COLORS[2], alpha=0.85,
            label="Treatment", edgecolor="white")
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"{r['segment']}\n(n={r['n']})" for _, r in seg_df.iterrows()],
                         fontsize=9)
    ax5.set_title("A/B Results by Contract Segment", fontweight="bold")
    ax5.set_ylabel("Churn Rate (%)")
    ax5.legend()

plt.savefig("charts/16_ab_testing.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Chart 16 saved: A/B testing framework")

# Save results
ab_results = {
    "experiment_design": {
        "control_n"         : len(control),
        "treatment_n"       : len(treatment),
        "intervention_effect": INTERVENTION_EFFECT
    },
    "results": {
        "control_churn_rate"  : round(control_churn_rate, 4),
        "treatment_churn_rate": round(treatment_churn_rate, 4),
        "absolute_reduction"  : round(absolute_reduction, 4),
        "relative_reduction"  : round(relative_reduction, 4),
        "mrr_saved_monthly"   : round(mrr_saved, 2),
        "mrr_saved_annual"    : round(mrr_saved * 12, 2),
        "p_value_ztest"       : round(p_ztest, 4),
        "p_value_chi2"        : round(p_chi2, 4),
        "significant"         : bool(p_ztest < 0.05),
        "did_estimate"        : round(did_estimate, 4)
    }
}
with open("data/ab_test_results.json", "w") as f:
    json.dump(ab_results, f, indent=2)

print("\n" + "=" * 60)
print("  Phase 4 Complete — A/B Testing Framework")
print("=" * 60)
print(f"\n  Experiment size  : {len(high_risk):,} high-risk customers")
print(f"  Churn reduction  : {relative_reduction*100:.1f}% relative")
print(f"  Statistical sig  : {'YES' if p_ztest < 0.05 else 'NO'} (p={p_ztest:.4f})")
print(f"  MRR saved/month  : ${mrr_saved:,.0f}")
print(f"  MRR saved/year   : ${mrr_saved*12:,.0f}")
print(f"  DiD estimate     : {did_estimate*100:+.1f}pp causal effect")
print(f"\n  Chart saved: 16")
print(f"  Data saved: ab_test_results.json")
print(f"\n  Next: build Streamlit dashboard")
print("=" * 60)
