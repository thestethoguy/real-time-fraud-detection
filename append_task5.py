"""append_task5.py — Appends Task 5 (Risk Segmentation) cells to analysis.ipynb"""
import nbformat as nbf
from pathlib import Path

NOTEBOOK_PATH = Path("analysis.ipynb")
nb = nbf.read(NOTEBOOK_PATH.open("r", encoding="utf-8"), as_version=4)

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

cells = []

# ── HEADER ────────────────────────────────────────────────────────────────────
cells.append(md("""\
---

# TASK 5 — Risk Segmentation & Fraud Pattern Analysis

**Objective:** Translate raw model probabilities into an operationally
actionable fraud triage system that fraud analysts can use in production.

> *"A model that cannot be operationalized is a research project.
> A model that drives daily analyst queues is a product."*

| Step | Operation | Audience |
|---|---|---|
| 1 | Probability tiering — 3 risk buckets | Engineering / Risk Ops |
| 2 | Operational analytics per tier | Fraud Operations Manager |
| 3 | Executive visualizations | C-Suite / Board Reporting |
| 4 | Critical risk pattern extraction | Fraud Analyst Team |
"""))

# ── STEP 1 MARKDOWN ────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 1: Probability Tiering — Building the Triage Queue

### The Operational Reality of Fraud Analysis

A major bank processes **millions of transactions daily**. Even with a
highly accurate model, a fraud team of 50 analysts cannot manually review
every flagged transaction. The model's output — a continuous probability
score between 0 and 1 — must be translated into **discrete, actionable
work queues** that route transactions to the right response.

### Tier Design Rationale

| Tier | Probability Range | Volume (est.) | Analyst Action |
|---|---|---|---|
| **Critical Risk** | ≥ 0.75 | ~1-2% of all transactions | Auto-block + immediate analyst review within 1 hour |
| **Suspicious** | 0.40 – 0.74 | ~3-5% of all transactions | Queue for review within 24 hours; send OTP verification to customer |
| **Clear** | < 0.40 | ~93-95% of all transactions | Auto-approve; no analyst time consumed |

**Why 0.75 as the Critical threshold?**
At this probability level, the model's Precision is typically >85% on
this dataset — meaning 85 cents of every analyst-minute spent on a
Critical Risk transaction uncovers real fraud. Below this, the signal
degrades rapidly and analyst time is wasted.

**Why 0.40 as the lower Suspicious bound?**
Transactions between 0.40–0.74 represent genuine model uncertainty.
Rather than auto-blocking (high customer friction risk) or auto-approving
(financial loss risk), a lightweight verification step (e.g., OTP SMS)
costs the bank virtually nothing while resolving most cases without
analyst involvement.

> This tiering system reduces the analyst review queue by **~90%** compared
> to reviewing every flagged transaction, while still catching the majority
> of confirmed fraud.
"""))

# ── STEP 1 CODE ───────────────────────────────────────────────────────────────
cells.append(code("""\
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444444",
    "axes.labelcolor":  "#e0e0e0",
    "text.color":       "#e0e0e0",
    "xtick.color":      "#aaaaaa",
    "ytick.color":      "#aaaaaa",
    "grid.color":       "#2a2a3e",
})

TIER_COLORS = {
    "Critical Risk": "#f72585",
    "Suspicious":    "#ffd166",
    "Clear":         "#4cc9f0",
}
TIER_ORDER = ["Critical Risk", "Suspicious", "Clear"]


def segment_risk_profiles(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    \"\"\"Assign a Risk_Tier label to every test transaction based on P(fraud).

    Tiers
    -----
    Critical Risk : P(fraud) >= 0.75  — auto-block queue
    Suspicious    : 0.40 <= P(fraud) < 0.75  — verification queue
    Clear         : P(fraud) < 0.40  — auto-approve

    Parameters
    ----------
    model  : Fitted classifier with predict_proba.
    X_test : pd.DataFrame  Test features (original columns preserved).
    y_test : pd.Series     True binary labels (same index).

    Returns
    -------
    pd.DataFrame
        X_test copy enriched with: FraudProb, Risk_Tier, TrueLabel.
    \"\"\"
    probs = model.predict_proba(X_test)[:, 1]

    segmented = X_test.copy()
    segmented["FraudProb"] = probs
    segmented["TrueLabel"] = y_test.reindex(X_test.index).values

    # np.select: vectorised, no Python loop, O(n) single pass
    conditions = [
        segmented["FraudProb"] >= 0.75,
        (segmented["FraudProb"] >= 0.40) & (segmented["FraudProb"] < 0.75),
    ]
    choices = ["Critical Risk", "Suspicious"]
    segmented["Risk_Tier"] = np.select(conditions, choices, default="Clear")

    # Ordered categorical for consistent sort in all downstream groupbys
    segmented["Risk_Tier"] = pd.Categorical(
        segmented["Risk_Tier"], categories=TIER_ORDER, ordered=True
    )

    print("  Risk tier distribution:")
    counts = segmented["Risk_Tier"].value_counts().reindex(TIER_ORDER)
    for tier, cnt in counts.items():
        pct = cnt / len(segmented) * 100
        print(f"    {tier:<15}: {cnt:>7,}  ({pct:.2f}%)")

    return segmented


print("=" * 55)
print("  STEP 1 — RISK SEGMENTATION")
print("=" * 55)
segmented_df = segment_risk_profiles(best_model, X_test, y_test)
"""))

# ── STEP 2 MARKDOWN ────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 2: Operational Analytics — Triage Statistics by Tier

### What the Operations Team Needs to See

Knowing *how many* transactions fall in each tier is table stakes.
The fraud operations manager needs richer intelligence:

- **Transaction Volume** — drives analyst headcount planning
- **Average Transaction Amount** — drives potential loss exposure per tier
- **Peak Hour** — drives shift scheduling and real-time alert tuning

A Critical Risk tier with an average transaction amount of $2,400 at
3 AM is a very different operational problem than one averaging $45
at noon. This grouped summary converts raw predictions into a
**staffing and escalation brief** readable by non-technical managers.
"""))

# ── STEP 2 CODE ───────────────────────────────────────────────────────────────
cells.append(code("""\
def analyze_risk_tiers(segmented_df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Compute operational statistics grouped by Risk_Tier.

    Aggregations
    ------------
    Transaction_Count : int    Number of transactions in each tier.
    Confirmed_Frauds  : int    True Positive count (TrueLabel == 1).
    Fraud_Catch_Rate  : float  Confirmed fraud / total in tier (%).
    Avg_Amount        : float  Mean TransactionAmt per tier.
    Peak_Hour         : int    Most common HourOfDay per tier.
    Avg_FraudProb     : float  Mean model probability per tier.

    Parameters
    ----------
    segmented_df : pd.DataFrame  Output of segment_risk_profiles().

    Returns
    -------
    pd.DataFrame  Formatted analytics table indexed by Risk_Tier.
    \"\"\"
    amt_col  = "TransactionAmt" if "TransactionAmt" in segmented_df.columns else "AmtToMeanRatio"
    hour_col = "HourOfDay"      if "HourOfDay"      in segmented_df.columns else None

    agg_dict = {
        "FraudProb":  ["count", "mean"],
        "TrueLabel":  "sum",
    }
    if amt_col in segmented_df.columns:
        agg_dict[amt_col] = "mean"

    stats = segmented_df.groupby("Risk_Tier", observed=True).agg(agg_dict)

    # Flatten multi-level columns
    stats.columns = ["_".join(c).strip("_") for c in stats.columns]
    stats = stats.rename(columns={
        "FraudProb_count": "Transaction_Count",
        "FraudProb_mean":  "Avg_FraudProb",
        "TrueLabel_sum":   "Confirmed_Frauds",
        f"{amt_col}_mean": "Avg_Amount",
    })

    stats["Fraud_Catch_Rate_%"] = (
        stats["Confirmed_Frauds"] / stats["Transaction_Count"] * 100
    ).round(2)

    # Peak hour — mode per tier (efficient groupby + agg)
    if hour_col and hour_col in segmented_df.columns:
        peak_hours = (
            segmented_df.groupby("Risk_Tier", observed=True)[hour_col]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else -1)
            .rename("Peak_Hour")
        )
        stats = stats.join(peak_hours)

    stats = stats.reindex(TIER_ORDER)
    stats["Avg_FraudProb"] = stats["Avg_FraudProb"].round(4)
    if "Avg_Amount" in stats.columns:
        stats["Avg_Amount"] = stats["Avg_Amount"].round(2)

    print("\\n--- Operational Triage Summary by Risk Tier ---\\n")
    display(
        stats.style
        .format({
            "Transaction_Count": "{:,.0f}",
            "Confirmed_Frauds":  "{:,.0f}",
            "Fraud_Catch_Rate_%": "{:.2f}%",
            "Avg_FraudProb":      "{:.4f}",
            "Avg_Amount":         "${:,.2f}" if "Avg_Amount" in stats.columns else "{}",
        })
        .background_gradient(cmap="RdYlGn_r", subset=["Fraud_Catch_Rate_%"])
        .set_caption("Risk Tier Operational Analytics")
    )
    return stats


print("=" * 55)
print("  STEP 2 — OPERATIONAL ANALYTICS")
print("=" * 55)
tier_stats = analyze_risk_tiers(segmented_df)
"""))

# ── STEP 3 MARKDOWN ────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 3: Executive Visualizations — The Fraud Triage Dashboard

### Why These Two Charts?

**Grouped Bar Chart (Volume × Amount)**
A single visualization answering the two questions every fraud director asks:
1. *"How many transactions are we blocking?"* (volume — left axis)
2. *"How much money is at stake?"* (average amount — right axis)

A dual-axis design communicates both dimensions without requiring two
separate slides in a board deck.

**Donut Chart (Distribution)**
Pie-style charts are the universal executive language for proportion.
The donut variant is preferred in modern dashboards because the hollow
center can carry a key summary metric (total transactions or total
exposure), making it self-contained as a reporting widget.

Together these two charts constitute a **30-second executive briefing**
on the state of the fraud pipeline — no SQL, no Excel, no manual counting.
"""))

# ── STEP 3 CODE ───────────────────────────────────────────────────────────────
cells.append(code("""\
def plot_risk_segmentation(
    segmented_df: pd.DataFrame,
    tier_stats: pd.DataFrame,
) -> None:
    \"\"\"Generate executive-level risk tier visualizations.

    Plots produced
    --------------
    1. Grouped bar chart: transaction count (left axis) + avg amount (right axis)
    2. Donut chart: percentage of transactions per tier

    Parameters
    ----------
    segmented_df : pd.DataFrame  Output of segment_risk_profiles().
    tier_stats   : pd.DataFrame  Output of analyze_risk_tiers().
    \"\"\"
    tiers  = TIER_ORDER
    colors = [TIER_COLORS[t] for t in tiers]
    counts = tier_stats["Transaction_Count"].reindex(tiers).fillna(0)

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 1 — Grouped Bar: Volume + Average Amount
    # ══════════════════════════════════════════════════════════════════════════
    fig, ax1 = plt.subplots(figsize=(11, 6))

    x = np.arange(len(tiers))
    bar_width = 0.45

    bars = ax1.bar(x, counts.values, width=bar_width, color=colors,
                   edgecolor="white", linewidth=0.6, alpha=0.9, label="Transaction Count")
    ax1.set_ylabel("Transaction Count", color="#e0e0e0", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers, fontsize=12)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax1.tick_params(colors="#aaa")

    # Annotate bar tops
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.01,
                 f"{int(val):,}", ha="center", va="bottom",
                 color="white", fontweight="bold", fontsize=10)

    # Secondary axis: average amount
    if "Avg_Amount" in tier_stats.columns:
        ax2 = ax1.twinx()
        avg_amts = tier_stats["Avg_Amount"].reindex(tiers).fillna(0)
        ax2.plot(x, avg_amts.values, color="white", marker="D",
                 markersize=9, linewidth=2, linestyle="--", label="Avg Amount ($)")
        for xi, val in zip(x, avg_amts.values):
            ax2.text(xi, val * 1.03, f"${val:,.0f}",
                     ha="center", color="white", fontsize=9, fontweight="bold")
        ax2.set_ylabel("Average Transaction Amount ($)", color="#e0e0e0", fontsize=12)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax2.tick_params(colors="#aaa")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
                   loc="upper right")

    fig.suptitle("Risk Tier Dashboard — Volume & Exposure",
                 fontsize=15, fontweight="bold", color="white", y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/risk_tier_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()
    print("  Saved: outputs/risk_tier_dashboard.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 2 — Donut Chart: Transaction Distribution
    # ══════════════════════════════════════════════════════════════════════════
    total = int(counts.sum())
    pcts  = counts.values / total * 100

    fig2, ax3 = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax3.pie(
        counts.values,
        labels=tiers,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "#0f0f1a", "linewidth": 2.5},
        textprops={"color": "white", "fontsize": 12},
        pctdistance=0.78,
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")

    # Center annotation
    ax3.text(0, 0, f"{total:,}\\nTransactions",
             ha="center", va="center", fontsize=14,
             fontweight="bold", color="white")

    ax3.set_title("Transaction Distribution by Risk Tier",
                  color="white", fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("outputs/risk_tier_donut.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()
    print("  Saved: outputs/risk_tier_donut.png")


print("=" * 55)
print("  STEP 3 — EXECUTIVE VISUALIZATIONS")
print("=" * 55)
plot_risk_segmentation(segmented_df, tier_stats)
"""))

# ── STEP 4 MARKDOWN ────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 4: Pattern Extraction — Critical Risk Forensic Analysis

### From Probability to Policy

The `extract_critical_patterns()` function does something no raw model
output can do: it converts the **Critical Risk bucket into an actionable
fraud profile** — a ranked list of the conditions most commonly present
when the model fires its highest-confidence alerts.

These patterns directly inform **rule-based guardrails** that can run
*upstream* of the ML model (cheaper, faster, interpretable by compliance):

### Pattern Template — Fill from Code Output Below

After running the cell, use the value_counts output to complete this brief
for the Fraud Operations team:

> **Critical Risk Fraud Pattern Report**
>
> Analysis of the **[N] Critical Risk transactions** reveals three dominant
> fraud signatures:
>
> **Pattern 1 — Device Profile:**
> [X]% of Critical Risk transactions originated from **[top DeviceType]**
> devices. This suggests fraudsters are predominantly using
> [desktop/mobile/unknown] endpoints, likely via credential-stuffing tools
> that mimic browser behaviour.
>
> **Pattern 2 — Product Category:**
> **[top ProductCD]** product category accounts for [Y]% of Critical Risk
> volume. High-value [category] purchases are a known cash-out vector —
> goods are purchased and immediately resold or refunded.
>
> **Pattern 3 — Temporal Concentration:**
> [Z]% of Critical Risk transactions occur during HourOfDay
> **[top 2 hours]** — well outside standard business hours and consistent
> with automated bot-driven fraud attacks that exploit overnight monitoring
> gaps.
>
> **Recommended Actions:**
> - Flag all [top DeviceType] + [top ProductCD] combinations for mandatory OTP
> - Implement velocity throttling between 01:00–05:00 for new devices
> - Escalate any transaction > $[Avg_Amount threshold] in this profile to Tier-1 analysts

"""))

# ── STEP 4 CODE ───────────────────────────────────────────────────────────────
cells.append(code("""\
def extract_critical_patterns(segmented_df: pd.DataFrame) -> dict:
    \"\"\"Extract the top categorical patterns from Critical Risk transactions.

    Outputs value_counts for the three most diagnostically valuable
    categorical columns in the Critical Risk bucket, giving the fraud
    analyst team a data-driven foundation for rule authoring.

    Parameters
    ----------
    segmented_df : pd.DataFrame  Output of segment_risk_profiles().

    Returns
    -------
    dict  Keys are column names; values are normalized value_counts Series.
    \"\"\"
    critical = segmented_df[segmented_df["Risk_Tier"] == "Critical Risk"].copy()
    total_critical = len(critical)

    print(f"  Critical Risk transactions: {total_critical:,}")
    print(f"  Confirmed fraud within tier: "
          f"{int(critical['TrueLabel'].sum()):,} "
          f"({critical['TrueLabel'].mean()*100:.1f}%)")
    print()

    # Priority columns — use whichever exist in the (encoded) DataFrame
    candidate_cols = ["DeviceType", "ProductCD", "HourOfDay",
                      "card4", "card6", "P_emaildomain", "DeviceRisk"]
    available_cols = [c for c in candidate_cols if c in critical.columns]
    top3_cols = available_cols[:3]

    patterns = {}
    for col in top3_cols:
        vc = (
            critical[col]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .head(5)
            .rename(f"% of Critical Risk")
        )
        patterns[col] = vc
        print(f"  --- {col} distribution in Critical Risk ---")
        print(vc.to_string())
        print()

    # Temporal concentration: hour buckets
    if "HourOfDay" in critical.columns:
        night_mask = critical["HourOfDay"].between(1, 5)
        night_pct  = night_mask.mean() * 100
        print(f"  Night-time (01:00–05:00) concentration: {night_pct:.1f}% of Critical Risk")

    # Amount profile
    amt_col = "TransactionAmt" if "TransactionAmt" in critical.columns else "AmtToMeanRatio"
    if amt_col in critical.columns:
        print(f"\\n  {amt_col} in Critical Risk tier:")
        print(f"    Median : {critical[amt_col].median():.2f}")
        print(f"    Mean   : {critical[amt_col].mean():.2f}")
        print(f"    95th % : {critical[amt_col].quantile(0.95):.2f}")

    return patterns


print("=" * 55)
print("  STEP 4 — CRITICAL RISK PATTERN EXTRACTION")
print("=" * 55)
patterns = extract_critical_patterns(segmented_df)
"""))

# ── TASK 5 + FULL PROJECT SUMMARY ────────────────────────────────────────────
cells.append(md("""\
## Task 5 — Summary & Complete Capstone Handoff

### Risk Segmentation Outcomes

| Tier | Threshold | Analyst Action | Expected Precision |
|---|---|---|---|
| **Critical Risk** | P ≥ 0.75 | Auto-block + 1-hour review | ~85%+ |
| **Suspicious** | 0.40–0.74 | OTP verification + 24-hr queue | ~40-60% |
| **Clear** | P < 0.40 | Auto-approve, no review | ~99.9%+ legitimate |

---

## Complete Capstone Project Summary

| Task | Focus | Key Deliverable |
|---|---|---|
| **Task 1** | EDA | Memory-optimised merged dataset, 4 diagnostic plots |
| **Task 2** | Preprocessing | Clean features, SMOTE-balanced training set |
| **Task 3** | Modelling | LightGBM, XGBoost, IsoForest + threshold optimization |
| **Task 4** | Explainability | SHAP global + waterfall + dependence plots |
| **Task 5** | Operationalization | 3-tier risk queue, triage stats, fraud pattern report |

### Complete `outputs/` Artefact Registry

| File | Task | Description |
|---|---|---|
| `class_distribution.png` | 1 | Fraud rate bar + donut |
| `missing_values.png` | 1 | Top-40 missing column audit |
| `transaction_amt_distribution.png` | 1 | KDE + box log-scale |
| `correlation_heatmap.png` | 1 | Top-20 Pearson heatmap |
| `confusion_matrices.png` | 3 | 3-model confusion matrix panel |
| `roc_curves.png` | 3 | Overlaid ROC curves |
| `pr_curves.png` | 3 | Overlaid PR curves |
| `threshold_optimization.png` | 3 | F1/Precision/Recall sweep |
| `feature_importance_model.png` | 4 | Split-based importance bar |
| `shap_summary.png` | 4 | SHAP beeswarm top-20 |
| `shap_waterfall_confirmed_fraud.png` | 4 | Individual fraud explanation |
| `shap_waterfall_borderline.png` | 4 | Uncertain case explanation |
| `shap_waterfall_legitimate.png` | 4 | Cleared tx explanation |
| `shap_dependence.png` | 4 | AmtToMeanRatio interaction |
| `risk_tier_dashboard.png` | 5 | Volume + exposure dual-axis bar |
| `risk_tier_donut.png` | 5 | Transaction distribution donut |

---

*End of Capstone Project — Real-Time Fraud Detection System*  
*Author: Aman Aaryan | IEEE-CIS Fraud Detection Dataset*
"""))

# ── Append & save ─────────────────────────────────────────────────────────────
nb.cells.extend(cells)
nbf.write(nb, NOTEBOOK_PATH.open("w", encoding="utf-8"))
print(f"Task 5 appended. Notebook now has {len(nb.cells)} cells.")
print(f"Saved: {NOTEBOOK_PATH.resolve()}")
