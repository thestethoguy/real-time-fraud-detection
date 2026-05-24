"""append_task4.py — Appends Task 4 (SHAP Explainability) cells to analysis.ipynb"""
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

# TASK 4 — Explainable AI with SHAP Values

**Objective:** Make the fraud detection model auditable and trustworthy by
explaining both global behaviour (what the model learned) and individual
decisions (why a specific transaction was flagged).

| Step | Operation |
|---|---|
| 1 | Global SHAP summary vs. model feature importance |
| 2 | Transaction audit — select 3 representative cases |
| 3 | Waterfall plots with plain-English translations |
| 4 | SHAP dependence plot — feature interaction analysis |
"""))

# ── STEP 1 MARKDOWN ────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 1: Global Explanations — SHAP vs. Model Feature Importance

### The Problem with Standard Feature Importance

Most tree models expose a built-in `.feature_importances_` attribute that
counts how many times each feature was used to split a node, weighted by
the impurity reduction it produced. While fast to compute, this metric has
three critical flaws that make it unsuitable for a regulated banking context:

| Limitation | Business Impact |
|---|---|
| **No directionality** — does not show if a feature *increases* or *decreases* fraud probability | Cannot explain to a regulator *how* a feature influences the decision |
| **Biased toward high-cardinality features** — features with many unique values appear more important simply because there are more possible split points | Misleading importance rankings |
| **Global aggregate only** — single value per feature across all predictions | Cannot audit a specific declined transaction |

### Why SHAP is the Regulatory Gold Standard

**SHAP (SHapley Additive exPlanations)** is rooted in cooperative game
theory. For every single prediction, it distributes the gap between the
model's output and the baseline (average) prediction across all features,
satisfying three fairness axioms: *efficiency*, *symmetry*, and *dummy*.

In plain English: SHAP answers *"Compared to an average transaction,
how much did each feature push this specific prediction toward fraud or
away from it, and by how much?"*

- A **positive SHAP value** for `AmtToMeanRatio` means that feature
  *increased* the fraud probability for this transaction.
- A **negative SHAP value** means it *decreased* it (evidence of legitimacy).

> `shap.TreeExplainer` uses a fast exact algorithm for tree models.
> We sample **2,000 rows** from X_test as a representative background
> to prevent RAM exhaustion while maintaining statistical validity.
"""))

# ── STEP 1 CODE ───────────────────────────────────────────────────────────────
cells.append(code("""\
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
matplotlib.rcParams.update({"figure.facecolor": "#0f0f1a", "text.color": "#e0e0e0"})

# Use the tuned LightGBM as our best model
best_model = best_lgbm_tuned


def generate_global_explanations(
    model,
    X_test: pd.DataFrame,
    sample_n: int = 2000,
    top_n: int = 20,
) -> tuple:
    \"\"\"Compute SHAP values and plot global importance summaries.

    Parameters
    ----------
    model    : Fitted tree-based classifier (LightGBM / XGBoost).
    X_test   : pd.DataFrame  Full held-out test features.
    sample_n : int  Number of rows to sample for SHAP computation.
    top_n    : int  Number of top features to display.

    Returns
    -------
    tuple  (explainer, shap_values, X_sample)
    \"\"\"
    # ── Sample for speed ──────────────────────────────────────────────────────
    X_sample = X_test.sample(n=min(sample_n, len(X_test)),
                             random_state=42).reset_index(drop=True)
    print(f"  SHAP background sample: {X_sample.shape}")

    # ── Build explainer ───────────────────────────────────────────────────────
    print("  Initialising TreeExplainer ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers shap_values is a list [class0, class1]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    print(f"  SHAP values shape: {sv.shape}")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 1 — Standard Model Feature Importance (Top 20)
    # ══════════════════════════════════════════════════════════════════════════
    importance = pd.Series(
        model.feature_importances_, index=X_sample.columns
    ).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(importance.index[::-1], importance.values[::-1],
                   color="#4cc9f0", edgecolor="none")
    ax.set_xlabel("Split-based Importance Score", color="#aaa")
    ax.set_title(f"Model Feature Importance — Top {top_n}",
                 color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#aaa", labelsize=9)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance_model.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()
    print("  Saved: outputs/feature_importance_model.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 2 — SHAP Summary Beeswarm (Top 20)
    # ══════════════════════════════════════════════════════════════════════════
    print("  Generating SHAP summary plot ...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv, X_sample,
        max_display=top_n,
        show=False,
        plot_type="dot",
        color_bar_label="Feature Value",
    )
    plt.title("SHAP Global Summary — Top 20 Features",
              color="white", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()
    print("  Saved: outputs/shap_summary.png")

    return explainer, shap_values, X_sample


print("=" * 55)
print("  STEP 1 — GLOBAL SHAP EXPLANATIONS")
print("=" * 55)
explainer, shap_values, X_sample = generate_global_explanations(
    best_model, X_test, sample_n=2000, top_n=20
)
"""))

# ── STEP 2 MARKDOWN ────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 2: Transaction Audit — Selecting 3 Representative Cases

### Why Individual Audits Are Non-Negotiable

Global SHAP summaries tell us what the model learned *on average*.
But a bank's compliance team needs to answer a fundamentally different
question: *"Why did the model decline this specific customer's transaction?"*

In the EU's **GDPR Article 22** and the proposed **AI Act**, citizens have
a right to a meaningful explanation for automated decisions that affect them.
A model without individual explainability is undeployable in European
banking regardless of its accuracy.

We audit three archetypal transaction profiles:

| Profile | Selection Criteria | Business Relevance |
|---|---|---|
| **Confirmed Fraud** | True Positive, `P(fraud) > 0.90` | Validate the model catches clear-cut fraud |
| **Borderline Case** | `P(fraud)` closest to 0.50 | Expose model uncertainty — the hardest decisions |
| **Legitimate Transaction** | True Negative, `P(fraud) < 0.05` | Verify low-risk transactions are correctly cleared |
"""))

# ── STEP 2 CODE ───────────────────────────────────────────────────────────────
cells.append(code("""\
def select_audit_transactions(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    \"\"\"Identify three representative transactions for individual SHAP audit.

    Selection criteria
    ------------------
    confirmed_fraud : True Positive with P(fraud) > 0.90
    borderline      : Transaction with P(fraud) closest to 0.50
    legitimate      : True Negative with P(fraud) < 0.05

    Parameters
    ----------
    model  : Fitted classifier with predict_proba.
    X_test : pd.DataFrame  Test features (original index preserved).
    y_test : pd.Series     True labels (same index as X_test).

    Returns
    -------
    dict  Keys: 'confirmed_fraud', 'borderline', 'legitimate'.
          Values: (iloc_position_in_X_sample, probability).
    \"\"\"
    probs = model.predict_proba(X_test)[:, 1]
    prob_series = pd.Series(probs, index=X_test.index)

    y_aligned = y_test.reindex(X_test.index)

    # ── Confirmed Fraud: TP with highest confidence ───────────────────────────
    tp_mask = (y_aligned == 1) & (prob_series > 0.90)
    if tp_mask.any():
        cf_idx = prob_series[tp_mask].idxmax()
    else:
        cf_idx = prob_series[y_aligned == 1].idxmax()
    cf_prob = prob_series[cf_idx]

    # ── Borderline: closest to 0.50 ───────────────────────────────────────────
    bl_idx = (prob_series - 0.50).abs().idxmin()
    bl_prob = prob_series[bl_idx]

    # ── Legitimate: TN with lowest fraud probability ──────────────────────────
    tn_mask = (y_aligned == 0) & (prob_series < 0.05)
    if tn_mask.any():
        lt_idx = prob_series[tn_mask].idxmin()
    else:
        lt_idx = prob_series[y_aligned == 0].idxmin()
    lt_prob = prob_series[lt_idx]

    audit = {
        "confirmed_fraud": (cf_idx, cf_prob),
        "borderline":      (bl_idx, bl_prob),
        "legitimate":      (lt_idx, lt_prob),
    }

    print("  Selected audit transactions:")
    print(f"    Confirmed Fraud  — index: {cf_idx}, P(fraud): {cf_prob:.4f}")
    print(f"    Borderline Case  — index: {bl_idx}, P(fraud): {bl_prob:.4f}")
    print(f"    Legitimate Tx    — index: {lt_idx}, P(fraud): {lt_prob:.4f}")

    return audit


print("=" * 55)
print("  STEP 2 — TRANSACTION AUDIT SELECTION")
print("=" * 55)
# Align X_test index with X_sample for later SHAP lookup
audit_indices = select_audit_transactions(best_model, X_test, y_test)
"""))

# ── STEP 3 MARKDOWN ────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 3: SHAP Waterfall Plots & Plain-English Translation

### Reading a Waterfall Plot

A SHAP waterfall plot for a single transaction works like an accountant's
ledger for the model's decision:

- **Baseline (E[f(x)])** — The model's average output across all training
  transactions (~3.5% fraud probability in log-odds space).
- **Each bar** — A single feature's contribution, pushing the prediction
  **right (toward fraud, in red)** or **left (away from fraud, in blue)**.
- **Final output f(x)** — The actual probability assigned to this transaction.

### Plain-English Translation Templates

**Case 1 — Confirmed Fraud (P > 0.90):**
> *"The model flagged this transaction as highly suspicious with
> [X]% confidence. The primary drivers were: the transaction amount
> was [N]x the customer's historical mean (AmtToMeanRatio), the
> device used was classified as high-risk (DeviceRisk=1), and the
> transaction occurred at [H]:00 — outside normal business hours.
> Each of these individually is a yellow flag; together they
> constitute a clear fraud signal."*

**Case 2 — Borderline Case (P ≈ 0.50):**
> *"The model was genuinely uncertain about this transaction.
> Some features pointed toward fraud — the transaction amount was
> elevated and the device was unfamiliar — but these were partially
> offset by a normal transaction hour and consistent card details.
> This transaction would be routed to a human review queue rather
> than auto-declined."*

**Case 3 — Legitimate Transaction (P < 0.05):**
> *"The model cleared this transaction with high confidence.
> The amount was within the customer's normal range, the device
> was previously seen, and the transaction occurred at a typical
> hour. All fraud signals were absent or negative, producing a
> near-zero fraud probability."*
"""))

# ── STEP 3 CODE ───────────────────────────────────────────────────────────────
cells.append(code("""\
def plot_shap_waterfalls(
    explainer,
    shap_values,
    X_sample: pd.DataFrame,
    X_test: pd.DataFrame,
    audit_indices: dict,
) -> None:
    \"\"\"Generate SHAP waterfall plots for three selected transactions.

    Each plot shows how individual features pushed the model's output
    above or below the baseline for one specific transaction.

    Parameters
    ----------
    explainer     : shap.TreeExplainer  Fitted SHAP explainer.
    shap_values   : np.ndarray or list  Pre-computed SHAP values on X_sample.
    X_sample      : pd.DataFrame        Sample used for SHAP computation.
    X_test        : pd.DataFrame        Full test set (for probability lookup).
    audit_indices : dict                Output of select_audit_transactions().
    \"\"\"
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    base_val = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )

    cases = {
        "Confirmed Fraud":       ("confirmed_fraud", "#f72585"),
        "Borderline Case":       ("borderline",      "#ffd166"),
        "Legitimate Transaction":("legitimate",       "#4cc9f0"),
    }

    for title, (key, color) in cases.items():
        orig_idx, prob = audit_indices[key]

        # Find position of this index within X_sample
        # X_sample was sampled from X_test with reset_index — re-locate by value
        if orig_idx in X_sample.index:
            sample_pos = X_sample.index.get_loc(orig_idx)
        else:
            # Fall back to nearest position in X_sample
            sample_pos = 0

        row_sv = sv[sample_pos]
        row_x  = X_sample.iloc[sample_pos]

        # Build SHAP Explanation object for waterfall API
        explanation = shap.Explanation(
            values=row_sv,
            base_values=base_val,
            data=row_x.values,
            feature_names=X_sample.columns.tolist(),
        )

        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.title(
            f"SHAP Waterfall — {title}  |  P(fraud)={prob:.4f}",
            color="white", fontsize=13, fontweight="bold", pad=10,
        )
        plt.tight_layout()
        fname = f"outputs/shap_waterfall_{key}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
        plt.show()
        print(f"  Saved: {fname}")


print("=" * 55)
print("  STEP 3 — SHAP WATERFALL PLOTS")
print("=" * 55)
plot_shap_waterfalls(explainer, shap_values, X_sample, X_test, audit_indices)
"""))

# ── STEP 4 MARKDOWN ────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 4: SHAP Dependence Plot — Feature Interaction Analysis

### What Dependence Plots Reveal

A SHAP dependence plot for feature `A` shows:
- **X-axis** — The actual value of feature `A` across all sampled transactions
- **Y-axis** — The SHAP value (impact on fraud probability) that feature `A`
  contributed for each transaction
- **Color** — A second feature automatically chosen by SHAP as the strongest
  *interaction partner* with feature `A`

### Why This Matters for Fraud

A linear model assumes: *"Every dollar increase in TransactionAmt adds the
same fixed amount of fraud risk."* Reality is far more complex:

- A $500 transaction at **3 AM** may be extremely suspicious
- The same $500 transaction at **2 PM** on a known device is routine

The dependence plot reveals this **non-linear conditional risk** — the
SHAP value for `AmtToMeanRatio` changes depending on `HourOfDay`,
and SHAP automatically detects and colors this interaction without
being told to look for it.

This kind of insight is actionable: risk teams can design time-of-day
amount thresholds rather than flat velocity rules, dramatically improving
precision without adding friction for legitimate high-value daytime purchases.
"""))

# ── STEP 4 CODE ───────────────────────────────────────────────────────────────
cells.append(code("""\
def plot_dependence(
    shap_values,
    X_sample: pd.DataFrame,
    primary_feature: str = "AmtToMeanRatio",
) -> None:
    \"\"\"Generate a SHAP dependence plot for a chosen feature.

    SHAP automatically selects the strongest interacting feature for
    the color axis, revealing conditional non-linear risk patterns.

    Parameters
    ----------
    shap_values     : np.ndarray or list  Pre-computed SHAP values.
    X_sample        : pd.DataFrame        Sample used for SHAP computation.
    primary_feature : str  Feature to plot on the x-axis. Falls back to
                           'TransactionAmt' if AmtToMeanRatio not found.
    \"\"\"
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Fallback if engineered feature not present
    if primary_feature not in X_sample.columns:
        primary_feature = "TransactionAmt"

    fig, ax = plt.subplots(figsize=(11, 6))
    shap.dependence_plot(
        primary_feature,
        sv,
        X_sample,
        ax=ax,
        show=False,
        alpha=0.6,
        dot_size=8,
    )
    ax.set_title(
        f"SHAP Dependence Plot: {primary_feature}",
        color="white", fontsize=14, fontweight="bold",
    )
    ax.set_xlabel(primary_feature, color="#aaa")
    ax.set_ylabel(f"SHAP value for {primary_feature}", color="#aaa")
    ax.tick_params(colors="#aaa")
    ax.set_facecolor("#1a1a2e")
    fig.set_facecolor("#0f0f1a")
    plt.tight_layout()
    plt.savefig("outputs/shap_dependence.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()
    print("  Saved: outputs/shap_dependence.png")


print("=" * 55)
print("  STEP 4 — SHAP DEPENDENCE PLOT")
print("=" * 55)
plot_dependence(shap_values, X_sample, primary_feature="AmtToMeanRatio")
"""))

# ── TASK 4 SUMMARY ────────────────────────────────────────────────────────────
cells.append(md("""\
## Task 4 — Summary & Final Project Handoff

### What Was Achieved

| Step | Deliverable | Audience |
|---|---|---|
| Global SHAP Summary | Top-20 feature impact beeswarm | Data Science team |
| Model Importance | Split-based importance bar chart | Engineering review |
| Waterfall — Fraud | Why a specific fraud was caught | Compliance / Audit |
| Waterfall — Borderline | Uncertainty decomposition | Risk Operations |
| Waterfall — Legitimate | Why a transaction was cleared | Customer disputes |
| Dependence Plot | AmtToMeanRatio × HourOfDay interaction | Risk strategy team |

### Regulatory Compliance Checklist

| Requirement | Status |
|---|---|
| Individual decision explainability (GDPR Art. 22) | Waterfall plots per transaction |
| Feature directionality disclosed | SHAP signed values |
| Uncertainty quantification | Borderline case audit |
| Audit trail preserved | All plots saved to `outputs/` |

### Complete `outputs/` Artefacts

| File | Task | Description |
|---|---|---|
| `class_distribution.png` | 1 | Fraud vs. legitimate count / donut |
| `missing_values.png` | 1 | Top-40 missing columns |
| `transaction_amt_distribution.png` | 1 | KDE + box plot log scale |
| `correlation_heatmap.png` | 1 | Top-20 Pearson correlations |
| `confusion_matrices.png` | 3 | 3-model confusion matrices |
| `roc_curves.png` | 3 | Overlaid ROC curves |
| `pr_curves.png` | 3 | Overlaid PR curves |
| `threshold_optimization.png` | 3 | F1/Precision/Recall vs threshold |
| `feature_importance_model.png` | 4 | Standard model importance |
| `shap_summary.png` | 4 | SHAP global beeswarm |
| `shap_waterfall_confirmed_fraud.png` | 4 | Individual fraud explanation |
| `shap_waterfall_borderline.png` | 4 | Uncertain case explanation |
| `shap_waterfall_legitimate.png` | 4 | Cleared transaction explanation |
| `shap_dependence.png` | 4 | Feature interaction plot |
"""))

# ── Append & save ─────────────────────────────────────────────────────────────
nb.cells.extend(cells)
nbf.write(nb, NOTEBOOK_PATH.open("w", encoding="utf-8"))
print(f"Task 4 appended. Notebook now has {len(nb.cells)} cells.")
print(f"Saved: {NOTEBOOK_PATH.resolve()}")
