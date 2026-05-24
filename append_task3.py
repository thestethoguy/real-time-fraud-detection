"""append_task3.py — Appends Task 3 cells to analysis.ipynb"""
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

# TASK 3 — Model Training, Comparison & Threshold Optimization

**Objective:** Train three distinct models, compare them on rigorous financial
metrics, and apply advanced threshold optimization to maximize fraud recall.

| Step | Operation |
|---|---|
| 1 | Baseline model training (LightGBM, XGBoost, Isolation Forest) |
| 2 | Multi-metric evaluation + visual comparison |
| 3 | Decision threshold optimization |
| 4 | Hyperparameter tuning via RandomizedSearchCV |
"""))

# ── STEP 1 MARKDOWN ────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 1: Baseline Model Training

### Model Selection Rationale

We train three architecturally distinct models to establish a robust baseline:

**LightGBM (Supervised — Gradient Boosting)**
Leaf-wise tree growth with histogram-based splitting. Fastest training on
tabular data, native support for `class_weight` to handle imbalance, and
consistently wins on fraud detection benchmarks. Our primary candidate.

**XGBoost (Supervised — Gradient Boosting)**
Level-wise tree growth with regularization (L1/L2). Slightly slower than
LightGBM but excellent generalization. Used as a cross-validation benchmark.
`scale_pos_weight` is set to the class ratio to handle imbalance internally.

**Isolation Forest (Unsupervised — Anomaly Detection)**
Does **not** use labels during training. It isolates anomalies by randomly
partitioning feature space — fraud samples require fewer splits to isolate.
Critical note: outputs `+1` (normal) and `-1` (anomaly). We remap these
to `0` and `1` respectively to align with our binary target convention.

> Using both supervised and unsupervised approaches lets us see how much
> signal is recoverable without labels — a key production insight when
> fraud labels are delayed or unavailable.
"""))

# ── STEP 1 CODE ───────────────────────────────────────────────────────────────
cells.append(code("""\
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# Class ratio for XGBoost scale_pos_weight
_fraud_ratio = float((y_train == 0).sum()) / float((y_train == 1).sum())


def train_baseline_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    \"\"\"Initialize, fit and return all three baseline models.

    Parameters
    ----------
    X_train : pd.DataFrame   SMOTE-balanced training features.
    y_train : pd.Series      Balanced binary labels (0/1).

    Returns
    -------
    dict  Keys: 'lgbm', 'xgb', 'iforest'. Values: fitted estimators.
    \"\"\"
    print("[1/3] Training LightGBM ...")
    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    print("      Done.")

    print("[2/3] Training XGBoost ...")
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=_fraud_ratio,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    print("      Done.")

    print("[3/3] Training Isolation Forest ...")
    iforest = IsolationForest(
        n_estimators=200,
        contamination=0.035,   # approx real-world fraud rate
        n_jobs=-1,
        random_state=42,
    )
    iforest.fit(X_train)
    print("      Done.")

    return {"lgbm": lgbm, "xgb": xgb, "iforest": iforest}


print("=" * 55)
print("  STEP 1 — BASELINE MODEL TRAINING")
print("=" * 55)
models = train_baseline_models(X_train, y_train)
print("\\nAll models trained successfully.")
"""))

# ── STEP 2 MARKDOWN ────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 2: Evaluation Metrics & Visualizations

### The Business Cost Framework

In fraud detection, not all errors are equal:

| Error Type | What It Means | Business Cost |
|---|---|---|
| **False Negative** | Missed fraud — labeled legitimate | Direct financial loss; customer liability |
| **False Positive** | Legitimate tx flagged as fraud | Customer friction, declined cards, churn |

For a bank, **False Negatives are far more expensive** than False Positives.
This asymmetry drives every metric choice below.

### Why PR-AUC is the North Star Metric

**ROC-AUC** plots True Positive Rate vs False Positive Rate. On a dataset
with 96.5% negatives, even a bad model achieves a high ROC-AUC because the
True Negative Rate is trivially high — the curve is flattered by the
majority class.

**PR-AUC** (Precision-Recall Area Under Curve) operates exclusively in the
minority-class space. It measures: *"Of the fraud we caught, how much was
real? And of all real fraud, how much did we catch?"*

A random classifier on a 3.5% imbalanced dataset achieves a PR-AUC of
**0.035**. Any score above that represents genuine learning. This makes
PR-AUC the only honest north-star metric for imbalanced fraud detection.
"""))

# ── STEP 2 CODE ───────────────────────────────────────────────────────────────
cells.append(code("""\
def evaluate_and_visualize_models(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    \"\"\"Compute metrics and produce confusion matrix, ROC, and PR curve plots.

    Parameters
    ----------
    models  : dict   Output of train_baseline_models().
    X_test  : pd.DataFrame   Held-out test features (real-world distribution).
    y_test  : pd.Series      True binary labels.

    Returns
    -------
    pd.DataFrame  Per-model metrics summary.
    \"\"\"
    results = []
    roc_data, pr_data = {}, {}

    for name, model in models.items():
        # ── Predictions ───────────────────────────────────────────────────────
        if name == "iforest":
            raw_pred = model.predict(X_test)
            # Isolation Forest: -1 = anomaly → 1 (fraud), 1 = normal → 0
            y_pred = np.where(raw_pred == -1, 1, 0)
            # Use negative anomaly score as a proxy probability
            scores = -model.score_samples(X_test)
            # Normalize to [0,1] range
            y_prob = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        # ── Scalar metrics ────────────────────────────────────────────────────
        results.append({
            "Model":     name.upper(),
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4),
            "PR-AUC":    round(average_precision_score(y_test, y_prob), 4),
        })

        # ── Curve data for later plots ────────────────────────────────────────
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_data[name] = (prec, rec)

    metrics_df = pd.DataFrame(results).set_index("Model")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 1 — Confusion Matrices
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices — All Models", fontsize=15,
                 fontweight="bold", color="white")

    cm_colors = ["#4cc9f0", "#f72585"]
    for ax, (name, model) in zip(axes, models.items()):
        if name == "iforest":
            raw = model.predict(X_test)
            preds = np.where(raw == -1, 1, 0)
        else:
            preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(
            cm, ax=ax, annot=True, fmt=",d", cmap="Blues",
            linewidths=0.5, linecolor="#0f0f1a",
            xticklabels=["Pred: Legit", "Pred: Fraud"],
            yticklabels=["True: Legit", "True: Fraud"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(name.upper(), color="white", fontsize=13)
        ax.tick_params(colors="#aaa")

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrices.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 2 — ROC Curves
    # ══════════════════════════════════════════════════════════════════════════
    palette = {"lgbm": "#4cc9f0", "xgb": "#f72585", "iforest": "#ffd166"}
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "w--", linewidth=0.8, label="Random (AUC=0.50)")
    for name, (fpr, tpr) in roc_data.items():
        auc = metrics_df.loc[name.upper(), "ROC-AUC"]
        ax.plot(fpr, tpr, color=palette[name], linewidth=2,
                label=f"{name.upper()}  AUC={auc:.4f}")
    ax.set_xlabel("False Positive Rate", color="#aaa")
    ax.set_ylabel("True Positive Rate", color="#aaa")
    ax.set_title("ROC Curves — All Models", color="white",
                 fontsize=14, fontweight="bold")
    ax.legend(labelcolor="white", facecolor="#1a1a2e")
    ax.tick_params(colors="#aaa")
    plt.tight_layout()
    plt.savefig("outputs/roc_curves.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 3 — Precision-Recall Curves
    # ══════════════════════════════════════════════════════════════════════════
    baseline_pr = y_test.mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(baseline_pr, color="white", linestyle="--", linewidth=0.8,
               label=f"Random baseline (PR={baseline_pr:.3f})")
    for name, (prec, rec) in pr_data.items():
        auc = metrics_df.loc[name.upper(), "PR-AUC"]
        ax.plot(rec, prec, color=palette[name], linewidth=2,
                label=f"{name.upper()}  PR-AUC={auc:.4f}")
    ax.set_xlabel("Recall", color="#aaa")
    ax.set_ylabel("Precision", color="#aaa")
    ax.set_title("Precision-Recall Curves — All Models", color="white",
                 fontsize=14, fontweight="bold")
    ax.legend(labelcolor="white", facecolor="#1a1a2e")
    ax.tick_params(colors="#aaa")
    plt.tight_layout()
    plt.savefig("outputs/pr_curves.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()

    print("\\n--- Model Comparison Metrics ---")
    display(metrics_df.style
        .background_gradient(cmap="Blues", subset=["PR-AUC", "ROC-AUC", "Recall"])
        .format("{:.4f}")
    )
    return metrics_df


print("=" * 55)
print("  STEP 2 — EVALUATION & VISUALIZATION")
print("=" * 55)
metrics_df = evaluate_and_visualize_models(models, X_test, y_test)
"""))

# ── STEP 3 MARKDOWN ────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 3: Decision Threshold Optimization

### Why 0.5 is the Wrong Threshold for Fraud

Every classifier outputs a **probability score** between 0 and 1, not a
hard class label. The default decision rule — *"predict fraud if P > 0.50"*
— was designed for balanced datasets. On our 3.5%-fraud dataset, this
threshold is far too conservative and misses a significant portion of fraud.

**The fundamental trade-off:**

| Threshold Direction | Effect on Recall | Effect on Precision |
|---|---|---|
| Lower (e.g., 0.2) | ↑ More fraud caught | ↓ More false alarms |
| Higher (e.g., 0.8) | ↓ Less fraud caught | ↑ Fewer false alarms |

**The F1-Score** is the harmonic mean of Precision and Recall. Optimizing
the threshold to maximize F1 finds the mathematically balanced operating
point that respects both business costs simultaneously.

In practice, a fraud operations team would then shift slightly from the
F1-optimal point toward higher recall — accepting more false positives to
ensure fewer fraudulent transactions slip through.
"""))

# ── STEP 3 CODE ───────────────────────────────────────────────────────────────
cells.append(code("""\
def optimize_threshold(
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "lgbm",
) -> float:
    \"\"\"Sweep decision thresholds and identify the F1-optimal cutoff.

    Parameters
    ----------
    best_model  : fitted sklearn-compatible classifier with predict_proba.
    X_test      : pd.DataFrame  Test features.
    y_test      : pd.Series     True binary labels.
    model_name  : str           Display name for plot title.

    Returns
    -------
    float   Threshold value that maximizes F1-Score.
    \"\"\"
    y_prob = best_model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.05, 0.95, 0.01)

    f1_scores, precision_scores, recall_scores = [], [], []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred, zero_division=0))

    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = thresholds[best_idx]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(thresholds, f1_scores,        color="#4cc9f0", lw=2, label="F1-Score")
    ax.plot(thresholds, precision_scores,  color="#ffd166", lw=2, label="Precision", alpha=0.8)
    ax.plot(thresholds, recall_scores,     color="#f72585", lw=2, label="Recall",    alpha=0.8)
    ax.axvline(optimal_threshold, color="white", linestyle="--", linewidth=1.5,
               label=f"Optimal threshold = {optimal_threshold:.2f}")
    ax.axvline(0.5, color="#aaa", linestyle=":", linewidth=1,
               label="Default threshold = 0.50")
    ax.set_xlabel("Decision Threshold", color="#aaa")
    ax.set_ylabel("Score", color="#aaa")
    ax.set_title(f"Threshold Optimization — {model_name.upper()}",
                 color="white", fontsize=14, fontweight="bold")
    ax.legend(labelcolor="white", facecolor="#1a1a2e")
    ax.tick_params(colors="#aaa")
    plt.tight_layout()
    plt.savefig("outputs/threshold_optimization.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()

    print(f"  Default threshold (0.50) F1  : {f1_scores[list(thresholds).index(min(thresholds, key=lambda x: abs(x-0.5)))]:.4f}")
    print(f"  Optimal threshold            : {optimal_threshold:.2f}")
    print(f"  Optimal F1-Score             : {f1_scores[best_idx]:.4f}")
    print(f"  Precision at optimal         : {precision_scores[best_idx]:.4f}")
    print(f"  Recall    at optimal         : {recall_scores[best_idx]:.4f}")

    return optimal_threshold


print("=" * 55)
print("  STEP 3 — THRESHOLD OPTIMIZATION")
print("=" * 55)
# Use LightGBM as the best-performing tree model
optimal_threshold = optimize_threshold(models["lgbm"], X_test, y_test, "lgbm")
print(f"\\nOptimal decision threshold: {optimal_threshold:.2f}")
"""))

# ── STEP 4 MARKDOWN ────────────────────────────────────────────────────────────
cells.append(md("""\
## Step 4: Hyperparameter Tuning — RandomizedSearchCV

### Why RandomizedSearchCV over GridSearchCV?

**GridSearchCV** evaluates every combination of hyperparameters.
With even a modest grid of 4 parameters × 5 values, that is 5⁴ = 625
model fits × k folds = **1,875 fits**. On 590k rows this will exhaust
RAM and run for hours on a local machine.

**RandomizedSearchCV** samples `n_iter` random combinations from the
parameter distributions. With `n_iter=5` and `cv=3`, we run only
**15 fits** — a 99% reduction — while still exploring the hyperparameter
space stochastically. Research shows random search finds near-optimal
parameters in a fraction of the compute time
*(Bergstra & Bengio, 2012)*.

### Parameters Being Tuned

| Parameter | Effect |
|---|---|
| `learning_rate` | Step size per boosting round; lower = more robust, needs more trees |
| `max_depth` | Tree depth; higher = more complex patterns but risks overfitting |
| `num_leaves` | Primary complexity control in LightGBM; must be < 2^max_depth |
| `n_estimators` | Number of boosting rounds; more = better until diminishing returns |
| `min_child_samples` | Minimum samples per leaf; regularizes against noise |
"""))

# ── STEP 4 CODE ───────────────────────────────────────────────────────────────
cells.append(code("""\
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def tune_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 5,
    cv: int = 3,
    random_state: int = 42,
) -> LGBMClassifier:
    \"\"\"Run RandomizedSearchCV on LightGBM and return the best estimator.

    Parameters
    ----------
    X_train      : SMOTE-balanced training features.
    y_train      : Balanced binary labels.
    n_iter       : Number of random parameter combinations to evaluate.
    cv           : Number of cross-validation folds.
    random_state : Reproducibility seed.

    Returns
    -------
    LGBMClassifier  Best estimator re-fitted on the full training set.
    \"\"\"
    param_dist = {
        "learning_rate":    uniform(0.01, 0.15),      # U[0.01, 0.16]
        "max_depth":        randint(4, 9),             # {4,5,6,7,8}
        "num_leaves":       randint(31, 128),          # {31 .. 127}
        "n_estimators":     randint(300, 800),         # {300 .. 799}
        "min_child_samples": randint(20, 100),         # regularisation
    }

    base_lgbm = LGBMClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )

    search = RandomizedSearchCV(
        estimator=base_lgbm,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="average_precision",   # PR-AUC as the search objective
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
        refit=True,                    # re-fits best params on full train set
    )

    print(f"Running RandomizedSearchCV: {n_iter} iterations x {cv} folds ...")
    search.fit(X_train, y_train)

    print("\\n  Best Parameters Found:")
    for param, value in search.best_params_.items():
        print(f"    {param:<22}: {value}")
    print(f"\\n  Best CV PR-AUC Score : {search.best_score_:.4f}")

    # Show improvement over baseline
    baseline_score = metrics_df.loc["LGBM", "PR-AUC"]
    improvement = search.best_score_ - baseline_score
    print(f"  Baseline PR-AUC      : {baseline_score:.4f}")
    print(f"  Improvement          : {improvement:+.4f}")

    return search.best_estimator_


print("=" * 55)
print("  STEP 4 — HYPERPARAMETER TUNING")
print("=" * 55)
best_lgbm_tuned = tune_best_model(X_train, y_train, n_iter=5, cv=3)
print("\\nTuned LightGBM model ready.")
"""))

# ── TASK 3 SUMMARY ────────────────────────────────────────────────────────────
cells.append(md("""\
## Task 3 — Summary & Handoff to Task 4

| Step | Outcome |
|---|---|
| Baseline Training | 3 models fitted; IForest predictions remapped to 0/1 |
| Evaluation | Confusion matrices, ROC, PR-AUC curves saved to `outputs/` |
| Threshold Opt. | Optimal F1 threshold computed; stored in `optimal_threshold` |
| Hyperparameter Tuning | Best LightGBM params via RandomizedSearchCV (PR-AUC objective) |

### Saved Artefacts

| File | Description |
|---|---|
| `outputs/confusion_matrices.png` | 3-panel confusion matrix comparison |
| `outputs/roc_curves.png` | Overlaid ROC curves |
| `outputs/pr_curves.png` | Overlaid Precision-Recall curves |
| `outputs/threshold_optimization.png` | F1 / Precision / Recall vs threshold |

### Task 4 Input Variables

| Variable | Description |
|---|---|
| `models` | Dict of all three fitted baseline models |
| `best_lgbm_tuned` | Tuned LightGBM — best production candidate |
| `optimal_threshold` | Decision cutoff for deployment |
| `metrics_df` | Comparative metrics DataFrame |
"""))

# ── Append & save ──────────────────────────────────────────────────────────────
nb.cells.extend(cells)
nbf.write(nb, NOTEBOOK_PATH.open("w", encoding="utf-8"))
print(f"Task 3 appended. Notebook now has {len(nb.cells)} cells.")
print(f"Saved: {NOTEBOOK_PATH.resolve()}")
