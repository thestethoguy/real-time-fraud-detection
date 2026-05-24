"""
make_notebook.py
Programmatically builds analysis.ipynb for the IEEE-CIS Fraud Detection
Capstone – Task 1.  Run once, then open the notebook in Jupyter.
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── helpers ──────────────────────────────────────────────────────────────────
def md(src: str):
    return nbf.v4.new_markdown_cell(src)

def code(src: str):
    return nbf.v4.new_code_cell(src)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 0 – Title
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
# TASK 1 — Data Loading, Merging & Exploratory Analysis

**Project:** Real-Time Fraud Detection System | **Dataset:** IEEE-CIS Fraud Detection  
**Author:** Aman Aaryan | **Role:** Lead ML Engineer  

---

This notebook covers the foundational phase of the fraud detection pipeline:

1. Memory-optimised data loading and merging  
2. Class imbalance analysis  
3. Missing value audit  
4. Transaction amount distribution (log-scale)  
5. Correlation analysis with the target variable  
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 – Imports & Theme
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## Step 0: Environment Setup

We configure a professional dark visual theme for all plots and declare the
canonical data paths so every subsequent cell stays path-agnostic.
"""))

cells.append(code("""\
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Dark professional plot theme ──────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor":  "#0f0f1a",
    "axes.facecolor":    "#1a1a2e",
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   "#e0e0e0",
    "text.color":        "#e0e0e0",
    "xtick.color":       "#aaaaaa",
    "ytick.color":       "#aaaaaa",
    "grid.color":        "#2a2a3e",
    "legend.facecolor":  "#1a1a2e",
    "legend.edgecolor":  "#444444",
    "figure.titlesize":  16,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR          = Path("data")
TRANSACTION_PATH  = DATA_DIR / "train_transaction.csv"
IDENTITY_PATH     = DATA_DIR / "train_identity.csv"
os.makedirs("outputs", exist_ok=True)

print("Libraries imported and theme configured.")
print(f"  Transaction file found : {TRANSACTION_PATH.exists()}")
print(f"  Identity file found    : {IDENTITY_PATH.exists()}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 – Markdown: Step 1 rationale
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## Step 1: Optimised Data Loading & Merging

### Business Context
The IEEE-CIS dataset ships as **two separate CSVs**:

| File | Rows | Columns | Size on disk |
|---|---|---|---|
| `train_transaction.csv` | 590,540 | 394 | ~652 MB |
| `train_identity.csv`    | 144,233 |  41 | ~25 MB  |

Naively loading both with default dtypes consumes **4–6 GB of RAM** —
enough to crash a typical laptop kernel mid-merge.

### Technical Strategy

**`reduce_mem_usage()`**  
Iterates every column and downcasts to the smallest safe numeric type:

- `float64` → `float32` — halves memory for every continuous feature  
- `int64` → `int32 / int16 / int8` — chosen based on observed min/max  
- `object` with low cardinality → `category` — efficient string storage  

Typical saving: **~55–65 % reduction** before a single row is dropped.

**Left join on `TransactionID`**  
We use a *left* join to retain **all 590,540 transactions**.  
Identity data enriches rows where available; the ~59 % of transactions  
with no identity record produce NaN — which is itself a signal  
(unidentifiable devices are a known fraud vector).

> Memory optimisation is applied *immediately after the merge* so we  
> never hold two large frames in RAM simultaneously.
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 – reduce_mem_usage + load_and_merge_data
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    \"\"\"Reduce DataFrame memory by downcasting numeric and object columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to optimise (mutated in-place).
    verbose : bool
        Print before/after memory statistics when True.

    Returns
    -------
    pd.DataFrame
        Memory-optimised DataFrame.
    \"\"\"
    start_mem: float = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_dtype = df[col].dtype

        if col_dtype == object:
            # Low-cardinality strings -> category saves substantial memory
            if df[col].nunique() / max(len(df[col]), 1) < 0.50:
                df[col] = df[col].astype("category")
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if str(col_dtype).startswith("int"):
            for int_type in [np.int8, np.int16, np.int32]:
                if (c_min >= np.iinfo(int_type).min and
                        c_max <= np.iinfo(int_type).max):
                    df[col] = df[col].astype(int_type)
                    break

        elif str(col_dtype).startswith("float"):
            if (c_min >= np.finfo(np.float32).min and
                    c_max <= np.finfo(np.float32).max):
                df[col] = df[col].astype(np.float32)

    end_mem: float = df.memory_usage(deep=True).sum() / 1024 ** 2

    if verbose:
        pct = 100 * (start_mem - end_mem) / start_mem
        print(f"  Memory before : {start_mem:8.2f} MB")
        print(f"  Memory after  : {end_mem:8.2f} MB")
        print(f"  Reduction     : {pct:.1f} %")

    return df


def load_and_merge_data(
    transaction_path: Path,
    identity_path: Path,
) -> pd.DataFrame:
    \"\"\"Load, merge, and memory-optimise the IEEE-CIS fraud dataset.

    Parameters
    ----------
    transaction_path : Path
        Path to train_transaction.csv.
    identity_path : Path
        Path to train_identity.csv.

    Returns
    -------
    pd.DataFrame
        Merged, memory-optimised DataFrame ready for EDA.
    \"\"\"
    print("[1/4] Loading transaction data ...")
    df_trans = pd.read_csv(transaction_path)
    print(f"      Shape: {df_trans.shape}")

    print("[2/4] Loading identity data ...")
    df_id = pd.read_csv(identity_path)
    print(f"      Shape: {df_id.shape}")

    print("[3/4] Merging on TransactionID (LEFT JOIN) ...")
    df = df_trans.merge(df_id, on="TransactionID", how="left")
    print(f"      Merged shape: {df.shape}")

    # Free constituent frames to reclaim RAM before optimisation
    del df_trans, df_id

    print("[4/4] Applying memory optimisation ...")
    df = reduce_mem_usage(df, verbose=True)

    return df


# ── Execute ───────────────────────────────────────────────────────────────────
print("=" * 55)
print("  LOADING & MERGING IEEE-CIS FRAUD DATASET")
print("=" * 55)
df = load_and_merge_data(TRANSACTION_PATH, IDENTITY_PATH)
print("\\nDataset ready.")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 – Quick dataset overview
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
print(f"Shape   : {df.shape[0]:,} rows x {df.shape[1]:,} columns")
print(f"\\nisFraud value counts:")
print(df["isFraud"].value_counts())
print("\\nDtype summary:")
print(df.dtypes.value_counts())
print("\\nFirst 10 rows:")
df.head(10)
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 – Markdown: class imbalance
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## Step 2: Target Variable Analysis — Class Imbalance

### Business Context
Financial fraud datasets are **severely imbalanced by design** — the system  
would be worthless if fraud were common. In IEEE-CIS, fraud accounts for  
only **~3.5 %** of all transactions.

### Why This Is Dangerous for Modelling

| Naive classifier behaviour | Result |
|---|---|
| Predict "Not Fraud" for every row | **96.5 % accuracy** — but catches 0 frauds |
| Optimise cross-entropy on raw counts | Model ignores the minority class |

### Evaluation Metrics We Will Use
- **PR-AUC** (Precision-Recall Area Under Curve) — most informative for imbalanced data  
- **ROC-AUC** — standard benchmark, less sensitive to imbalance  
- **F1-Score** at tuned threshold  
- *Not* raw accuracy  

### Mitigation Plan (Task 2 & 3)
- `class_weight="balanced"` in all sklearn estimators  
- Stratified K-Fold to preserve the 3.5 % ratio in every fold  
- Probability-threshold tuning to maximise recall at acceptable precision  
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 – plot_fraud_distribution
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
def plot_fraud_distribution(df: pd.DataFrame) -> None:
    \"\"\"Visualise the binary class imbalance for isFraud.

    Produces a dual-panel figure:
      - Left  : Annotated bar chart of absolute counts
      - Right : Donut chart showing percentage split

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the ``isFraud`` column.
    \"\"\"
    counts = df["isFraud"].value_counts().sort_index()
    labels = ["Non-Fraud (0)", "Fraud (1)"]
    colors = ["#4cc9f0", "#f72585"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Class Distribution: isFraud",
        fontsize=18, fontweight="bold", color="white", y=1.02,
    )

    # ── Left: bar chart ───────────────────────────────────────────────────────
    ax1 = axes[0]
    bars = ax1.bar(labels, counts.values, color=colors,
                   edgecolor="white", linewidth=0.8, width=0.5)
    ax1.set_title("Absolute Count", color="white", fontsize=13)
    ax1.set_ylabel("Number of Transactions", color="#aaa")
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    for bar, val in zip(bars, counts.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:,}", ha="center", va="bottom",
            color="white", fontweight="bold",
        )

    # ── Right: donut chart ────────────────────────────────────────────────────
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=labels,
        autopct="%1.2f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"width": 0.6, "edgecolor": "white", "linewidth": 1.5},
        textprops={"color": "white"},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax2.set_title("Percentage Split", color="white", fontsize=13)

    plt.tight_layout()
    plt.savefig("outputs/class_distribution.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()
    fraud_rate = counts[1] / counts.sum() * 100
    print(f"  Fraud rate: {fraud_rate:.2f}%")


plot_fraud_distribution(df)
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 – Markdown: missing values
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## Step 3: Missing Value Analysis & Threshold Logic

### Technical Rationale

Not all missing values are equal.  We apply a tiered strategy:

| Missing % Range | Treatment |
|---|---|
| **0 – 20 %** | Safe to impute (median / mode / KNN) |
| **20 – 50 %** | Impute with caution; add binary missingness-indicator feature |
| **> 50 %** | **Flag for removal** — imputation injects more noise than signal |

### Why 50 % Is the Hard Cutoff
A column that is absent for more than half the dataset was **not collected**  
for most customers. Any imputed value is essentially fabricated, and a model  
trained on it learns the imputation algorithm's behaviour, not real fraud  
patterns.

> **Decision:** Columns above the 50 % threshold are **identified now**  
> and stored in `cols_to_drop`. They will be **removed in Task 2** during  
> feature engineering so the raw audit trail is preserved here.
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 – analyze_missing_values
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
def analyze_missing_values(
    df: pd.DataFrame,
    threshold: float = 0.50,
) -> list:
    \"\"\"Audit missing values and flag columns above the drop threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset.
    threshold : float
        Fraction (0-1) above which a column is flagged.  Default: 0.50.

    Returns
    -------
    list
        Column names whose missing fraction exceeds ``threshold``.
    \"\"\"
    missing = (
        df.isnull()
          .mean()
          .rename_axis("Column")
          .reset_index(name="Missing_Fraction")
    )
    missing["Missing_Pct"] = (missing["Missing_Fraction"] * 100).round(2)
    missing = missing.sort_values("Missing_Fraction", ascending=False)

    flagged: list = (
        missing.loc[missing["Missing_Fraction"] > threshold, "Column"]
        .tolist()
    )

    # ── Visualise top-40 most-missing columns ─────────────────────────────────
    top40 = missing.head(40)
    bar_colors = [
        "#f72585" if v > threshold else "#4cc9f0"
        for v in top40["Missing_Fraction"]
    ]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.barh(top40["Column"], top40["Missing_Pct"],
            color=bar_colors, edgecolor="none")
    ax.axvline(
        threshold * 100, color="#ffd166",
        linestyle="--", linewidth=1.8,
        label=f"Drop threshold ({threshold*100:.0f}%)",
    )
    ax.set_xlabel("Missing Value Percentage (%)", color="#aaa")
    ax.set_title("Top 40 Columns by Missing Value %",
                 color="white", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(labelcolor="white")
    ax.tick_params(colors="#aaa")
    plt.tight_layout()
    plt.savefig("outputs/missing_values.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()

    print(f"  Total columns           : {df.shape[1]}")
    print(f"  Columns above threshold : {len(flagged)}")
    print(f"  Columns retained        : {df.shape[1] - len(flagged)}")
    return flagged


cols_to_drop = analyze_missing_values(df, threshold=0.50)
print(f"\\nSample of flagged columns ({len(cols_to_drop)} total):")
print(cols_to_drop[:15])
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 – Markdown: log scale rationale
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## Step 4: Transaction Amount Distribution

### Why Log Scale?
`TransactionAmt` is a textbook **right-skewed, heavy-tailed distribution**:

- The bulk of transactions cluster between **$10 – $200**  
- A small fraction reach **$10,000+** (wire transfers, high-value purchases)  
- On a **linear axis**, extreme outliers compress 95 % of the data into  
  a thin band — intra-class patterns become invisible  

**Log scale** (log1p to handle near-zero values) spreads values across  
orders of magnitude, making structural differences between fraud and  
legitimate amounts clearly visible.

### Expected Business Insight
Fraudsters exhibit two behavioural signatures:
1. **Micro-transactions** (e.g., $1) — testing a stolen card before cashing out  
2. **Macro-transactions** — maximising extraction before the card is blocked  

This bimodal pattern on a log scale is one of the strongest early signals  
available without any feature engineering.
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 – plot_transaction_amt
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
def plot_transaction_amt(df: pd.DataFrame) -> None:
    \"\"\"Plot TransactionAmt distributions for fraud vs. non-fraud.

    Left panel  : Overlapping KDE on log1p-transformed amounts.
    Right panel : Side-by-side box plots on a log-scale y-axis.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing ``TransactionAmt`` and ``isFraud``.
    \"\"\"
    fraud_palette = {0: "#4cc9f0", 1: "#f72585"}
    fraud_labels  = {0: "Non-Fraud", 1: "Fraud"}

    log_amt = np.log1p(df["TransactionAmt"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Transaction Amount Distribution by Fraud Label",
        fontsize=16, fontweight="bold", color="white",
    )

    # ── Left: overlapping KDE ─────────────────────────────────────────────────
    ax1 = axes[0]
    for label in [0, 1]:
        mask = df["isFraud"] == label
        sns.kdeplot(
            log_amt[mask].dropna(),
            ax=ax1,
            label=fraud_labels[label],
            color=fraud_palette[label],
            fill=True, alpha=0.35, linewidth=2,
        )
    ax1.set_xlabel("log1p(TransactionAmt)", color="#aaa")
    ax1.set_ylabel("Density", color="#aaa")
    ax1.set_title("KDE — Log Amount by Class", color="white", fontsize=13)
    ax1.legend(labelcolor="white")

    # ── Right: box plot (log-scale y-axis) ───────────────────────────────────
    ax2 = axes[1]
    legit_amt = df.loc[df["isFraud"] == 0, "TransactionAmt"].dropna()
    fraud_amt = df.loc[df["isFraud"] == 1, "TransactionAmt"].dropna()
    bp = ax2.boxplot(
        [legit_amt, fraud_amt],
        labels=["Non-Fraud", "Fraud"],
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.3},
    )
    for patch, color in zip(bp["boxes"], ["#4cc9f0", "#f72585"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_yscale("log")
    ax2.set_ylabel("TransactionAmt (log scale)", color="#aaa")
    ax2.set_title("Box Plot — Log Scale", color="white", fontsize=13)
    ax2.tick_params(colors="#aaa")

    plt.tight_layout()
    plt.savefig("outputs/transaction_amt_distribution.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()


plot_transaction_amt(df)
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11 – Markdown: correlation
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## Step 5: Correlation Heatmap — Top Features vs. isFraud

### Technical Rationale
Before investing in complex feature engineering, we compute **Pearson  
correlations** between every numeric feature and `isFraud`.  
This lightweight analysis serves three purposes:

1. **Feature ranking** — identifies the strongest linear predictors,  
   giving us a high-quality baseline feature set without any ML overhead  
2. **Multicollinearity detection** — the heatmap exposes clusters of  
   highly correlated features that could cause instability in logistic  
   regression or inflate coefficient variance  
3. **Domain expert communication** — correlation bar charts are intuitive  
   to stakeholders who do not have an ML background  

> **Limitation:** Pearson only captures linear associations.  
> Non-linear patterns require mutual information or model-based importance  
> (Shapley values) — addressed in Task 3.
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12 – plot_top_correlations
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code("""\
def plot_top_correlations(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    top_n: int = 20,
) -> pd.Series:
    \"\"\"Compute and visualise features most correlated with the target.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset (numeric columns used).
    target_col : str
        Name of the binary target column.
    top_n : int
        Number of highest-correlated features to display.

    Returns
    -------
    pd.Series
        Signed correlations for the top-N features (descending |r|).
    \"\"\"
    numeric_df = df.select_dtypes(include=[np.number])

    # Full correlation with target, drop NaNs and self-correlation
    full_corr: pd.Series = (
        numeric_df.corr()[target_col]
        .drop(labels=[target_col], errors="ignore")
        .dropna()
    )

    # Rank by absolute correlation, select top_n
    top_idx = full_corr.abs().sort_values(ascending=False).head(top_n).index
    top_corr = full_corr[top_idx]

    # Sub-matrix for heatmap
    cols_for_heatmap = top_idx.tolist() + [target_col]
    corr_matrix = numeric_df[cols_for_heatmap].corr()

    fig, axes = plt.subplots(
        1, 2, figsize=(20, 9),
        gridspec_kw={"width_ratios": [1, 2.2]},
    )
    fig.suptitle(
        f"Top {top_n} Features Correlated with {target_col}",
        fontsize=16, fontweight="bold", color="white",
    )

    # ── Left: signed correlation bar ─────────────────────────────────────────
    ax1 = axes[0]
    bar_colors = ["#f72585" if v > 0 else "#4cc9f0" for v in top_corr.values[::-1]]
    ax1.barh(top_corr.index[::-1], top_corr.values[::-1],
             color=bar_colors, edgecolor="none")
    ax1.axvline(0, color="white", linewidth=0.8)
    ax1.set_xlabel(f"Pearson Correlation with {target_col}", color="#aaa")
    ax1.set_title("Signed Correlation", color="white", fontsize=12)
    ax1.tick_params(colors="#aaa", labelsize=8)

    # ── Right: correlation heatmap ────────────────────────────────────────────
    ax2 = axes[1]
    mask = np.eye(len(corr_matrix), dtype=bool)          # hide diagonal
    sns.heatmap(
        corr_matrix,
        ax=ax2,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.4,
        linecolor="#0f0f1a",
        cbar_kws={"shrink": 0.8},
        mask=mask,
    )
    ax2.set_title(
        f"Correlation Matrix — Top {top_n} Features",
        color="white", fontsize=12,
    )
    ax2.tick_params(colors="#aaa", labelsize=7)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax2.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.show()

    return top_corr


top_features = plot_top_correlations(df, target_col="isFraud", top_n=20)
print("\\nTop 10 most correlated features with isFraud:")
print(top_features.head(10).to_string())
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13 – Summary & handoff
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## Task 1 — Summary & Handoff to Task 2

| Step | Key Outcome |
|---|---|
| Memory Optimisation | ~60 % RAM reduction via dtype downcasting |
| Dataset Shape | 590,540 rows x 434 columns after merge |
| Fraud Rate | ~3.5 % — severe class imbalance confirmed |
| Missing Value Audit | Columns >50 % missing flagged in `cols_to_drop` |
| Amount Distribution | Bimodal fraud pattern visible on log scale |
| Top Correlations | `top_features` Series ready for baseline modelling |

### Artefacts Produced

| Artefact | Path |
|---|---|
| Class distribution chart | `outputs/class_distribution.png` |
| Missing value chart | `outputs/missing_values.png` |
| Amount distribution chart | `outputs/transaction_amt_distribution.png` |
| Correlation heatmap | `outputs/correlation_heatmap.png` |

### Task 2 Input Variables
- `df` — memory-optimised merged DataFrame  
- `cols_to_drop` — columns earmarked for removal  
- `top_features` — ranked feature list for baseline model selection  
"""))

# ── Write notebook ─────────────────────────────────────────────────────────────
nb.cells = cells
output_path = "analysis.ipynb"
nbf.write(nb, output_path)
print(f"Notebook written to: {output_path}")
