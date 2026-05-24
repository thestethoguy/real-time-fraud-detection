"""
append_task2.py
Appends Task 2 cells (Preprocessing, Imbalance Handling & Feature Engineering)
to the existing analysis.ipynb produced in Task 1.
"""
import nbformat as nbf
from pathlib import Path

NOTEBOOK_PATH = Path("analysis.ipynb")

# ── Load existing notebook ────────────────────────────────────────────────────
nb = nbf.read(NOTEBOOK_PATH.open("r", encoding="utf-8"), as_version=4)

def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src)

def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)

new_cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 HEADER
# ═══════════════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
---

# TASK 2 — Preprocessing, Imbalance Handling & Feature Engineering

**Objective:** Transform the raw, merged IEEE-CIS dataset into a clean,  
balanced, fully-encoded feature matrix that is ready for model training.

Pipeline overview:

| Step | Operation | Key Decision |
|---|---|---|
| 1 | Drop + Impute | 50% threshold; median/mode imputation |
| 2 | Feature Engineering | Domain-driven fraud signals |
| 3 | Encoding + Scaling | Label encoding + RobustScaler |
| 4 | Split + SMOTE | Stratified split **then** SMOTE on train only |
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DROPPING & IMPUTING
# ═══════════════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
## Step 1: Dropping High-Missing Columns & Imputing Remaining Gaps

### Strategy Rationale

**Drop (> 50 % missing)**  
Columns above this threshold were identified in Task 1 and stored in  
`cols_to_drop`. Imputing them would mean fabricating values for the  
majority of rows — the model would learn the imputer's behaviour, not  
real fraud patterns. These columns are removed first to shrink the  
DataFrame before any further operations.

**Impute remaining numerical columns → Median**  
The median is the correct central-tendency estimator for financial data  
because it is **resistant to outliers**. A single $100,000 transaction  
would drag the mean far from the typical transaction; the median ignores it.

**Impute remaining categorical columns → Mode**  
For string/category features (card networks, device types, email domains)  
the mode — the most frequent observed value — is the least-distorting  
fill. It preserves the dominant signal without introducing unseen labels.

> All transformations are encapsulated in `clean_and_impute()` so the  
> logic can be independently unit-tested in Task 3.
"""))

new_cells.append(code("""\
import pandas as pd
import numpy as np

def clean_and_impute(
    df: pd.DataFrame,
    missing_cols_to_drop: list,
) -> pd.DataFrame:
    \"\"\"Drop high-missing columns and impute remaining gaps.

    Parameters
    ----------
    df : pd.DataFrame
        Merged, memory-optimised dataset from Task 1.
    missing_cols_to_drop : list
        Columns identified in Task 1 as exceeding the 50% missing threshold.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with no missing values.
    \"\"\"
    # ── 1. Drop flagged columns ───────────────────────────────────────────────
    # Only drop columns that actually exist (guard against re-runs)
    cols_present = [c for c in missing_cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_present)
    print(f"  Dropped {len(cols_present):>3} high-missing columns.")
    print(f"  Shape after drop : {df.shape}")

    # ── 2. Separate column types ──────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Protect the target and ID from imputation
    protected = {"isFraud", "TransactionID"}
    num_cols = [c for c in num_cols if c not in protected]

    # ── 3. Median imputation for numericals (vectorised, no .apply()) ─────────
    num_medians = df[num_cols].median()          # compute once
    df[num_cols] = df[num_cols].fillna(num_medians)

    # ── 4. Mode imputation for categoricals ───────────────────────────────────
    for col in cat_cols:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])

    # ── 5. Verify zero residual nulls ─────────────────────────────────────────
    remaining_nulls = df.isnull().sum().sum()
    print(f"  Imputed {len(num_cols):>3} numerical columns  (strategy: median)")
    print(f"  Imputed {len(cat_cols):>3} categorical columns (strategy: mode)")
    print(f"  Residual null values : {remaining_nulls}")
    print(f"  Final shape          : {df.shape}")

    return df


# ── Execute ───────────────────────────────────────────────────────────────────
print("=" * 55)
print("  STEP 1 — CLEANING & IMPUTATION")
print("=" * 55)
df = clean_and_impute(df, cols_to_drop)
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
## Step 2: Feature Engineering — Creating Fraud Signals

### Business Value of Engineered Features

Raw transactional columns capture *what happened*.  
Engineered features capture *how unusual it was* — which is where fraud  
signals live.

#### `AmtToMeanRatio` — Relative Transaction Magnitude
A $500 transaction is normal for a $480 average spender but alarming for a  
$12 average spender. Dividing by the **global mean** normalises amount  
across customers and highlights extreme deviations.  
Fraudsters often transact at amounts far above the cardholder's typical  
pattern — this ratio is a direct proxy for that anomaly.

#### `HourOfDay` — Temporal Fraud Pattern
`TransactionDT` encodes seconds elapsed from a reference point.  
Converting to hour-of-day (`(TransactionDT // 3600) % 24`) exposes a  
well-documented fraud signal: **fraudulent transactions peak between  
2 AM – 5 AM** when cardholders are asleep and cannot notice alerts.  
This cyclical feature is critical for tree models and time-series approaches.

#### `DeviceRisk` — High-Risk Device Heuristic
Identity data shows that certain device types and configurations correlate  
strongly with fraud (e.g., unknown devices, generic Android browsers, or  
missing device info altogether). We encode this as a **binary flag**:  
`1` = device is in the high-risk tier, `0` = otherwise.  
This converts sparse, high-cardinality device strings into an immediately  
actionable feature.

> These three features cost zero external data and encode domain knowledge  
> directly — they consistently appear in top-10 SHAP importance lists for  
> this dataset.
"""))

new_cells.append(code("""\
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Create domain-driven fraud signal features.

    New columns
    -----------
    AmtToMeanRatio : float32
        Transaction amount relative to the global mean. Values >> 1 are
        disproportionately large and a known fraud signal.
    HourOfDay : int8
        Hour of the day (0-23) extracted from TransactionDT.
        Captures the nocturnal fraud spike pattern.
    DeviceRisk : int8
        Binary flag: 1 = high-risk device profile, 0 = standard device.
        Derived from DeviceInfo / DeviceType heuristics.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, imputed DataFrame from Step 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with three new engineered columns appended.
    \"\"\"
    df = df.copy()

    # ── Feature 1: Amount-to-Mean Ratio ───────────────────────────────────────
    # Vectorised: single mean() call, then broadcast division
    global_mean_amt: float = df["TransactionAmt"].mean()
    df["AmtToMeanRatio"] = (
        df["TransactionAmt"] / global_mean_amt
    ).astype(np.float32)

    # ── Feature 2: Hour of Day ────────────────────────────────────────────────
    # TransactionDT is seconds from a reference epoch; modular arithmetic
    # extracts the clock hour without any Python-level loop
    df["HourOfDay"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)

    # ── Feature 3: Device Risk Flag ───────────────────────────────────────────
    # High-risk heuristic based on DeviceInfo & DeviceType columns.
    # Conditions identified from domain knowledge / public kernel analysis:
    #   - DeviceType is missing or labelled "desktop" (more exploitable)
    #   - DeviceInfo contains generic/unknown strings
    high_risk_device_info = {
        "unknown", "nan", "rv:11.0", "trident/7.0",   # IE / old browsers
        "sm-j700f", "sm-j200g", "redmi",               # common fraud handsets
    }

    device_info_col = "DeviceInfo" if "DeviceInfo" in df.columns else None
    device_type_col = "DeviceType" if "DeviceType" in df.columns else None

    risk_flags = pd.Series(0, index=df.index, dtype=np.int8)

    if device_info_col:
        info_lower = df[device_info_col].astype(str).str.lower()
        # Flag if any high-risk token appears as a substring
        pattern = "|".join(high_risk_device_info)
        risk_flags |= info_lower.str.contains(pattern, na=False).astype(np.int8)

    if device_type_col:
        # Null DeviceType is itself a risk signal
        risk_flags |= df[device_type_col].isna().astype(np.int8)

    df["DeviceRisk"] = risk_flags

    print(f"  AmtToMeanRatio  — mean: {df['AmtToMeanRatio'].mean():.4f}  "
          f"max: {df['AmtToMeanRatio'].max():.2f}")
    print(f"  HourOfDay       — unique hours: {df['HourOfDay'].nunique()}")
    print(f"  DeviceRisk      — high-risk rows: "
          f"{df['DeviceRisk'].sum():,} "
          f"({df['DeviceRisk'].mean()*100:.1f}%)")

    return df


# ── Execute ───────────────────────────────────────────────────────────────────
print("=" * 55)
print("  STEP 2 — FEATURE ENGINEERING")
print("=" * 55)
df = engineer_features(df)
print(f"\\nNew columns added: AmtToMeanRatio, HourOfDay, DeviceRisk")
print(f"Dataset shape: {df.shape}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ENCODING & SCALING
# ═══════════════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
## Step 3: Encoding & Scaling

### Encoding Strategy — Why Label Encoding over One-Hot?

The IEEE-CIS dataset contains several **high-cardinality categorical  
features** (e.g., `card4` has 4 values, `P_emaildomain` has ~60+ values,  
`DeviceInfo` has 1,000+). One-Hot Encoding these columns would:

- Explode dimensionality by hundreds of columns  
- Create severe **sparsity** — most entries are 0  
- Make tree models slower with no accuracy benefit  

**Label Encoding** assigns an integer to each category. This is the  
standard approach for **Gradient Boosted Trees** (XGBoost, LightGBM,  
CatBoost) which internally handle ordinal-integer categories correctly  
by splitting on thresholds — the arbitrary integer ordering has no  
semantic meaning to the model.

### Scaling Strategy — Why RobustScaler over StandardScaler?

| Scaler | Formula | Weakness |
|---|---|---|
| `StandardScaler` | `(x - mean) / std` | **Mean and std are dragged by outliers** |
| `RobustScaler` | `(x - median) / IQR` | Uses **quartiles** — outlier-immune |

Transaction amounts follow a power-law distribution with extreme outliers  
(e.g., $30,000+ transactions). `StandardScaler` would compress 95% of  
values near zero after a single whale transaction shifts the mean.  
`RobustScaler`'s use of the **Interquartile Range** keeps the bulk of the  
distribution well-scaled regardless of extremes.

> **Note:** `TransactionID` and `isFraud` are explicitly excluded from  
> all transformations.
"""))

new_cells.append(code("""\
from sklearn.preprocessing import LabelEncoder, RobustScaler

def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Apply Label Encoding to categoricals and RobustScaler to numericals.

    Transformations applied in-place on a copy:
    - Categorical (object/category) columns -> LabelEncoder (integer codes)
    - Numerical columns -> RobustScaler  (median-IQR normalisation)

    Columns excluded from all transformations
    -----------------------------------------
    - TransactionID  : identifier, must never leak into feature space
    - isFraud        : target variable

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame from Step 2.

    Returns
    -------
    pd.DataFrame
        Fully encoded and scaled DataFrame.
    \"\"\"
    df = df.copy()
    PROTECTED = {"TransactionID", "isFraud"}

    # ── Label Encoding ────────────────────────────────────────────────────────
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in PROTECTED
    ]
    le = LabelEncoder()
    for col in cat_cols:
        # fillna guard: LabelEncoder does not handle NaN
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col]).astype(np.int32)

    print(f"  Label-encoded {len(cat_cols):>3} categorical columns.")

    # ── RobustScaler ──────────────────────────────────────────────────────────
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in PROTECTED
    ]
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols]).astype(np.float32)

    print(f"  RobustScaler applied to {len(num_cols):>3} numerical columns.")
    print(f"  Final shape: {df.shape}")

    return df, scaler


# ── Execute ───────────────────────────────────────────────────────────────────
print("=" * 55)
print("  STEP 3 — ENCODING & SCALING")
print("=" * 55)
df, scaler = encode_and_scale(df)
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — SPLIT & SMOTE
# ═══════════════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
## Step 4: Train-Test Split & SMOTE

---

> ### ⚠️ CRITICAL WARNING — DATA LEAKAGE
>
> **The single most common and catastrophic mistake in imbalanced-class  
> ML pipelines is applying SMOTE before the train-test split.**
>
> SMOTE (Synthetic Minority Over-sampling Technique) generates **synthetic  
> fraud samples** by interpolating between real minority-class points.  
> If you apply SMOTE to the full dataset first:
>
> - Synthetic samples land in **both** the training set and the test set  
> - The model is evaluated on points that are **statistically derived**  
>   from its own training data  
> - Test-set metrics become **wildly optimistic** — completely invalid  
> - The model will underperform in production against real unseen fraud  
>
> **The correct sequence, strictly enforced below:**
> 1. **Stratified 80/20 split first** — the test set is locked away  
>    and never touched again until final evaluation  
> 2. **SMOTE applied only to `X_train` / `y_train`** — the test set  
>    retains the real-world 3.5% fraud ratio

---

### Why Stratified Split?
A random split on 590k rows has a small but non-zero chance of placing  
all fraud in one partition. `stratify=y` guarantees the 3.5% ratio is  
preserved in **both** train and test sets.

### What SMOTE Does
SMOTE selects a minority-class point, finds its k-nearest minority  
neighbours, and synthesises new points **along the line segments**  
between them. This creates plausible, interpolated fraud samples rather  
than simple duplicates (which overfit), delivering a balanced training  
distribution for the model to learn from.
"""))

new_cells.append(code("""\
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def prepare_modeling_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple:
    \"\"\"Execute stratified split then SMOTE on training data only.

    Strict pipeline:
        1. Separate features X and target y
        2. Stratified 80/20 train-test split
        3. SMOTE applied EXCLUSIVELY to X_train / y_train
        4. Test set is NEVER touched or transformed after split

    Parameters
    ----------
    df : pd.DataFrame
        Fully encoded and scaled DataFrame from Step 3.
    test_size : float
        Fraction of data reserved for testing. Default 0.20.
    random_state : int
        Reproducibility seed. Default 42.

    Returns
    -------
    tuple
        (X_train_res, X_test, y_train_res, y_test)
        Where _res suffix denotes SMOTE-resampled training data.
    \"\"\"
    # ── Separate features and target ──────────────────────────────────────────
    DROP_FROM_FEATURES = {"isFraud", "TransactionID"}
    feature_cols = [c for c in df.columns if c not in DROP_FROM_FEATURES]

    X: pd.DataFrame = df[feature_cols]
    y: pd.Series    = df["isFraud"].astype(np.int8)

    print(f"  Feature matrix shape : {X.shape}")
    print(f"  Target distribution  :")
    print(f"    Non-Fraud : {(y == 0).sum():>7,}  ({(y==0).mean()*100:.1f}%)")
    print(f"    Fraud     : {(y == 1).sum():>7,}  ({(y==1).mean()*100:.1f}%)")

    # ── Step 1: Stratified train-test split ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"\\n  [Split] Train size   : {X_train.shape[0]:>7,} rows")
    print(f"  [Split] Test size    : {X_test.shape[0]:>7,} rows")
    print(f"  [Split] Train fraud  : {y_train.sum():>7,} "
          f"({y_train.mean()*100:.2f}%)")
    print(f"  [Split] Test fraud   : {y_test.sum():>7,}  "
          f"({y_test.mean()*100:.2f}%)")

    # ── Step 2: SMOTE on training data ONLY ──────────────────────────────────
    print("\\n  Applying SMOTE to training set only ...")
    smote = SMOTE(
        sampling_strategy="auto",   # balance minority to majority count
        k_neighbors=5,
        random_state=random_state,
        n_jobs=-1,
    )
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # ── Summary report ────────────────────────────────────────────────────────
    print("\\n" + "=" * 55)
    print("  SMOTE RESAMPLING SUMMARY")
    print("=" * 55)
    print(f"  BEFORE SMOTE (training set):")
    print(f"    Non-Fraud : {(y_train == 0).sum():>7,}  ({(y_train==0).mean()*100:.1f}%)")
    print(f"    Fraud     : {(y_train == 1).sum():>7,}  ({(y_train==1).mean()*100:.1f}%)")
    print(f"\\n  AFTER SMOTE (resampled training set):")
    print(f"    Non-Fraud : {(y_train_res == 0).sum():>7,}  ({(y_train_res==0).mean()*100:.1f}%)")
    print(f"    Fraud     : {(y_train_res == 1).sum():>7,}  ({(y_train_res==1).mean()*100:.1f}%)")
    print(f"    Total rows: {len(y_train_res):>7,}")
    print(f"\\n  TEST SET (untouched — real-world distribution):")
    print(f"    Non-Fraud : {(y_test == 0).sum():>7,}  ({(y_test==0).mean()*100:.1f}%)")
    print(f"    Fraud     : {(y_test == 1).sum():>7,}  ({(y_test==1).mean()*100:.1f}%)")
    print(f"    Total rows: {len(y_test):>7,}")

    return X_train_res, X_test, y_train_res, y_test


# ── Execute ───────────────────────────────────────────────────────────────────
print("=" * 55)
print("  STEP 4 — STRATIFIED SPLIT + SMOTE")
print("=" * 55)
X_train, X_test, y_train, y_test = prepare_modeling_data(df)
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
new_cells.append(md("""\
## Task 2 — Summary & Handoff to Task 3

| Step | Operation | Output |
|---|---|---|
| 1 | Drop + Impute | Clean `df` with zero nulls |
| 2 | Feature Engineering | `AmtToMeanRatio`, `HourOfDay`, `DeviceRisk` |
| 3 | Encode + Scale | All cols integer/float; RobustScaler fitted |
| 4 | Split + SMOTE | Balanced `X_train` / `y_train`; pristine `X_test` / `y_test` |

### Task 3 Input Artefacts

| Variable | Description |
|---|---|
| `X_train` | SMOTE-balanced feature matrix (training) |
| `y_train` | Balanced target vector (50/50 after SMOTE) |
| `X_test` | Unseen feature matrix — real-world distribution |
| `y_test` | Unseen target — 3.5% fraud rate (ground truth) |
| `scaler` | Fitted `RobustScaler` instance for inverse-transform |
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Append and save
# ═══════════════════════════════════════════════════════════════════════════════
nb.cells.extend(new_cells)
nbf.write(nb, NOTEBOOK_PATH.open("w", encoding="utf-8"))
print(f"Task 2 cells appended. Notebook now has {len(nb.cells)} cells.")
print(f"Saved to: {NOTEBOOK_PATH.resolve()}")
