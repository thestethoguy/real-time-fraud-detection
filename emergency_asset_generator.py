"""
emergency_asset_generator.py
Generates dashboard/model.pkl and dashboard/sample_transactions.csv
using only 50,000 rows to stay within local RAM limits.
"""
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

os.makedirs("dashboard", exist_ok=True)

# ── Step 1: Low-memory load ───────────────────────────────────────────────────
print("[1/6] Loading 50,000 rows from train_transaction.csv ...")
df_trans = pd.read_csv("data/train_transaction.csv", nrows=50000)
print(f"      Transaction shape: {df_trans.shape}")

print("[1/6] Loading train_identity.csv ...")
df_id = pd.read_csv("data/train_identity.csv")
print(f"      Identity shape   : {df_id.shape}")

# ── Step 2: Merge ─────────────────────────────────────────────────────────────
print("[2/6] Merging on TransactionID (left join) ...")
df = df_trans.merge(df_id, on="TransactionID", how="left")
del df_trans, df_id
print(f"      Merged shape: {df.shape}")

# ── Step 3: Quick prep ────────────────────────────────────────────────────────
print("[3/6] Filling NAs and encoding categoricals ...")

# Separate target & ID before any transforms
y = df["isFraud"].astype(np.int8)
drop_cols = ["isFraud", "TransactionID"]

# Fill NAs
num_cols = df.select_dtypes(include=[np.number]).columns.difference(drop_cols)
cat_cols = df.select_dtypes(include=["object"]).columns

df[num_cols] = df[num_cols].fillna(0)
df[cat_cols] = df[cat_cols].fillna("missing")

# Label encode all categoricals (in-place, single pass per column)
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str)).astype(np.int32)

# Feature matrix
X = df.drop(columns=drop_cols)
print(f"      Feature matrix: {X.shape}")
print(f"      Fraud rate    : {y.mean()*100:.2f}%")

# ── Step 4: Train LightGBM ────────────────────────────────────────────────────
print("[4/6] Training LGBMClassifier ...")
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=63,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
model.fit(X, y)
print("      Training complete.")

# ── Step 5: Export model ──────────────────────────────────────────────────────
model_path = Path("dashboard/model.pkl")
joblib.dump(model, model_path, compress=3)
print(f"[5/6] Model saved -> {model_path.resolve()}  "
      f"({model_path.stat().st_size / 1e6:.1f} MB)")

# ── Step 6: Build & export sample CSV ─────────────────────────────────────────
print("[6/6] Building 2,000-row sample with predictions ...")

SAMPLE_N = 2000
sample_X = X.sample(n=SAMPLE_N, random_state=42).copy()
sample_y = y.loc[sample_X.index].copy()

probs = model.predict_proba(sample_X)[:, 1]

sample_out = sample_X.copy()
sample_out["TransactionID"]  = df.loc[sample_X.index, "TransactionID"] \
                                   if "TransactionID" in df.columns \
                                   else sample_X.index + 3_663_549
sample_out["TrueLabel"]  = sample_y.values
sample_out["FraudProb"]  = probs

# Risk tier via np.select (vectorised)
conditions = [probs >= 0.75, (probs >= 0.40) & (probs < 0.75)]
choices    = ["Critical Risk", "Suspicious"]
sample_out["Risk_Tier"]  = np.select(conditions, choices, default="Clear")

# Ensure TransactionAmt is present for the dashboard amount column
if "TransactionAmt" in sample_X.columns:
    sample_out["TransactionAmt"] = sample_X["TransactionAmt"].values

# Add engineered columns the dashboard expects
sample_out["AmtToMeanRatio"] = (
    sample_X["TransactionAmt"] / sample_X["TransactionAmt"].mean()
    if "TransactionAmt" in sample_X.columns
    else 1.0
)
if "TransactionDT" in sample_X.columns:
    sample_out["HourOfDay"] = ((sample_X["TransactionDT"] // 3600) % 24).astype(np.int8)

csv_path = Path("dashboard/sample_transactions.csv")
sample_out.to_csv(csv_path, index=False)
print(f"      CSV saved  -> {csv_path.resolve()}  ({csv_path.stat().st_size / 1e6:.1f} MB)")

# ── Verify ────────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  ASSET GENERATION COMPLETE")
print("=" * 55)
print(f"  dashboard/model.pkl              : {'OK' if model_path.exists() else 'MISSING'}")
print(f"  dashboard/sample_transactions.csv: {'OK' if csv_path.exists() else 'MISSING'}")
print()
print("  Risk tier breakdown in sample:")
print(sample_out["Risk_Tier"].value_counts().to_string())
print("=" * 55)
