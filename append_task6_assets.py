"""append_task6_assets.py — Appends asset-export cells to analysis.ipynb (Tasks 6/7)"""
import nbformat as nbf
from pathlib import Path

NOTEBOOK_PATH = Path("analysis.ipynb")
nb = nbf.read(NOTEBOOK_PATH.open("r", encoding="utf-8"), as_version=4)

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

cells = []

cells.append(md("""\
---

# TASK 6 — Dashboard Asset Extraction

**Objective:** Serialize the trained model and a representative data sample
so the Streamlit dashboard (`dashboard/app.py`) can load them without
re-running the entire ML pipeline.

| Asset | Path | Description |
|---|---|---|
| `model.pkl` | `dashboard/model.pkl` | Tuned LightGBM — joblib serialized |
| `sample_transactions.csv` | `dashboard/sample_transactions.csv` | 2,000-row enriched sample for UI |
"""))

cells.append(code("""\
import os
import joblib
import pandas as pd
import numpy as np

os.makedirs("dashboard", exist_ok=True)

# ── 1. Save the tuned LightGBM model ─────────────────────────────────────────
model_path = "dashboard/model.pkl"
joblib.dump(best_lgbm_tuned, model_path, compress=3)
print(f"Model saved : {model_path}")

# ── 2. Build enriched sample for the dashboard ────────────────────────────────
# Start from segmented_df which already has FraudProb and Risk_Tier columns
SAMPLE_N = 2000
sample = segmented_df.sample(n=min(SAMPLE_N, len(segmented_df)),
                              random_state=42).copy()

# Ensure TransactionAmt exists (may have been scaled — add raw if possible)
if "TransactionAmt" not in sample.columns and "AmtToMeanRatio" in sample.columns:
    # Approximate raw amount from ratio (for display only)
    global_mean = 151.0   # approximate IEEE-CIS global mean
    sample["TransactionAmt"] = (sample["AmtToMeanRatio"] * global_mean).round(2)

# Add a synthetic TransactionID column for the SHAP Explainer page
# (The real TransactionID was dropped during encoding — reconstruct positional IDs)
sample = sample.reset_index(drop=True)
sample.insert(0, "TransactionID", sample.index + 3_663_549)  # IEEE-CIS start ID

# Keep Risk_Tier as string (Categorical → str for CSV portability)
sample["Risk_Tier"] = sample["Risk_Tier"].astype(str)

csv_path = "dashboard/sample_transactions.csv"
sample.to_csv(csv_path, index=False)
print(f"Sample CSV  : {csv_path}  ({len(sample):,} rows x {sample.shape[1]} cols)")

# ── 3. Quick sanity check ─────────────────────────────────────────────────────
print("\\nRisk tier distribution in sample:")
print(sample["Risk_Tier"].value_counts().to_string())
print("\\nColumns available to dashboard:")
print(list(sample.columns[:15]), "...")
"""))

cells.append(md("""\
## Assets Saved Successfully

The `dashboard/` directory now contains everything needed to run the
Streamlit application independently of this notebook:

```
dashboard/
├── model.pkl                  ← Tuned LightGBM (joblib)
├── sample_transactions.csv    ← 2,000-row enriched sample
└── app.py                     ← Multi-page Streamlit application
```

### Running the Dashboard

```bash
cd dashboard
streamlit run app.py
```

The app will launch at `http://localhost:8501` with three pages:
- **Overview** — Executive KPI summary + Plotly charts
- **Transaction Explorer** — Filterable triage queue
- **SHAP Explainer** — Per-transaction AI explanation
"""))

nb.cells.extend(cells)
nbf.write(nb, NOTEBOOK_PATH.open("w", encoding="utf-8"))
print(f"Task 6 asset cells appended. Notebook now has {len(nb.cells)} cells.")
