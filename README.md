# 🛡️ Real-Time Fraud Detection System
### IEEE-CIS Fraud Detection — Capstone Project

**Author:** Aman Aaryan |  
**Dataset:** IEEE-CIS Fraud Detection (Kaggle)  
**Live Dashboard URL:** https://real-time-fraud-detection-psfsa6cwf9svdd7auvjz8n.streamlit.app

---

## 📋 Project Overview

A production-grade, end-to-end machine learning pipeline for real-time
financial fraud detection, built on the IEEE-CIS Fraud Detection dataset
(590,540 transactions, 434 features after merge).

| Task | Focus | Key Output |
|------|-------|-----------|
| Task 1 | EDA & Memory Optimization | Merged dataset, 4 diagnostic plots |
| Task 2 | Preprocessing & Feature Engineering | SMOTE-balanced training set |
| Task 3 | Model Training & Threshold Optimization | LightGBM, XGBoost, IsoForest |
| Task 4 | Explainable AI (SHAP) | Waterfall + dependence plots |
| Task 5 | Risk Segmentation | 3-tier triage queue |
| Task 6/7 | Streamlit Dashboard | Multi-page FraudOps UI |
| Task 8 | Business Report | summary.docx |

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Processing | `pandas`, `numpy` |
| Machine Learning | `scikit-learn`, `lightgbm`, `xgboost` |
| Imbalance Handling | `imbalanced-learn` (SMOTE) |
| Explainability | `shap` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Dashboard | `streamlit` |
| Serialization | `joblib` |

---

## 🚀 Installation & Setup

```bash
# 1. Clone / extract the project
cd FraudDetection_AmanAaryan

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Place dataset files
#    Download from https://www.kaggle.com/c/ieee-fraud-detection
#    Place train_transaction.csv and train_identity.csv inside data/

# 5. Run the notebook end-to-end
jupyter notebook analysis.ipynb

# 6. Launch the dashboard
cd dashboard
streamlit run app.py
```

---

## 📁 Project Structure

```
FraudDetection_AmanAaryan/
├── data/
│   ├── train_transaction.csv
│   └── train_identity.csv
├── dashboard/
│   ├── app.py                    ← Streamlit multi-page dashboard
│   ├── model.pkl                 ← Serialized LightGBM model
│   └── sample_transactions.csv   ← Dashboard data sample
├── outputs/                      ← All generated plots (16 files)
├── analysis.ipynb                ← Master notebook (Tasks 1–6)
├── requirements.txt
├── summary.docx                  ← Business insights report (Task 8)
└── README.md
```

---

## 📊 Model Performance Summary

| Model | ROC-AUC | PR-AUC | Recall |
|-------|---------|--------|--------|
| LightGBM (Tuned) | ~0.94 | ~0.72 | ~0.85 |
| XGBoost | ~0.93 | ~0.70 | ~0.83 |
| Isolation Forest | ~0.76 | ~0.12 | ~0.65 |

> **North Star Metric:** PR-AUC (Precision-Recall AUC)  
> Accuracy is misleading on a 3.5%-fraud dataset. PR-AUC measures
> real performance in the minority class space.

---

## ⚖️ License
For academic/capstone submission purposes only.
