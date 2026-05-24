"""
dashboard/app.py
Real-Time Fraud Operations Dashboard — Streamlit Multi-Page Application
Capstone Project: IEEE-CIS Fraud Detection System
Author: Aman Aaryan | Role: Lead ML Engineer
"""

# ── Standard imports ──────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib
import streamlit as st

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # non-interactive backend required for st.pyplot()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FraudOps System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Dark gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 60%, #16213e 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
    border-right: 1px solid #2a2a3e;
}
/* KPI metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 18px 20px;
    backdrop-filter: blur(6px);
}
/* Section headers */
h1, h2, h3 { color: #e0e0e0 !important; }
/* Dataframe styling */
[data-testid="stDataFrame"] { border-radius: 10px; }
/* Sidebar selectbox */
.stSelectbox label { color: #aaa !important; }
/* Success / warning boxes */
.stSuccess, .stWarning, .stInfo { border-radius: 10px; }
/* Hide Streamlit default hamburger + footer */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
TIER_COLORS = {
    "Critical Risk": "#f72585",
    "Suspicious":    "#ffd166",
    "Clear":         "#4cc9f0",
}
TIER_ORDER = ["Critical Risk", "Suspicious", "Clear"]

# Absolute paths anchored to this file's directory — works regardless of CWD.
# On Streamlit Cloud the working directory is the repo root, not dashboard/,
# so bare filenames would silently fail. __file__ always resolves correctly.
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(_DASHBOARD_DIR, "model.pkl")
DATA_PATH      = os.path.join(_DASHBOARD_DIR, "sample_transactions.csv")

# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS  (load once, reuse across reruns)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading fraud detection model…")
def load_model():
    """Load serialized LightGBM model via joblib."""
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading transaction sample…")
def load_data() -> pd.DataFrame:
    """Load the 2,000-row enriched transaction sample."""
    df = pd.read_csv(DATA_PATH)
    df["Risk_Tier"] = pd.Categorical(
        df["Risk_Tier"], categories=TIER_ORDER, ordered=True
    )
    return df


# ── Load assets ───────────────────────────────────────────────────────────────
try:
    model = load_model()
    df    = load_data()
    assets_ok = True
except FileNotFoundError as exc:
    assets_ok = False
    missing = str(exc)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/security-shield-green.png",
        width=64,
    )
    st.markdown("## 🛡️ FraudOps System")
    st.markdown("*IEEE-CIS Real-Time Fraud Detection*")
    st.divider()

    page = st.selectbox(
        "Navigate to",
        options=["Overview", "Transaction Explorer", "SHAP Explainer"],
        index=0,
    )

    st.divider()
    st.markdown("**Model:** LightGBM (Tuned)")
    st.markdown("**Dataset:** IEEE-CIS Fraud")
    if assets_ok:
        st.success(f"✅ {len(df):,} transactions loaded")
    else:
        st.error("❌ Assets missing — run the notebook first")

# ── Guard: assets not loaded ──────────────────────────────────────────────────
if not assets_ok:
    st.error(f"**Asset loading failed:** `{missing}`")
    st.info(
        "Please run all cells in `analysis.ipynb` first to generate "
        "`dashboard/model.pkl` and `dashboard/sample_transactions.csv`."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    st.markdown("# 🛡️ FraudOps — Executive Overview")
    st.markdown("*Real-time risk intelligence dashboard · IEEE-CIS Fraud Detection System*")
    st.divider()

    # ── KPI Metrics Row ───────────────────────────────────────────────────────
    total_txn     = len(df)
    critical_cnt  = int((df["Risk_Tier"] == "Critical Risk").sum())
    suspicious_cnt = int((df["Risk_Tier"] == "Suspicious").sum())
    at_risk_cnt   = critical_cnt + suspicious_cnt

    amt_col = "TransactionAmt" if "TransactionAmt" in df.columns else "AmtToMeanRatio"
    revenue_at_risk = df.loc[
        df["Risk_Tier"].isin(["Critical Risk", "Suspicious"]), amt_col
    ].sum()

    fraud_rate = df["TrueLabel"].mean() * 100 if "TrueLabel" in df.columns else 3.5
    avg_prob   = df["FraudProb"].mean() * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📊 Total Transactions",  f"{total_txn:,}")
    k2.metric("🚨 Critical Risk",       f"{critical_cnt:,}",
              delta=f"{critical_cnt/total_txn*100:.1f}% of total",
              delta_color="inverse")
    k3.metric("⚠️ Suspicious",          f"{suspicious_cnt:,}",
              delta=f"{suspicious_cnt/total_txn*100:.1f}% of total",
              delta_color="inverse")
    k4.metric("💰 Revenue at Risk",     f"${revenue_at_risk:,.0f}")
    k5.metric("🎯 Avg Fraud Prob",      f"{avg_prob:.2f}%")

    st.divider()

    # ── Charts Row ────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        st.markdown("### Risk Tier Distribution")
        tier_counts = df["Risk_Tier"].value_counts().reindex(TIER_ORDER)
        fig_donut = go.Figure(go.Pie(
            labels=tier_counts.index.tolist(),
            values=tier_counts.values.tolist(),
            hole=0.55,
            marker=dict(
                colors=[TIER_COLORS[t] for t in tier_counts.index],
                line=dict(color="#0f0f1a", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=13, color="white"),
            hovertemplate="%{label}: %{value:,} transactions<br>%{percent}<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{total_txn:,}</b><br>Transactions",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="white"),
        )
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=380,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown("### Transaction Amount vs. Hour of Day")
        if "HourOfDay" in df.columns and amt_col in df.columns:
            fig_scatter = px.scatter(
                df.sample(min(1000, len(df)), random_state=42),
                x="HourOfDay",
                y=amt_col,
                color="FraudProb",
                color_continuous_scale="RdYlGn_r",
                size_max=8,
                opacity=0.65,
                labels={
                    "HourOfDay": "Hour of Day (0–23)",
                    amt_col:     "Transaction Amount ($)",
                    "FraudProb": "P(Fraud)",
                },
                hover_data={"Risk_Tier": True, "FraudProb": ":.3f"},
            )
            fig_scatter.update_layout(
                font=dict(color="white"),
                coloraxis_colorbar=dict(
                    title="P(Fraud)", tickfont=dict(color="white"),
                    titlefont=dict(color="white"),
                ),
                xaxis=dict(gridcolor="#2a2a3e", title_font=dict(color="#aaa")),
                yaxis=dict(gridcolor="#2a2a3e", title_font=dict(color="#aaa")),
                margin=dict(t=10, b=40, l=10, r=10),
                height=380,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("HourOfDay / TransactionAmt columns not found in sample.")

    st.divider()

    # ── Tier stats table ──────────────────────────────────────────────────────
    st.markdown("### Triage Queue Summary")
    tier_summary = (
        df.groupby("Risk_Tier", observed=True)
        .agg(
            Count=("FraudProb", "count"),
            Avg_Probability=("FraudProb", "mean"),
            Avg_Amount=(amt_col, "mean"),
        )
        .reindex(TIER_ORDER)
        .round(4)
    )
    tier_summary.index.name = "Risk Tier"
    st.dataframe(
        tier_summary.style
        .format({"Count": "{:,}", "Avg_Probability": "{:.4f}", "Avg_Amount": "${:,.2f}"})
        .background_gradient(cmap="RdYlGn_r", subset=["Avg_Probability"]),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: TRANSACTION EXPLORER — TRIAGE QUEUE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Transaction Explorer":

    st.markdown("# 🔍 Transaction Explorer")
    st.markdown("*Filter and sort the live triage queue. Critical Risk rows appear in red.*")
    st.divider()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filters")
        selected_tiers = st.multiselect(
            "Risk Tier",
            options=TIER_ORDER,
            default=TIER_ORDER,
        )

        amt_col = "TransactionAmt" if "TransactionAmt" in df.columns else "AmtToMeanRatio"
        amt_min = float(df[amt_col].min())
        amt_max = float(df[amt_col].max())
        amt_range = st.slider(
            f"{amt_col} Range ($)",
            min_value=amt_min,
            max_value=amt_max,
            value=(amt_min, amt_max),
            step=1.0,
        )

        prob_threshold = st.slider(
            "Min Fraud Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
        )

    # ── Apply filters (vectorised boolean mask) ───────────────────────────────
    mask = (
        df["Risk_Tier"].isin(selected_tiers) &
        df[amt_col].between(amt_range[0], amt_range[1]) &
        (df["FraudProb"] >= prob_threshold)
    )
    filtered = df[mask].copy()

    # ── Summary metrics ───────────────────────────────────────────────────────
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Transactions in View", f"{len(filtered):,}")
    mc2.metric("Critical Risk in View",
               f"{(filtered['Risk_Tier'] == 'Critical Risk').sum():,}")
    mc3.metric("Total Amount at Risk",
               f"${filtered[amt_col].sum():,.0f}")

    st.divider()

    # ── Display columns (prioritise the most useful) ──────────────────────────
    priority_cols = [
        "TransactionID", "Risk_Tier", "FraudProb", amt_col,
        "HourOfDay", "AmtToMeanRatio", "DeviceRisk", "TrueLabel",
    ]
    display_cols = [c for c in priority_cols if c in filtered.columns]
    display_df   = filtered[display_cols].sort_values("FraudProb", ascending=False)

    # ── Row highlighting via Pandas Styler ───────────────────────────────────
    def highlight_tier(row):
        tier = row.get("Risk_Tier", "")
        if tier == "Critical Risk":
            return ["background-color: rgba(247,37,133,0.15); color: #f72585"] * len(row)
        elif tier == "Suspicious":
            return ["background-color: rgba(255,209,102,0.10); color: #ffd166"] * len(row)
        return [""] * len(row)

    styled = (
        display_df.style
        .apply(highlight_tier, axis=1)
        .format({
            "FraudProb":    "{:.4f}",
            amt_col:        "${:,.2f}",
            "AmtToMeanRatio": "{:.3f}",
        }, na_rep="—")
    )

    st.dataframe(styled, use_container_width=True, height=520)

    st.caption(
        f"Showing {len(filtered):,} of {len(df):,} transactions "
        f"· Sorted by P(Fraud) descending"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SHAP EXPLAINER — INDIVIDUAL TRANSACTION AI EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "SHAP Explainer":

    st.markdown("# 🧠 SHAP Explainer")
    st.markdown(
        "*Select any transaction to understand **why** the model assigned "
        "its fraud probability. Required for GDPR Article 22 compliance.*"
    )
    st.divider()

    # ── Transaction selector ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Select Transaction")
        txn_ids = df["TransactionID"].astype(str).tolist() if "TransactionID" in df.columns \
                  else df.index.astype(str).tolist()

        selected_id = st.selectbox(
            "Transaction ID",
            options=txn_ids,
            index=0,
        )
        st.markdown("---")
        st.markdown("**Interpreting the Waterfall:**")
        st.markdown("- 🔴 Red bars → **push toward fraud**")
        st.markdown("- 🔵 Blue bars → **push away from fraud**")
        st.markdown("- Bottom value = model baseline (avg prediction)")
        st.markdown("- Top value = final P(fraud) for this transaction")

    # ── Isolate selected transaction ──────────────────────────────────────────
    if "TransactionID" in df.columns:
        row_mask = df["TransactionID"].astype(str) == selected_id
    else:
        row_mask = df.index.astype(str) == selected_id

    if not row_mask.any():
        st.error("Transaction not found in sample.")
        st.stop()

    txn_row = df[row_mask].iloc[[0]]

    # ── Drop non-feature columns before SHAP ─────────────────────────────────
    DROP_COLS = {"TransactionID", "FraudProb", "Risk_Tier", "TrueLabel"}
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X_row = txn_row[feature_cols].select_dtypes(include=[np.number])

    # ── Transaction summary card ──────────────────────────────────────────────
    tier    = str(txn_row["Risk_Tier"].iloc[0])
    prob    = float(txn_row["FraudProb"].iloc[0])
    amt_col = "TransactionAmt" if "TransactionAmt" in txn_row.columns else "AmtToMeanRatio"
    amount  = float(txn_row[amt_col].iloc[0]) if amt_col in txn_row.columns else 0.0

    tier_emoji = {"Critical Risk": "🚨", "Suspicious": "⚠️", "Clear": "✅"}.get(tier, "🔍")
    st.markdown(f"### {tier_emoji} Transaction `{selected_id}`")

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Risk Tier",          tier)
    sc2.metric("P(Fraud)",           f"{prob:.4f}")
    sc3.metric("Transaction Amount", f"${amount:,.2f}")

    st.divider()

    # ── Compute SHAP values ───────────────────────────────────────────────────
    with st.spinner("Computing SHAP values — this may take a few seconds…"):
        try:
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer.shap_values(X_row)

            # Handle list output from binary classifiers
            sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
            base_val = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )

            explanation = shap.Explanation(
                values=sv[0],
                base_values=base_val,
                data=X_row.values[0],
                feature_names=X_row.columns.tolist(),
            )
            shap_ok = True
        except Exception as exc:
            shap_ok = False
            shap_err = str(exc)

    if not shap_ok:
        st.error(f"SHAP computation failed: `{shap_err}`")
        st.stop()

    # ── Waterfall plot ────────────────────────────────────────────────────────
    st.markdown("#### Feature Contribution Waterfall")
    fig_wf, ax_wf = plt.subplots(figsize=(11, 6),
                                  facecolor="#1a1a2e")
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Transaction {selected_id}",
              color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_wf, use_container_width=True)
    plt.close(fig_wf)

    st.divider()

    # ── Plain-English Generator ───────────────────────────────────────────────
    st.markdown("#### 📝 Plain-English Explanation (Auto-Generated)")

    # Rank features by absolute SHAP contribution
    shap_series = pd.Series(sv[0], index=X_row.columns).abs().sort_values(ascending=False)
    top_feature  = shap_series.index[0]
    top_val_raw  = sv[0][X_row.columns.tolist().index(top_feature)]
    top_direction = "increased" if top_val_raw > 0 else "decreased"
    top_data_val  = float(X_row[top_feature].iloc[0])

    # Secondary feature
    sec_feature  = shap_series.index[1] if len(shap_series) > 1 else ""
    sec_val_raw  = sv[0][X_row.columns.tolist().index(sec_feature)] if sec_feature else 0
    sec_direction = "increased" if sec_val_raw > 0 else "decreased"

    # Hour context
    hour_context = ""
    if "HourOfDay" in X_row.columns:
        hour = int(X_row["HourOfDay"].iloc[0])
        if 1 <= hour <= 5:
            hour_context = f" The transaction occurred at **{hour:02d}:00**, during the high-risk overnight window (01:00–05:00)."
        else:
            hour_context = f" The transaction occurred at **{hour:02d}:00**, within normal business hours."

    explanation_text = (
        f"The model assigned this transaction a fraud probability of **{prob:.1%}** "
        f"and classified it as **{tier}**. "
        f"The most influential factor was `{top_feature}` (value: `{top_data_val:.3f}`), "
        f"which **{top_direction}** the fraud probability significantly."
    )
    if sec_feature:
        sec_data_val = float(X_row[sec_feature].iloc[0])
        explanation_text += (
            f" The second most important driver was `{sec_feature}` "
            f"(value: `{sec_data_val:.3f}`), which **{sec_direction}** the risk score."
        )
    explanation_text += hour_context

    # Choose alert level by tier
    if tier == "Critical Risk":
        st.error(f"🚨 **Critical Risk Detected**\n\n{explanation_text}\n\n"
                 f"**Recommended Action:** Auto-block and escalate to Tier-1 analyst within 1 hour.")
    elif tier == "Suspicious":
        st.warning(f"⚠️ **Suspicious Transaction**\n\n{explanation_text}\n\n"
                   f"**Recommended Action:** Send OTP verification to customer. "
                   f"Queue for analyst review within 24 hours.")
    else:
        st.success(f"✅ **Transaction Cleared**\n\n{explanation_text}\n\n"
                   f"**Recommended Action:** Auto-approve. No analyst review required.")

    # ── Top-5 SHAP feature table ──────────────────────────────────────────────
    st.divider()
    st.markdown("#### Top 5 Feature Contributions")
    top5 = pd.DataFrame({
        "Feature":    shap_series.head(5).index,
        "SHAP Value": sv[0][shap_series.head(5).index.map(
            lambda f: X_row.columns.tolist().index(f)
        )],
        "Feature Value": [float(X_row[f].iloc[0]) for f in shap_series.head(5).index],
    })
    top5["Direction"] = top5["SHAP Value"].apply(
        lambda v: "🔴 Toward Fraud" if v > 0 else "🔵 Away from Fraud"
    )
    st.dataframe(
        top5.style.format({"SHAP Value": "{:+.4f}", "Feature Value": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
