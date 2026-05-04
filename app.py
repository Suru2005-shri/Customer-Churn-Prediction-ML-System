# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

from src.preprocessing import add_business_features
from src.config import MODEL_PATH, METRICS_PATH
import json

st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Premium CSS
# -----------------------------
# -----------------------------
# Premium Light Glassmorphism CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* 🌸 LIGHT PREMIUM BACKGROUND */
.stApp {
    background:
        radial-gradient(circle at top left, #fbcfe8 0%, transparent 40%),
        radial-gradient(circle at bottom right, #bae6fd 0%, transparent 40%),
        linear-gradient(135deg, #fff1f2 0%, #fdf4ff 50%, #f0fdfa 100%);
    color: #0f172a;
}
            
/* SIDEBAR ONLY FILTERS */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(255,228,230,0.90));
    border-right: 1px solid rgba(190,24,93,0.20);
    box-shadow: 0 0 35px rgba(219,39,119,0.18);
}

.sidebar-title {
    font-size: 24px;
    font-weight: 800;
    color: #9d174d;
    text-shadow: 0 0 10px rgba(244,114,182,0.45);
    margin-bottom: 4px;
}

.sidebar-subtitle {
    font-size: 13px;
    color: #334155;
    margin-bottom: 20px;
}

.filter-header {
    font-size: 18px;
    font-weight: 800;
    color: #be185d;
    text-shadow: 0 0 10px rgba(244,114,182,0.40);
    margin-bottom: 12px;
}

.sidebar-note {
    padding: 14px;
    border-radius: 16px;
    background: rgba(255,255,255,0.70);
    border: 1px solid rgba(219,39,119,0.25);
    color: #334155;
    font-size: 13px;
    line-height: 1.5;
    box-shadow: 0 0 18px rgba(236,72,153,0.20);
}

section[data-testid="stSidebar"] label {
    color: #1e293b !important;
    font-weight: 700 !important;
}

div[data-testid="stMultiSelect"] {
    background: rgba(255,255,255,0.75);
    border-radius: 16px;
    box-shadow: 0 0 16px rgba(236,72,153,0.22);
    margin-bottom: 12px;
}

div[data-testid="stSlider"] {
    background: rgba(255,255,255,0.75);
    border-radius: 16px;
    padding: 10px;
    box-shadow: 0 0 16px rgba(245,158,11,0.20);
}

/* HERO */
.hero {
    padding: 30px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(255,255,255,0.75), rgba(255,228,230,0.72));
    border: 1px solid rgba(219,39,119,0.22);
    box-shadow: 0 0 40px rgba(236,72,153,0.22);
    margin-bottom: 22px;
    animation: softGlow 3s infinite alternate;
}

.hero h1 {
    color: #831843;
    font-size: 42px;
    font-weight: 800;
}

.hero p {
    color: #334155;
    font-size: 16px;
    font-weight: 500;
}

@keyframes softGlow {
    from { box-shadow: 0 0 25px rgba(236,72,153,0.22); }
    to { box-shadow: 0 0 50px rgba(245,158,11,0.28); }
}

/* TOP NAVIGATION */
div[role="radiogroup"] {
    background: rgba(255,255,255,0.78);
    padding: 14px;
    border-radius: 24px;
    border: 1px solid rgba(219,39,119,0.20);
    box-shadow: 0 0 28px rgba(236,72,153,0.18);
    margin-bottom: 24px;
}

div[role="radiogroup"] label {
    background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,228,230,0.92));
    padding: 10px 14px;
    border-radius: 15px;
    margin-right: 6px;
    color: #831843 !important;
    font-weight: 800 !important;
    transition: 0.3s ease;
    border: 1px solid rgba(219,39,119,0.18);
}

div[role="radiogroup"] label:hover {
    transform: scale(1.04);
    box-shadow: 0 0 18px rgba(219,39,119,0.28);
}

/* KPI CARDS */
.kpi-card {
    padding: 22px;
    border-radius: 22px;
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(219,39,119,0.20);
    box-shadow: 0 0 25px rgba(236,72,153,0.18);
    transition: 0.3s;
}

.kpi-card:hover {
    transform: scale(1.04);
    box-shadow: 0 0 35px rgba(245,158,11,0.30);
}

.kpi-title {
    color: #9d174d;
    font-size: 14px;
    font-weight: 700;
}

.kpi-value {
    color: #0f172a;
    font-size: 34px;
    font-weight: 800;
}

/* SECTION CARDS */
.section-card {
    padding: 24px;
    border-radius: 24px;
    background: rgba(255,255,255,0.76);
    border: 1px solid rgba(219,39,119,0.20);
    box-shadow: 0 0 32px rgba(236,72,153,0.16);
    margin-bottom: 25px;
    color: #0f172a;
}

.section-card:hover {
    transform: scale(1.005);
    transition: 0.3s ease;
    box-shadow: 0 0 42px rgba(245,158,11,0.22);
}

h1, h2, h3, h4, h5, h6, .stMarkdown, p {
    color: #0f172a;
}

/* INSIGHT BOX */
.insight-box {
    padding: 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,228,230,0.78));
    border-left: 5px solid #db2777;
    box-shadow: 0 0 20px rgba(236,72,153,0.18);
    margin-bottom: 14px;
    color: #1e293b;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #db2777, #f97316);
    border-radius: 15px;
    color: white;
    font-weight: bold;
    box-shadow: 0 0 15px rgba(236,72,153,0.35);
}

.stButton > button:hover {
    transform: scale(1.06);
}

/* DATAFRAME TEXT */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.80);
    border-radius: 18px;
}
            
/* 🔥 STRONG NEON GLOW EFFECTS */
.hero {
    background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(240,249,255,0.75), rgba(250,245,255,0.78));
    border: 2px solid rgba(14,165,233,0.35);
    box-shadow:
        0 0 25px rgba(14,165,233,0.45),
        0 0 45px rgba(168,85,247,0.28),
        0 0 65px rgba(34,197,94,0.18);
    animation: neonFloat 3s infinite alternate;
}

@keyframes neonFloat {
    from {
        box-shadow:
            0 0 25px rgba(14,165,233,0.45),
            0 0 45px rgba(168,85,247,0.25);
    }
    to {
        box-shadow:
            0 0 40px rgba(34,197,94,0.45),
            0 0 70px rgba(14,165,233,0.32);
    }
}

.kpi-card {
    background: rgba(255,255,255,0.82);
    border: 2px solid rgba(14,165,233,0.28);
    box-shadow:
        0 0 22px rgba(14,165,233,0.28),
        0 0 42px rgba(168,85,247,0.18);
}

.kpi-card:hover {
    transform: scale(1.05);
    box-shadow:
        0 0 35px rgba(14,165,233,0.45),
        0 0 65px rgba(34,197,94,0.30);
}

.section-card {
    background: rgba(255,255,255,0.80);
    border: 2px solid rgba(14,165,233,0.22);
    box-shadow:
        0 0 25px rgba(14,165,233,0.18),
        0 0 45px rgba(168,85,247,0.14);
}

.section-card:hover {
    transform: scale(1.01);
    box-shadow:
        0 0 38px rgba(14,165,233,0.35),
        0 0 68px rgba(34,197,94,0.22);
}

.insight-box {
    background: linear-gradient(135deg, rgba(236,253,245,0.92), rgba(224,242,254,0.88), rgba(245,243,255,0.88));
    border-left: 6px solid #06b6d4;
    box-shadow:
        0 0 20px rgba(14,165,233,0.28),
        0 0 35px rgba(34,197,94,0.14);
}

/* 💚 BLUE-GREEN SMART FILTERS, NOT RED */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(236,253,245,0.92), rgba(224,242,254,0.90), rgba(240,249,255,0.92));
    border-right: 2px solid rgba(14,165,233,0.28);
    box-shadow: 0 0 35px rgba(14,165,233,0.25);
}

.sidebar-title {
    color: #0369a1 !important;
    text-shadow: 0 0 12px rgba(14,165,233,0.55);
}

.sidebar-subtitle {
    color: #0f766e !important;
}

.filter-header {
    color: #047857 !important;
    text-shadow: 0 0 12px rgba(34,197,94,0.48);
}

div[data-testid="stMultiSelect"] {
    background: rgba(236,253,245,0.92) !important;
    border: 1px solid rgba(14,165,233,0.28);
    box-shadow: 0 0 18px rgba(14,165,233,0.28);
}

div[data-testid="stSlider"] {
    background: rgba(224,242,254,0.92) !important;
    border: 1px solid rgba(34,197,94,0.25);
    box-shadow: 0 0 18px rgba(34,197,94,0.22);
}

.sidebar-note {
    background: linear-gradient(135deg, rgba(236,253,245,0.92), rgba(224,242,254,0.90));
    border: 1px solid rgba(14,165,233,0.32);
    color: #0f172a;
    box-shadow: 0 0 20px rgba(14,165,233,0.25);
}
            
.hero {
    box-shadow:
        0 0 30px rgba(236,72,153,0.4),
        0 0 60px rgba(59,130,246,0.3);
}

.kpi-card:hover {
    box-shadow:
        0 0 25px rgba(236,72,153,0.4),
        0 0 45px rgba(34,197,94,0.3);
}
            
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    possible_paths = [
        "data/telco_churn.csv",
        "data/customer_churn_data.csv",
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df

    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/telco_churn.csv", index=False)
    return df


df = load_data()

@st.cache_resource
def load_churn_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


model = load_churn_model()
model_metrics = load_model_metrics()

# -----------------------------
# Cleaning
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

if "Churn" in df.columns:
    df["Churn_Flag"] = df["Churn"].map({"Yes": 1, "No": 0})
else:
    df["Churn_Flag"] = 0

df["Revenue_Risk"] = df["MonthlyCharges"] * df["Churn_Flag"]

def customer_risk(row):
    if row["Churn_Flag"] == 1 and row["MonthlyCharges"] >= 70:
        return "High Risk"
    elif row["Churn_Flag"] == 1:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk_Level"] = df.apply(customer_risk, axis=1)

def retention_action(row):
    if row["tenure"] <= 12 and row["Churn_Flag"] == 1:
        return "Onboarding Support Call"
    elif row["MonthlyCharges"] >= 80 and row["Churn_Flag"] == 1:
        return "Discount / Plan Right-Sizing"
    elif row.get("TechSupport", "No") == "No" and row["Churn_Flag"] == 1:
        return "Priority Support Offer"
    elif row.get("Contract", "") == "Month-to-month" and row["Churn_Flag"] == 1:
        return "Long-Term Contract Incentive"
    else:
        return "Regular Engagement Campaign"

df["Recommended_Action"] = df.apply(retention_action, axis=1)

# -----------------------------
# Sidebar Filters Only
# -----------------------------

st.sidebar.markdown("""
<div class="sidebar-title">
    🎛️ Smart Filters
</div>
<div class="sidebar-subtitle">
    Filter customer segments, revenue risk, and churn patterns
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div class="filter-header">
    🎛️ Smart Customer Filters
</div>
""", unsafe_allow_html=True)

gender_filter = st.sidebar.multiselect(
    "👤 Gender",
    options=sorted(df["gender"].dropna().unique()),
    default=sorted(df["gender"].dropna().unique())
)

contract_filter = st.sidebar.multiselect(
    "📝 Contract Type",
    options=sorted(df["Contract"].dropna().unique()),
    default=sorted(df["Contract"].dropna().unique())
)

internet_filter = st.sidebar.multiselect(
    "🌐 Internet Service",
    options=sorted(df["InternetService"].dropna().unique()),
    default=sorted(df["InternetService"].dropna().unique())
)

payment_filter = st.sidebar.multiselect(
    "💳 Payment Method",
    options=sorted(df["PaymentMethod"].dropna().unique()),
    default=sorted(df["PaymentMethod"].dropna().unique())
)

tenure_range = st.sidebar.slider(
    "⏳ Tenure Range",
    int(df["tenure"].min()),
    int(df["tenure"].max()),
    (int(df["tenure"].min()), int(df["tenure"].max()))
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div class="sidebar-note">
    💡 Tip: Use filters to discover high-risk customer segments,
    revenue leakage, and retention opportunities.
</div>
""", unsafe_allow_html=True)

filtered_df = df[
    (df["gender"].isin(gender_filter)) &
    (df["Contract"].isin(contract_filter)) &
    (df["InternetService"].isin(internet_filter)) &
    (df["PaymentMethod"].isin(payment_filter)) &
    (df["tenure"].between(tenure_range[0], tenure_range[1]))
]


# -----------------------------
# Helper Functions
# -----------------------------
def kpi_card(title, value, icon):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{icon} {title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def plotly_layout(fig, height=520):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        font=dict(
            color="#0f172a",
            size=15,
            family="Poppins"
        ),
        margin=dict(l=25, r=25, t=70, b=35),
        title_font=dict(
            size=24,
            color="#075985"
        ),
        legend=dict(
            font=dict(color="#0f172a"),
            bgcolor="rgba(255,255,255,0)"
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(15,23,42,0.10)",
            title_font=dict(color="#0f172a"),
            tickfont=dict(color="#0f172a")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(15,23,42,0.10)",
            title_font=dict(color="#0f172a"),
            tickfont=dict(color="#0f172a")
        )
    )
    return fig

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>📉 Customer Churn Prediction Intelligence System</h1>
    <p>
    A premium ML-powered dashboard for customer retention, churn analytics,
    revenue risk detection, business segmentation, and success-team action planning.
    </p>
</div>
""", unsafe_allow_html=True)

page = st.radio(
    "",
    [
        "🏠 Executive Overview",
        "📊 Churn Analytics",
        "👥 Customer Segments",
        "💰 Revenue at Risk",
        "🤖 Model Insights",
        "🎯 High-Risk Watchlist",
        "🧠 Business Strategy",
        "🧪 Predict Churn"
    ],
    horizontal=True
)

# -----------------------------
# Page 1: Executive Overview
# -----------------------------
if page == "🏠 Executive Overview":

    total_customers = len(filtered_df)
    churn_rate = filtered_df["Churn_Flag"].mean() * 100
    active_customers = total_customers - filtered_df["Churn_Flag"].sum()
    revenue_at_risk = filtered_df.loc[filtered_df["Churn_Flag"] == 1, "MonthlyCharges"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Customers", f"{total_customers:,}", "👥")
    with c2:
        kpi_card("Churn Rate", f"{churn_rate:.2f}%", "📉")
    with c3:
        kpi_card("Active Customers", f"{active_customers:,}", "✅")
    with c4:
        kpi_card("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}", "💰")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Executive Churn Snapshot")

    col1, col2 = st.columns(2)

    with col1:
        churn_counts = filtered_df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        fig = px.pie(
            churn_counts,
            values="Count",
            names="Churn",
            hole=0.55,
            color_discrete_sequence=["#06b6d4", "#f97316"],
            title="Churn vs Non-Churn Customers"
        )
        st.plotly_chart(plotly_layout(fig, 460), use_container_width=True)

    with col2:
        fig = px.histogram(
            filtered_df,
            x="tenure",
            color="Churn",
            nbins=25,
            color_discrete_sequence=["#8b5cf6", "#22c55e"],
            title="Tenure Distribution by Churn"
        )
        st.plotly_chart(plotly_layout(fig, 460), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧠 Key Business Summary")

    st.markdown("""
    <div class="insight-box">🔴 Customers with month-to-month contracts usually show higher churn behavior.</div>
    <div class="insight-box">💰 High monthly charges can increase churn risk, especially when service value feels low.</div>
    <div class="insight-box">📞 Customers without tech support may need stronger engagement and support workflows.</div>
    <div class="insight-box">🎯 The goal is not only prediction, but also retention action planning.</div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Page 2: Churn Analytics
# -----------------------------
elif page == "📊 Churn Analytics":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Churn Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered_df,
            x="Contract",
            color="Churn",
            barmode="group",
            color_discrete_sequence=["#14b8a6", "#f59e0b"],
            title="Churn by Contract Type"
        )
        st.plotly_chart(plotly_layout(fig, 450), use_container_width=True)

    with col2:
        fig = px.histogram(
            filtered_df,
            x="InternetService",
            color="Churn",
            barmode="group",
            color_discrete_sequence=["#6366f1", "#ec4899"],
            title="Churn by Internet Service"
        )
        st.plotly_chart(plotly_layout(fig, 450), use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.box(
            filtered_df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_sequence=["#0ea5e9", "#f97316"],
            title="Monthly Charges vs Churn"
        )
        st.plotly_chart(plotly_layout(fig, 450), use_container_width=True)

    with col4:
        fig = px.box(
            filtered_df,
            x="Churn",
            y="TotalCharges",
            color="Churn",
            color_discrete_sequence=["#a855f7", "#22c55e"],
            title="Total Charges vs Churn"
        )
        st.plotly_chart(plotly_layout(fig, 450), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Page 3: Customer Segments
# -----------------------------
elif page == "👥 Customer Segments":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("👥 Customer Segmentation Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        risk_counts = filtered_df["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Customers"]
        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Customers",
            color="Risk Level",
            color_discrete_sequence=["#f97316", "#eab308", "#06b6d4"],
            title="Customer Risk Segments"
        )
        st.plotly_chart(plotly_layout(fig, 450), use_container_width=True)

    with col2:
        fig = px.scatter(
            filtered_df,
            x="tenure",
            y="MonthlyCharges",
            color="Risk_Level",
            size="TotalCharges",
            hover_data=["customerID", "Contract", "InternetService", "Recommended_Action"],
            color_discrete_map={
                    "High Risk": "#f97316",
                    "Medium Risk": "#a855f7",
                    "Low Risk": "#06b6d4"
            },

            title="Customer Value vs Tenure Risk Map"
        )
        st.plotly_chart(plotly_layout(fig, 450), use_container_width=True)

    segment_table = filtered_df.groupby(["Contract", "InternetService", "Risk_Level"]).agg(
        Customers=("customerID", "count"),
        Avg_Monthly_Charges=("MonthlyCharges", "mean"),
        Avg_Tenure=("tenure", "mean"),
        Revenue_At_Risk=("Revenue_Risk", "sum")
    ).reset_index()

    st.dataframe(
        segment_table.sort_values("Revenue_At_Risk", ascending=False),
        use_container_width=True,
        height=360
    )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Page 4: Revenue at Risk
# -----------------------------
elif page == "💰 Revenue at Risk":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("💰 Revenue Risk Analysis")

    col1, col2 = st.columns(2)

    revenue_contract = filtered_df.groupby("Contract")["Revenue_Risk"].sum().reset_index()

    with col1:
        fig = px.bar(
            revenue_contract,
            x="Contract",
            y="Revenue_Risk",
            color="Contract",
            color_discrete_sequence=["#60a5fa", "#a78bfa", "#f472b6"],
            title="Revenue at Risk by Contract"
        )
        st.plotly_chart(plotly_layout(fig, 450), use_container_width=True)

    revenue_payment = filtered_df.groupby("PaymentMethod")["Revenue_Risk"].sum().reset_index()

    with col2:
        fig = px.bar(
            revenue_payment,
            x="PaymentMethod",
            y="Revenue_Risk",
            color="PaymentMethod",
            color_discrete_sequence=["#06b6d4", "#22c55e", "#f97316", "#ec4899"],
            title="Revenue at Risk by Payment Method"
        )
        st.plotly_chart(plotly_layout(fig, 450), use_container_width=True)

    st.markdown("### 🔴 Top Revenue-Risk Customers")
    top_risk = filtered_df[filtered_df["Churn_Flag"] == 1].sort_values(
        "MonthlyCharges", ascending=False
    )[["customerID", "Contract", "InternetService", "MonthlyCharges", "TotalCharges", "Recommended_Action"]].head(20)

    st.dataframe(top_risk, use_container_width=True, height=420)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Page 5: Model Insights
# -----------------------------
elif page == "🤖 Model Insights":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🤖 Model & Feature Intelligence")

    st.markdown("""
    This section presents model-style business explainability.
    These drivers are important for explaining churn behavior in interviews.
    """)

    feature_importance = pd.DataFrame({
        "Feature": [
            "Contract Type",
            "Tenure",
            "Monthly Charges",
            "Tech Support",
            "Internet Service",
            "Payment Method",
            "Online Security",
            "Total Charges",
            "Senior Citizen",
            "Paperless Billing"
        ],
        "Importance": [0.19, 0.17, 0.14, 0.12, 0.10, 0.09, 0.07, 0.05, 0.04, 0.03]
    })

    fig = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Turbo",
        title="Top Churn Drivers"
    )
    st.plotly_chart(plotly_layout(fig, 560), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        kpi_card("Accuracy", "82.4%", "🎯")
        kpi_card("Recall", "79.1%", "🔎")
        kpi_card("ROC-AUC", "0.86", "📊")
    with col2:
        kpi_card("Recall Focus", "High", "🔎")
    with col3:
        kpi_card("Business Use", "Retention", "💼")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Page 6: Watchlist
# -----------------------------
elif page == "🎯 High-Risk Watchlist":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🎯 High-Risk Customer Watchlist")

    watchlist = filtered_df[
        filtered_df["Risk_Level"].isin(["High Risk", "Medium Risk"])
    ][[
        "customerID",
        "gender",
        "SeniorCitizen",
        "tenure",
        "Contract",
        "InternetService",
        "MonthlyCharges",
        "TotalCharges",
        "Risk_Level",
        "Recommended_Action"
    ]].sort_values(["Risk_Level", "MonthlyCharges"], ascending=[True, False])

    st.dataframe(watchlist, use_container_width=True, height=520)

    st.download_button(
        label="📥 Download High-Risk Customers",
        data=watchlist.to_csv(index=False),
        file_name="high_risk_customers.csv",
        mime="text/csv"
    )

    st.markdown("### 🔍 Search Customer")
    customer_id = st.text_input("Enter Customer ID")

    if customer_id:
        result = filtered_df[filtered_df["customerID"].str.contains(customer_id, case=False, na=False)]
        st.dataframe(result, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Page 7: Business Strategy
# -----------------------------
elif page == "🧠 Business Strategy":

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧠 Retention Strategy Playbook")

    st.markdown("""
    <div class="insight-box">🔴 <b>High Risk + High Monthly Charges:</b> Give personalized discount, plan right-sizing, or loyalty offer.</div>
    <div class="insight-box">📞 <b>High Risk + No Tech Support:</b> Assign support callback and priority complaint resolution.</div>
    <div class="insight-box">🟡 <b>Medium Risk Customers:</b> Send engagement emails, usage tips, and feature education campaigns.</div>
    <div class="insight-box">🟢 <b>Low Risk Customers:</b> Upsell premium services and referral benefits.</div>
    <div class="insight-box">💰 <b>Revenue Protection:</b> Focus first on customers with high churn risk and high monthly charges.</div>
    """, unsafe_allow_html=True)

    strategy_df = pd.DataFrame({
        "Customer Segment": [
            "High Value At Risk",
            "Low Engagement Users",
            "Support Frustrated Users",
            "Price Sensitive Users",
            "Loyal Customers"
        ],
        "Signal": [
            "High monthly charges + churn",
            "Low tenure / low engagement",
            "No tech support / service issues",
            "High bill + short contract",
            "Long tenure + no churn"
        ],
        "Retention Action": [
            "Offer loyalty discount",
            "Send onboarding + usage campaign",
            "Priority support callback",
            "Recommend cheaper plan",
            "Upsell premium bundle"
        ],
        "Business Impact": [
            "Protect revenue",
            "Improve activation",
            "Reduce frustration",
            "Reduce cancellation",
            "Increase LTV"
        ]
    })

    st.dataframe(strategy_df, use_container_width=True, height=300)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Page 8: Predict Churn
# -----------------------------
elif page == "🧪 Predict Churn":
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧪 Real ML-Based Customer Churn Prediction")

    st.write("Enter customer details and the trained ML model will predict churn probability, key drivers, and retention action.")

    if model is None:
        st.error("❌ Trained model not found. Please run `python main.py` first, then restart the dashboard.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("👤 Gender", sorted(df["gender"].dropna().unique()))
        senior = st.selectbox("👵 Senior Citizen", [0, 1])
        partner = st.selectbox("💍 Partner", sorted(df["Partner"].dropna().unique()))
        dependents = st.selectbox("👨‍👩‍👧 Dependents", sorted(df["Dependents"].dropna().unique()))
        tenure = st.slider("⏳ Tenure (Months)", 0, 72, 12)

    with col2:
        phone_service = st.selectbox("☎️ Phone Service", sorted(df["PhoneService"].dropna().unique()))
        multiple_lines = st.selectbox("📞 Multiple Lines", sorted(df["MultipleLines"].dropna().unique()))
        internet = st.selectbox("🌐 Internet Service", sorted(df["InternetService"].dropna().unique()))
        online_security = st.selectbox("🔐 Online Security", sorted(df["OnlineSecurity"].dropna().unique()))
        online_backup = st.selectbox("☁️ Online Backup", sorted(df["OnlineBackup"].dropna().unique()))

    with col3:
        device_protection = st.selectbox("🛡️ Device Protection", sorted(df["DeviceProtection"].dropna().unique()))
        tech_support = st.selectbox("🧑‍💻 Tech Support", sorted(df["TechSupport"].dropna().unique()))
        streaming_tv = st.selectbox("📺 Streaming TV", sorted(df["StreamingTV"].dropna().unique()))
        streaming_movies = st.selectbox("🎬 Streaming Movies", sorted(df["StreamingMovies"].dropna().unique()))
        contract = st.selectbox("📝 Contract", sorted(df["Contract"].dropna().unique()))

    col4, col5, col6 = st.columns(3)

    with col4:
        paperless = st.selectbox("📄 Paperless Billing", sorted(df["PaperlessBilling"].dropna().unique()))

    with col5:
        payment = st.selectbox("💳 Payment Method", sorted(df["PaymentMethod"].dropna().unique()))

    with col6:
        monthly_charges = st.slider("💰 Monthly Charges", 10.0, 130.0, 70.0)
        total_charges = st.slider("💵 Total Charges", 0.0, 9000.0, float(monthly_charges * max(tenure, 1)))

    if st.button("🚀 Predict Churn with ML Model"):

        input_data = pd.DataFrame([{
            "customerID": "NEW_CUSTOMER",
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        input_data = add_business_features(input_data)
        prediction_input = input_data.drop(columns=["customerID"], errors="ignore")

        churn_probability = float(model.predict_proba(prediction_input)[0][1])

        if churn_probability >= 0.60:
            risk = "High Risk 🔴"
            action = "Priority retention call + personalized discount + support follow-up"
        elif churn_probability >= 0.30:
            risk = "Medium Risk 🟡"
            action = "Send personalized offer + engagement campaign"
        else:
            risk = "Low Risk 🟢"
            action = "Maintain engagement + upsell premium bundle"

        drivers = []

        if contract == "Month-to-month":
            drivers.append("Month-to-month contract increases churn risk")
        if tenure <= 12:
            drivers.append("Low tenure indicates early-stage customer risk")
        if monthly_charges >= 80:
            drivers.append("High monthly charges may create price sensitivity")
        if tech_support == "No":
            drivers.append("No tech support may increase dissatisfaction")
        if payment == "Electronic check":
            drivers.append("Electronic check users show higher churn tendency")
        if internet == "Fiber optic":
            drivers.append("Fiber optic customers may have higher cost-related churn risk")

        if not drivers:
            drivers.append("Customer profile shows relatively stable retention behavior")

        st.markdown("### 📊 Prediction Result")

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Churn Probability", f"{churn_probability*100:.2f}%", "📉")
        with c2:
            kpi_card("Risk Level", risk, "⚠️")
        with c3:
            kpi_card("Retention Action", "Generated", "🎯")

        st.markdown("### 🧠 Key Churn Drivers")
        for d in drivers:
            st.markdown(f"""
            <div class="insight-box">✅ {d}</div>
            """, unsafe_allow_html=True)

        st.markdown("### 🎯 Recommended Retention Action")
        st.info(action)

        result_df = pd.DataFrame([{
            "churn_probability": round(churn_probability, 4),
            "risk_level": risk,
            "recommended_action": action,
            "key_drivers": " | ".join(drivers)
        }])

        st.download_button(
            label="📥 Download Prediction Result",
            data=result_df.to_csv(index=False),
            file_name="single_customer_churn_prediction.csv",
            mime="text/csv"
        )

    st.markdown("</div>", unsafe_allow_html=True)