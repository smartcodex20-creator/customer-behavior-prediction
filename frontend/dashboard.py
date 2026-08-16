"""
Customer Behavior Prediction Platform – Streamlit Dashboard
Phase 5 – Deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Behavior Prediction",
    page_icon="📊",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    features = pd.read_csv("data/processed/customer_features.csv")
    return features

df = load_data()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Customer Lookup", "Churn Leaderboard", "About"]
)

st.sidebar.markdown("---")
st.sidebar.info("Customer Behavior Prediction Platform\nVantara Retail Solutions")

# --------------------------------------------------
# Page: Overview
# --------------------------------------------------
if page == "Overview":
    st.title("Customer Behavior Prediction Platform")
    st.markdown("### Overview Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(df)
    churn_rate = df["Churn"].mean() * 100
    high_risk = (df["Engagement_Score"] < 2.0).sum()
    avg_monetary = df["Monetary_Positive"].mean()

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Low Engagement", f"{high_risk:,}")
    col4.metric("Avg. Customer Value", f"£{avg_monetary:,.0f}")

    st.markdown("---")

    st.subheader("Feature Distributions")
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("**Recency Distribution**")
        st.bar_chart(df["Recency"].clip(upper=400).value_counts().sort_index())

    with col_b:
        st.write("**Engagement Score Distribution**")
        st.bar_chart(df["Engagement_Score"].round(1).value_counts().sort_index())

# --------------------------------------------------
# Page: Customer Lookup
# --------------------------------------------------
elif page == "Customer Lookup":
    st.title("Customer Lookup & Prediction")

    customer_id = st.number_input("Enter Customer ID", min_value=1, value=12347, step=1)

    if st.button("Get Prediction"):
        try:
            response = requests.get(f"{API_URL}/customer/{customer_id}", timeout=10)

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction generated successfully")

                col1, col2, col3 = st.columns(3)
                col1.metric("Churn Probability", f"{result['Churn_Probability']:.1%}")
                col2.metric("Prediction", "Churn" if result["Churn_Prediction"] == 1 else "Active")
                col3.metric("Risk Level", result["Risk_Level"])

                st.info(result["Message"])

                # Show customer features
                customer_row = df[df["Customer_ID"] == customer_id]
                if not customer_row.empty:
                    st.subheader("Customer Features")
                    st.dataframe(customer_row.T.rename(columns={customer_row.index[0]: "Value"}))

            elif response.status_code == 404:
                st.error(f"Customer ID {customer_id} not found.")
            else:
                st.error(f"API Error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# --------------------------------------------------
# Page: Churn Leaderboard
# --------------------------------------------------
elif page == "Churn Leaderboard":
    st.title("Churn Risk Leaderboard")
    st.markdown("Customers sorted by highest churn risk indicators (high Recency + low Engagement)")

    leaderboard = df.copy()
    leaderboard["Risk_Score"] = (
        leaderboard["Recency"] * 0.5 +
        (5 - leaderboard["Engagement_Score"]) * 30
    )

    top_risk = leaderboard.nlargest(20, "Risk_Score")[
        ["Customer_ID", "Recency", "Frequency", "Monetary_Positive", 
         "Engagement_Score", "Churn", "Risk_Score"]
    ].round(2)

    st.dataframe(top_risk, use_container_width=True)

# --------------------------------------------------
# Page: About
# --------------------------------------------------
elif page == "About":
    st.title("About This Platform")
    st.markdown("""
    ### Customer Behavior Prediction Platform
    **Vantara Retail Solutions – Data Science & Analytics Division**

    This platform predicts customer churn risk and provides actionable insights for the Marketing and Retention teams.

    **Key Capabilities:**
    - Churn probability scoring
    - Customer Lifetime Value estimation
    - Customer segmentation
    - Model explainability (SHAP & LIME)
    - Anomaly detection

    **Tech Stack:**
    - Python, Pandas, Scikit-learn, XGBoost, LightGBM
    - TensorFlow (ANN, LSTM, Autoencoder)
    - FastAPI + Streamlit
    - SHAP & LIME for explainability
    """)