"""
Feature Engineering Module (PRD-Aligned)
Customer Behavior Prediction Platform

- Features calculated using data BEFORE the cutoff date
- Churn label defined using the NEXT 90 days AFTER the cutoff date
- Includes Purchase Frequency Trend (slope) as required by PRD
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from sklearn.linear_model import LinearRegression


def load_cleaned_data(path: str = "data/interim/customer_transactions.csv") -> pd.DataFrame:
    """Load the cleaned customer-level transaction data."""
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    print(f"Loaded {len(df):,} transactions")
    print(f"Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
    print(f"Unique customers: {df['Customer_ID'].nunique():,}")
    return df


def calculate_frequency_trend(group: pd.DataFrame) -> float:
    """
    Calculate the slope of purchase frequency over time for one customer.
    Uses the month number as X and number of invoices per month as Y.
    Returns 0 if the customer has less than 2 active months.
    """
    if len(group) < 2:
        return 0.0

    monthly = group.groupby(group["InvoiceDate"].dt.to_period("M")).size().reset_index(name="count")
    monthly["month_num"] = range(len(monthly))

    if len(monthly) < 2:
        return 0.0

    X = monthly[["month_num"]]
    y = monthly["count"]

    model = LinearRegression()
    model.fit(X, y)
    return float(model.coef_[0])


def create_features_and_target(df: pd.DataFrame, cutoff_date: str = "2011-09-01") -> pd.DataFrame:
    """
    Create customer features + churn label using point-in-time discipline.
    """
    cutoff = pd.Timestamp(cutoff_date)
    observation_end = cutoff + timedelta(days=90)

    print(f"\nCutoff date (features before)     : {cutoff.date()}")
    print(f"Observation window (churn label) : {cutoff.date()} → {observation_end.date()}")

    # Split data
    df_past = df[df["InvoiceDate"] < cutoff].copy()
    df_future = df[(df["InvoiceDate"] >= cutoff) & (df["InvoiceDate"] <= observation_end)].copy()

    print(f"Transactions used for features   : {len(df_past):,}")
    print(f"Transactions used for churn label: {len(df_future):,}")

    df_past["Revenue"] = df_past["Quantity"] * df_past["Price"]

    # --------------------------------------------------
    # Basic Aggregations
    # --------------------------------------------------
    features = df_past.groupby("Customer_ID").agg(
        Recency=("InvoiceDate", lambda x: (cutoff - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        Avg_Basket_Size=("Revenue", "mean"),
        Total_Quantity=("Quantity", "sum"),
        N_Transactions=("Invoice", "count"),
        First_Purchase=("InvoiceDate", "min"),
        Last_Purchase=("InvoiceDate", "max"),
    ).reset_index()

    # --------------------------------------------------
    # Return Features
    # --------------------------------------------------
    returns = df_past[df_past["Is_Return"] == True].groupby("Customer_ID").size().reset_index(name="N_Returns")
    features = features.merge(returns, on="Customer_ID", how="left")
    features["N_Returns"] = features["N_Returns"].fillna(0).astype(int)
    features["Return_Rate"] = features["N_Returns"] / features["N_Transactions"]

    # --------------------------------------------------
    # Derived Features
    # --------------------------------------------------
    features["Customer_Age_Days"] = (features["Last_Purchase"] - features["First_Purchase"]).dt.days
    features["Avg_Days_Between_Orders"] = features["Customer_Age_Days"] / features["Frequency"].replace(0, np.nan)
    features["Monetary_Positive"] = features["Monetary"].clip(lower=0)

    # Engagement Score
    features["Engagement_Score"] = (
        (1 / (1 + features["Recency"])) * 0.40 +
        np.log1p(features["Frequency"]) * 0.30 +
        np.log1p(features["Monetary_Positive"]) * 0.30
    )

    # --------------------------------------------------
    # Purchase Frequency Trend (PRD Required)
    # --------------------------------------------------
    print("Calculating Purchase Frequency Trend (this may take a moment)...")
    trend = df_past.groupby("Customer_ID").apply(calculate_frequency_trend, include_groups=False)
    trend = trend.reset_index(name="Frequency_Trend")
    features = features.merge(trend, on="Customer_ID", how="left")
    features["Frequency_Trend"] = features["Frequency_Trend"].fillna(0)

    # --------------------------------------------------
    # Churn Label
    # --------------------------------------------------
    future_customers = set(df_future["Customer_ID"].unique())
    features["Churn"] = features["Customer_ID"].apply(lambda x: 0 if x in future_customers else 1)

    print(f"\nTotal customers with features : {len(features):,}")
    print(f"Churned customers (Churn=1)   : {features['Churn'].sum():,}")
    print(f"Active customers (Churn=0)    : {(features['Churn']==0).sum():,}")
    print(f"Churn rate                    : {features['Churn'].mean()*100:.2f}%")

    return features


def save_features(features: pd.DataFrame, path: str = "data/processed/customer_features.csv") -> None:
    """Save the final feature table."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"\nFeature table saved → {output_path}")


if __name__ == "__main__":
    df = load_cleaned_data()
    features = create_features_and_target(df, cutoff_date="2011-09-01")

    print("\nSample of final features + target:")
    cols_to_show = ["Customer_ID", "Recency", "Frequency", "Monetary", "Return_Rate", 
                    "Frequency_Trend", "Engagement_Score", "Churn"]
    print(features[cols_to_show].head(8).round(3))

    print("\nFrequency Trend Summary:")
    print(features["Frequency_Trend"].describe().round(3))

    save_features(features)