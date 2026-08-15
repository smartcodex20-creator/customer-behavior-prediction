"""
Feature Engineering Module
Customer Behavior Prediction Platform

Creates customer-level features with point-in-time discipline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta


def load_cleaned_data(path: str = "data/interim/customer_transactions.csv") -> pd.DataFrame:
    """Load the cleaned customer-level transaction data."""
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    print(f"Loaded {len(df):,} transactions")
    print(f"Unique customers: {df['Customer_ID'].nunique():,}")
    return df


def create_customer_features(df: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Create customer-level features.
    """
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + timedelta(days=1)

    print(f"\nReference date for Recency: {reference_date.date()}")

    df = df.copy()
    df["Revenue"] = df["Quantity"] * df["Price"]

    # --------------------------------------------------
    # 1. Basic RFM + Activity Features
    # --------------------------------------------------
    features = df.groupby("Customer_ID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        Avg_Basket_Size=("Revenue", "mean"),
        Total_Quantity=("Quantity", "sum"),
        N_Transactions=("Invoice", "count"),
        First_Purchase=("InvoiceDate", "min"),
        Last_Purchase=("InvoiceDate", "max"),
    ).reset_index()

    # --------------------------------------------------
    # 2. Return Related Features
    # --------------------------------------------------
    returns = df[df["Is_Return"] == True].groupby("Customer_ID").agg(
        N_Returns=("Invoice", "count"),
        Return_Value=("Revenue", "sum")
    ).reset_index()

    features = features.merge(returns, on="Customer_ID", how="left")
    features["N_Returns"] = features["N_Returns"].fillna(0).astype(int)
    features["Return_Value"] = features["Return_Value"].fillna(0)

    # Return Rate = number of return transactions / total transactions
    features["Return_Rate"] = features["N_Returns"] / features["N_Transactions"]

    # --------------------------------------------------
    # 3. Derived Features
    # --------------------------------------------------
    features["Customer_Age_Days"] = (features["Last_Purchase"] - features["First_Purchase"]).dt.days
    features["Avg_Days_Between_Orders"] = features["Customer_Age_Days"] / features["Frequency"].replace(0, np.nan)

    # Handle customers with negative monetary value (heavy returners)
    features["Monetary_Positive"] = features["Monetary"].clip(lower=0)

    # Simple Engagement Score (higher is better)
    # Normalized roughly for interpretability
    features["Engagement_Score"] = (
        (1 / (1 + features["Recency"])) * 0.4 +
        np.log1p(features["Frequency"]) * 0.3 +
        np.log1p(features["Monetary_Positive"]) * 0.3
    )

    print(f"\nFeatures created for {len(features):,} customers")
    return features


def save_features(features: pd.DataFrame, path: str = "data/processed/customer_features.csv") -> None:
    """Save the feature table."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"\nFeature table saved to: {output_path}")


if __name__ == "__main__":
    df = load_cleaned_data()
    features = create_customer_features(df)

    print("\nSample features:")
    print(features.head(8).round(3))

    print("\nFeature Summary:")
    print(features[["Recency", "Frequency", "Monetary", "Return_Rate", "Engagement_Score"]].describe().round(3))

    save_features(features)