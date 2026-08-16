"""
Feature Engineering Module (PRD-Aligned) - Final Phase 2 Version
Customer Behavior Prediction Platform
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from sklearn.linear_model import LinearRegression


def load_cleaned_data(path: str = "data/interim/customer_transactions.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    print(f"Loaded {len(df):,} transactions")
    print(f"Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
    print(f"Unique customers: {df['Customer_ID'].nunique():,}")
    return df


def calculate_frequency_trend(group: pd.DataFrame) -> float:
    """Calculate slope of monthly purchase frequency."""
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


def calculate_purchase_interval_stats(group: pd.DataFrame) -> pd.Series:
    """Calculate average and standard deviation of days between consecutive purchases."""
    dates = group["InvoiceDate"].sort_values().drop_duplicates()
    
    if len(dates) < 2:
        return pd.Series({"Avg_Days_Between": 0.0, "Std_Days_Between": 0.0})
    
    diffs = dates.diff().dt.days.dropna()
    
    return pd.Series({
        "Avg_Days_Between": diffs.mean(),
        "Std_Days_Between": diffs.std(ddof=0)  # population std
    })


def create_features_and_target(df: pd.DataFrame, cutoff_date: str = "2011-09-01") -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff_date)
    observation_end = cutoff + timedelta(days=90)

    print(f"\nCutoff date (features before)     : {cutoff.date()}")
    print(f"Observation window (churn label) : {cutoff.date()} → {observation_end.date()}")

    df_past = df[df["InvoiceDate"] < cutoff].copy()
    df_future = df[(df["InvoiceDate"] >= cutoff) & (df["InvoiceDate"] <= observation_end)].copy()

    print(f"Transactions used for features   : {len(df_past):,}")
    print(f"Transactions used for churn label: {len(df_future):,}")

    df_past["Revenue"] = df_past["Quantity"] * df_past["Price"]

    # Basic aggregations
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

    # Return features
    returns = df_past[df_past["Is_Return"] == True].groupby("Customer_ID").size().reset_index(name="N_Returns")
    features = features.merge(returns, on="Customer_ID", how="left")
    features["N_Returns"] = features["N_Returns"].fillna(0).astype(int)
    features["Return_Rate"] = features["N_Returns"] / features["N_Transactions"]

    # Derived features
    features["Customer_Age_Days"] = (features["Last_Purchase"] - features["First_Purchase"]).dt.days
    features["Monetary_Positive"] = features["Monetary"].clip(lower=0)

    # Engagement Score
    features["Engagement_Score"] = (
        (1 / (1 + features["Recency"])) * 0.40 +
        np.log1p(features["Frequency"]) * 0.30 +
        np.log1p(features["Monetary_Positive"]) * 0.30
    )

    # Frequency Trend
    print("Calculating Purchase Frequency Trend...")
    trend = df_past.groupby("Customer_ID").apply(calculate_frequency_trend, include_groups=False)
    trend = trend.reset_index(name="Frequency_Trend")
    features = features.merge(trend, on="Customer_ID", how="left")
    features["Frequency_Trend"] = features["Frequency_Trend"].fillna(0)

    # Time Between Purchases (Avg + Std) - PRD required
    print("Calculating Time Between Purchases statistics...")
    interval_stats = df_past.groupby("Customer_ID").apply(calculate_purchase_interval_stats, include_groups=False)
    interval_stats = interval_stats.reset_index()
    features = features.merge(interval_stats, on="Customer_ID", how="left")
    features["Avg_Days_Between"] = features["Avg_Days_Between"].fillna(0)
    features["Std_Days_Between"] = features["Std_Days_Between"].fillna(0)

    # Churn label
    future_customers = set(df_future["Customer_ID"].unique())
    features["Churn"] = features["Customer_ID"].apply(lambda x: 0 if x in future_customers else 1)

    print(f"\nTotal customers with features : {len(features):,}")
    print(f"Churned customers (Churn=1)   : {features['Churn'].sum():,}")
    print(f"Active customers (Churn=0)    : {(features['Churn']==0).sum():,}")
    print(f"Churn rate                    : {features['Churn'].mean()*100:.2f}%")

    return features


def save_features(features: pd.DataFrame, path: str = "data/processed/customer_features.csv") -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"\nFeature table saved → {output_path}")


if __name__ == "__main__":
    df = load_cleaned_data()
    features = create_features_and_target(df, cutoff_date="2011-09-01")

    print("\nSample of final features:")
    cols = ["Customer_ID", "Recency", "Frequency", "Monetary", "Return_Rate", 
            "Frequency_Trend", "Avg_Days_Between", "Std_Days_Between", "Engagement_Score", "Churn"]
    print(features[cols].head(8).round(3))

    print("\nNew features summary:")
    print(features[["Frequency_Trend", "Avg_Days_Between", "Std_Days_Between"]].describe().round(3))

    save_features(features)