"""
Phase 3 – Customer Lifetime Value (CLV) Regression
Customer Behavior Prediction Platform
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


def load_features(path: str = "data/processed/customer_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded feature table: {df.shape[0]:,} customers")
    return df


def prepare_clv_data(df: pd.DataFrame):
    """
    For CLV we predict Monetary_Positive (historical value).
    We remove Monetary-related columns from features to avoid leakage.
    """
    target = "Monetary_Positive"

    feature_cols = [
        "Recency", "Frequency",
        "Avg_Basket_Size", "Total_Quantity", "N_Transactions",
        "N_Returns", "Return_Rate", "Customer_Age_Days",
        "Frequency_Trend", "Avg_Days_Between", "Std_Days_Between",
        "Engagement_Score"
    ]

    X = df[feature_cols].copy().fillna(0)
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples : {len(X_test):,}")
    print(f"Target (Monetary_Positive) mean: £{y.mean():,.2f}")

    return X_train, X_test, y_train, y_test


def evaluate_regression(model, X, y, model_name: str):
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print(f"\n{'='*55}")
    print(f"Model: {model_name}")
    print(f"{'='*55}")
    print(f"MAE  : £{mae:,.2f}")
    print(f"RMSE : £{rmse:,.2f}")
    print(f"R²   : {r2:.4f}")

    return {"Model": model_name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}


if __name__ == "__main__":
    df = load_features()
    X_train, X_test, y_train, y_test = prepare_clv_data(df)

    results = []

    # 1. Random Forest Regressor
    print("\nTraining Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate_regression(rf, X_test, y_test, "Random Forest Regressor"))

    # 2. XGBoost Regressor
    print("\nTraining XGBoost Regressor...")
    xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    xgb.fit(X_train, y_train)
    results.append(evaluate_regression(xgb, X_test, y_test, "XGBoost Regressor"))

    # Comparison
    print("\n" + "="*55)
    print("CLV REGRESSION MODEL COMPARISON")
    print("="*55)
    print(pd.DataFrame(results).to_string(index=False))