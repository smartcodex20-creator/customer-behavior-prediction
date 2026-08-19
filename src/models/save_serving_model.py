"""
Save a production-style serving model for the API.
Customer Behavior Prediction Platform
"""

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


FEATURES_PATH = Path("data/processed/customer_features.csv")
MODEL_DIR = Path("models_artifacts")
MODEL_PATH = MODEL_DIR / "churn_serving_model.joblib"
FEATURES_META_PATH = MODEL_DIR / "serving_features.json"

FEATURE_COLS = [
    "Recency",
    "Frequency",
    "Monetary_Positive",
    "Avg_Basket_Size",
    "Total_Quantity",
    "N_Transactions",
    "N_Returns",
    "Return_Rate",
    "Customer_Age_Days",
    "Frequency_Trend",
    "Avg_Days_Between",
    "Std_Days_Between",
    "Engagement_Score",
]


def main():
    print("Loading feature table...")
    df = pd.read_csv(FEATURES_PATH)

    missing = [c for c in FEATURE_COLS + ["Churn"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLS].fillna(0)
    y = df["Churn"]

    print(f"Training serving model on {len(df):,} customers...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    with open(FEATURES_META_PATH, "w", encoding="utf-8") as f:
        json.dump({"feature_cols": FEATURE_COLS}, f, indent=2)

    print(f"Model saved → {MODEL_PATH}")
    print(f"Feature metadata saved → {FEATURES_META_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()