"""
Phase 4 – Model Explainability (SHAP + LIME)
Customer Behavior Prediction Platform
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings("ignore")


def load_features(path: str = "data/processed/customer_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} customers")
    return df


def prepare_data(df: pd.DataFrame):
    feature_cols = [
        "Recency", "Frequency", "Monetary_Positive",
        "Avg_Basket_Size", "Total_Quantity", "N_Transactions",
        "N_Returns", "Return_Rate", "Customer_Age_Days",
        "Frequency_Trend", "Avg_Days_Between", "Std_Days_Between",
        "Engagement_Score"
    ]

    X = df[feature_cols].copy().fillna(0)
    y = df["Churn"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train, y_train):
    """Train a strong and interpretable model for explanations."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Random Forest model trained for explainability.")
    return model


def shap_explanations(model, X_train, X_test, feature_cols):
    """Generate SHAP global and local explanations."""
    print("\nCalculating SHAP values (this may take a moment)...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # Older format: list of arrays (one per class)
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        # Newer format: (n_samples, n_features, n_classes)
        shap_values = shap_values[:, :, 1]

    print(f"SHAP values shape: {shap_values.shape}")
    print("SHAP values calculated successfully.")

    # Global feature importance
    print("\n" + "="*60)
    print("SHAP GLOBAL FEATURE IMPORTANCE (Top Features)")
    print("="*60)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_|SHAP|": mean_abs_shap
    }).sort_values("Mean_|SHAP|", ascending=False)

    print(importance_df.to_string(index=False))

    # Save SHAP importance
    output_path = Path("data/processed/shap_global_importance.csv")
    importance_df.to_csv(output_path, index=False)
    print(f"\nSHAP global importance saved → {output_path}")

    return explainer, shap_values, importance_df


def lime_explanation(model, X_train, X_test, feature_cols, customer_index: int = 0):
    """Generate a LIME explanation for one customer."""
    print("\n" + "="*60)
    print(f"LIME LOCAL EXPLANATION (Customer index {customer_index})")
    print("="*60)

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_cols,
        class_names=["Active", "Churn"],
        mode="classification"
    )

    exp = explainer.explain_instance(
        data_row=X_test.iloc[customer_index].values,
        predict_fn=model.predict_proba,
        num_features=8
    )

    print(exp.as_list())

    # Plain language template
    print("\n" + "-"*60)
    print("PLAIN LANGUAGE EXPLANATION (Example)")
    print("-"*60)
    print("This customer is flagged as high churn risk mainly because of:")
    for feature, weight in exp.as_list()[:4]:
        direction = "increased" if weight > 0 else "decreased"
        print(f"  - {feature} ({direction} the churn probability)")

    return exp


if __name__ == "__main__":
    df = load_features()
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)

    model = train_model(X_train, y_train)

    # SHAP
    explainer, shap_values, importance_df = shap_explanations(
        model, X_train, X_test, feature_cols
    )

    # LIME – explain the first customer in the test set as an example
    lime_exp = lime_explanation(model, X_train, X_test, feature_cols, customer_index=0)

    print("\nExplainability completed successfully.")