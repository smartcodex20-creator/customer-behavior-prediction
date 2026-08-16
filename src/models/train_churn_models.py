"""
Phase 3 – Classical Machine Learning Models (Final Version)
Customer Behavior Prediction Platform
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")


def load_features(path: str = "data/processed/customer_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded feature table: {df.shape[0]:,} customers × {df.shape[1]} columns")
    print(f"Churn rate: {df['Churn'].mean()*100:.2f}%")
    return df


def prepare_data(df: pd.DataFrame):
    feature_cols = [
        "Recency", "Frequency", "Monetary", "Monetary_Positive",
        "Avg_Basket_Size", "Total_Quantity", "N_Transactions",
        "N_Returns", "Return_Rate", "Customer_Age_Days",
        "Frequency_Trend", "Avg_Days_Between", "Std_Days_Between",
        "Engagement_Score"
    ]

    X = df[feature_cols].copy().fillna(0)
    y = df["Churn"].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print("\nData Split Summary:")
    print(f"Train      : {len(X_train):,}")
    print(f"Validation : {len(X_val):,}")
    print(f"Test       : {len(X_test):,}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def evaluate_model(model, X, y, model_name: str, scaler=None):
    if scaler is not None:
        X = scaler.transform(X)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y, y_pred), 4),
        "Precision": round(precision_score(y, y_pred), 4),
        "Recall": round(recall_score(y, y_pred), 4),
        "F1": round(f1_score(y, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y, y_prob), 4)
    }
    return metrics


if __name__ == "__main__":
    df = load_features()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_data(df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # Train all models
    # ==========================================
    models = {}

    print("\nTraining all models...")

    # 1. Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = (log_reg, True)

    # 2. Decision Tree
    tree = DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced")
    tree.fit(X_train, y_train)
    models["Decision Tree"] = (tree, False)

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    models["Random Forest"] = (rf, False)

    # 4. XGBoost
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                        random_state=42, eval_metric="logloss",
                        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
    xgb.fit(X_train, y_train)
    models["XGBoost"] = (xgb, False)

    # 5. LightGBM
    lgbm = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                          random_state=42, class_weight="balanced", verbose=-1)
    lgbm.fit(X_train, y_train)
    models["LightGBM"] = (lgbm, False)

    # 6. KNN (Distance-based model required by PRD)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train_scaled, y_train)
    models["KNN"] = (knn, True)

    # ==========================================
    # Evaluate on Validation Set
    # ==========================================
    print("\n" + "="*70)
    print("VALIDATION SET RESULTS")
    print("="*70)

    val_results = []
    for name, (model, use_scaler) in models.items():
        metrics = evaluate_model(model, X_val, y_val, name, scaler if use_scaler else None)
        val_results.append(metrics)

    val_df = pd.DataFrame(val_results).sort_values("ROC-AUC", ascending=False)
    print(val_df.to_string(index=False))

    # ==========================================
    # Evaluate on Test Set (Final)
    # ==========================================
    print("\n" + "="*70)
    print("TEST SET RESULTS (Final Held-out Evaluation)")
    print("="*70)

    test_results = []
    for name, (model, use_scaler) in models.items():
        metrics = evaluate_model(model, X_test, y_test, name, scaler if use_scaler else None)
        test_results.append(metrics)

    test_df = pd.DataFrame(test_results).sort_values("ROC-AUC", ascending=False)
    print(test_df.to_string(index=False))

    print("\nBest model on Test Set:", test_df.iloc[0]["Model"])
    print(f"Best ROC-AUC on Test Set: {test_df.iloc[0]['ROC-AUC']}")