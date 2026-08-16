"""
Phase 4 – Artificial Neural Network (ANN) for Churn Prediction
Customer Behavior Prediction Platform
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, metrics, optimizers
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)


def load_features(path: str = "data/processed/customer_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded feature table: {df.shape[0]:,} customers")
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test, scaler, feature_cols)


def build_ann(input_dim: int):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", metrics.AUC(name="auc")]
    )
    return model


def evaluate_ann(model, X, y, set_name: str):
    y_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"\n{'='*55}")
    print(f"ANN Results – {set_name}")
    print(f"{'='*55}")
    print(f"Accuracy   : {accuracy_score(y, y_pred):.4f}")
    print(f"Precision  : {precision_score(y, y_pred):.4f}")
    print(f"Recall     : {recall_score(y, y_pred):.4f}")
    print(f"F1-Score   : {f1_score(y, y_pred):.4f}")
    print(f"ROC-AUC    : {roc_auc_score(y, y_prob):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    df = load_features()
    (X_train, X_val, X_test,
     y_train, y_val, y_test, scaler, feature_cols) = prepare_data(df)

    model = build_ann(input_dim=X_train.shape[1])
    print("\nANN Architecture:")
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_auc", patience=10, mode="max", restore_best_weights=True
    )

    print("\nTraining ANN...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    evaluate_ann(model, X_val, y_val, "Validation")
    evaluate_ann(model, X_test, y_test, "Test")

    model_path = Path("models_artifacts/ann_churn_model.keras")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"\nANN model saved → {model_path}")