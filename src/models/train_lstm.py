"""
Phase 4 – LSTM for Sequential Purchase Behavior
Customer Behavior Prediction Platform

Creates purchase sequences for each customer and trains an LSTM
to predict churn (PRD requirement).
"""

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

SEQUENCE_LENGTH = 5   # last 5 purchases
CUTOFF_DATE = "2011-09-01"


def load_transactions(path: str = "data/interim/customer_transactions.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    print(f"Loaded {len(df):,} transactions")
    return df


def create_sequences(df: pd.DataFrame, cutoff: str = CUTOFF_DATE, seq_len: int = SEQUENCE_LENGTH):
    """
    Create fixed-length purchase sequences for each customer.
    Features per step: [Revenue, Days_Since_Previous, Quantity]
    """
    cutoff = pd.Timestamp(cutoff)
    df = df[df["InvoiceDate"] < cutoff].copy()
    df["Revenue"] = df["Quantity"] * df["Price"]
    df = df.sort_values(["Customer_ID", "InvoiceDate"])

    # Calculate days since previous purchase
    df["Days_Since_Prev"] = df.groupby("Customer_ID")["InvoiceDate"].diff().dt.days.fillna(0)

    sequences = []
    customer_ids = []

    for cust_id, group in df.groupby("Customer_ID"):
        # Aggregate to invoice level
        inv = group.groupby("Invoice").agg({
            "Revenue": "sum",
            "Quantity": "sum",
            "Days_Since_Prev": "first",
            "InvoiceDate": "first"
        }).sort_values("InvoiceDate")

        if len(inv) < 1:
            continue

        feats = inv[["Revenue", "Days_Since_Prev", "Quantity"]].values

        # Pad or truncate to fixed length
        if len(feats) >= seq_len:
            seq = feats[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(feats), 3))
            seq = np.vstack([pad, feats])

        sequences.append(seq)
        customer_ids.append(cust_id)

    X = np.array(sequences)
    print(f"Created sequences for {len(customer_ids):,} customers")
    print(f"Sequence shape: {X.shape}")  # (n_customers, seq_len, n_features)
    return X, customer_ids


def load_churn_labels(customer_ids, features_path: str = "data/processed/customer_features.csv"):
    """Match sequences with the existing churn labels."""
    feat = pd.read_csv(features_path)
    label_map = dict(zip(feat["Customer_ID"], feat["Churn"]))
    y = np.array([label_map.get(cid, 0) for cid in customer_ids])
    return y


def build_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", metrics.AUC(name="auc")]
    )
    return model


def evaluate(model, X, y, set_name: str):
    y_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"\n{'='*55}")
    print(f"LSTM Results – {set_name}")
    print(f"{'='*55}")
    print(f"Accuracy   : {accuracy_score(y, y_pred):.4f}")
    print(f"Precision  : {precision_score(y, y_pred):.4f}")
    print(f"Recall     : {recall_score(y, y_pred):.4f}")
    print(f"F1-Score   : {f1_score(y, y_pred):.4f}")
    print(f"ROC-AUC    : {roc_auc_score(y, y_prob):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    # 1. Create sequences
    df = load_transactions()
    X, customer_ids = create_sequences(df)
    y = load_churn_labels(customer_ids)

    print(f"Churn rate in sequences: {y.mean()*100:.2f}%")

    # 2. Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 3. Scale features across the sequence
    # Reshape for scaling → scale → reshape back
    n_samples, n_steps, n_feats = X_train.shape
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, n_feats)
    scaler.fit(X_train_reshaped)

    def scale_sequences(X):
        shape = X.shape
        X_reshaped = X.reshape(-1, shape[-1])
        X_scaled = scaler.transform(X_reshaped)
        return X_scaled.reshape(shape)

    X_train = scale_sequences(X_train)
    X_val = scale_sequences(X_val)
    X_test = scale_sequences(X_test)

    # 4. Build & train LSTM
    model = build_lstm(input_shape=(SEQUENCE_LENGTH, 3))
    print("\nLSTM Architecture:")
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_auc", patience=8, mode="max", restore_best_weights=True
    )

    print("\nTraining LSTM...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # 5. Evaluate
    evaluate(model, X_val, y_val, "Validation")
    evaluate(model, X_test, y_test, "Test")

    # 6. Save
    model_path = Path("models_artifacts/lstm_churn_model.keras")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"\nLSTM model saved → {model_path}")