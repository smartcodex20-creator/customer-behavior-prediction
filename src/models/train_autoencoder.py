"""
Phase 4 – Autoencoder for Anomaly Detection
Customer Behavior Prediction Platform
"""
# pyright: reportMissingModuleSource=false, reportMissingImports=false

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)


def load_features(path: str = "data/processed/customer_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} customers")
    return df


def prepare_data(df: pd.DataFrame):
    feature_cols = [
        "Recency", "Frequency", "Monetary_Positive",
        "Avg_Basket_Size", "Total_Quantity", "N_Transactions",
        "Return_Rate", "Customer_Age_Days",
        "Frequency_Trend", "Avg_Days_Between", "Std_Days_Between",
        "Engagement_Score"
    ]

    X = df[feature_cols].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val = train_test_split(X_scaled, test_size=0.20, random_state=42)

    print(f"Train: {len(X_train):,} | Validation: {len(X_val):,}")
    return X_train, X_val, scaler, feature_cols, X_scaled, df


def build_autoencoder(input_dim: int):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation="relu")(input_layer)
    encoded = layers.Dense(16, activation="relu")(encoded)
    encoded = layers.Dense(8, activation="relu")(encoded)

    # Decoder
    decoded = layers.Dense(16, activation="relu")(encoded)
    decoded = layers.Dense(32, activation="relu")(decoded)
    decoded = layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="mse")
    return autoencoder


if __name__ == "__main__":
    df = load_features()
    X_train, X_val, scaler, feature_cols, X_scaled, df_full = prepare_data(df)

    model = build_autoencoder(input_dim=X_train.shape[1])
    print("\nAutoencoder Architecture:")
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    print("\nTraining Autoencoder...")
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Calculate reconstruction error for all customers
    X_pred = model.predict(X_scaled, verbose=0)
    mse = np.mean(np.square(X_scaled - X_pred), axis=1)

    # Set threshold at 95th percentile
    threshold = np.percentile(mse, 95)
    anomalies = mse > threshold

    print(f"\n{'='*55}")
    print("AUTOENCODER ANOMALY DETECTION RESULTS")
    print(f"{'='*55}")
    print(f"Reconstruction error threshold (95th percentile): {threshold:.4f}")
    print(f"Number of anomalous customers flagged: {anomalies.sum():,}")
    print(f"Percentage of customers flagged: {anomalies.mean()*100:.2f}%")

    # Show some anomalous customers
    df_full = df_full.copy()
    df_full["Reconstruction_Error"] = mse
    df_full["Is_Anomaly"] = anomalies

    print("\nTop 10 most anomalous customers:")
    print(df_full.nlargest(10, "Reconstruction_Error")[
        ["Customer_ID", "Recency", "Frequency", "Monetary_Positive", "Return_Rate", "Reconstruction_Error"]
    ].round(2).to_string(index=False))

    # Save model and results
    model_path = Path("models_artifacts/autoencoder_anomaly.keras")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"\nAutoencoder model saved → {model_path}")

    # Save anomaly flags
    anomaly_path = Path("data/processed/customer_anomalies.csv")
    df_full[["Customer_ID", "Reconstruction_Error", "Is_Anomaly"]].to_csv(anomaly_path, index=False)
    print(f"Anomaly results saved → {anomaly_path}")