"""
Phase 3 – Customer Segmentation
Customer Behavior Prediction Platform

- K-Means with Elbow + Silhouette analysis
- Gaussian Mixture Model (GMM)
- Business-readable segment profiles
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


def load_features(path: str = "data/processed/customer_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} customers")
    return df


def prepare_segmentation_data(df: pd.DataFrame):
    """Select and scale features for clustering."""
    feature_cols = [
        "Recency", "Frequency", "Monetary_Positive",
        "Avg_Basket_Size", "Return_Rate", "Engagement_Score",
        "Frequency_Trend", "Avg_Days_Between", "Std_Days_Between"
    ]

    X = df[feature_cols].copy().fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, feature_cols, scaler


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Find a good number of clusters using Inertia (Elbow) and Silhouette Score."""
    inertias = []
    silhouettes = []

    print("\nFinding optimal number of clusters...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        print(f"k={k} | Inertia: {kmeans.inertia_:.0f} | Silhouette: {silhouettes[-1]:.4f}")

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\nBest k according to Silhouette Score: {best_k}")
    return best_k, inertias, silhouettes


def create_segment_profiles(df: pd.DataFrame, labels: np.ndarray, method_name: str):
    """Create business-readable profiles for each segment."""
    df = df.copy()
    df["Segment"] = labels

    profile = df.groupby("Segment").agg(
        N_Customers=("Customer_ID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary_Positive", "mean"),
        Avg_Return_Rate=("Return_Rate", "mean"),
        Avg_Engagement=("Engagement_Score", "mean"),
        Churn_Rate=("Churn", "mean")
    ).round(2)

    profile["Churn_Rate"] = (profile["Churn_Rate"] * 100).round(1).astype(str) + "%"
    profile["Pct_of_Customers"] = (profile["N_Customers"] / len(df) * 100).round(1).astype(str) + "%"

    print(f"\n{'='*70}")
    print(f"{method_name} – Segment Profiles")
    print(f"{'='*70}")
    print(profile.to_string())

    return df, profile


if __name__ == "__main__":
    df = load_features()
    X, X_scaled, feature_cols, scaler = prepare_segmentation_data(df)

    # 1. Find optimal k
    best_k, inertias, silhouettes = find_optimal_k(X_scaled)

    # 2. K-Means with best k
    print(f"\nRunning K-Means with k={best_k}...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    df_kmeans, profile_kmeans = create_segment_profiles(df, kmeans_labels, "K-Means")

    # 3. Gaussian Mixture Model
    print(f"\nRunning Gaussian Mixture Model with {best_k} components...")
    gmm = GaussianMixture(n_components=best_k, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    df_gmm, profile_gmm = create_segment_profiles(df, gmm_labels, "Gaussian Mixture Model")

    # 4. Save segmented data
    output_path = Path("data/processed/customer_segments.csv")
    df_kmeans.to_csv(output_path, index=False)
    print(f"\nSegmented data saved → {output_path}")