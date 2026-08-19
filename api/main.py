"""
Customer Behavior Prediction Platform – FastAPI Service
Phase 5 – Deployment (Serving Quality Upgrade)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Customer Behavior Prediction API",
    description="Churn prediction and customer insights API for Vantara Retail Solutions",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Paths and startup loading
# --------------------------------------------------
FEATURES_PATH = Path("data/processed/customer_features.csv")
MODEL_PATH = Path("models_artifacts/churn_serving_model.joblib")
FEATURES_META_PATH = Path("models_artifacts/serving_features.json")

if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not FEATURES_META_PATH.exists():
    raise FileNotFoundError(f"Feature metadata not found: {FEATURES_META_PATH}")

df = pd.read_csv(FEATURES_PATH)

with open(FEATURES_META_PATH, "r", encoding="utf-8") as f:
    feature_cols = json.load(f)["feature_cols"]

model = joblib.load(MODEL_PATH)

print("Saved model loaded and ready.")
print(f"Features used: {len(feature_cols)}")
print(f"Customers available: {len(df):,}")


# --------------------------------------------------
# Schemas
# --------------------------------------------------
class CustomerFeatures(BaseModel):
    Customer_ID: Optional[int] = None
    Recency: float
    Frequency: float
    Monetary_Positive: float
    Avg_Basket_Size: float
    Total_Quantity: float
    N_Transactions: float
    N_Returns: float
    Return_Rate: float
    Customer_Age_Days: float
    Frequency_Trend: float
    Avg_Days_Between: float
    Std_Days_Between: float
    Engagement_Score: float


class PredictionResponse(BaseModel):
    Customer_ID: Optional[int]
    Churn_Probability: float
    Churn_Prediction: int
    Risk_Level: str
    Message: str


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def get_risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "High"
    elif prob >= 0.45:
        return "Medium"
    return "Low"


def row_to_feature_vector(row) -> np.ndarray:
    return np.array([[float(row[col]) if pd.notna(row[col]) else 0.0 for col in feature_cols]])


def predict_from_row(row) -> dict:
    data = row_to_feature_vector(row)
    prob = float(model.predict_proba(data)[0][1])
    pred = int(prob >= 0.5)
    risk = get_risk_level(prob)
    customer_id = int(row["Customer_ID"]) if "Customer_ID" in row and pd.notna(row["Customer_ID"]) else None

    return {
        "Customer_ID": customer_id,
        "Churn_Probability": round(prob, 4),
        "Churn_Prediction": pred,
        "Risk_Level": risk,
        "Message": f"Customer is at {risk} risk of churning (probability: {prob:.1%}).",
    }


# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Customer Behavior Prediction API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    return {
        "status": "healthy",
        "model": "RandomForestClassifier (saved artifact)",
        "version": "1.1.0",
    }


@app.get("/model-info", tags=["Monitoring"])
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "model_source": str(MODEL_PATH),
        "features_used": feature_cols,
        "n_features": len(feature_cols),
        "customers_available": len(df),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(customer: CustomerFeatures):
    try:
        data = np.array([[
            customer.Recency,
            customer.Frequency,
            customer.Monetary_Positive,
            customer.Avg_Basket_Size,
            customer.Total_Quantity,
            customer.N_Transactions,
            customer.N_Returns,
            customer.Return_Rate,
            customer.Customer_Age_Days,
            customer.Frequency_Trend,
            customer.Avg_Days_Between,
            customer.Std_Days_Between,
            customer.Engagement_Score,
        ]])

        prob = float(model.predict_proba(data)[0][1])
        pred = int(prob >= 0.5)
        risk = get_risk_level(prob)

        return {
            "Customer_ID": customer.Customer_ID,
            "Churn_Probability": round(prob, 4),
            "Churn_Prediction": pred,
            "Risk_Level": risk,
            "Message": f"Customer is at {risk} risk of churning (probability: {prob:.1%}).",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: List[CustomerFeatures]):
    results = [predict_single(customer) for customer in customers]
    return {"predictions": results, "count": len(results)}


@app.get("/customer/{customer_id}", tags=["Customer Lookup"])
def get_customer(customer_id: int):
    row = df[df["Customer_ID"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return predict_from_row(row.iloc[0])


@app.get("/metrics", tags=["Monitoring"])
def get_overview_metrics():
    return {
        "total_customers": len(df),
        "churn_rate": round(df["Churn"].mean() * 100, 1),
        "low_engagement": int((df["Engagement_Score"] < 2.0).sum()),
        "avg_customer_value": round(df["Monetary_Positive"].mean(), 0),
    }


@app.get("/leaderboard", tags=["Customer Lookup"])
def get_leaderboard(limit: int = 20):
    """
    Return top high-risk customers ranked by model churn probability.
    """
    temp = df.copy()
    X = temp[feature_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1]
    temp["Churn_Probability"] = probs
    temp["Risk_Score"] = (temp["Churn_Probability"] * 100).round(1)

    top = temp.nlargest(limit, "Churn_Probability")[
        ["Customer_ID", "Recency", "Frequency", "Monetary_Positive",
         "Engagement_Score", "Churn", "Churn_Probability", "Risk_Score"]
    ]

    customers = []
    for _, row in top.iterrows():
        customers.append({
            "Customer_ID": int(row["Customer_ID"]),
            "Recency": int(row["Recency"]),
            "Frequency": int(row["Frequency"]),
            "Monetary_Positive": round(float(row["Monetary_Positive"]), 2),
            "Engagement_Score": round(float(row["Engagement_Score"]), 2),
            "Churn": int(row["Churn"]),
            "Churn_Probability": round(float(row["Churn_Probability"]), 4),
            "Risk_Score": round(float(row["Risk_Score"]), 1),
        })

    return {
        "count": len(customers),
        "ranking_method": "model_churn_probability",
        "customers": customers,
    }


@app.get("/charts/distributions", tags=["Monitoring"])
def get_chart_distributions():
    recency_bins = [0, 15, 30, 45, 60, 90, float("inf")]
    recency_labels = ["0-15", "15-30", "30-45", "45-60", "60-90", "90+"]
    recency_counts = []
    for i in range(len(recency_bins) - 1):
        count = int(((df["Recency"] >= recency_bins[i]) & (df["Recency"] < recency_bins[i + 1])).sum())
        recency_counts.append(count)

    engagement_bins = [0, 1, 2, 3, 4, 5, float("inf")]
    engagement_labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5+"]
    engagement_counts = []
    for i in range(len(engagement_bins) - 1):
        count = int(((df["Engagement_Score"] >= engagement_bins[i]) & (df["Engagement_Score"] < engagement_bins[i + 1])).sum())
        engagement_counts.append(count)

    return {
        "recency": {"labels": recency_labels, "values": recency_counts},
        "engagement": {"labels": engagement_labels, "values": engagement_counts},
    }