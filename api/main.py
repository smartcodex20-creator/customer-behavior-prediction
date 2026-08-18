"""
Customer Behavior Prediction Platform – FastAPI Service
Phase 5 – Deployment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="Customer Behavior Prediction API",
    description="Churn prediction and customer insights API for Vantara Retail Solutions",
    version="1.0.0"
)

# Enable CORS so the frontend can talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------------------
# Load model and data at startup
# --------------------------------------------------
FEATURES_PATH = Path("data/processed/customer_features.csv")
feature_cols = [
    "Recency", "Frequency", "Monetary_Positive",
    "Avg_Basket_Size", "Total_Quantity", "N_Transactions",
    "N_Returns", "Return_Rate", "Customer_Age_Days",
    "Frequency_Trend", "Avg_Days_Between", "Std_Days_Between",
    "Engagement_Score"
]

# Train a lightweight model at startup (in production this would be a saved artifact)
df = pd.read_csv(FEATURES_PATH)
X = df[feature_cols].fillna(0)
y = df["Churn"]

model = RandomForestClassifier(
    n_estimators=150, max_depth=8, random_state=42,
    class_weight="balanced", n_jobs=-1
)
model.fit(X, y)

print("Model loaded and ready.")


# --------------------------------------------------
# Pydantic Schemas
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
# Helper
# --------------------------------------------------
def get_risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "High"
    elif prob >= 0.45:
        return "Medium"
    else:
        return "Low"


# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Customer Behavior Prediction API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    return {
        "status": "healthy",
        "model": "RandomForestClassifier",
        "version": "1.0.0"
    }


@app.get("/model-info", tags=["Monitoring"])
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "features_used": feature_cols,
        "n_features": len(feature_cols),
        "training_customers": len(df)
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
            customer.Engagement_Score
        ]])

        prob = float(model.predict_proba(data)[0][1])
        pred = int(prob >= 0.5)
        risk = get_risk_level(prob)

        message = (
            f"Customer is at {risk} risk of churning "
            f"(probability: {prob:.1%})."
        )

        return {
            "Customer_ID": customer.Customer_ID,
            "Churn_Probability": round(prob, 4),
            "Churn_Prediction": pred,
            "Risk_Level": risk,
            "Message": message
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: List[CustomerFeatures]):
    results = []
    for customer in customers:
        result = predict_single(customer)
        results.append(result)
    return {"predictions": results, "count": len(results)}


@app.get("/customer/{customer_id}", tags=["Customer Lookup"])
def get_customer(customer_id: int):
    row = df[df["Customer_ID"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    row = row.iloc[0]
    features = CustomerFeatures(
        Customer_ID=int(row["Customer_ID"]),
        Recency=float(row["Recency"]),
        Frequency=float(row["Frequency"]),
        Monetary_Positive=float(row["Monetary_Positive"]),
        Avg_Basket_Size=float(row["Avg_Basket_Size"]),
        Total_Quantity=float(row["Total_Quantity"]),
        N_Transactions=float(row["N_Transactions"]),
        N_Returns=float(row["N_Returns"]),
        Return_Rate=float(row["Return_Rate"]),
        Customer_Age_Days=float(row["Customer_Age_Days"]),
        Frequency_Trend=float(row["Frequency_Trend"]),
        Avg_Days_Between=float(row["Avg_Days_Between"]),
        Std_Days_Between=float(row["Std_Days_Between"]),
        Engagement_Score=float(row["Engagement_Score"])
    )


    return predict_single(features)
@app.get("/metrics", tags=["Monitoring"])
def get_overview_metrics():
    """Return real overview metrics for the dashboard."""
    total_customers = len(df)
    churn_rate = round(df["Churn"].mean() * 100, 1)
    low_engagement = int((df["Engagement_Score"] < 2.0).sum())
    avg_value = round(df["Monetary_Positive"].mean(), 0)

    return {
        "total_customers": total_customers,
        "churn_rate": churn_rate,
        "low_engagement": low_engagement,
        "avg_customer_value": avg_value
    }

@app.get("/leaderboard", tags=["Customer Lookup"])
def get_leaderboard(limit: int = 20):
    """
    Return top high-risk customers sorted by a simple Risk Score.
    Risk Score = Recency * 0.5 + (5 - Engagement_Score) * 30
    """
    temp = df.copy()
    temp["Risk_Score"] = (
        temp["Recency"] * 0.5 +
        (5 - temp["Engagement_Score"]) * 30
    )

    top = temp.nlargest(limit, "Risk_Score")[
        ["Customer_ID", "Recency", "Frequency", "Monetary_Positive",
         "Engagement_Score", "Churn", "Risk_Score"]
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
            "Risk_Score": round(float(row["Risk_Score"]), 1)
        })

    return {
        "count": len(customers),
        "customers": customers
    }

@app.get("/charts/distributions", tags=["Monitoring"])
def get_chart_distributions():
    """
    Return real distribution data for Recency and Engagement Score
    to be used in the Overview charts.
    """
    # Recency bins
    recency_bins = [0, 15, 30, 45, 60, 90, float("inf")]
    recency_labels = ["0-15", "15-30", "30-45", "45-60", "60-90", "90+"]
    recency_counts = []

    for i in range(len(recency_bins) - 1):
        count = int(
            ((df["Recency"] >= recency_bins[i]) & (df["Recency"] < recency_bins[i+1])).sum()
        )
        recency_counts.append(count)

    # Engagement Score bins
    engagement_bins = [0, 1, 2, 3, 4, 5, float("inf")]
    engagement_labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5+"]
    engagement_counts = []

    for i in range(len(engagement_bins) - 1):
        count = int(
            ((df["Engagement_Score"] >= engagement_bins[i]) &
             (df["Engagement_Score"] < engagement_bins[i+1])).sum()
        )
        engagement_counts.append(count)

    return {
        "recency": {
            "labels": recency_labels,
            "values": recency_counts
        },
        "engagement": {
            "labels": engagement_labels,
            "values": engagement_counts
        }
    }