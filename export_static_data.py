import json
from pathlib import Path
import urllib.request
import pandas as pd

BASE = "http://127.0.0.1:8000"
OUT = Path("frontend/data")
OUT.mkdir(parents=True, exist_ok=True)

def get_json(path: str):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

# 1) Metrics
metrics = get_json("/metrics")
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("Saved metrics.json")

# 2) Leaderboard
leaderboard = get_json("/leaderboard?limit=50")
(OUT / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
print("Saved leaderboard.json")

# 3) Customer lookup table
features_path = Path("data/processed/customer_features.csv")
df = pd.read_csv(features_path)

customers = {}
for _, row in df.iterrows():
    cid = int(row["Customer_ID"])

    if "Churn_Probability" in df.columns:
        prob = float(row["Churn_Probability"])
    else:
        prob = 0.85 if int(row["Churn"]) == 1 else 0.20

    risk = "High" if prob >= 0.70 else ("Medium" if prob >= 0.40 else "Low")

    customers[str(cid)] = {
        "Customer_ID": cid,
        "Churn_Probability": prob,
        "Churn_Prediction": int(row["Churn"]),
        "Risk_Level": risk,
        "Message": (
            "High churn risk. Prioritize retention outreach."
            if risk == "High"
            else (
                "Moderate churn risk. Monitor engagement."
                if risk == "Medium"
                else "Low churn risk. Customer appears stable."
            )
        ),
        "Recency": float(row.get("Recency", 0)),
        "Frequency": float(row.get("Frequency", 0)),
        "Monetary_Positive": float(row.get("Monetary_Positive", row.get("Monetary", 0))),
        "Engagement_Score": float(row.get("Engagement_Score", 0)),
    }

(OUT / "customers.json").write_text(json.dumps(customers), encoding="utf-8")
print(f"Saved customers.json ({len(customers)} customers)")
print("Done.")