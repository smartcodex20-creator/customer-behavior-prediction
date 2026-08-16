# Phase 4 – Deep Learning & Explainability Documentation

**Project:** Customer Behavior Prediction Platform  
**Last Updated:** 17 August 2026

---

## 1. Objective

Build the deep learning models required by the PRD and provide model explainability using SHAP and LIME.

---

## 2. Models Implemented

### 2.1 Artificial Neural Network (ANN)
- Feed-forward network with Batch Normalization and Dropout
- Architecture: 64 → 32 → 16 → 1
- Early stopping on validation AUC

**Test Set Results:**
- ROC-AUC: 0.7968
- Recall: 0.7751
- F1-Score: 0.7626

### 2.2 LSTM (Sequential Model)
- Trained on last 5 purchase sequences per customer
- Features per time step: Revenue, Days Since Previous Purchase, Quantity
- Used for modeling sequential purchase behavior

**Test Set Results:**
- ROC-AUC: 0.7429
- Recall: 0.7926
- F1-Score: 0.7423

**Note:** LSTM performance is lower than classical models and ANN, which is expected given the limited number of customers and short sequences. This aligns with the PRD guidance that deep learning models serve partly as comparative exercises.

### 2.3 Autoencoder (Anomaly Detection)
- Unsupervised model trained on customer spending-pattern features
- Reconstruction error used to flag anomalies
- Threshold set at 95th percentile

**Results:**
- Anomalous customers flagged: 266 (5.01%)
- Successfully identified extreme high-value customers and high-return customers

---

## 3. Explainability

### 3.1 SHAP (Global Feature Importance)

| Rank | Feature            | Mean \|SHAP\| |
|------|--------------------|---------------|
| 1    | Recency            | 0.1034        |
| 2    | Engagement_Score   | 0.0428        |
| 3    | N_Transactions     | 0.0344        |
| 4    | Total_Quantity     | 0.0301        |
| 5    | Frequency          | 0.0229        |

**Key Insight:** Recency is by far the strongest predictor of churn.

### 3.2 LIME
- Local explanation generated for individual customers
- Plain-language explanation template created for business users

---

## 4. Output Artifacts

- `models_artifacts/ann_churn_model.keras`
- `models_artifacts/lstm_churn_model.keras`
- `models_artifacts/autoencoder_anomaly.keras`
- `data/processed/shap_global_importance.csv`
- `data/processed/customer_anomalies.csv`
- `src/models/train_ann.py`
- `src/models/train_lstm.py`
- `src/models/train_autoencoder.py`
- `src/explainability/model_explainability.py`

---

## 5. Summary

Phase 4 successfully delivered all required deep learning models and explainability components as specified in the PRD. The ANN performed competitively with classical models, while the LSTM and Autoencoder fulfilled their comparative and anomaly-detection roles.

---

## 6. Next Steps

- Phase 5: FastAPI service + Streamlit Dashboard + Docker