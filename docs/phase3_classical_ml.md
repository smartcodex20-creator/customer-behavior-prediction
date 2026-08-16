# Phase 3 – Classical Machine Learning & Segmentation Documentation

**Project:** Customer Behavior Prediction Platform  
**Last Updated:** 16 August 2026

---

## 1. Objective

Train and evaluate classical machine learning models for Churn Prediction and Customer Lifetime Value (CLV), and perform customer segmentation as required by the PRD.

---

## 2. Data Split Strategy

- Method: Stratified Train / Validation / Test split
- Ratio: 70% / 15% / 15%
- Random State: 42
- Target: Churn

| Set        | Customers |
|------------|-----------|
| Train      | 3,719     |
| Validation | 797       |
| Test       | 798       |

---

## 3. Churn Classification Models

### Models Trained
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- KNN (distance-based model required by PRD)

### Test Set Results (Final)

| Rank | Model                 | ROC-AUC | Recall | F1     |
|------|-----------------------|---------|--------|--------|
| 1    | Logistic Regression   | 0.7997  | 0.7162 | 0.7497 |
| 2    | Random Forest         | 0.7996  | 0.7620 | 0.7696 |
| 3    | XGBoost               | 0.7944  | 0.7467 | 0.7500 |
| 4    | LightGBM              | 0.7855  | 0.7533 | 0.7533 |
| 5    | Decision Tree         | 0.7712  | 0.8275 | 0.7758 |
| 6    | KNN                   | 0.7599  | 0.7555 | 0.7465 |

**Best Models:** Logistic Regression and Random Forest (virtually tied)

---

## 4. Customer Lifetime Value (CLV) Regression

### Models Trained
- Random Forest Regressor
- XGBoost Regressor

### Results

| Model                     | MAE     | RMSE    | R²     |
|---------------------------|---------|---------|--------|
| Random Forest Regressor   | £377    | £2,271  | 0.9351 |
| XGBoost Regressor         | £459    | £3,881  | 0.8105 |

**Best Model:** Random Forest Regressor (R² = 0.9351)  
This significantly exceeds the PRD target of R² ≥ 0.60.

---

## 5. Customer Segmentation

### Methods Used
- K-Means (primary)
- Gaussian Mixture Model (GMM)

### Optimal Number of Clusters
- Best k = 8 (based on Silhouette Score = 0.3389)

### Key Segment Insights (K-Means)

| Segment | % of Customers | Character                        | Churn Rate |
|---------|----------------|----------------------------------|------------|
| 0       | 42.3%          | Low frequency, high recency      | 77%        |
| 1       | 21.1%          | Medium engagement                | 55%        |
| 2       | 25.3%          | High frequency & monetary        | 22%        |
| 3       | 0.4%           | Extremely high monetary (VIP)    | 5%         |
| 4       | 1.8%           | High return rate                 | 95%        |

---

## 6. Output Files

- `src/models/train_churn_models.py`
- `src/models/train_clv_model.py`
- `src/segmentation/customer_segmentation.py`
- `data/processed/customer_segments.csv` (generated)

---

## 7. Next Steps

- Phase 4: Deep Learning models (ANN, LSTM, Autoencoder)
- Explainability (SHAP & LIME)