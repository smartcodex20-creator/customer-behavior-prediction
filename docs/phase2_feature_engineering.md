# Phase 2 – Feature Engineering Documentation

**Project:** Customer Behavior Prediction Platform  
**Last Updated:** 16 August 2026

---

## 1. Objective

Create a leakage-free customer-level feature table and a proper churn target variable, fully aligned with the PRD (Section 7).

---

## 2. Point-in-Time Approach (PRD Requirement)

To avoid target leakage, we used a strict time-based cutoff:

- **Cutoff Date:** 2011-09-01
- **Features** calculated using data **before** 2011-09-01
- **Churn Label** defined using the next 90 days (2011-09-01 to 2011-11-30)

### Logic:
- If a customer made **at least one purchase** in the 90-day observation window → `Churn = 0`
- If a customer made **no purchase** in that window → `Churn = 1`

---

## 3. Features Created

| Feature                  | Description                                      | Type      |
|--------------------------|--------------------------------------------------|-----------|
| Recency                  | Days since last purchase before cutoff           | Numeric   |
| Frequency                | Number of unique invoices before cutoff          | Numeric   |
| Monetary                 | Total net revenue before cutoff                  | Numeric   |
| Monetary_Positive        | Monetary value clipped at 0                      | Numeric   |
| Avg_Basket_Size          | Average revenue per transaction                  | Numeric   |
| Total_Quantity           | Total items purchased                            | Numeric   |
| N_Transactions           | Total number of transaction rows                 | Numeric   |
| N_Returns                | Number of return transactions                    | Numeric   |
| Return_Rate              | N_Returns / N_Transactions                       | Numeric   |
| Customer_Age_Days        | Days between first and last purchase             | Numeric   |
| Avg_Days_Between_Orders  | Customer_Age_Days / Frequency                    | Numeric   |
| Engagement_Score         | Weighted combination of Recency, Frequency, Monetary | Numeric |
| Churn                    | Target variable (1 = churned, 0 = active)        | Binary    |

---

## 4. Key Results

- Customers with features: **5,314**
- Churned customers: **3,052** (57.43%)
- Active customers: **2,262** (42.57%)

**Note:** The relatively high churn rate is expected because many customers in this dataset are infrequent or one-time buyers. Class imbalance will be handled during modeling.

---

## 5. Output File

- `data/processed/customer_features.csv` (generated, not committed to Git)

---

## 6. Next Steps

- Review feature distributions and correlations
- Decide if additional features from the PRD are required before modeling
- Prepare data for train/validation/test split