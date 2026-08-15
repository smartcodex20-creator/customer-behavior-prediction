# Phase 1 – Data Pipeline Documentation

**Project:** Customer Behavior Prediction Platform  
**Dataset:** Online Retail II (UCI Machine Learning Repository)  
**Last Updated:** 16 August 2026

---

## 1. Phase 0 Summary (Completed)

- Project initialized on `D:\customer-behavior-prediction`
- Exact folder structure created as per PRD Section 15
- Git repository initialized and connected to GitHub
- Python 3.12.9 virtual environment created
- First clean commit pushed to GitHub

**GitHub Repository:**  
https://github.com/smartcodex20-creator/customer-behavior-prediction

---

## 2. Data Loading

**Source File:** `data/raw/online_retail_II.xlsx`  
**Method:** Manual download (ucimlrepo programmatic method was not available for this dataset)

**Loading Steps:**
- Loaded sheet `Year 2009-2010`
- Loaded sheet `Year 2010-2011`
- Combined both sheets into a single DataFrame

**Result:**
- Total rows loaded: **1,067,371**
- Total columns: **8**
- Columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`

Combined raw data saved as:  
`data/raw/online_retail_ii_combined.csv`

---

## 3. Initial Data Inspection

| Issue                        | Count / Percentage      | Notes                                      |
|-----------------------------|-------------------------|--------------------------------------------|
| Missing Customer ID         | 243,007 (22.77%)        | Must be excluded for customer-level models |
| Missing Description         | 4,382 (0.41%)           | Minor issue                                |
| Negative Quantity (Returns) | Present                 | Important predictive signal – keep & flag  |
| Zero / Negative Price       | Present                 | Requires cleaning rule                     |

---

## 4. Data Cleaning Performed

**Script:** `src/data/clean_data.py`

### Operations Performed:
1. Standardized column names (removed spaces → `Customer_ID`)
2. Converted `Customer_ID` to nullable integer (`Int64`)
3. Flagged returns (`Is_Return = Quantity < 0`)
4. Flagged invalid prices (`Invalid_Price = Price <= 0`)
5. Removed exact duplicate rows

### Cleaning Results:

| Metric                        | Value       |
|------------------------------|-------------|
| Original rows                | 1,067,371   |
| Exact duplicates removed     | 34,335      |
| Rows after cleaning          | 1,033,036   |
| Rows with valid Customer ID  | 797,885     |
| Rows without Customer ID     | 235,151     |
| Return transactions          | 22,496      |
| Invalid price rows           | 6,019       |

### Output Files:
- `data/interim/customer_transactions.csv` → Used for customer-level modeling
- `data/interim/all_transactions.csv` → Full data for product-level analysis

---

## 5. Deep Data Validation

**Script:** `src/data/validate_data.py`

### Key Validation Results:
- All expected columns present
- Date range: 2009-12-01 → 2011-12-09
- Unique customers: 5,942
- Unique invoices: 44,876
- Returns remaining: 18,390
- Invalid prices remaining: 70

---

## 6. Exploratory Data Analysis (EDA)

**Notebook:** `notebooks/01_eda.ipynb`

### Key Findings:
- Total Revenue: **£16.29 million**
- Average Order Value: £363 | Median: £236
- Returns: 2.30% of transactions (18,390 rows)
- 2,572 customers made at least one return
- Strong seasonal peaks in November (2010 & 2011)
- United Kingdom accounts for ~90% of transactions
- No missing values in the cleaned customer-level dataset

### Hypotheses Generated:
1. High Recency → higher churn risk
2. Low Frequency → higher churn risk
3. High Monetary value customers should be prioritized even if at risk
4. Higher return rate → higher churn risk
5. Country and season affect purchase behavior and churn
6. Declining purchase frequency over time → higher churn risk

---

## 7. Next Steps

- Feature Engineering (RFM, CLV, engagement score, return rate, trends)
- Create customer-level feature table
- Add leakage prevention tests

---

**Status:** Phase 1 (Data Loading, Cleaning, Validation & EDA) completed successfully.