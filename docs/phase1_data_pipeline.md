\# Phase 1 – Data Pipeline Documentation



\*\*Project:\*\* Customer Behavior Prediction Platform  

\*\*Dataset:\*\* Online Retail II (UCI Machine Learning Repository)  

\*\*Last Updated:\*\* 15 August 2026



\---



\## 1. Phase 0 Summary (Completed)



\- Project initialized on `D:\\customer-behavior-prediction`

\- Exact folder structure created as per PRD Section 15

\- Git repository initialized and connected to GitHub

\- Python 3.12.9 virtual environment created

\- First clean commit pushed to GitHub



\*\*GitHub Repository:\*\*  

https://github.com/smartcodex20-creator/customer-behavior-prediction



\---



\## 2. Data Loading



\*\*Source File:\*\* `data/raw/online\_retail\_II.xlsx`  

\*\*Method:\*\* Manual download (ucimlrepo programmatic method was not available for this dataset)



\*\*Loading Steps:\*\*

\- Loaded sheet `Year 2009-2010`

\- Loaded sheet `Year 2010-2011`

\- Combined both sheets into a single DataFrame



\*\*Result:\*\*

\- Total rows loaded: \*\*1,067,371\*\*

\- Total columns: \*\*8\*\*

\- Columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`



Combined raw data saved as:  

`data/raw/online\_retail\_ii\_combined.csv`



\---



\## 3. Initial Data Inspection



| Issue                        | Count / Percentage      | Notes                                      |

|-----------------------------|-------------------------|--------------------------------------------|

| Missing Customer ID         | 243,007 (22.77%)        | Must be excluded for customer-level models |

| Missing Description         | 4,382 (0.41%)           | Minor issue                                |

| Negative Quantity (Returns) | Present                 | Important predictive signal – keep \& flag  |

| Zero / Negative Price       | Present                 | Requires cleaning rule                     |



\---



\## 4. Data Cleaning Performed



\*\*Script:\*\* `src/data/clean\_data.py`



\### Operations Performed:

1\. Standardized column names (removed spaces → `Customer\_ID`)

2\. Converted `Customer\_ID` to nullable integer (`Int64`)

3\. Flagged returns (`Is\_Return = Quantity < 0`)

4\. Flagged invalid prices (`Invalid\_Price = Price <= 0`)

5\. Removed exact duplicate rows



\### Cleaning Results:



| Metric                        | Value       |

|------------------------------|-------------|

| Original rows                | 1,067,371   |

| Exact duplicates removed     | 34,335      |

| Rows after cleaning          | 1,033,036   |

| Rows with valid Customer ID  | 797,885     |

| Rows without Customer ID     | 235,151     |

| Return transactions          | 22,496      |

| Invalid price rows           | 6,019       |



\### Output Files:

\- `data/interim/customer\_transactions.csv` → Used for customer-level modeling

\- `data/interim/all\_transactions.csv` → Full data for product-level analysis



\---



\## 5. Next Steps



\- Deep Data Validation (schema checks, outlier analysis, date range validation)

\- Exploratory Data Analysis (EDA)

\- Feature Engineering (RFM, CLV, etc.)



\---



\*\*Status:\*\* Data loading and basic cleaning completed successfully.

