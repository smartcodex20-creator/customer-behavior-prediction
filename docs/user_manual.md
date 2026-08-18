# User Manual
## Customer Behavior Prediction Platform

This manual explains how to run and use the Customer Behavior Prediction Platform.

---

## 1. System Overview

The platform has two main parts:

- **Backend API** - provides metrics, predictions, charts, and leaderboard data
- **Frontend Dashboard** - browser interface for business users

The system can be started in two ways:

- Local development mode
- Docker mode

---

## 2. Prerequisites

Before running the system, ensure the following are available:

- Python 3.12
- Project dependencies installed from `requirements.txt`
- For Docker mode: Docker Desktop installed and running

---

## 3. How to Start the System

### Option A: Local Development Mode

**Step 1: Open project folder**

- `D:`
- `cd D:\customer-behavior-prediction`

**Step 2: Activate virtual environment**

- `.venv\Scripts\activate`

**Step 3: Start the API**

- `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`

**Step 4: Start the Frontend**

Open a second terminal, then run:

- `D:`
- `cd D:\customer-behavior-prediction\frontend`
- `python -m http.server 5500`

**Step 5: Open the application**

- Dashboard: http://127.0.0.1:5500/index.html
- API Docs: http://127.0.0.1:8000/docs
- Health Check: http://127.0.0.1:8000/health

### Option B: Docker Mode

**Step 1: Open project folder**

- `D:`
- `cd D:\customer-behavior-prediction`

**Step 2: Start services**

- `docker compose up --build`

**Step 3: Open the application**

- Dashboard: http://127.0.0.1:5500/index.html
- API Docs: http://127.0.0.1:8000/docs
- Health Check: http://127.0.0.1:8000/health

**Step 4: Stop services**

- `docker compose down`

---

## 4. Dashboard Guide

### 4.1 Overview Page

**Purpose**

- View high-level customer metrics
- Understand overall churn risk
- Inspect Recency and Engagement distributions

**Main sections**

- Total Customers
- Churn Rate
- Low Engagement count
- Average Customer Value
- Key Insights
- Distribution Charts

**Notes**

- Metrics are loaded from the API
- Charts are loaded from real distribution data
- The prediction window is fixed as defined by the project methodology

### 4.2 Customer Lookup Page

**Purpose**

- Check churn risk for one customer

**How to use**

1. Enter a Customer ID
2. Click Get Prediction
3. Review the result

**Output includes**

- Churn Probability
- Prediction label
- Risk Level
- Explanation message

If the Customer ID is not found, the system shows an error message.

### 4.3 Churn Leaderboard Page

**Purpose**

- View customers with the highest estimated churn risk

**How to use**

1. Open the Leaderboard page
2. Review ranked customers
3. Use the search box to filter by Customer ID

**Displayed fields**

- Customer ID
- Recency
- Frequency
- Monetary value
- Engagement Score
- Churn label
- Risk Score

### 4.4 About Page

**Purpose**

- Provide project summary
- Explain platform capabilities
- List the technology stack

---

## 5. API Usage Guide

### Health Check

- Endpoint: `GET /health`
- Purpose: confirm API is running

### Overview Metrics

- Endpoint: `GET /metrics`
- Purpose: supply dashboard KPI values

### Chart Distributions

- Endpoint: `GET /charts/distributions`
- Purpose: supply Recency and Engagement chart data

### Customer Prediction by ID

- Endpoint: `GET /customer/{customer_id}`
- Purpose: return churn prediction for one customer

### Leaderboard

- Endpoint: `GET /leaderboard`
- Purpose: return high-risk customers

### Single Prediction

- Endpoint: `POST /predict`
- Purpose: predict using manually provided feature values

### Batch Prediction

- Endpoint: `POST /predict/batch`
- Purpose: predict for multiple customers in one request

Interactive API documentation is available at:

- http://127.0.0.1:8000/docs

---

## 6. Troubleshooting

### Dashboard loads but metrics do not appear

- Check whether the API is running on port 8000
- Open http://127.0.0.1:8000/health

### Customer Lookup fails

- Confirm the API is running
- Confirm the Customer ID exists in the processed feature table
- Check browser console for connection errors

### Docker command fails

- Confirm Docker Desktop is running
- Confirm Dockerfile is a file, not a folder
- Confirm requirements.txt exists in the project root

### Port already in use

- Stop previous API/frontend terminals
- Or stop Docker containers using `docker compose down`

---

## 7. Important Project Notes

- The system uses a point-in-time churn labeling approach
- Dashboard values are connected to live API responses
- The shown prediction window follows the project-defined methodology
- Docker is provided for reproducible local deployment

---

## 8. Quick Reference

- Dashboard: http://127.0.0.1:5500/index.html
- API Docs: http://127.0.0.1:8000/docs
- Health Check: http://127.0.0.1:8000/health

---

## 9. Author

**Robin Dsilva**  
Internship Project – Customer Behavior Prediction Platform