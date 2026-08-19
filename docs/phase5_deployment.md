# Phase 5 – Deployment Documentation

**Project:** Customer Behavior Prediction Platform  
**Last Updated:** 19 August 2026

---

## 1. Objective

Deploy the Customer Behavior Prediction Platform as a usable system with:

- FastAPI backend for local/real-time prediction
- Custom frontend dashboard
- Docker-based local packaging
- Public free hosting for demo access without requiring the developer PC to stay online

---

## 2. Backend – FastAPI

### Endpoints Implemented

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Service root |
| `/health` | GET | Health check |
| `/model-info` | GET | Model metadata |
| `/metrics` | GET | Overview metrics |
| `/charts/distributions` | GET | Recency and Engagement distributions |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch prediction |
| `/customer/{customer_id}` | GET | Customer lookup and prediction |
| `/leaderboard` | GET | High-risk customer leaderboard |

### Notes
- The API uses the engineered feature table from earlier phases
- RandomForestClassifier is used for serving predictions

---

## 3. Frontend Dashboard

### Pages Delivered

| Page | File | Description |
|------|------|-------------|
| Overview | `frontend/index.html` | Metrics, insights, distribution charts |
| Customer Lookup | `frontend/customer.html` | Predict churn by Customer ID |
| Churn Leaderboard | `frontend/leaderboard.html` | High-risk customers with search |
| About | `frontend/about.html` | Platform summary |

---

## 4. Docker Deployment (Local Full Stack)

### Files Added
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `requirements.txt`

### Services

| Service | Container Name | Port |
|---------|----------------|------|
| API | `cbpp-api` | 8000 |
| Frontend | `cbpp-frontend` | 5500 |

### Run Command
```bash
docker compose up --build
```

### Access URLs
- Frontend: http://127.0.0.1:5500/index.html
- API Docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

---

## 5. Public Deployment – Firebase Static Mode

### Constraint
- No paid hosting services
- Public HTTPS required
- Must work without developer PC being online

### Solution
Export API responses into static JSON files and host frontend + data on Firebase Hosting.

### Static Data Files
- `frontend/data/metrics.json`
- `frontend/data/charts.json`
- `frontend/data/leaderboard.json`
- `frontend/data/customers.json`

### Export Scripts
- `export_static_data.py`
- `export_charts.py`

### Production Behavior
- Overview metrics load from `metrics.json`
- Charts load from `charts.json`
- Customer Lookup loads from `customers.json`
- Leaderboard loads from `leaderboard.json`

### Public URL
- https://customer-behavior-pred.web.app

### Refresh Production Data
1. Start local API: `uvicorn api.main:app --host 127.0.0.1 --port 8000`
2. Run:
   - `python export_static_data.py`
   - `python export_charts.py`
3. Deploy:
   - `firebase deploy --only hosting`

---

## 6. Verification

| Check | Result |
|------|--------|
| API health endpoint | Working |
| API docs | Working |
| Frontend dashboard (local) | Working |
| Real metrics | Working |
| Real charts | Working |
| Customer lookup | Working |
| Leaderboard and search | Working |
| Docker Compose | Working |
| Public Firebase site | Working |
| Works without local PC | Working |

---

## 7. Status

Phase 5 deployment deliverables are complete for:
- Local API + frontend
- Docker packaging
- Public Firebase static production deployment