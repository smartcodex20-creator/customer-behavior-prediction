# Phase 5 – Deployment Documentation

**Project:** Customer Behavior Prediction Platform  
**Last Updated:** 18 August 2026

---

## 1. Objective

Deploy the Customer Behavior Prediction Platform as a usable local system with:

- FastAPI backend
- Custom frontend dashboard
- Docker-based packaging

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

### Frontend Behavior
- Metrics are loaded from the API
- Charts are loaded from the API
- Customer predictions are loaded from the API
- Leaderboard data is loaded from the API

---

## 4. Docker Deployment

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

Access URLs

Frontend: http://127.0.0.1:5500/index.html
API Docs: http://127.0.0.1:8000/docs
Health: http://127.0.0.1:8000/health


## 5. Verification

Check,Result
API health endpoint,Working
API docs,Working
Frontend dashboard,Working
Real metrics,Working
Real charts,Working
Customer lookup,Working
Leaderboard and search,Working
Docker Compose,Working

## 6. Status
Phase 5 deployment deliverables are complete.