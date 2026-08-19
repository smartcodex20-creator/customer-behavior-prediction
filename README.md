# Customer Behavior Prediction Platform

AI-powered customer analytics system for churn prediction, customer value estimation, segmentation, and decision support.

Built as a complete end-to-end internship project covering data processing, machine learning, deep learning, explainability, API deployment, and dashboard delivery.

## Project Overview

This platform analyzes retail customer transaction behavior and helps identify customers who are at risk of churning.

It provides:

- Churn probability scoring
- Customer risk ranking
- Behavioral feature analysis
- Model explainability
- Interactive dashboard for business users

The system is designed as a practical decision-support tool for marketing and retention teams.

## Key Features

- End-to-end data pipeline on real retail transaction data
- Point-in-time feature engineering to avoid target leakage
- Classical ML models for churn prediction
- Deep learning models for comparison and anomaly detection
- SHAP and LIME explainability
- FastAPI backend for real-time prediction
- Custom frontend dashboard
- Docker-based local deployment

## Tech Stack

**Data and ML**

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM
- TensorFlow / Keras

**Explainability**

- SHAP
- LIME

**Backend**

- FastAPI
- Uvicorn

**Frontend**

- HTML, CSS, JavaScript
- Chart.js

**Deployment**

- Docker
- Docker Compose

## Project Structure

- `api/` - FastAPI backend
- `frontend/` - Dashboard UI
- `src/` - Data, features, models, segmentation
- `data/raw/` - Raw source data
- `data/interim/` - Cleaned intermediate data
- `data/processed/` - Final feature tables
- `models_artifacts/` - Saved model outputs
- `notebooks/` - EDA and analysis notebooks
- `docs/` - Phase documentation and user manual
- `Dockerfile` - API container build file
- `docker-compose.yml` - Multi-service local deployment
- `requirements.txt` - Python dependencies
- `README.md` - Project overview

## How to Run

### Local Development

1. Create and activate virtual environment:

- `python -m venv .venv`
- `.venv\Scripts\activate`

2. Install dependencies:

- `pip install -r requirements.txt`

3. Start API:

- `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`

4. Start Frontend:

- `cd frontend`
- `python -m http.server 5500`

5. Open in browser:

- Dashboard: http://127.0.0.1:5500/index.html
- API Docs: http://127.0.0.1:8000/docs

### Docker

1. Run:

- `docker compose up --build`

2. Open:

- Dashboard: http://127.0.0.1:5500/index.html
- API Docs: http://127.0.0.1:8000/docs
- Health Check: http://127.0.0.1:8000/health

## Main API Endpoints

- `GET /health` - Service health check
- `GET /metrics` - Overview metrics
- `GET /charts/distributions` - Chart distribution data
- `GET /customer/{id}` - Predict by Customer ID
- `GET /leaderboard` - High-risk customer list
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch prediction

## Dashboard Pages

- **Overview** - KPI cards, insights, and distribution charts
- **Customer Lookup** - Real-time prediction by Customer ID
- **Churn Leaderboard** - Ranked high-risk customers with search
- **About** - Project and technology summary

## Project Phases

- **Phase 0** - Setup and repository structure - Completed
- **Phase 1** - Data pipeline and EDA - Completed
- **Phase 2** - Feature engineering and churn labeling - Completed
- **Phase 3** - Classical ML, CLV, segmentation - Completed
- **Phase 4** - Deep learning and explainability - Completed
- **Phase 5** - API, dashboard, Docker - Completed
- **Phase 6** - Final polish and presentation - In progress

## Documentation

Detailed documentation is available in the `docs/` folder:

- Phase-wise technical documentation
- User manual for running and using the system

## Notes

- Churn labels were created using a point-in-time approach
- Dashboard metrics and charts are connected to live API responses
- API serves a saved RandomForest model artifact
- Leaderboard is ranked by model-predicted churn probability
- Docker is included for reproducible local deployment

## Author

**Robin Dsilva**  
Internship Project – Customer Behavior Prediction Platform