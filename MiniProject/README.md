# StreamPulse — AI Music Stream Forecasting & Hit Potential Engine

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com)
[![Tests](https://img.shields.io/badge/Tests-8%20Passed-brightgreen.svg)](https://pytest.org)

> **Individual Mini-Project:** Feature Engineering & MLOps  
> **Author:** Sakshi Kore  
> **Instructor:** Murshid Khan  
> **Product Name:** StreamPulse Music Analytics Platform  

---

## 🎧 Executive Summary & Product Vision

In the modern music streaming industry, record labels, artists, and A&R executives invest substantial capital into production and marketing campaigns without reliable forecasts of streaming traction. 

**StreamPulse** is an end-to-end machine learning microservice that bridges this gap. It takes song acoustic attributes (tempo, energy, danceability, acousticness, key, mode) and cross-platform playlist reach (Spotify, Apple Music, Deezer) to predict **total lifetime Spotify streams** and classify the track into actionable commercial hit tiers.

---

## 🏗️ System Architecture

```
[ User Input / Producers ]
            │
            ▼
┌─────────────────────────┐
│ Streamlit Web Dashboard │ (Port 8501)
│ (Interactive Sliders)   │
└────────────┬────────────┘
             │ HTTP REST (JSON)
             ▼
┌─────────────────────────┐
│   FastAPI Backend API   │ (Port 8000)
│   (Pydantic Validation) │
└────────────┬────────────┘
             │
             ▼
┌───────────────────────────────────────────────────────────┐
│                Scikit-Learn ML Pipeline                   │
│                                                           │
│  [ Raw Data ] ──► [ SpotifyFeatureCreator ] (features.py) │
│                          │                                │
│                          ▼                                │
│       [ ColumnTransformer: StandardScaler + OHE ]         │
│                          │                                │
│                          ▼                                │
│          [ Regularized Ridge Regressor (α=10) ]           │
│                          │                                │
│                          ▼                                │
│           [ log1p Inversion: expm1(pred) ]                │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
                   [ Predicted Streams ]
               [ Hit Potential Category ]
```

---

## 📂 Project Organization

```
MiniProject/
├── data/
│   └── raw/
│       ├── generate_data.py                    # Verification and cleaning utility
│       └── spotify_2023.csv                    # Cleaned Spotify dataset
├── notebooks/
│   └── spotify_hit_pipeline.ipynb              # Executed end-to-end Jupyter notebook
├── features.py                                 # Scikit-learn compliant FeatureCreator
├── train.py                                    # Model training script with MLflow logging
├── app/
│   ├── __init__.py
│   └── main.py                                 # FastAPI microservice (/health, /predict)
├── streamlit_app.py                            # Interactive Spotify-themed Streamlit UI
├── models/
│   └── spotify_pipeline.joblib                 # Serialized production pipeline artifact
├── tests/
│   ├── test_pipeline.py                        # Pipeline unit tests (range, directionality)
│   └── test_api.py                             # API endpoint & schema validation tests
├── Dockerfile                                  # Multi-service production containerfile
├── entrypoint.sh                               # Container supervisor script
├── dvc.yaml                                    # DVC data and training pipeline stages
├── .github/
│   └── workflows/
│       └── docker-build.yml                    # Automated CI/CD test and build workflow
├── requirements.txt                            # Pinned Python dependencies
└── README.md                                   # Comprehensive documentation
```

---

## 🧠 Feature Engineering Decisions & Justifications

Every technique employed in StreamPulse addresses a specific real-world characteristic of music streaming data:

| Feature / Technique | Problem in Raw Data | Transformation Applied | Why It's Better Than Alternatives |
|---|---|---|---|
| **Target Log Transform** | Streams span 1M to 3.7B+ (severe right-skew). | $y = \log(1 + \text{streams})$ | Linear models assume homoscedastic errors. $\log(1+x)$ stabilizes variance without clipping extreme hits. |
| **Track Age Feature** | Release year is an absolute calendar label. | `2023 - released_year` | Older tracks have had more years to accumulate streams. Age turns year into a cumulative time exposure feature. |
| **Cyclical Month Encoding** | Plain integers 1–12 treat Dec (12) and Jan (1) as maximally distant. | $\sin(2\pi m/12)$, $\cos(2\pi m/12)$ | Preserves the continuous annual cycle of music release timing (summer anthems vs holiday releases). |
| **Playlist Aggregation** | Platform inclusions are fragmented across services. | $\log(1 + \sum \text{playlists})$ | Total reach represents aggregate playlist exposure; $\log(1+x)$ prevents mega-tracks from having outsized leverage. |
| **Party Energy Synergy** | Danceability alone without energy doesn't indicate a club hit. | $(\text{danceability} \times \text{energy}) / 100$ | Captures the synergistic effect between high rhythm and high acoustic intensity. |
| **Musical Key Imputation** | Missing keys in 10% of tracks. | Filled with `'Unknown'` + One-Hot Encoding (`handle_unknown='ignore'`) | Avoids dropping valid tracks while preventing unseen keys from crashing inference at runtime. |

---

## 🛠️ Step-by-Step Execution Guide

### 1. Environment Setup
```bash
cd MiniProject
pip install -r requirements.txt
```

### 2. Verify Dataset
```bash
python data/raw/generate_data.py
```

### 3. Train Model with MLflow Tracking
```bash
python train.py
```
*Output: Logs parameters and metrics ($R^2 \approx 0.61$, RMSE $\approx 0.63$) to MLflow and exports `models/spotify_pipeline.joblib`.*

### 4. Run Automated Test Suite
```bash
pytest tests/ -v
```
*Runs 8 tests covering model loading, prediction sanity, directional consistency, unknown categories, and FastAPI endpoints.*

### 5. Launch the FastAPI Microservice
```bash
uvicorn app.main:app --reload --port 8000
```
- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### 6. Launch the Streamlit Web Dashboard
```bash
streamlit run streamlit_app.py
```
- Dashboard URL: [http://localhost:8501](http://localhost:8501)

### 7. Run with Docker
```bash
docker build -t streampulse-api:latest .
docker run -p 8000:8000 -p 8501:8501 streampulse-api:latest
```

---

## 📊 The "Rule That Matters Most" — Justification Matrix

| Tool / Choice | Why we used it for this problem | Why it is better than the obvious alternative |
|---|---|---|
| **Ridge Regression** | Multi-platform playlist metrics are collinear. $L_2$ penalty stabilizes coefficients. | **vs OLS / Random Forest:** OLS overfits to collinear features. Random forest produces opaque trees that lose interpretable per-feature marginal impact. |
| **FastAPI** | Asynchronous execution, high throughput, and automatic Pydantic request validation. | **vs Flask:** Flask requires manual payload parsing and validation; FastAPI automatically generates Swagger documentation. |
| **Streamlit** | Rapid, stateful UI in Python that shares models and data structures with backend pipelines. | **vs Plain HTML/JS:** Allows live slider-based prototyping and metric visualizers without building a separate frontend build pipeline. |
| **MLflow** | Centralized, code-driven tracking of hyperparameters ($\alpha$) and evaluation metrics ($R^2$, RMSE). | **vs Manual Spreadsheets:** Eliminates human recording error and saves the exact model binary associated with each run. |
| **DVC** | Versions raw data and ties pipeline execution to git commit hashes. | **vs Committing CSV to Git:** Prevents git repositories from becoming bloated with binary datasets. |
| **Docker Multi-Stage Container** | Packages backend, frontend, models, and dependencies into an identical runtime environment. | **vs Local Virtualenv:** Eliminates "it works on my machine" issues when sharing code or deploying to cloud servers. |

