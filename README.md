# G2G Model — Good‑to‑Go Scoring System

Production‑grade machine‑learning system that scores business readiness/compliance with interpretability, observability, and automation built‑in. This document is a complete, single‑source guide: the why, what, and how for every folder and file, all dependencies, data flow, APIs, training, deployment, monitoring, databases, CI, and troubleshooting.

Quick start navigation
- Getting Started Cheatsheet: see the Quick Commands section → [jump to cheatsheet](#cheatsheet)
- API Docs locally: http://localhost:8000/docs (when the API is running)
- Full stack: run `docker compose up --build` (API + Prometheus + Grafana)

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd g2g-modelling

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train and save the model
python scripts/train_and_save_model.py

# Start the API server
uvicorn src.g2g_model.api.main:app --host 0.0.0.0 --port 8000
```

## 📋 Overview

The G2G (Good‑to‑Go) model evaluates organizational readiness using a mix of financial metrics, operational KPIs, compliance signals, and short business descriptions. It produces a score in [0, 1] and optional explanations for why the model predicted that score. The project is structured for clarity, testability, and deployment in real environments.

### Key Features

- **🤖 Advanced ML Pipeline**: CatBoost-based regression model with 241 engineered features
- **📊 Multi-Modal Input**: Handles numerical, categorical, and text features seamlessly  
- **🔍 SHAP Explainability**: Provides detailed explanations for every prediction
- **🚀 Production-Ready API**: FastAPI-based REST API with comprehensive validation
- **📈 Professional Architecture**: Modular, testable, and maintainable codebase
- **⚡ High Performance**: Optimized for both accuracy and inference speed

## 🏗️ Repository Structure (What and Why)

```
g2g-modelling/
├── src/g2g_model/
│   ├── api/
│   │   └── main.py            # FastAPI app (routes, schemas, startup lifecycle)
│   ├── data/
│   │   └── data_generator.py  # Synthetic tabular + text data generator
│   ├── evaluation/
│   │   └── evaluator.py       # Metrics + SHAP explainability (global/local)
│   ├── models/
│   │   └── catboost_model.py  # CatBoost wrapper: train/CV/tune/save/load
│   ├── preprocessing/
│   │   └── preprocessor.py    # TF‑IDF, label encoding, scaling, outlier capping
│   └── config.py              # Typed configuration loader (Pydantic v2)
├── scripts/
│   ├── train_and_save_model.py# Train → evaluate (SHAP) → save artifacts (with versions)
│   ├── test_api.py            # API smoke tester (hits all endpoints)
│   ├── validate_artifacts.py  # Validate artifacts vs environment (feature dims, versions)
│   └── entrypoint.sh          # Container startup: validate then start API
├── deploy/k8s/
│   ├── deployment.yaml        # K8s Deployment with liveness/readiness probes
│   └── service.yaml           # K8s Service (ClusterIP)
├── docs/                      # Topic docs (overview, API, training, ops, etc.)
├── tests/                     # Pytest unit tests (preprocessor/model/API)
├── config/
│   └── config.yaml            # Central config: features, preprocessing, model, API
├── models/                    # Artifacts produced by training (mounted in container)
│   ├── g2g_model.pkl          # CatBoost model file
│   ├── g2g_model.json         # Model metadata (incl. version stamps)
│   └── preprocessor.pkl       # Preprocessing pipeline (TF‑IDF, scalers, encoders)
├── data/                      # Local sample / generated data
│   └── raw/g2g_dataset.csv
├── .github/workflows/ci.yml   # CI: lint, type checks, tests, Docker build
├── Dockerfile                 # Python 3.11 slim; healthcheck; entrypoint start
├── docker-compose.yml         # One‑command local run; mounts models; strict validate
├── .dockerignore              # Speed up builds, keep images small
├── .env.example               # Example env vars (host/port, paths)
├── requirements.txt           # Pinned dependencies for reproducibility
└── README.md                  # You are here
```

Why this layout (beginner‑friendly)
- src/g2g_model/api: the web service boundary. Validates requests, runs preprocessing/model, returns predictions and SHAP explanations. Also exposes health, metrics, and optional admin endpoints.
- src/g2g_model/preprocessing: the feature factory. Text TF‑IDF, categorical encoders, numerical scaling/outlier capping; persisted to reuse in serving.
- src/g2g_model/models: the modeling core. CatBoost wrapper with training, CV/tuning, prediction clamping to [0,1], model persistence, and metadata stamping.
- src/g2g_model/evaluation: the explainer and critic. Regression metrics and SHAP explanations (global/importances and local/top‑3 reasons).
- src/g2g_model/storage: local persistence adapters (SQLite) for training data and inference logs.
- scripts: the toolbox. Train/evaluate/log artifacts; validate bundles; daily drift monitoring; DB exports; API smoke tests; container entrypoint.
- models: the artifact vault. Saved model (pkl + json metadata) and preprocessor (pkl); volume‑mounted in Docker for reproducibility and rollbacks.
- config: the recipe. YAML with consistent lists of features and preprocessing/model settings.
- deploy/k8s: optional manifests to run in Kubernetes with probes.
- docs: supporting notes. This README is the canonical guide.

## 🎯 Model Performance

The current model achieves:
- **R² Score**: 0.728 (72.8% variance explained)
- **RMSE**: 0.058 (highly accurate predictions)
- **Features**: 241 engineered features from 20 input variables
- **Training Time**: ~2 minutes on standard hardware

### Feature Importance (Top 10)

1. **debt_to_equity_ratio** (0.0267) - Financial leverage indicator
2. **liquidity_ratio** (0.0236) - Short-term financial health
3. **profitability_margin** (0.0189) - Business profitability
4. **regulatory_status** (0.0175) - Compliance standing
5. **growth_rate** (0.0165) - Business growth trajectory
6. **risk_category** (0.0155) - Overall risk assessment
7. **credit_score** (0.0145) - Creditworthiness indicator
8. **operational_efficiency** (0.0140) - Operational performance
9. **regulatory_violations** (0.0104) - Compliance history
10. **text_tfidf_87** (0.0098) - Text-derived insights

## 📊 Input Features

### Categorical Features (5)
- `risk_category`: Low, Medium, High, Critical
- `region`: Geographic region
- `business_type`: Industry classification
- `regulatory_status`: Current compliance status
- `compliance_level`: Overall compliance grade

### Numerical Features (13)
- `revenue`: Annual revenue
- `employee_count`: Number of employees
- `years_in_business`: Company age
- `credit_score`: Credit rating (300-850)
- `debt_to_equity_ratio`: Financial leverage
- `liquidity_ratio`: Short-term liquidity
- `market_share`: Market position (0-1)
- `growth_rate`: Revenue growth rate
- `profitability_margin`: Profit margins
- `customer_satisfaction`: Customer satisfaction score (0-100)
- `operational_efficiency`: Operational efficiency score (0-100)
- `regulatory_violations`: Number of violations
- `audit_score`: Latest audit score (0-100)

### Text Features (1)
- `description`: Business description (processed via TF-IDF)

Notes
- Numerical features are standardized and outliers capped (IQR by default).
- Categoricals are label‑encoded in preprocessing to keep CatBoost inputs consistent.
- Text features use TF‑IDF (unigrams/bigrams) capped at 1000 terms for performance.

End‑to‑end data flow (simple mental model)
- Training: data_generator → preprocessor.fit_transform → model.fit → evaluator (SHAP) → save
- Serving: JSON → preprocessor.transform → model.predict → (optional SHAP via evaluator)

Database flow (optional, local SQLite)
- Training import: USE_DB_TRAIN=1 loads train_data table into a dataframe as the training set (or seeds synthetic then saves into DB on first run).
- Inference logging: /predict and /explain write inputs + prediction into inference_logs (best‑effort; never blocks serving).
- Batch: /batch_predict does not log by default (can be enabled if required).

## 🔧 FastAPI Interface

OpenAPI Docs
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Core endpoints
- `GET /health`: liveness/readiness including model load status and version
- `GET /model/info`: model metadata (feature count, config, training history)
- `POST /predict`: single prediction
- `POST /explain`: single prediction with explanations (top 3 positive/negative)
- `POST /batch_predict`: batch predictions; can include explanations

Request/response validation
- Pydantic models enforce input types and ranges (e.g., `credit_score` 300–850).
- Categorical fields validated against allowed sets.

Health Check
```bash
curl http://localhost:8000/health
```

Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gid": "test_case_1",
    "risk_category": "Medium",
    "region": "North America",
    "business_type": "Technology",
    "regulatory_status": "Compliant",
    "compliance_level": "Good",
    "revenue": 5000000,
    "employee_count": 50,
    "years_in_business": 8,
    "credit_score": 720,
    "debt_to_equity_ratio": 0.5,
    "liquidity_ratio": 1.8,
    "market_share": 0.15,
    "growth_rate": 0.12,
    "profitability_margin": 0.15,
    "customer_satisfaction": 85,
    "operational_efficiency": 78,
    "regulatory_violations": 1,
    "audit_score": 82,
    "description": "A technology company specializing in cloud-based solutions with strong growth and good compliance record."
  }'
```

Prediction with SHAP Explanation (Top 3 ±)
```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{
    # Same payload as above
  }'
```

Batch Predictions
```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cases": [
      {
        # Multiple cases here
      }
    ],
    "include_explanations": true
  }'
```

Online vs batch usage
- Online (low latency): send a single JSON to `/predict` or `/explain` (online explains are more expensive due to SHAP).
- Batch (throughput): submit `cases` array to `/batch_predict` with `include_explanations=true` only when needed; consider chunking.

Schemas (inputs/outputs — plain English)
- Input (G2GInput):
  - Identifiers: `gid` (optional)
  - Categorical: `risk_category`, `region`, `business_type`, `regulatory_status`, `compliance_level`
  - Numerical: `revenue`, `employee_count`, `years_in_business`, `credit_score`, `debt_to_equity_ratio`, `liquidity_ratio`, `market_share`, `growth_rate`, `profitability_margin`, `customer_satisfaction`, `operational_efficiency`, `regulatory_violations`, `audit_score`
  - Text: `description`
- Output (/predict → G2GPrediction):
  - `gid`, `g2g_score` (0–1), `confidence_level` (High/Medium/Low), `prediction_timestamp`, `model_version`
- Output (/explain → G2GExplanation):
  - Everything in prediction plus `explanation_summary`, `top_positive_features` (3), `top_negative_features` (3)

Batch payload example
- A ready file is included at `tests/batch_payload.json` (include_explanations=true). Run locally:
  - `curl -X POST http://localhost:8000/batch_predict -H "Content-Type: application/json" --data @tests/batch_payload.json`
  - Or use Swagger UI and paste the file content into the body.

## 🧪 Development

### Environment Setup
```bash
# Development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Code formatting
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ scripts/ tests/
```

### Training a New Model
```bash
# Generate new synthetic data and train model
python scripts/train_and_save_model.py

# The script will:
# 1. Generate 5000 synthetic samples
# 2. Preprocess the data
# 3. Train the CatBoost model
# 4. Evaluate with SHAP
# 5. Save model artifacts
```

### Model Evaluation
```python
from g2g_model.evaluation.evaluator import G2GModelEvaluator

# Create evaluator
evaluator = G2GModelEvaluator(model, feature_names)

# Run comprehensive evaluation
results = evaluator.evaluate_with_shap(X_train, y_train, X_test, y_test)

# Generate detailed report
report_path = evaluator.generate_model_report("evaluation_results/")
```

## 🐳 Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t g2g-model:latest .

# Run container
docker run -p 8000:8000 \
  -v "$(pwd)/models:/app/models" \
  g2g-model:latest
```

Docker Compose
```bash
docker compose up --build   # or: docker-compose up --build
```

Kubernetes (optional)
- `kubectl apply -f deploy/k8s/`
- Manifests include readiness/liveness probes on `/health`.

Environment variables (serving)
- `VALIDATE_STRICT=true` to fail fast on invalid/missing artifacts at startup
- `ADMIN_ENDPOINTS=true` to enable `/admin/recent_inferences`
- `G2G_DB_PATH` to override SQLite path (default: `data/g2g.db`; in Docker `/app/data/g2g.db`)
- `PYTHONPATH=/app/src` is set in the image for imports

### Production Considerations

1. **Monitoring**: Implement model drift detection
2. **Scaling**: Use multiple worker processes
3. **Security**: Add authentication and rate limiting
4. **Logging**: Comprehensive request/response logging
5. **Model Updates**: Implement A/B testing for model versions

## 📈 Model Interpretability

The system provides three levels of interpretability:

### 1. Global Feature Importance
- SHAP-based feature importance across all predictions
- Identifies which features drive model decisions most

### 2. Local Explanations
- SHAP values for individual predictions
- Shows how each feature contributed to a specific score

### 3. Natural Language Explanations
- Human-readable summaries of predictions
- Highlights key positive and negative factors

Example explanation:
> "G2G Score: 0.742 (High confidence); Main positive factors: liquidity_ratio, profitability_margin; Main risk factors: debt_to_equity_ratio, regulatory_violations"

API explain output
- For clarity and payload size, `/explain` returns only the top 3 positive and top 3 negative features, plus a concise summary. The API also ensures feature alignment when building explanations.

## 🔍 Model Validation

### Data Quality Checks
- Input validation using Pydantic models
- Range checking for numerical features
- Category validation for categorical features
- Text preprocessing and cleaning

### Model Performance
- Cross-validation during training
- Hold-out test set evaluation
- SHAP additivity checking
- Prediction consistency tests

### Monitoring
- Input drift detection
- Output distribution monitoring
- Performance degradation alerts
- Feature importance stability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions, issues, or contributions:
- Create an issue in the repository
- Contact the development team
- Check the documentation in the `docs/` folder

---

**Built with ❤️ for professional ML deployments**
 
## 📚 Documentation

- Overview: docs/overview.md
- Architecture: docs/architecture.md
- API: docs/api.md
- Training: docs/training.md
- Testing: docs/testing.md
- Deployment: docs/deployment.md
- Operations: docs/operations.md

---

## 📦 Dependencies — What and Why

Core ML
- catboost: Gradient‑boosted decision trees optimized for categorical/tabular data; robust accuracy and speed.
- scikit-learn: Common ML utilities (metrics, splitters, model selection); also interoperable with joblib.
- numpy: Vectorized numerical computations; the base of most ML stacks.
- pandas: DataFrame operations for tabular data preparation.

Text Processing
- nltk: Tokenization/stopwords support (TF‑IDF preprocessing convenience).

Explainability
- shap: SHAP value computation for local/global model explanations.

API & Validation
- fastapi: High‑performance web framework with pydantic validation and OpenAPI docs.
- uvicorn[standard]: ASGI server; the [standard] extra includes useful uvloop/watchfiles.
- pydantic: Data validation and settings schema models (v2).
- pydantic-settings: 12‑factor style settings via env/ini files.
- python-dotenv: Load `.env` files in local/dev environments.

Data Quality & Ops
- great-expectations: Optional data validation suite if integrating with real datasets.
- loguru: Simple, powerful structured logging.
- prometheus-client: Export Prometheus metrics for monitoring.
 - mlflow: Experiment tracking; logs params/metrics/artifacts during training for auditability.
 - evidently: Data drift (and model drift if ground truth available) reporting; generates HTML/JSON reports.

Dev & Tooling
- pytest, pytest-cov: Unit tests + coverage.
- black, isort, flake8, mypy: Formatting, import sorting, linting, static typing.
- joblib: Persistence for preprocessing artifacts.
- tqdm: Progress bars during longer tasks.
- click: CLI helper if/when additional scripts are added.

Versioning Notes
- Train and serve with the same dependency set. Artifacts saved with one stack (e.g., sklearn 1.7 + numpy 2.x) may not load under older stacks. Use Dockerized training to guarantee compatibility.

Line‑by‑line (requirements.txt)
- catboost==1.2.2: core tabular model (fast, accurate, handles non‑linearities well)
- scikit-learn==1.3.2: splitters, metrics, CV/tuning scaffolding
- numpy==1.24.3: numerical backbone (vectors/matrices)
- pandas==2.1.4: tabular data manipulation
- nltk==3.8.1: text utilities (stopwords/tokenization) for TF‑IDF
- shap==0.43.0: explainability toolkit (TreeExplainer used by default)
- fastapi==0.104.1: API framework (routing, validation, OpenAPI docs)
- uvicorn[standard]==0.24.0: ASGI server (production‑grade with uvloop)
- pydantic==2.5.0: request/response schemas; configuration models
- pydantic-settings==2.1.0: environment‑based settings loader
- python-dotenv==1.0.0: loads .env in local/dev runs
- great-expectations==0.18.8: optional data quality checks (pipelines)
- loguru==0.7.2: ergonomic logging
- pytest==7.4.3 / pytest-cov==4.1.0: testing/coverage
- black==23.11.0 / isort==5.12.0 / flake8==6.1.0 / mypy==1.7.1: code quality
- prometheus-client==0.19.0: metrics export for scraping
- joblib==1.3.2: save/load preprocessing artifacts
- tqdm==4.66.1: progress bars
- click==8.1.7: simple CLIs if needed later

---

## 🔌 Using the Service: Online & Batch Inference

Online (single request)
- Use `/predict` for fastest response; `/explain` adds SHAP overhead.
- Example (Python):
```python
import requests

payload = {"gid":"test_case_1", "risk_category":"Medium", "region":"North America", "business_type":"Technology",
           "regulatory_status":"Compliant", "compliance_level":"Good", "revenue":5000000, "employee_count":50,
           "years_in_business":8, "credit_score":720, "debt_to_equity_ratio":0.5, "liquidity_ratio":1.8,
           "market_share":0.15, "growth_rate":0.12, "profitability_margin":0.15, "customer_satisfaction":85,
           "operational_efficiency":78, "regulatory_violations":1, "audit_score":82,
           "description":"A technology company specializing in cloud-based solutions with strong growth and good compliance record."}

r = requests.post("http://localhost:8000/predict", json=payload)
print(r.json())
```

Batch (many cases)
- Submit an array of `cases` to `/batch_predict`.
- Set `include_explanations` to `true` only when required; consider chunk sizes based on latency targets.

Operational tips
- Health: `GET /health` is used by Docker/K8s probes.
- Model info: `GET /model/info` to verify features, version, and training metadata.

Example batch (Python)
```python
import requests

batch = {
  "cases": [payload1, payload2, payload3],
  "include_explanations": True
}
r = requests.post("http://localhost:8000/batch_predict", json=batch)
print(r.json())
```

---

## 🧰 Troubleshooting

- Artifact version mismatch (pickle load error):
  - Symptom: warnings about TfidfTransformer version; `No module named numpy._core`.
  - Fix: retrain with the pinned requirements or use Dockerized training. The API will fall back to a demo model if artifacts fail to load.
- Explanations payload too large:
  - We already limit to top 3 ± features and return a concise `explanation_summary`.
- Slow startup due to demo training:
  - Save real artifacts to `models/` and mount into the container (`-v $(pwd)/models:/app/models`).

Bundle validator
- Run `python scripts/validate_artifacts.py` to verify that your model/preprocessor and environment are compatible.
- The Docker/Compose startup runs this validator and logs warnings if validation fails, then continues (API may auto‑train a demo model).
- Strict mode: set `VALIDATE_STRICT=true` (env var) to fail startup on validation errors (enabled in docker-compose by default).

CI
- GitHub Actions workflow `.github/workflows/ci.yml` runs lint, type checks, tests, and builds the Docker image on pushes/PRs.

---

## 🗺️ File‑by‑File Guide (Deep Dive)

src/g2g_model/api/main.py
- Declares Pydantic request/response models with validation.
- Startup lifecycle: logs artifact summary, loads artifacts (or trains demo), wires SHAP evaluator.
- Endpoints: `/health`, `/model/info`, `/predict`, `/explain`, `/batch_predict`.

src/g2g_model/preprocessing/preprocessor.py
- Fits and applies TF‑IDF (text), LabelEncoder (categoricals), StandardScaler (numericals).
- Handles outliers via IQR clip (configurable). Persists fitted components via joblib.
- Exposes `fit`, `transform`, `fit_transform`, and `get_target`.

src/g2g_model/models/catboost_model.py
- Wraps CatBoostRegressor; supports CV and randomized/grid tuning.
- Saves model file and JSON metadata (incl. library versions and training stats).
- `predict` clamps outputs to [0,1] to match G2G score range.

src/g2g_model/evaluation/evaluator.py
- Computes regression metrics (RMSE/MAE/R², etc.).
- Builds SHAP explainer (TreeExplainer by default). `/explain` returns top 3+/− only.

src/g2g_model/data/data_generator.py
- Creates realistic synthetic rows (categoricals/numericals/text) with correlated target.

src/g2g_model/config.py
- Loads `config/config.yaml` with Pydantic v2 models and validates ranges/types.

scripts/train_and_save_model.py
- End‑to‑end training: generate → preprocess → train → evaluate (SHAP) → save → load‑test.

scripts/test_api.py
- Calls all API endpoints with realistic payloads (including performance loop).

scripts/validate_artifacts.py
- Ensures model/preprocessor load and feature counts match metadata; reports version mismatches.

scripts/entrypoint.sh
- Container entrypoint: runs validation; optionally fails fast (VALIDATE_STRICT=true); launches API.

tests/
- test_preprocessor.py: shape/consistency checks.
- test_model.py: quick train/inference with clamped outputs.
- test_api.py: TestClient with stubbed lightweight model to keep tests fast.

deploy/k8s/
- deployment.yaml: container, env, readiness/liveness probes, replicas.
- service.yaml: cluster‑internal Service for the API.

docs/
- overview.md, architecture.md, api.md, training.md, testing.md, deployment.md, operations.md.
---

## 🧱 Deep Dive — Every Folder and Key File (with Rationale)

- `src/g2g_model/api/main.py`
  - Owns the FastAPI app, request/response schemas, and lifecycle.
  - Endpoints:
    - `/health` (liveness/readiness), `/model/info` (metadata), `/predict`, `/explain`, `/batch_predict`.
    - `/metrics` (Prometheus exposition) and `/admin/recent_inferences` (optional; enable via `ADMIN_ENDPOINTS`).
  - Startup:
    - Logs artifact validation summary (feature counts, versions), then loads model/preprocessor or trains a demo model.
  - Observability:
    - Exposes counters/histograms for request totals and latency for predict/explain/batch.
  - DB logging:
    - Writes `/predict` and `/explain` calls into SQLite (`inference_logs`) when `G2G_DB_PATH` is available.

- `src/g2g_model/preprocessing/preprocessor.py`
  - TF‑IDF for text (n‑grams, max features), LabelEncoder for categoricals, StandardScaler + IQR capping for numericals.
  - Persists fitted components via joblib; exposes `fit`, `transform`, `fit_transform`, and `get_target`.
  - Produces explicit `feature_names` to align with SHAP.

- `src/g2g_model/models/catboost_model.py`
  - Wraps CatBoostRegressor with configurable hyperparameters, CV/tuning, and evaluation.
  - Saves model and a JSON metadata file with training history, CV results, feature names, and environment versions (python, numpy, sklearn, catboost).
  - `predict` clamps outputs to [0, 1] to match the G2G score definition.

- `src/g2g_model/evaluation/evaluator.py`
  - Computes regression metrics (RMSE/MAE/R², etc.).
  - Builds a SHAP explainer (TreeExplainer preferred) and returns concise explanations:
    - Only top‑3 positive and top‑3 negative features with a human‑readable summary, plus a limited contribution list.

- `src/g2g_model/data/data_generator.py`
  - Generates realistic synthetic data: tabular categoricals/numericals and a contextual text description.
  - Encodes reasonable correlations with the target to produce learnable signals.

- `src/g2g_model/storage/db.py`
  - Self‑contained SQLite adapter (no extra deps) to persist:
    - `train_data` (JSON payload + optional target)
    - `inference_logs` (request payload + prediction + confidence + optional explanation)
  - Helpers to insert/fetch/export records and create tables on first use.

- `scripts/train_and_save_model.py`
  - End‑to‑end training: generate (or load DB) → preprocess → train (CatBoost) → evaluate (metrics + SHAP) → save artifacts → smoke reload test.
  - MLflow logging: logs hyperparameters, metrics, feature count, and artifacts to `./mlruns` (or `MLFLOW_TRACKING_URI`).
  - Seeds the SQLite DB (if empty) so future runs can `USE_DB_TRAIN=1` to train from persisted data.

- `scripts/validate_artifacts.py`
  - Validates presence and loadability of the model/preprocessor and checks feature dimensionality.
  - Compares versions in metadata vs runtime; prints warnings for mismatches.
  - Used by the container entrypoint and available as a manual preflight.

- `scripts/daily_batch_monitor.py`
  - Daily automation:
    - Generates `DAILY_SAMPLES` synthetic rows (default 5), saves raw/predictions to `data/monitoring/<date>/`.
    - Adds 5 new rows into `train_data` daily and logs their inferences in `inference_logs`.
    - Computes Evidently data‑drift report (HTML + JSON), compresses artifacts, and cleans up according to `RETAIN_DAYS`.

- `scripts/export_db.py`
  - Exports `train_data` and `inference_logs` to CSV/Parquet for analysis/audit.

- `scripts/test_api.py`
  - Local API smoke tests (health, model info, predict, explain, batch, small perf loop).

- `scripts/entrypoint.sh`
  - Validates artifacts on container start; if `VALIDATE_STRICT=true`, fails on validation errors; otherwise logs and continues.
  - Starts Uvicorn.

- `pages/` (GitHub Pages site)
  - `index.html`, `app.js`, `styles.css`: a static UI that embeds the latest Evidently drift report and includes a client‑side form for online inference.
  - The UI lets you set an API base URL and optional auth token (sent as `Authorization` header) and call `/predict` or `/explain` from the browser.

- `monitoring/`
  - `prometheus.yml`: scrapes `/metrics` on the API.
  - Grafana provisioning: preloads a Prometheus data source, a dashboard for RPS and p95 latency, and a sample alert rule (predict p95 > 0.5s).

- `.github/workflows/ci.yml`
  - CI pipeline: formats (isort/black), lints (flake8), runs mypy (non‑blocking), tests (pytest), builds Docker, and smokes `/health` + `/predict` in a container.

- `.github/workflows/daily-monitor.yml`
  - Daily job (02:00 UTC) to run the monitoring script, upload artifacts, email success/failure, and deploy the latest drift report + inference UI to GitHub Pages.
  - Requires SMTP secrets and enabling Pages; job is attached to the `github-pages` environment.

---

## 🗄️ Databases, Persistence, and Artifacts

- SQLite DB path: default `data/g2g.db` (override via `G2G_DB_PATH`). In Docker use a volume mount: `-v $(pwd)/data:/app/data`.
- Creation: tables are created on first insert or via helper calls; the DB appears after you run at least one `/predict` or `/explain`, training script, or the daily monitor.
- Tables:
  - `train_data(gid, payload_json, target, created_at)` — training rows (optional target for synthetic data)
  - `inference_logs(gid, payload_json, prediction, confidence, explanation_summary, created_at)` — audit of served predictions
- Export:
  - `python scripts/export_db.py --db data/g2g.db --out-dir exports --format both`
- Models and preprocessor:
  - `models/g2g_model.pkl` (CatBoost), `models/g2g_model.json` (metadata), `models/preprocessor.pkl` (pipeline)
  - Mount into Docker for consistent serving and rollbacks.

---

## 📡 Monitoring, Observability, and Alerts

- Prometheus metrics (API):
  - Counters: `g2g_predict_requests_total`, `g2g_explain_requests_total`, `g2g_batch_requests_total`
  - Histograms: `g2g_predict_latency_seconds`, `g2g_explain_latency_seconds`, `g2g_batch_latency_seconds`
  - Endpoint: `/metrics`
- Grafana:
  - Preprovisioned Prometheus data source and dashboard under folder “G2G”.
  - Sample alert: Predict p95 > 0.5s for 5 minutes.
- Evidently drift:
  - Daily drift report HTML + JSON saved to `data/monitoring/<date>/` and published to GitHub Pages.
  - Customize thresholds/rules by editing the monitoring script or adding Evidently performance metrics (requires labels).
- MLflow:
  - Training logs to `./mlruns` by default (override via `MLFLOW_TRACKING_URI`).
  - UI: `mlflow ui --backend-store-uri ./mlruns --port 5000`.

---

## ⚙️ Configuration and Environment Variables

- Serving:
  - `VALIDATE_STRICT=true` — fail on invalid/missing model artifacts at container start
  - `ADMIN_ENDPOINTS=true` — enable `/admin/recent_inferences` (for quick inspection)
  - `G2G_DB_PATH` — SQLite file path (default `data/g2g.db`)
- Training:
  - `USE_DB_TRAIN=1` — train from SQLite `train_data` (fallback to synthetic if empty)
  - `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`
- Monitoring:
  - `DAILY_SAMPLES` — number of synthetic rows for daily drift (default 5)
  - `RETAIN_DAYS` — retention for old monitoring artifacts (default 14 in CI; 30 local default)
  - `COMPRESS_REPORT=1` — compress daily folder into tar.gz
- GitHub Actions (email):
  - `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `NOTIFY_EMAIL`

---

## 🧪 Testing and CI/CD

- Local tests: `pytest`
- Lint & format: `flake8`, `black`, `isort`, `mypy` (CI runs isort/black then flake8 blocking; mypy non‑blocking).
- CI pipeline:
  - Runs on push/PR: format, lint, type‑check, tests, Docker build, and container smoke test (`/health`, `/predict`).
- Daily monitor:
  - Scheduled at 02:00 UTC; uploads artifacts, emails, and deploys Pages.

---

## 🧯 Troubleshooting (Expanded)

- No DB file in `data/`:
  - Ensure you ran `/predict` or the training/daily script. `/batch_predict` does not log by default.
  - In Docker, mount `-v $(pwd)/data:/app/data`. Otherwise the DB lives only inside the container.
- Model artifacts fail to load (pickle errors or version mismatch):
  - Use Dockerized training or recreate artifacts under the same dependency set as serving.
  - `python scripts/train_and_save_model.py` then restart the container with `-v $(pwd)/models:/app/models`.
- Healthcheck failing in Compose:
  - `docker compose logs -f g2g-api` to see why. If model missing, the API auto‑trains a small demo model.
- GitHub Pages deploy fails:
  - Enable Pages → Source: GitHub Actions; ensure the workflow has `environment: name: github-pages`.
  - Private repo on Free plan: Pages require public repo or upgraded plan.
- Grafana shows no data:
  - Ensure API is up and Prometheus is scraping `/metrics`. Visit http://localhost:9090 → Status → Targets.
- CORS blocks browser calls to the API:
  - By default we allow all (`allow_origins=['*']`). For stricter prod usage, list allowed origins explicitly.

---

## 🧭 Roadmap & Extensibility

- AuthN/Z: plug in FastAPI dependencies for tokens/roles or use a reverse proxy for auth.
- Batch DB logging: enable logging for `/batch_predict` if required by your audit policy.
- Versioned model registry: extend MLflow + GitHub Releases for promotion and rollbacks.
- Real reference data: swap synthetic reference in daily monitor for a curated dataset.

---

<a id="cheatsheet"></a>
## 🧾 Quick Commands (Cheatsheet)

- Train and save artifacts: `python scripts/train_and_save_model.py`
- Validate artifacts: `python scripts/validate_artifacts.py`
- Start API (local): `uvicorn src.g2g_model.api.main:app --host 0.0.0.0 --port 8000`
- Build + run (Docker): `docker build -t g2g-model:latest . && docker run -p 8000:8000 -v "$(pwd)/models:/app/models" -v "$(pwd)/data:/app/data" g2g-model:latest`
- Full stack (Compose): `docker compose up --build`
- Batch inference: `curl -X POST :8000/batch_predict -H 'Content-Type: application/json' --data @tests/batch_payload.json`
- DB export: `python scripts/export_db.py --db data/g2g.db --out-dir exports --format both`
