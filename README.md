# G2G Model — Good‑to‑Go Scoring System

Production‑grade ML system for evaluating business readiness and compliance, built on CatBoost with SHAP explainability and a FastAPI service for online and batch inference. This README guides you end‑to‑end: what every folder/file is for, why each dependency is used, how data flows through the system, how to train, test, deploy, monitor, and troubleshoot.

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
- src/g2g_model/api: “the web server.” Receives JSON, validates it, calls the model.
- src/g2g_model/preprocessing: “the kitchen.” Cleans data, vectorizes text, scales numbers.
- src/g2g_model/models: “the chef.” Trains CatBoost and knows how to save/load it.
- src/g2g_model/evaluation: “the food critic.” Scores the chef and explains decisions (SHAP).
- scripts: “utilities.” Train the model, validate artifacts, test the API, and start in Docker.
- models: “the fridge.” Where trained models/preprocessors live (mounted into the container).
- config: “the recipe.” Single YAML for feature lists, preprocessing, and model hyperparams.
- deploy/k8s: “the restaurant.” How to run this reliably in Kubernetes.
- docs: Deep dives for later; README is your primary map.

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
