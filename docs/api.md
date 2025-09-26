# API Usage

Base URL: `http://localhost:8000`

Endpoints:
- `GET /health`: service and model status
- `POST /predict`: single prediction
- `POST /explain`: prediction + SHAP explanation
- `POST /batch_predict`: multiple cases; optional explanations
- `GET /model/info`: model metadata

Run locally:
- `uvicorn src.g2g_model.api.main:app --reload --port 8000`

Request model (abbrev):
- Categorical: `risk_category`, `region`, `business_type`, `regulatory_status`, `compliance_level`
- Numerical: `revenue`, `employee_count`, `years_in_business`, `credit_score`, ...
- Text: `description`

See README for curl examples.

