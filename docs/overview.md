# G2G Model Overview

The G2G (Good-to-Go) Model evaluates business readiness and compliance using a CatBoost regressor with rich feature engineering and SHAP-based explainability.

- Purpose: production-grade scoring service for operational decisioning.
- Inputs: numerical, categorical, and text features.
- Outputs: a score in [0,1] with confidence and optional explanations.

Key capabilities:
- Advanced preprocessing (TF‑IDF, label encoding, scaling, outlier capping)
- CatBoost model with CV, tuning, persistence
- SHAP explainability (global and local)
- FastAPI service with batch and health endpoints

See `architecture.md` for structure and `api.md` for endpoints.

