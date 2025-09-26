from fastapi.testclient import TestClient
from typing import Any

from src.g2g_model.api import main as api_main
from src.g2g_model.preprocessing.preprocessor import G2GPreprocessor
from src.g2g_model.data.data_generator import G2GDataGenerator


def _bootstrap_lightweight_components():
    # Fit a minimal preprocessor so .transform works
    gen = G2GDataGenerator(random_state=10)
    df = gen.generate_data(n_samples=20)
    pre = G2GPreprocessor()
    pre.fit(df)

    class MiniModel:
        is_fitted = True
        config = {'hyperparameters': {}}
        training_history = {}

        def predict(self, X):
            import numpy as np
            return np.full((X.shape[0],), 0.5, dtype=float)

    api_main.model = MiniModel()
    api_main.preprocessor = pre
    api_main.evaluator = None
    api_main.feature_names = pre.text_feature_names + pre.config['categorical_features'] + pre.config['numerical_features']


def test_health_and_predict_endpoints():
    # Patch load_model to avoid heavy training in tests
    async def stub_load_model():
        _bootstrap_lightweight_components()

    api_main.load_model = stub_load_model  # type: ignore

    with TestClient(api_main.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["model_loaded"] is True

        sample = {
            "gid": "test_case_1",
            "risk_category": "Medium",
            "region": "North America",
            "business_type": "Technology",
            "regulatory_status": "Compliant",
            "compliance_level": "Good",
            "revenue": 1000000,
            "employee_count": 20,
            "years_in_business": 5,
            "credit_score": 700,
            "debt_to_equity_ratio": 0.3,
            "liquidity_ratio": 1.2,
            "market_share": 0.1,
            "growth_rate": 0.05,
            "profitability_margin": 0.12,
            "customer_satisfaction": 80,
            "operational_efficiency": 75,
            "regulatory_violations": 0,
            "audit_score": 88,
            "description": "Solid technology company with good compliance."
        }

        pr = client.post("/predict", json=sample)
        assert pr.status_code == 200
        pred_body = pr.json()
        assert pred_body["gid"] == sample["gid"]
        assert 0.0 <= pred_body["g2g_score"] <= 1.0

