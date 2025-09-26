# Architecture

```
repo/
├─ src/g2g_model/
│  ├─ data/            # Synthetic data generation
│  ├─ preprocessing/    # Preprocessor (text/cat/num)
│  ├─ models/          # CatBoost wrapper
│  ├─ evaluation/      # Metrics + SHAP
│  ├─ api/             # FastAPI app
│  └─ config.py        # Typed config loader
├─ scripts/            # Train and API test utilities
├─ config/             # YAML configuration
├─ models/             # Saved artifacts
├─ tests/              # Pytest suite
├─ docs/               # Documentation
└─ Dockerfile, docker-compose.yml
```

Core flows:
- Training: `data_generator -> preprocessor.fit_transform -> model.fit -> evaluator`.
- Serving: request -> `preprocessor.transform -> model.predict -> (optional SHAP)`.

Notable design:
- Preprocessed categoricals (label-encoded) keep CatBoost simple
- Text via TF‑IDF limited to max 1000 features for performance
- SHAP TreeExplainer preferred; falls back to generic when needed
- Strict Pydantic input validation in API

