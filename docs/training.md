# Training

Script: `python scripts/train_and_save_model.py`

Workflow:
1) Generate synthetic data (configurable sample size)
2) Fit `G2GPreprocessor` and transform
3) Train `G2GCatBoostModel`
4) Evaluate with `G2GModelEvaluator` + SHAP
5) Save artifacts to `models/`

Artifacts:
- `models/g2g_model.pkl` (CatBoost model)
- `models/preprocessor.pkl` (preprocessor)
- `models/g2g_model.json` (metadata)

Tuning:
- Use `G2GCatBoostModel.tune_hyperparameters` for grid/random search

