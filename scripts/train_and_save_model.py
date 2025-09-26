#!/usr/bin/env python3
"""
Train and Save G2G Model Script

This script trains the G2G model and saves it for use by the FastAPI application.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from g2g_model.data.data_generator import G2GDataGenerator
from g2g_model.preprocessing.preprocessor import G2GPreprocessor
from g2g_model.models.catboost_model import G2GCatBoostModel
from g2g_model.evaluation.evaluator import G2GModelEvaluator
import os
import mlflow
import mlflow.sklearn
from g2g_model.storage import db as g2gdb


def main():
    print("=== G2G Model Training and Saving Script ===")
    
    # Option A: load training data from local DB if enabled and present
    use_db = os.getenv("USE_DB_TRAIN", "0").lower() in {"1", "true", "yes"}
    db_path = os.getenv("G2G_DB_PATH", str(project_root / "data" / "g2g.db"))
    if use_db and Path(db_path).exists():
        print("\n1. Loading training data from local DB...")
        g2gdb.init_db(db_path)
        df = g2gdb.fetch_training_dataframe(db_path)
        if df.empty:
            print("Local DB is empty; falling back to synthetic generation.")
            use_db = False
        else:
            print(f"Loaded {len(df)} training rows from DB")
    if not use_db:
        # Generate synthetic data
        print("\n1. Generating synthetic data...")
        generator = G2GDataGenerator(random_state=42)
        df = generator.generate_data(n_samples=5000)
        print(f"Generated {len(df)} samples with {len(df.columns)} features")
        # Persist to DB for future training reuse
        g2gdb.init_db(db_path)
        cases = df.drop(columns=["g2g_score"]).to_dict(orient="records")
        targets = df["g2g_score"].tolist()
        inserted = g2gdb.insert_train_cases(db_path, cases, targets)
        print(f"Saved {inserted} rows to {db_path}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = G2GPreprocessor()
    X, feature_names = preprocessor.fit_transform(df)
    y = preprocessor.get_target(df)
    print(f"Preprocessed data shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Configure MLflow local tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", str(project_root / "mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "g2g-training"))

    # Train model
    print("\n3. Training CatBoost model...")
    config = {
        'hyperparameters': {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_strength': 1,
            'bagging_temperature': 1,
            'bootstrap_type': 'Bayesian',
            'eval_metric': 'RMSE',
            'early_stopping_rounds': 50,
            'verbose': 100
        }
    }
    
    with mlflow.start_run(run_name="train_catboost"):
        model = G2GCatBoostModel(config=config)
        model.fit(X_train, y_train, feature_names, X_test, y_test)
        # Log params and metrics
        for k, v in config['hyperparameters'].items():
            mlflow.log_param(f"catboost_{k}", v)
        eval_metrics = model.evaluate(X_test, y_test)
        for k, v in eval_metrics.items():
            mlflow.log_metric(k, float(v))
        # Log feature count
        mlflow.log_param("n_features", len(feature_names))
    
    # Evaluate model
    print("\n4. Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print(f"Test R² Score: {metrics['r2']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:.4f}")
    
    # Create evaluator and run SHAP analysis
    print("\n5. Creating evaluator with SHAP...")
    evaluator = G2GModelEvaluator(model, feature_names)
    results = evaluator.evaluate_with_shap(X_train, y_train, X_test, y_test)
    
    print("\nTop 10 most important features (SHAP):")
    top_features = results['shap_analysis']['global_importance']['top_10_features']
    for i, feature in enumerate(top_features, 1):
        print(f"{i:2d}. {feature['feature']}: {feature['importance']:.4f}")
    
    # Save model and preprocessor
    print("\n6. Saving model and preprocessor...")
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "g2g_model.pkl"
    preprocessor_path = models_dir / "preprocessor.pkl"
    
    # Save model
    model.save(str(model_path))
    
    # Save preprocessor
    preprocessor.save(str(preprocessor_path))
    # Log artifacts to MLflow
    mlflow.log_artifact(str(model_path))
    mlflow.log_artifact(str(preprocessor_path))
    
    print(f"Model saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    
    # Test loading
    print("\n7. Testing model loading...")
    loaded_model = G2GCatBoostModel.load(str(model_path))
    loaded_preprocessor = G2GPreprocessor.load(str(preprocessor_path))
    
    # Quick test prediction
    test_sample = X_test[:1]
    original_pred = model.predict(test_sample)[0]
    loaded_pred = loaded_model.predict(test_sample)[0]
    
    print(f"Original prediction: {original_pred:.4f}")
    print(f"Loaded model prediction: {loaded_pred:.4f}")
    print(f"Difference: {abs(original_pred - loaded_pred):.8f}")
    
    if abs(original_pred - loaded_pred) < 1e-6:
        print("✅ Model loading test passed!")
    else:
        print("❌ Model loading test failed!")
    
    print("\n=== Model training and saving completed successfully! ===")
    print("\nYou can now run the FastAPI application:")
    print("  cd src/g2g_model/api")
    print("  python main.py")
    print("\nOr use uvicorn directly:")
    print("  uvicorn src.g2g_model.api.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
