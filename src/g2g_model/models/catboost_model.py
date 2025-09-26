"""
CatBoost Model for G2G Prediction

Professional implementation of CatBoost regressor with hyperparameter tuning,
cross-validation, and model persistence capabilities.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import time

logger = logging.getLogger(__name__)


class G2GCatBoostModel:
    """
    Professional CatBoost model implementation for G2G score prediction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, random_state: int = 42):
        """
        Initialize the CatBoost model
        
        Args:
            config: Configuration dictionary containing model parameters
            random_state: Random seed for reproducibility
        """
        self.config = config or self._get_default_config()
        self.random_state = random_state
        
        # Model components
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.categorical_features = None
        
        # Training history
        self.training_history = {}
        self.cv_scores = {}
        self.best_params = {}
        
        logger.info("G2G CatBoost Model initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for CatBoost model"""
        return {
            'hyperparameters': {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_strength': 1,
                'bagging_temperature': 1,
                'border_count': 254,
                'bootstrap_type': 'Bayesian',
                'eval_metric': 'RMSE',
                'early_stopping_rounds': 100,
                'verbose': 100,
                'random_seed': 42,
                'loss_function': 'RMSE'
            },
            'cross_validation': {
                'cv_folds': 5,
                'scoring': 'neg_mean_squared_error',
                'shuffle': True
            },
            'hyperparameter_tuning': {
                'method': 'random',  # 'grid' or 'random'
                'n_iter': 50,
                'param_space': {
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'depth': [4, 5, 6, 7, 8],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'iterations': [500, 750, 1000, 1250, 1500]
                }
            }
        }
    
    def _create_catboost_pool(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                             feature_names: Optional[List[str]] = None,
                             categorical_features: Optional[List[int]] = None) -> Pool:
        """
        Create CatBoost Pool object
        
        Args:
            X: Feature matrix
            y: Target vector (optional for prediction)
            feature_names: List of feature names
            categorical_features: List of categorical feature indices
            
        Returns:
            CatBoost Pool object
        """
        return Pool(
            data=X,
            label=y,
            feature_names=feature_names,
            cat_features=categorical_features
        )
    
    def _identify_categorical_features(self, feature_names: List[str]) -> List[int]:
        """
        Identify categorical feature indices based on feature names
        Note: Since we're using preprocessed data where categorical features are 
        already label-encoded, we don't specify them as categorical to CatBoost
        to avoid conflicts.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            List of categorical feature indices (empty for preprocessed data)
        """
        # For preprocessed data with label-encoded categoricals, 
        # we don't specify categorical features to CatBoost
        categorical_indices = []
        
        # Uncomment below if you want to let CatBoost handle raw categorical features
        # categorical_keywords = [
        #     'risk_category', 'region', 'business_type', 
        #     'regulatory_status', 'compliance_level'
        # ]
        # 
        # for i, name in enumerate(feature_names):
        #     if any(keyword in name.lower() for keyword in categorical_keywords):
        #         categorical_indices.append(i)
        
        logger.info(f"Using {len(categorical_indices)} categorical features (preprocessed data)")
        return categorical_indices
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> 'G2GCatBoostModel':
        """
        Train the CatBoost model
        
        Args:
            X: Training feature matrix
            y: Training target vector
            feature_names: List of feature names
            X_val: Validation feature matrix (optional)
            y_val: Validation target vector (optional)
            
        Returns:
            Self for method chaining
        """
        logger.info("Training G2G CatBoost model...")
        start_time = time.time()
        
        # Store feature information
        self.feature_names = feature_names
        self.categorical_features = self._identify_categorical_features(feature_names)
        
        # Create training pool
        train_pool = self._create_catboost_pool(
            X, y, feature_names, self.categorical_features
        )
        
        # Create validation pool if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = self._create_catboost_pool(
                X_val, y_val, feature_names, self.categorical_features
            )
        
        # Initialize model with configuration
        hyperparams = self.config['hyperparameters'].copy()
        hyperparams['random_seed'] = self.random_state
        
        self.model = CatBoostRegressor(**hyperparams)
        
        # Train model
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            use_best_model=True if eval_set is not None else False,
            plot=False
        )
        
        # Store training history
        training_time = time.time() - start_time
        self.training_history = {
            'training_time': training_time,
            'iterations_completed': self.model.get_best_iteration() or hyperparams['iterations'],
            'final_train_score': self.model.get_best_score()
        }
        
        self.is_fitted = True
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create prediction pool
        pred_pool = self._create_catboost_pool(
            X, feature_names=self.feature_names, 
            categorical_features=self.categorical_features
        )
        
        predictions = self.model.predict(pred_pool)
        
        # Ensure predictions are within valid G2G score range [0, 1]
        predictions = np.clip(predictions, 0.0, 1.0)
        
        return predictions
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            Cross-validation results dictionary
        """
        logger.info("Performing cross-validation...")
        
        # Store feature information
        self.feature_names = feature_names
        self.categorical_features = self._identify_categorical_features(feature_names)
        
        # Initialize model
        hyperparams = self.config['hyperparameters'].copy()
        hyperparams['random_seed'] = self.random_state
        hyperparams['verbose'] = False  # Reduce verbosity for CV
        
        model = CatBoostRegressor(**hyperparams)
        
        # Create pool for cross-validation
        cv_pool = self._create_catboost_pool(
            X, y, feature_names, self.categorical_features
        )
        
        # Perform cross-validation
        cv_config = self.config['cross_validation']
        
        # CatBoost cv method
        cv_results = model.cv(
            cv_pool,
            fold_count=cv_config['cv_folds'],
            shuffle=cv_config['shuffle'],
            partition_random_seed=self.random_state,
            plot=False,
            verbose=False
        )
        
        # Extract final scores
        test_scores = cv_results['test-RMSE-mean'].iloc[-1]
        test_std = cv_results['test-RMSE-std'].iloc[-1]
        
        self.cv_scores = {
            'mean_rmse': test_scores,
            'std_rmse': test_std,
            'mean_r2': 1 - (test_scores ** 2),  # Approximation
            'cv_results': cv_results
        }
        
        logger.info(f"Cross-validation RMSE: {test_scores:.4f} (+/- {test_std * 2:.4f})")
        
        return self.cv_scores
    
    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Tune hyperparameters using GridSearchCV or RandomizedSearchCV
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            Best parameters and scores
        """
        logger.info("Starting hyperparameter tuning...")
        start_time = time.time()
        
        # Store feature information
        self.feature_names = feature_names
        self.categorical_features = self._identify_categorical_features(feature_names)
        
        # Prepare base model
        base_hyperparams = self.config['hyperparameters'].copy()
        base_hyperparams['random_seed'] = self.random_state
        base_hyperparams['verbose'] = False
        
        # Remove parameters that will be tuned
        tuning_config = self.config['hyperparameter_tuning']
        param_space = tuning_config['param_space']
        
        for param in param_space.keys():
            if param in base_hyperparams:
                del base_hyperparams[param]
        
        model = CatBoostRegressor(**base_hyperparams)
        
        # Prepare data - convert categorical features to appropriate format
        X_tuning = X.copy()
        
        # Perform hyperparameter search
        cv_config = self.config['cross_validation']
        
        if tuning_config['method'] == 'grid':
            search = GridSearchCV(
                model, param_space,
                cv=cv_config['cv_folds'],
                scoring=cv_config['scoring'],
                n_jobs=-1,
                verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                model, param_space,
                n_iter=tuning_config['n_iter'],
                cv=cv_config['cv_folds'],
                scoring=cv_config['scoring'],
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
        
        # Fit the search
        search.fit(X_tuning, y)
        
        # Store best parameters and results
        self.best_params = search.best_params_
        tuning_time = time.time() - start_time
        
        tuning_results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'cv_results': search.cv_results_,
            'tuning_time': tuning_time
        }
        
        logger.info(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        # Update model with best parameters
        best_hyperparams = {**base_hyperparams, **search.best_params_}
        self.model = CatBoostRegressor(**best_hyperparams)
        
        return tuning_results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-8))) * 100
        }
        
        logger.info("Model evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'PredictionValuesChange') -> Dict[str, float]:
        """
        Get feature importance from trained model
        
        Args:
            importance_type: Type of importance ('PredictionValuesChange' or 'LossFunctionChange')
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        # Get importance values
        importance_values = self.model.get_feature_importance(type=importance_type)
        
        # Map to feature names
        feature_importance = dict(zip(self.feature_names, importance_values))
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None) -> None:
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to plot
            save_path: Path to save the plot (optional)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to plot feature importance")
        
        # Get feature importance
        importance = self.get_feature_importance()
        
        # Select top features
        top_features = dict(list(importance.items())[:top_n])
        
        # Create plot
        plt.figure(figsize=(10, 8))
        features = list(top_features.keys())
        values = list(top_features.values())
        
        plt.barh(range(len(features)), values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - G2G CatBoost Model')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Save trained model and metadata
        
        Args:
            model_path: Path to save the model
            metadata_path: Path to save metadata (optional)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CatBoost model
        self.model.save_model(str(model_path))
        
        # Save metadata
        if metadata_path is None:
            metadata_path = model_path.with_suffix('.json')
        
        # Stamp environment versions for validation
        try:
            import platform
            import numpy as _np
            import sklearn as _sk
            import catboost as _cb
            versions = {
                'python': platform.python_version(),
                'numpy': getattr(_np, '__version__', 'unknown'),
                'sklearn': getattr(_sk, '__version__', 'unknown'),
                'catboost': getattr(_cb, '__version__', 'unknown'),
            }
        except Exception:
            versions = {}

        metadata = {
            'config': self.config,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'training_history': self.training_history,
            'cv_scores': self.cv_scores,
            'best_params': self.best_params,
            'model_type': 'CatBoostRegressor',
            'random_state': self.random_state,
            'versions': versions,
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, model_path: str, metadata_path: Optional[str] = None) -> 'G2GCatBoostModel':
        """
        Load trained model and metadata
        
        Args:
            model_path: Path to the saved model
            metadata_path: Path to metadata file (optional)
            
        Returns:
            Loaded model instance
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load metadata
        if metadata_path is None:
            metadata_path = model_path.with_suffix('.json')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            config=metadata['config'],
            random_state=metadata['random_state']
        )
        
        # Load CatBoost model
        instance.model = CatBoostRegressor()
        instance.model.load_model(str(model_path))
        
        # Restore metadata
        instance.feature_names = metadata['feature_names']
        instance.categorical_features = metadata['categorical_features']
        instance.training_history = metadata['training_history']
        instance.cv_scores = metadata['cv_scores']
        instance.best_params = metadata['best_params']
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {model_path}")
        
        return instance


if __name__ == "__main__":
    print("Run training via scripts/train_and_save_model.py")
