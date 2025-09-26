"""
G2G Model Evaluator with SHAP Explainability

Provides comprehensive model evaluation metrics and SHAP-based explanations
for individual predictions and global model behavior.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import shap
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import json
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    explained_variance_score, max_error
)

logger = logging.getLogger(__name__)


class G2GModelEvaluator:
    """
    Comprehensive evaluator for G2G models with SHAP explainability
    """
    
    def __init__(self, model, feature_names: List[str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator
        
        Args:
            model: Trained G2G model (must have predict method)
            feature_names: List of feature names
            config: Configuration dictionary for explainability settings
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config or self._get_default_config()
        
        # SHAP explainer
        self.explainer = None
        self.shap_values_cache = {}
        
        # Evaluation results storage
        self.evaluation_results = {}
        
        logger.info("G2G Model Evaluator initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'shap_explainer': 'tree',  # 'tree', 'kernel', 'linear', 'deep'
            'max_display': 10,
            'save_explanations': True,
            'explanation_path': 'data/processed/explanations',
            'background_samples': 100,  # For kernel explainer
            'check_additivity': False
        }
    
    def _initialize_shap_explainer(self, X_background: np.ndarray) -> None:
        """
        Initialize SHAP explainer based on configuration
        
        Args:
            X_background: Background dataset for explainer initialization
        """
        explainer_type = self.config.get('shap_explainer', 'tree')
        
        if explainer_type == 'tree':
            # For tree-based models like CatBoost
            try:
                self.explainer = shap.TreeExplainer(self.model.model)
                logger.info("TreeExplainer initialized successfully")
            except Exception as e:
                logger.warning(f"TreeExplainer failed: {e}. Falling back to Explainer")
                self.explainer = shap.Explainer(self.model.predict, X_background[:100])
        
        elif explainer_type == 'kernel':
            # For any model
            background_size = min(self.config.get('background_samples', 100), len(X_background))
            background_sample = X_background[:background_size]
            self.explainer = shap.KernelExplainer(self.model.predict, background_sample)
            logger.info(f"KernelExplainer initialized with {background_size} background samples")
        
        else:
            # Default: use generic explainer
            self.explainer = shap.Explainer(self.model.predict, X_background[:100])
            logger.info("Generic Explainer initialized successfully")
    
    def evaluate_regression_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Calculating regression evaluation metrics...")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'explained_variance': explained_variance_score(y_test, y_pred),
            'max_error': max_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-8))) * 100,
            'median_ae': np.median(np.abs(y_test - y_pred)),
            'std_residual': np.std(y_test - y_pred)
        }
        
        # Additional G2G-specific metrics
        # Accuracy within tolerance bands
        tolerances = [0.05, 0.1, 0.15, 0.2]
        for tol in tolerances:
            within_tolerance = np.abs(y_test - y_pred) <= tol
            metrics[f'accuracy_within_{tol}'] = np.mean(within_tolerance) * 100
        
        # Log metrics
        logger.info("Model evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        self.evaluation_results['regression_metrics'] = metrics
        return metrics
    
    def get_shap_values(self, X: np.ndarray, max_evals: int = 1000) -> np.ndarray:
        """
        Calculate SHAP values for given data
        
        Args:
            X: Input data
            max_evals: Maximum evaluations for kernel explainer
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized. Call evaluate_with_shap first.")
        
        logger.info(f"Calculating SHAP values for {len(X)} samples...")
        
        # Calculate SHAP values
        if isinstance(self.explainer, shap.KernelExplainer):
            shap_values = self.explainer.shap_values(X, nsamples=min(max_evals, 100))
        else:
            shap_values = self.explainer.shap_values(X)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # For multi-output models, take the first output
            shap_values = shap_values[0]
        
        logger.info(f"SHAP values calculated with shape: {shap_values.shape}")
        return shap_values
    
    def explain_prediction(self, X_instance: np.ndarray, 
                         instance_id: Optional[str] = None,
                         return_dict: bool = True) -> Dict[str, Any]:
        """
        Explain individual prediction with SHAP values
        
        Args:
            X_instance: Single instance to explain (1D array)
            instance_id: Optional identifier for the instance
            return_dict: Whether to return explanation as dictionary
            
        Returns:
            Dictionary containing explanation details
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")
        
        # Ensure X_instance is 2D
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Get prediction
        prediction = self.model.predict(X_instance)[0]
        
        # Get SHAP values
        shap_values = self.get_shap_values(X_instance)[0]  # Take first (and only) instance
        
        # Calculate base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[0] if len(expected_value) > 0 else 0
        else:
            expected_value = 0
        
        # Align lengths defensively
        names = self.feature_names or []
        if len(names) != len(shap_values):
            names = [f"feature_{i}" for i in range(len(shap_values))]

        feature_values = X_instance[0]
        if len(feature_values) != len(shap_values):
            # Truncate or pad feature_values to match shap_values
            fv = np.zeros_like(shap_values, dtype=float)
            n = min(len(feature_values), len(shap_values))
            fv[:n] = feature_values[:n]
            feature_values = fv

        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': names,
            'feature_value': feature_values,
            'shap_value': shap_values,
            'abs_shap_value': np.abs(shap_values)
        }).sort_values('abs_shap_value', ascending=False)
        
        # Select top 3 positive/negative features
        top_pos_df = feature_importance_df[feature_importance_df['shap_value'] > 0].head(3)
        top_neg_df = feature_importance_df[feature_importance_df['shap_value'] < 0].head(3)

        # Limit contributions to these six items to reduce payload size
        limited_contrib_df = pd.concat([top_pos_df, top_neg_df]).sort_values('abs_shap_value', ascending=False)

        explanation = {
            'instance_id': instance_id or 'unknown',
            'prediction': prediction,
            'expected_value': expected_value,
            'feature_contributions': limited_contrib_df.to_dict('records'),
            'top_positive_features': top_pos_df.to_dict('records'),
            'top_negative_features': top_neg_df.to_dict('records'),
            'explanation_summary': self._generate_explanation_text(feature_importance_df, prediction)
        }
        
        return explanation
    
    def _generate_explanation_text(self, feature_df: pd.DataFrame, prediction: float) -> str:
        """
        Generate human-readable explanation text
        
        Args:
            feature_df: DataFrame with feature contributions
            prediction: Model prediction
            
        Returns:
            Human-readable explanation string
        """
        # Get top positive and negative contributors
        top_positive = feature_df[feature_df['shap_value'] > 0].head(3)
        top_negative = feature_df[feature_df['shap_value'] < 0].head(3)
        
        explanation_parts = []
        
        # Prediction summary
        risk_level = "High" if prediction > 0.7 else "Medium" if prediction > 0.4 else "Low"
        explanation_parts.append(f"G2G Score: {prediction:.3f} ({risk_level} confidence)")
        
        # Top positive contributors
        if len(top_positive) > 0:
            pos_features = ", ".join(top_positive['feature'].head(2))
            explanation_parts.append(f"Main positive factors: {pos_features}")
        
        # Top negative contributors
        if len(top_negative) > 0:
            neg_features = ", ".join(top_negative['feature'].head(2))
            explanation_parts.append(f"Main risk factors: {neg_features}")
        
        return "; ".join(explanation_parts)
    
    def evaluate_with_shap(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive evaluation with SHAP explanations
        
        Args:
            X_train: Training features (used for SHAP background)
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Starting comprehensive evaluation with SHAP...")
        
        # Initialize SHAP explainer
        self._initialize_shap_explainer(X_train)
        
        # Calculate regression metrics
        regression_metrics = self.evaluate_regression_metrics(X_test, y_test)
        
        # Calculate SHAP values for test set
        test_shap_values = self.get_shap_values(X_test)
        
        # Global feature importance from SHAP
        global_importance = np.abs(test_shap_values).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': global_importance
        }).sort_values('importance', ascending=False)
        
        # Feature importance analysis
        importance_analysis = {
            'top_10_features': feature_importance_df.head(10).to_dict('records'),
            'feature_importance_scores': dict(zip(self.feature_names, global_importance))
        }
        
        # Compile results
        results = {
            'regression_metrics': regression_metrics,
            'shap_analysis': {
                'global_importance': importance_analysis,
                'shap_values_shape': test_shap_values.shape,
                'feature_names': self.feature_names
            }
        }
        
        # Store SHAP values for later use
        self.shap_values_cache['test'] = {
            'shap_values': test_shap_values,
            'X_test': X_test,
            'y_test': y_test
        }
        
        self.evaluation_results = results
        logger.info("Comprehensive evaluation completed")
        
        return results
